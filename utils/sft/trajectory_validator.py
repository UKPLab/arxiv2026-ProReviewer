"""
Trajectory Validator - Simulates trajectory execution and validates against gold review

This module:
1. Simulates executing a decision trajectory
2. Extracts the final review/assessment from simulated memory state
3. Compares with gold review to detect discrepancies
4. Provides detailed error reports for refinement
"""

import json
import re
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class TrajectorySimulator:
    """Simulates ReviewerR1 decision execution to build memory state."""

    def __init__(self):
        """Initialize empty memory state."""
        self.claims = {}  # claim_id -> claim_dict
        self.questions = {}  # question_id -> question_dict
        self.assessments = {}  # aspect -> assessment_dict
        self.outline = {
            "summary": "",
            "strengths": [],
            "weaknesses": [],
            "questions": []
        }
        self.sections_read = []
        self.research_requests = []
        self.final_review = None

    def execute_trajectory(self, trajectory: List[Dict]) -> Dict:
        """
        Execute a trajectory and return final memory state.

        Args:
            trajectory: List of decisions (each with memory_operations and action)

        Returns:
            Final memory state dict with claims, outline, assessments, etc.
        """
        for i, decision in enumerate(trajectory):
            try:
                self._execute_decision(decision, step=i)
            except Exception as e:
                print(f"Warning: Error executing decision {i}: {e}")
                continue

        return self._get_memory_state()

    def _execute_decision(self, decision: Dict, step: int):
        """Execute a single decision (memory operations + action)."""
        # Execute memory operations
        for op in decision.get("memory_operations", []):
            self._execute_memory_op(op)

        # Execute action
        action = decision.get("action", {})
        self._execute_action(action, step)

    def _execute_memory_op(self, op: Dict):
        """Execute a memory operation."""
        op_type = op.get("op")
        args = op.get("args", {})

        if op_type == "add_claim":
            claim_id = args.get("claim_id")
            if claim_id:
                self.claims[claim_id] = {
                    "text": args.get("claim_text", ""),
                    "source_section": args.get("source_section", ""),
                    "issues": args.get("issues", []),
                    "status": "pending",  # Default status
                    "reasoning": "",
                    "evidence": []
                }

        elif op_type == "update_claim_status":
            claim_id = args.get("claim_id")
            if claim_id in self.claims:
                self.claims[claim_id]["status"] = args.get("status", "pending")
                self.claims[claim_id]["reasoning"] = args.get("reasoning", "")
                self.claims[claim_id]["evidence"] = args.get("evidence", [])

        elif op_type == "add_question":
            question_id = args.get("question_id")
            if question_id:
                self.questions[question_id] = {
                    "text": args.get("question_text", ""),
                    "source_section": args.get("source_section", ""),
                    "resolved": False,
                    "answer": "",
                    "evidence": []
                }

        elif op_type == "resolve_question":
            question_id = args.get("question_id")
            if question_id in self.questions:
                self.questions[question_id]["resolved"] = True
                self.questions[question_id]["answer"] = args.get("answer", "")
                self.questions[question_id]["evidence"] = args.get("evidence", [])

        elif op_type == "update_assessment":
            aspect = args.get("aspect")
            if aspect:
                self.assessments[aspect] = {
                    "score": args.get("score", 3),
                    "reasoning": args.get("reasoning", "")
                }

        elif op_type == "update_outline":
            section = args.get("section")
            content = args.get("content", "")

            if section == "summary":
                self.outline["summary"] = content
            elif section == "strengths":
                if isinstance(content, list):
                    self.outline["strengths"].extend(content)
                else:
                    self.outline["strengths"].append(content)
            elif section == "weaknesses":
                if isinstance(content, list):
                    self.outline["weaknesses"].extend(content)
                else:
                    self.outline["weaknesses"].append(content)
            elif section == "questions":
                if isinstance(content, list):
                    self.outline["questions"].extend(content)
                else:
                    self.outline["questions"].append(content)

    def _execute_action(self, action: Dict, step: int):
        """Execute an external action."""
        action_name = action.get("name")
        args = action.get("args", {})

        if action_name == "read_section":
            section = args.get("section_name", "")
            if section:
                self.sections_read.append(section)

        elif action_name == "research":
            claim_id = args.get("claim_id")
            if claim_id:
                self.research_requests.append({
                    "claim_id": claim_id,
                    "focus": args.get("investigation_focus", ""),
                    "step": step
                })

        elif action_name == "write_review":
            # Mark that final review was written
            self.final_review = {
                "summary": self.outline.get("summary", ""),
                "strengths": self.outline.get("strengths", []),
                "weaknesses": self.outline.get("weaknesses", []),
                "questions": self.outline.get("questions", []),
                "scores": {
                    aspect: data["score"]
                    for aspect, data in self.assessments.items()
                }
            }

    def _get_memory_state(self) -> Dict:
        """Get current memory state."""
        return {
            "claims": self.claims,
            "questions": self.questions,
            "assessments": self.assessments,
            "outline": self.outline,
            "sections_read": self.sections_read,
            "research_requests": self.research_requests,
            "final_review": self.final_review or {
                "summary": self.outline.get("summary", ""),
                "strengths": self.outline.get("strengths", []),
                "weaknesses": self.outline.get("weaknesses", []),
                "questions": self.outline.get("questions", []),
                "scores": {
                    aspect: data["score"]
                    for aspect, data in self.assessments.items()
                }
            }
        }


class TrajectoryValidator:
    """Validates trajectory output against gold review."""

    def __init__(self):
        self.simulator = TrajectorySimulator()

    def validate(self, trajectory: List[Dict], gold_review: Dict) -> Tuple[bool, List[str]]:
        """
        Validate trajectory against gold review.

        Args:
            trajectory: List of decisions
            gold_review: Parsed gold review from review_parser

        Returns:
            (is_valid, discrepancies_list)
        """
        # Simulate trajectory execution
        memory_state = self.simulator.execute_trajectory(trajectory)
        final_output = memory_state["final_review"]

        # Compare with gold review
        discrepancies = []

        # 1. Check claims/strengths coverage
        strength_discrepancies = self._check_strengths_coverage(
            memory_state, gold_review
        )
        discrepancies.extend(strength_discrepancies)

        # 2. Check weaknesses coverage
        weakness_discrepancies = self._check_weaknesses_coverage(
            memory_state, gold_review
        )
        discrepancies.extend(weakness_discrepancies)

        # 3. Check scores alignment
        score_discrepancies = self._check_scores_alignment(
            memory_state, gold_review
        )
        discrepancies.extend(score_discrepancies)

        # 4. Check summary similarity
        summary_discrepancies = self._check_summary_similarity(
            memory_state, gold_review
        )
        discrepancies.extend(summary_discrepancies)

        # 5. Check trajectory quality (skepticism, judgment pattern)
        quality_discrepancies = self._check_trajectory_quality(
            trajectory, memory_state
        )
        discrepancies.extend(quality_discrepancies)

        is_valid = len(discrepancies) == 0
        return is_valid, discrepancies

    def _check_strengths_coverage(self, memory_state: Dict, gold_review: Dict) -> List[str]:
        """Check if trajectory captures all strengths from gold review."""
        discrepancies = []

        gold_claims = gold_review.get("claims", [])
        trajectory_strengths = memory_state["outline"].get("strengths", [])
        strong_claims = [
            c for c in memory_state["claims"].values()
            if c.get("status") == "strong"
        ]

        # Check count
        expected_count = len(gold_claims)
        actual_count = len(trajectory_strengths) + len(strong_claims)

        if actual_count < expected_count - 2:
            discrepancies.append(
                f"Missing strengths: expected ~{expected_count}, found {actual_count}"
            )

        # Check coverage (fuzzy match)
        gold_claim_texts = [c["text"].lower() for c in gold_claims]
        trajectory_strength_texts = [s.lower() if isinstance(s, str) else str(s).lower()
                                     for s in trajectory_strengths]

        uncovered = []
        for gold_text in gold_claim_texts:
            # Check if any trajectory strength contains key terms from gold claim
            key_terms = self._extract_key_terms(gold_text)
            matched = any(
                any(term in traj_text for term in key_terms)
                for traj_text in trajectory_strength_texts
            )
            if not matched:
                uncovered.append(gold_text[:60])

        if uncovered and len(uncovered) > len(gold_claim_texts) * 0.3:
            discrepancies.append(
                f"Uncovered strengths: {len(uncovered)}/{len(gold_claim_texts)} not mentioned"
            )

        return discrepancies

    def _check_weaknesses_coverage(self, memory_state: Dict, gold_review: Dict) -> List[str]:
        """Check if trajectory captures all weaknesses from gold review."""
        discrepancies = []

        gold_issues = gold_review.get("issues", [])
        trajectory_weaknesses = memory_state["outline"].get("weaknesses", [])
        weak_claims = [
            c for c in memory_state["claims"].values()
            if c.get("status") == "weak"
        ]

        # Check count
        expected_count = len(gold_issues)
        actual_count = len(trajectory_weaknesses) + len(weak_claims)

        if actual_count < expected_count - 2:
            discrepancies.append(
                f"Missing weaknesses: expected ~{expected_count}, found {actual_count}"
            )

        # Check coverage (fuzzy match)
        gold_issue_texts = [i["text"].lower() for i in gold_issues]
        trajectory_weakness_texts = [w.lower() if isinstance(w, str) else str(w).lower()
                                     for w in trajectory_weaknesses]

        uncovered = []
        for gold_text in gold_issue_texts:
            key_terms = self._extract_key_terms(gold_text)
            matched = any(
                any(term in traj_text for term in key_terms)
                for traj_text in trajectory_weakness_texts
            )
            if not matched:
                uncovered.append(gold_text[:60])

        if uncovered and len(uncovered) > len(gold_issue_texts) * 0.3:
            discrepancies.append(
                f"Uncovered weaknesses: {len(uncovered)}/{len(gold_issue_texts)} not mentioned"
            )

        return discrepancies

    def _check_scores_alignment(self, memory_state: Dict, gold_review: Dict) -> List[str]:
        """Check if final scores align with gold review."""
        discrepancies = []

        gold_scores = gold_review.get("scores", {})
        trajectory_scores = memory_state["final_review"].get("scores", {})

        for aspect in ["soundness", "contribution", "presentation"]:
            if aspect in gold_scores:
                gold_score = gold_scores[aspect]
                trajectory_score = trajectory_scores.get(aspect, 0)

                # Allow ±1 difference
                if abs(gold_score - trajectory_score) > 1:
                    discrepancies.append(
                        f"Score mismatch for {aspect}: gold={gold_score}, trajectory={trajectory_score}"
                    )

        return discrepancies

    def _check_summary_similarity(self, memory_state: Dict, gold_review: Dict) -> List[str]:
        """Check if summary is reasonably similar to gold review summary."""
        discrepancies = []

        gold_summary = gold_review.get("sections", {}).get("summary", "").lower()
        trajectory_summary = memory_state["outline"].get("summary", "").lower()

        if not trajectory_summary and gold_summary:
            discrepancies.append("Missing summary in trajectory outline")
            return discrepancies

        # Check if key terms from gold summary appear in trajectory summary
        gold_terms = self._extract_key_terms(gold_summary)
        trajectory_terms = set(trajectory_summary.split())

        overlap = sum(1 for term in gold_terms if term in trajectory_terms)
        coverage = overlap / len(gold_terms) if gold_terms else 0

        if coverage < 0.3 and len(gold_summary) > 50:
            discrepancies.append(
                f"Summary has low overlap with gold review ({coverage:.1%} key terms)"
            )

        return discrepancies

    def _check_trajectory_quality(self, trajectory: List[Dict], memory_state: Dict) -> List[str]:
        """Check trajectory quality (skepticism, judgment pattern)."""
        discrepancies = []

        # Count patterns
        claims_with_issues = sum(
            1 for c in memory_state["claims"].values()
            if c.get("issues") and len(c["issues"]) > 0
        )
        total_claims = len(memory_state["claims"])

        research_count = len(memory_state["research_requests"])
        judgment_count = sum(
            1 for c in memory_state["claims"].values()
            if c.get("status") in ["strong", "weak", "partial"]
        )

        # Check skeptical reading
        if total_claims > 0:
            skepticism_rate = claims_with_issues / total_claims
            if skepticism_rate < 0.4:
                discrepancies.append(
                    f"Low skepticism: only {claims_with_issues}/{total_claims} claims have issues flagged"
                )

        # Check 2-turn judgment pattern
        if research_count > 0 and judgment_count < research_count * 0.7:
            discrepancies.append(
                f"Incomplete judgment pattern: {research_count} research but only {judgment_count} judgments"
            )

        # Check trajectory length
        if len(trajectory) < 10:
            discrepancies.append(f"Trajectory too short: {len(trajectory)} decisions")
        elif len(trajectory) > 35:
            discrepancies.append(f"Trajectory too long: {len(trajectory)} decisions")

        return discrepancies

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text (simple approach: remove common words)."""
        # Remove common words
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "this", "that", "these", "those",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "should", "could", "may", "might",
            "i", "you", "he", "she", "it", "we", "they",
            "paper", "method", "approach", "work", "study"
        }

        # Tokenize and filter
        words = re.findall(r'\w+', text.lower())
        key_terms = [w for w in words if w not in common_words and len(w) > 3]

        return key_terms


def compare_trajectories(trajectory: List[Dict], gold_review: Dict) -> Tuple[bool, List[str], Dict]:
    """
    Convenience function to validate a trajectory.

    Args:
        trajectory: List of decisions
        gold_review: Parsed gold review

    Returns:
        (is_valid, discrepancies, memory_state)
    """
    validator = TrajectoryValidator()
    is_valid, discrepancies = validator.validate(trajectory, gold_review)
    memory_state = validator.simulator._get_memory_state()

    return is_valid, discrepancies, memory_state


# Example usage
if __name__ == "__main__":
    print("Trajectory Validator Module")
    print("=" * 60)
    print("\nThis module simulates trajectory execution and validates against gold reviews.")
    print("\nKey classes:")
    print("  - TrajectorySimulator: Simulates decision execution")
    print("  - TrajectoryValidator: Validates trajectory output")
    print("\nKey functions:")
    print("  - validate(): Main validation function")
    print("  - compare_trajectories(): Convenience function")
    print("\nUsage: Import this module in scripts/generate_sft_data.py")
