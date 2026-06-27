"""
Trajectory Generator - Uses GPT-4 to generate ProReviewer decision trajectories from gold reviews

This module generates realistic decision sequences that lead to a given final review.
Uses iterative refinement to ensure trajectory outputs match the gold review.
"""

import json
import os
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sft.review_parser import parse_complete_review
from utils.sft.trajectory_validator import TrajectoryValidator


# Prompt templates for GPT-4
TRAJECTORY_GENERATION_PROMPT = """You are generating SFT training data for ProReviewer, a paper review agent that uses explicit judgment.

Given a paper and its final gold-standard review, generate a realistic decision trajectory showing how the agent would systematically review the paper to arrive at this final review.

## ProReviewer Decision Format

Each decision has two parts:
1. **memory_operations**: Internal reasoning (add/update claims, questions, assessments)
2. **action**: External action (read_section, research, write_review)

Example decision:
```json
{
  "memory_operations": [
    {
      "op": "add_claim",
      "args": {
        "claim_id": "C1",
        "claim_text": "Method achieves SOTA on 3 tasks",
        "source_section": "Abstract",
        "issues": ["Which 3 tasks?", "Need to verify in results"]
      }
    }
  ],
  "action": {
    "name": "read_section",
    "args": {"section_name": "Introduction"}
  }
}
```

## Memory Operations (6 types)

1. **add_claim**: Record a paper claim with skeptical issues
   - Args: claim_id, claim_text, source_section, issues (list of concerns)

2. **update_claim_status**: Judge a claim after research
   - Args: claim_id, status ("supported"/"weak"/"invalid"/"unverified"), reasoning, evidence

3. **add_question**: Record a question for clarification
   - Args: question_id, question_text, source_section

4. **resolve_question**: Answer a question after investigation
   - Args: question_id, answer, evidence

5. **update_assessment**: Update aspect assessment
   - Args: aspect ("soundness"/"contribution"/"presentation"), score (1-5), reasoning

6. **update_outline**: Build review outline progressively
   - Args: section ("summary"/"strengths"/"weaknesses"), content

## External Actions (3 types)

1. **read_section**: Read a paper section
   - Args: section_name (e.g., "Abstract", "Introduction", "Results")

2. **research**: Delegate detailed verification to research subagent
   - Args: claim_id, investigation_focus
   - Returns: research findings (status, reasoning, evidence)

3. **write_review**: Generate final review from outline
   - Args: (none)

## Skeptical Reading Pattern

CRITICAL: The agent must follow skeptical reading, not passive recording:

1. **Read**: Encounter a claim in the paper
2. **Question**: Immediately identify issues/concerns
3. **Record**: add_claim WITH issues flagged
4. **Research**: Delegate verification (research action)
5. **Judge**: Receive findings → make judgment → update_claim_status
6. **Update**: Build review outline based on judgments

## Two-Turn Judgment Flow

Every research must be followed by agent judgment (2 turns):

Turn 1: Agent delegates
```json
{
  "memory_operations": [],
  "action": {
    "name": "research",
    "args": {
      "claim_id": "C1",
      "investigation_focus": "Verify SOTA claims in results tables"
    }
  }
}
```

Turn 2: Agent receives findings and judges
```json
{
  "memory_operations": [
    {
      "op": "update_claim_status",
      "args": {
        "claim_id": "C1",
        "status": "weak",
        "reasoning": "Only 2/3 tasks achieve SOTA, third task is below baseline",
        "evidence": ["Table 2 shows Task A: SOTA", "Table 3 shows Task B: SOTA", "Table 4 shows Task C: below BERT baseline"]
      }
    }
  ],
  "action": {
    "name": "read_section",
    "args": {"section_name": "Discussion"}
  }
}
```

## Your Task

**Paper Content:**
{paper_text}

**Gold Review (Target Output):**
Summary: {review_summary}

Strengths:
{review_strengths}

Weaknesses:
{review_weaknesses}

Questions:
{review_questions}

Scores:
{review_scores}

**Generate the decision trajectory:**

Generate 15-25 decisions that:
1. Follow skeptical reading (flag issues, don't passively record)
2. Use 2-turn judgment flow (research → judge → update)
3. Read sections strategically (not just linearly)
4. Progressively build the review outline
5. Ensure final review matches the gold review above

Output format: JSON array of decisions
```json
[
  {{decision 1}},
  {{decision 2}},
  ...
]
```

IMPORTANT: The trajectory must produce a final review that matches the gold review. Ensure:
- Claims in outline match gold review strengths
- Weaknesses in outline match gold review weaknesses
- Scores match gold review scores
- Summary content aligns with gold review summary
"""


TRAJECTORY_REFINEMENT_PROMPT = """The generated trajectory has discrepancies with the gold review. Fix the trajectory to match.

**Discrepancies:**
{discrepancies}

**Original Trajectory:**
{trajectory}

**Gold Review (Target):**
Summary: {review_summary}
Strengths: {review_strengths}
Weaknesses: {review_weaknesses}
Scores: {review_scores}

**Fix the trajectory** to ensure it produces the gold review. Focus on:
1. Ensuring claims recorded match gold review strengths
2. Ensuring weaknesses flagged match gold review weaknesses
3. Ensuring scores align with gold review assessments
4. Maintaining realistic agent behavior (skeptical reading, 2-turn judgment)

Output the corrected trajectory as a JSON array of decisions.
"""


def generate_trajectory(
    paper: Dict,
    review: Dict,
    parsed_review: Dict,
    llm_helper,
    max_refinements: int = 3,
    model: str = "gpt-4"
) -> Dict:
    """
    Generate trajectory with iterative refinement.

    Args:
        paper: Paper content (dict with sections)
        review: Raw review dict with text and scores
        parsed_review: Parsed review from parse_complete_review()
        llm_helper: LLM API wrapper
        max_refinements: Maximum refinement iterations
        model: Model name to use

    Returns:
        Generated trajectory dict:
        {
            "trajectory": [...],  # List of decisions
            "metadata": {
                "refinement_iterations": int,
                "validation_passed": bool,
                "discrepancies": []
            }
        }
    """
    # Format the prompt
    prompt = _format_generation_prompt(paper, parsed_review)

    # Generate initial trajectory
    print(f"Generating initial trajectory with {model}...")
    trajectory_json = llm_helper.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    try:
        trajectory = json.loads(trajectory_json)
        if isinstance(trajectory, dict) and "decisions" in trajectory:
            trajectory = trajectory["decisions"]
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse GPT-4 output as JSON: {trajectory_json[:200]}")

    # Iterative refinement
    best_trajectory = trajectory
    best_discrepancies = []
    validator = TrajectoryValidator()

    for iteration in range(max_refinements):
        print(f"Refinement iteration {iteration + 1}/{max_refinements}...")

        # Validate trajectory using full validator
        is_valid, discrepancies = validator.validate(trajectory, parsed_review)

        if not discrepancies:
            print("✓ Trajectory validated successfully!")
            return {
                "trajectory": trajectory,
                "metadata": {
                    "refinement_iterations": iteration,
                    "validation_passed": True,
                    "discrepancies": []
                }
            }

        print(f"  Found {len(discrepancies)} discrepancies")
        best_discrepancies = discrepancies

        # Refine trajectory
        refinement_prompt = TRAJECTORY_REFINEMENT_PROMPT.format(
            discrepancies="\n".join(f"- {d}" for d in discrepancies),
            trajectory=json.dumps(trajectory, indent=2),
            review_summary=parsed_review['sections'].get('summary', ''),
            review_strengths="\n".join(f"- {c['text']}" for c in parsed_review['claims']),
            review_weaknesses="\n".join(f"- {i['text']}" for i in parsed_review['issues']),
            review_scores=json.dumps(parsed_review['scores'], indent=2)
        )

        refined_json = llm_helper.chat(
            messages=[{"role": "user", "content": refinement_prompt}],
            model=model,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        try:
            trajectory = json.loads(refined_json)
            if isinstance(trajectory, dict) and "decisions" in trajectory:
                trajectory = trajectory["decisions"]
            best_trajectory = trajectory
        except json.JSONDecodeError:
            print(f"  Warning: Failed to parse refined trajectory, using previous version")
            break

    # Return best trajectory even if not perfect
    return {
        "trajectory": best_trajectory,
        "metadata": {
            "refinement_iterations": max_refinements,
            "validation_passed": False,
            "discrepancies": best_discrepancies
        }
    }


def _format_generation_prompt(paper: Dict, parsed_review: Dict) -> str:
    """Format the trajectory generation prompt with paper and review data."""
    # Extract paper text (simplified - assumes paper dict has text content)
    paper_text = _extract_paper_text(paper)

    # Format review components
    review_summary = parsed_review['sections'].get('summary', 'N/A')
    review_strengths = "\n".join(f"- {c['text']}" for c in parsed_review['claims'])
    review_weaknesses = "\n".join(f"- {i['text']}" for i in parsed_review['issues'])
    review_questions = "\n".join(f"- {q}" for q in parsed_review['questions'])
    review_scores = json.dumps(parsed_review['scores'], indent=2)

    return TRAJECTORY_GENERATION_PROMPT.format(
        paper_text=paper_text[:3000] + "..." if len(paper_text) > 3000 else paper_text,
        review_summary=review_summary,
        review_strengths=review_strengths or "N/A",
        review_weaknesses=review_weaknesses or "N/A",
        review_questions=review_questions or "N/A",
        review_scores=review_scores
    )


def _extract_paper_text(paper: Dict) -> str:
    """Extract readable text from paper dict."""
    # Simplified extraction - adjust based on actual paper format
    if isinstance(paper, dict):
        # NEW: Handle LaTeX triplet format
        if 'latex' in paper:
            if isinstance(paper['latex'], dict) and 'concatenated_content' in paper['latex']:
                return paper['latex']['concatenated_content']
            elif isinstance(paper['latex'], str):
                return paper['latex']

        # Existing logic (unchanged)
        if 'text' in paper:
            return paper['text']
        elif 'content' in paper:
            return paper['content']
        else:
            # Combine available fields
            parts = []
            for key in ['title', 'abstract', 'introduction', 'method', 'results', 'conclusion']:
                if key in paper and paper[key]:
                    parts.append(f"## {key.title()}\n{paper[key]}")
            return "\n\n".join(parts) if parts else str(paper)
    return str(paper)


def _quick_validate(trajectory: List[Dict], parsed_review: Dict) -> List[str]:
    """
    Quick validation of trajectory against gold review.
    Returns list of discrepancies.

    Full validation is in trajectory_validator.py - this is a simplified version.
    """
    discrepancies = []

    # Check trajectory structure
    if not isinstance(trajectory, list):
        discrepancies.append("Trajectory is not a list of decisions")
        return discrepancies

    if len(trajectory) < 10:
        discrepancies.append(f"Trajectory too short ({len(trajectory)} decisions, expected 15-25)")

    if len(trajectory) > 30:
        discrepancies.append(f"Trajectory too long ({len(trajectory)} decisions, expected 15-25)")

    # Count memory operations and actions
    add_claim_ops = []
    update_claim_ops = []
    research_actions = 0
    read_actions = 0

    for i, decision in enumerate(trajectory):
        if not isinstance(decision, dict):
            discrepancies.append(f"Decision {i} is not a dict")
            continue

        if "memory_operations" not in decision:
            discrepancies.append(f"Decision {i} missing 'memory_operations'")

        if "action" not in decision:
            discrepancies.append(f"Decision {i} missing 'action'")
            continue

        # Count operations
        for op in decision.get("memory_operations", []):
            if op.get("op") == "add_claim":
                add_claim_ops.append(op)
            elif op.get("op") == "update_claim_status":
                update_claim_ops.append(op)

        # Count actions
        action = decision.get("action", {})
        if action.get("name") == "research":
            research_actions += 1
        elif action.get("name") == "read_section":
            read_actions += 1

    # Validate claims coverage
    expected_claims = len(parsed_review['claims'])
    if len(add_claim_ops) < expected_claims - 2:
        discrepancies.append(
            f"Too few claims recorded ({len(add_claim_ops)}, expected ~{expected_claims})"
        )

    # Validate 2-turn judgment pattern
    if research_actions > 0 and len(update_claim_ops) < research_actions:
        discrepancies.append(
            f"Research without judgment: {research_actions} research but only {len(update_claim_ops)} judgments"
        )

    # Validate skeptical reading
    claims_with_issues = sum(1 for op in add_claim_ops if op.get("args", {}).get("issues"))
    if claims_with_issues < len(add_claim_ops) * 0.5:
        discrepancies.append(
            f"Not enough skepticism: only {claims_with_issues}/{len(add_claim_ops)} claims have issues flagged"
        )

    return discrepancies


# Example usage
if __name__ == "__main__":
    print("Trajectory Generator Module")
    print("=" * 60)
    print("\nThis module generates ProReviewer decision trajectories from gold reviews.")
    print("\nKey functions:")
    print("  - generate_trajectory(): Main generation function with refinement")
    print("  - _format_generation_prompt(): Format GPT-4 prompts")
    print("  - _quick_validate(): Validate trajectory structure")
    print("\nUsage: Import this module in scripts/generate_sft_data.py")
