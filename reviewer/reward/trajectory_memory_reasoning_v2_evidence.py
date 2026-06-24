"""Evidence-based trajectory memory reasoning reward.

This variant produces per-item scores for fine-grained credit assignment:
- factual_correctness: per-step hallucination penalties
- technical_depth: per-outline-item (W only) scores
- outline_grounding: per-outline-item (S/W only) scores

combined = mean of 3 dimensions (for fallback/logging)

Returns details dict with per-step and per-item scores for evidence-based credit.
"""

import asyncio
import json
import logging
import re
import warnings
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIGURABLE DIMENSIONS
# ---------------------------------------------------------------------------
# Comment out any dimension you want to disable. At least one must be active.
#
# Available dimensions:
# - "technical_depth": Evaluates quality/depth of weaknesses (W items only)
#                      Rewards high-quality critiques (1-5 scale normalized to 0-1)
#
# - "outline_grounding": Evaluates whether S/W items are grounded in memory
#                        Rewards well-cited, evidence-backed items (1-5 scale -> 0-1)
#
# - "factual_simple": Evaluates factual correctness against paper content (W/Q items)
#                     Penalty-only: correct=0, hallucinated=-1 (5->0, 4->-0.25, ..., 1->-1)
#
# Examples:
#   - All three: ["technical_depth", "outline_grounding", "factual_simple"]
#   - Quality only: ["technical_depth"]
#   - Grounding only: ["outline_grounding"]
#   - No penalties: ["technical_depth", "outline_grounding"]
# ---------------------------------------------------------------------------

ACTIVE_DIMENSIONS = [
    "technical_depth",      # Per-item quality scores for weaknesses (W only)
    # "outline_grounding",    # Per-item grounding scores for strengths/weaknesses (S/W)
    # "factual_simple",       # Per-item factual correctness (paper-based, penalty-only)
    "grounding",            # Per-item grounding scores for weaknesses (W only, uses paper content)
]

# Default weights for each dimension (will be normalized to sum to 1.0)
# Only weights for active dimensions will be used.
DIMENSION_WEIGHTS = {
    "technical_depth": 0.50,
    "outline_grounding": 0.25,
    "factual_simple": 0.25,
    "grounding": 0.50,
}

# Import trajectory summarization helpers from v2
from .trajectory_memory_reasoning_v2 import (
    _build_trajectory_summary,
    _trunc,
)

# Import evidence-based prompts
from .trajectory_memory_reasoning_v2_prompt_evidence import (
    TRAJECTORY_JUDGE_SYSTEM_PROMPT_EVIDENCE,
    TRAJECTORY_JUDGE_USER_PROMPT_EVIDENCE,
    TECHNICAL_DEPTH_USER_PROMPT_EVIDENCE,
    GROUNDING_USER_PROMPT_EVIDENCE,
    DIMS_EVIDENCE,
    get_evidence_dimension_prompt,
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_hallucinations_response(response: str) -> Tuple[Optional[List[dict]], str]:
    """Parse hallucinations list from <hallucinations>...</hallucinations> tags.

    Returns:
        (hallucinations_list, reasoning) where hallucinations_list is None on parse failure
    """
    content = response.strip()

    # Extract reasoning
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        # Fallback: extract partial content
        incomplete_match = re.search(r"<reasoning>\s*(.*?)(?:<|$)", content, re.DOTALL | re.IGNORECASE)
        reasoning = incomplete_match.group(1).strip() if incomplete_match else ""

    # Extract hallucinations array
    hall_match = re.search(r"<hallucinations>(.*?)</hallucinations>", content, re.DOTALL | re.IGNORECASE)
    if hall_match:
        hall_content = hall_match.group(1).strip()
    else:
        # Fallback: extract partial content
        incomplete_match = re.search(r"<hallucinations>\s*(.*?)(?:<reasoning>|$)", content, re.DOTALL | re.IGNORECASE)
        if incomplete_match:
            hall_content = incomplete_match.group(1).strip()
        else:
            warnings.warn(f"Failed to find <hallucinations> tag in response: {content[:200]}")
            return None, reasoning or "parse_error"

    # Try to parse JSON
    for attempt in [hall_content, re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', hall_content)]:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                # Validate entries
                validated = []
                for entry in parsed:
                    if isinstance(entry, dict) and "step" in entry:
                        validated.append({
                            "step": int(entry["step"]),
                            "evidence_id": str(entry.get("evidence_id", "")),
                            "severity": str(entry.get("severity", "major")),
                            "description": str(entry.get("description", ""))
                        })
                return validated, reasoning
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            pass

    warnings.warn(f"Failed to parse hallucinations JSON: {hall_content[:200]}")
    return None, reasoning or "parse_error"


def _parse_item_scores_response(response: str) -> Tuple[Optional[Dict[str, int]], str]:
    """Parse item scores from <item_scores>...</item_scores> tags.

    Returns:
        (scores_dict, reasoning) where scores_dict maps item_id -> score (1-5)
        scores_dict is None on parse failure
    """
    content = response.strip()

    # Extract reasoning
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        incomplete_match = re.search(r"<reasoning>\s*(.*?)(?:<|$)", content, re.DOTALL | re.IGNORECASE)
        reasoning = incomplete_match.group(1).strip() if incomplete_match else ""

    # Extract item_scores array
    scores_match = re.search(r"<item_scores>(.*?)</item_scores>", content, re.DOTALL | re.IGNORECASE)
    if scores_match:
        scores_content = scores_match.group(1).strip()
    else:
        incomplete_match = re.search(r"<item_scores>\s*(.*?)(?:<reasoning>|$)", content, re.DOTALL | re.IGNORECASE)
        if incomplete_match:
            scores_content = incomplete_match.group(1).strip()
        else:
            warnings.warn(f"Failed to find <item_scores> tag in response: {content[:200]}")
            return None, reasoning or "parse_error"

    # Try to parse JSON
    for attempt in [scores_content, re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', scores_content)]:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                scores_dict = {}
                for entry in parsed:
                    if isinstance(entry, dict) and "item_id" in entry and "score" in entry:
                        item_id = str(entry["item_id"])
                        score = int(entry["score"])
                        score = max(1, min(5, score))  # Clamp to [1, 5]
                        scores_dict[item_id] = score
                return scores_dict, reasoning
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            pass

    warnings.warn(f"Failed to parse item_scores JSON: {scores_content[:200]}")
    return None, reasoning or "parse_error"


# ---------------------------------------------------------------------------
# Memory and outline formatting helpers
# ---------------------------------------------------------------------------

def _build_memory_summary(claims: List[dict], questions: List[dict], notes: List[dict]) -> str:
    """Build concise memory summary for grounding evaluation.

    Only includes ID, text, and key fields needed to verify outline grounding.
    """
    lines = ["## Memory Records (Final State)\n"]

    # Use very large limits to preserve evidence detail for grounding checks.
    # (Still bounded to avoid pathological prompt size blow-ups.)
    claim_text_limit = 10000
    verifier_reason_limit = 10000
    question_text_limit = 10000
    answer_text_limit = 10000
    note_text_limit = 10000

    # Claims
    if claims:
        lines.append("### Claims")
        for c in claims:
            status = c.get("status", "unknown")
            text = _trunc(c.get("text", ""), claim_text_limit)
            verifier_reason = c.get("verifier_reason", "")
            verifier_str = f" — {_trunc(verifier_reason, verifier_reason_limit)}" if verifier_reason else ""
            lines.append(f"  {c.get('id', '?')} (status={status}): {text}{verifier_str}")
        lines.append("")

    # Questions
    if questions:
        lines.append("### Questions")
        for q in questions:
            status = q.get("status", "unknown")
            text = _trunc(q.get("question", ""), question_text_limit)
            answer = q.get("answer", "")
            answer_str = f" — {_trunc(answer, answer_text_limit)}" if answer else ""
            lines.append(f"  {q.get('id', '?')} (status={status}): {text}{answer_str}")
        lines.append("")

    # Notes
    if notes:
        lines.append("### Notes")
        for n in notes:
            text = _trunc(n.get("text", ""), note_text_limit)
            lines.append(f"  {n.get('id', '?')}: {text}")
        lines.append("")

    return "\n".join(lines)


def _format_outline_section(items: List[dict], prefix: str) -> str:
    """Format outline items with item_id prefix (W1, W2, ...).

    Args:
        items: List of outline item dicts
        prefix: "S", "W", or "Q"

    Returns:
        Formatted string with numbered items and their memory tags
    """
    if not items:
        return "(none)"

    lines = []
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            text = item.get("text", str(item))
            refs = []
            refs += item.get("related_claims", [])
            refs += item.get("related_questions", [])
            refs += item.get("related_notes", [])
            ref_str = f" [{', '.join(refs)}]" if refs else ""
            lines.append(f"{prefix}{i}. {text}{ref_str}")
        else:
            lines.append(f"{prefix}{i}. {str(item)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

async def compute_trajectory_memory_reasoning_reward_evidence_async(
    log_snapshot: Dict,
    llm_judge_fn,
    step_snapshots: List[Dict],
    format: str = "scirm",
    paper_content: Optional[str] = None,
    evidence_based: bool = False,
) -> Tuple[Optional[float], Dict]:
    """Compute evidence-based trajectory memory reasoning reward.

    This variant produces per-item scores for fine-grained credit assignment.

    Active dimensions are configured via ACTIVE_DIMENSIONS at the top of this file:
    - technical_depth: Score for each weakness only (per-item, reward for quality)
    - outline_grounding: Score for each strength/weakness only (per-item, reward for grounding)
    - factual_simple: Paper-based factual correctness (per-item, penalty-only)

    Args:
        log_snapshot: Final log snapshot dict
        llm_judge_fn: Async callable(system_prompt, user_prompt) -> str
        step_snapshots: List of per-step log snapshot dicts
        format: "scirm" (only format supported for evidence-based)
        paper_content: Paper content string (required for factual_simple dimension)

    Returns:
        (combined_score, details_dict) where details_dict contains:
        - technical_depth_per_item: {item_id: score_0_1} for each W (if active)
        - outline_grounding_per_item: {item_id: score_0_1} for each S/W (if active)
        - factual_simple_per_item: {item_id: penalty_score} for each W/Q (if active)
        - evidence_based: True (flag for workflow to detect)
        - trajectory_quality: combined scalar for logging
        - active_dimensions: list of dimensions that were evaluated
    """
    # Build trajectory summary
    trajectory_summary = _build_trajectory_summary(step_snapshots)

    # Extract final state stats
    claims = log_snapshot.get("claims", [])
    questions = log_snapshot.get("questions", [])
    notes = log_snapshot.get("notes", [])
    outline = log_snapshot.get("review_outline", {})
    section_visits = log_snapshot.get("section_visits", {})

    outline_strengths = outline.get("strengths", [])
    outline_weaknesses = outline.get("weaknesses", [])
    outline_questions = outline.get("questions", [])

    # If there are no strengths and no weaknesses, per-item judging is not meaningful.
    # Also fail if there are no weaknesses: technical_depth scores W-items only,
    # and outline_grounding needs at least some items to be meaningful.
    # Treat as judge_failed so upstream can skip this instance consistently.
    if not outline_weaknesses:
        missing = "no_weaknesses" if outline_strengths else "no_strengths_or_weaknesses"
        logger.warning(f"[Evidence] {missing} in final outline; skipping instance")
        return None, {
            "judge_failed": True,
            "failed_dims": ["technical_depth", "outline_grounding"],
            "individual_reasons": {
                "technical_depth": missing,
                "outline_grounding": missing,
                "factual_simple": missing,
            },
        }

    # Format outline sections
    strengths_str = _format_outline_section(outline_strengths, "S")
    weaknesses_str = _format_outline_section(outline_weaknesses, "W")
    questions_str = _format_outline_section(outline_questions, "Q")

    system_prompt = TRAJECTORY_JUDGE_SYSTEM_PROMPT_EVIDENCE

    # Evaluate each dimension separately in parallel
    async def evaluate_factual_correctness():
        """Evaluate factual correctness and return (dim, hallucinations_list, reasoning)."""
        query, criteria, examples = get_evidence_dimension_prompt("factual_correctness")

        user_prompt = TRAJECTORY_JUDGE_USER_PROMPT_EVIDENCE.format(
            query=query,
            criteria=criteria,
            examples=examples,
            trajectory_summary=trajectory_summary,
            outline_strengths=strengths_str,
            outline_weaknesses=weaknesses_str,
            outline_questions=questions_str,
            n_steps=len(step_snapshots),
            n_claims=len(claims),
            n_supported=sum(1 for c in claims if c.get("status") == "supported"),
            n_weak=sum(1 for c in claims if c.get("status") == "weak"),
            n_pending=sum(1 for c in claims if c.get("status") == "to_be_verified"),
            n_questions=len(questions),
            n_resolved=sum(1 for q in questions if q.get("status") == "resolved"),
            n_open=sum(1 for q in questions if q.get("status") == "open"),
            n_notes=len(notes),
            sections_visited=', '.join(list(section_visits.keys())[:10]) if section_visits else "(none)",
        )

        try:
            response = await llm_judge_fn(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"[Evidence] LLM call failed for factual_correctness: {e}")
            return "factual_correctness", None, f"llm_error: {e}"

        if response is None:
            logger.error(f"[Evidence] LLM returned None for factual_correctness")
            return "factual_correctness", None, "llm_returned_none"

        hallucinations_list, reasoning = _parse_hallucinations_response(response)
        return "factual_correctness", hallucinations_list, reasoning

    async def evaluate_per_item_dimension(dim):
        """Evaluate technical_depth, outline_grounding, or grounding and return (dim, scores_dict, reasoning)."""
        query, criteria, examples = get_evidence_dimension_prompt(dim)

        if dim == "technical_depth":
            # Technical depth needs paper content to verify technical engagement.
            if not paper_content:
                logger.warning("[Evidence] No paper_content provided for technical_depth, skipping")
                return dim, None, "no_paper_content"
            user_prompt = TECHNICAL_DEPTH_USER_PROMPT_EVIDENCE.format(
                query=query,
                criteria=criteria,
                examples=examples,
                paper_content=paper_content,
                outline_weaknesses=weaknesses_str,
            )
        elif dim == "grounding":
            # Grounding needs paper content to verify whether weaknesses reference specific parts.
            if not paper_content:
                logger.warning("[Evidence] No paper_content provided for grounding, skipping")
                return dim, None, "no_paper_content"
            user_prompt = GROUNDING_USER_PROMPT_EVIDENCE.format(
                query=query,
                criteria=criteria,
                examples=examples,
                paper_content=paper_content,
                outline_weaknesses=weaknesses_str,
            )
        else:
            # Outline grounding needs memory entries to verify cited details.
            context = _build_memory_summary(claims, questions, notes)
            user_prompt = TRAJECTORY_JUDGE_USER_PROMPT_EVIDENCE.format(
                query=query,
                criteria=criteria,
                examples=examples,
                trajectory_summary=context,
                outline_strengths=strengths_str,
                outline_weaknesses=weaknesses_str,
                n_steps=len(step_snapshots),
                n_claims=len(claims),
                n_supported=sum(1 for c in claims if c.get("status") == "supported"),
                n_weak=sum(1 for c in claims if c.get("status") == "weak"),
                n_pending=sum(1 for c in claims if c.get("status") == "to_be_verified"),
                n_questions=len(questions),
                n_resolved=sum(1 for q in questions if q.get("status") == "resolved"),
                n_open=sum(1 for q in questions if q.get("status") == "open"),
                n_notes=len(notes),
                sections_visited=', '.join(list(section_visits.keys())[:10]) if section_visits else "(none)",
            )

        try:
            response = await llm_judge_fn(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"[Evidence] LLM call failed for {dim}: {e}")
            return dim, None, f"llm_error: {e}"

        if response is None:
            logger.error(f"[Evidence] LLM returned None for {dim}")
            return dim, None, "llm_returned_none"

        scores_dict, reasoning = _parse_item_scores_response(response)
        return dim, scores_dict, reasoning

    async def evaluate_factual_simple():
        """Evaluate simple factual correctness (paper-based) per-item."""
        from reviewer.reward.factual_correctness_simple import compute_factual_correctness_simple_async

        # Need paper_content parameter
        if not paper_content:
            logger.warning("[Evidence] No paper_content provided for factual_simple, skipping")
            return "factual_simple", None, "no_paper_content"

        try:
            scores_dict, reasoning = await compute_factual_correctness_simple_async(
                paper_content=paper_content,
                weaknesses=outline_weaknesses,
                questions=outline_questions,
                llm_judge_fn=llm_judge_fn,
            )
            if scores_dict is None:
                return "factual_simple", None, reasoning
        except Exception as e:
            logger.error(f"[Evidence] factual_simple call failed: {e}")
            return "factual_simple", None, f"error: {e}"

        return "factual_simple", scores_dict, reasoning

    # Execute only active dimensions in parallel
    tasks = []
    if "technical_depth" in ACTIVE_DIMENSIONS:
        tasks.append(evaluate_per_item_dimension("technical_depth"))
    if "outline_grounding" in ACTIVE_DIMENSIONS:
        tasks.append(evaluate_per_item_dimension("outline_grounding"))
    if "factual_simple" in ACTIVE_DIMENSIONS:
        tasks.append(evaluate_factual_simple())
    if "grounding" in ACTIVE_DIMENSIONS:
        tasks.append(evaluate_per_item_dimension("grounding"))

    if not tasks:
        logger.error("[Evidence] No active dimensions configured in ACTIVE_DIMENSIONS")
        return None, {
            "judge_failed": True,
            "failed_dims": ["config_error"],
            "individual_reasons": {"config_error": "no_active_dimensions"},
        }

    results = await asyncio.gather(*tasks)

    # Collect results
    dim_results = {dim: (data, reason) for dim, data, reason in results}

    # Check if any dimension failed
    failed_dims = [dim for dim, (data, _) in dim_results.items() if data is None]
    if failed_dims:
        logger.error(f"[Evidence] Judge failed for dims {failed_dims}, skipping instance")
        return None, {
            "judge_failed": True,
            "failed_dims": failed_dims,
            "individual_reasons": {dim: reason for dim, (_, reason) in dim_results.items()},
        }

    # Extract per-dimension data conditionally based on active dimensions
    tech_scores = None
    ground_scores = None
    factual_simple_scores = None
    grounding_scores = None

    if "technical_depth" in dim_results:
        tech_scores, tech_reason = dim_results["technical_depth"]
    if "outline_grounding" in dim_results:
        ground_scores, ground_reason = dim_results["outline_grounding"]
    if "factual_simple" in dim_results:
        factual_simple_scores, factual_simple_reason = dim_results["factual_simple"]
    if "grounding" in dim_results:
        grounding_scores, grounding_reason = dim_results["grounding"]

    # Filter out malformed / out-of-scope item IDs from judge outputs
    tech_per_item_normalized = {}
    ground_per_item_normalized = {}
    factual_simple_per_item = {}
    grounding_per_item_normalized = {}

    if tech_scores is not None:
        # technical_depth expects W1, W2, ... only
        raw_tech_count = len(tech_scores)
        tech_scores = {
            item_id: score
            for item_id, score in tech_scores.items()
            if re.fullmatch(r"W\d+", item_id)
        }
        if len(tech_scores) < raw_tech_count:
            logger.warning(
                f"[Evidence] Dropped {raw_tech_count - len(tech_scores)} invalid technical_depth item_ids"
            )
        if not tech_scores:
            logger.error("[Evidence] No valid item_ids after filtering for technical_depth")
            return None, {
                "judge_failed": True,
                "failed_dims": ["technical_depth"],
                "individual_reasons": {"technical_depth": "no_valid_item_ids_after_filtering"},
            }
        # Normalize tech scores (1-5 -> 0-1, reward for quality)
        tech_per_item_normalized = {item_id: (score - 1) / 4.0 for item_id, score in tech_scores.items()}

    if ground_scores is not None:
        # outline_grounding expects S1/S2/... and W1/W2/... only
        raw_ground_count = len(ground_scores)
        ground_scores = {
            item_id: score
            for item_id, score in ground_scores.items()
            if re.fullmatch(r"[SW]\d+", item_id)
        }
        if len(ground_scores) < raw_ground_count:
            logger.warning(
                f"[Evidence] Dropped {raw_ground_count - len(ground_scores)} invalid outline_grounding item_ids"
            )
        if not ground_scores:
            logger.error("[Evidence] No valid item_ids after filtering for outline_grounding")
            return None, {
                "judge_failed": True,
                "failed_dims": ["outline_grounding"],
                "individual_reasons": {"outline_grounding": "no_valid_item_ids_after_filtering"},
            }
        # Normalize grounding scores (1-5 -> 0-1, reward for quality)
        ground_per_item_normalized = {item_id: (score - 1) / 4.0 for item_id, score in ground_scores.items()}

    if factual_simple_scores is not None:
        # Factual simple: penalty-only (score 5 -> 0, score 1 -> -1)
        # Being factually correct (5) = no reward, hallucinating (1) = -1.0 penalty
        factual_simple_per_item = {
            item_id: -(5 - score) / 4.0  # 5->0.0, 4->-0.25, 3->-0.5, 2->-0.75, 1->-1.0
            for item_id, score in factual_simple_scores.items()
        }
        if not factual_simple_per_item:
            logger.error("[Evidence] factual_simple returned no valid scores")
            return None, {
                "judge_failed": True,
                "failed_dims": ["factual_simple"],
                "individual_reasons": {"factual_simple": "no_valid_scores"},
            }

    if grounding_scores is not None:
        # grounding expects W1, W2, ... only (same as technical_depth)
        raw_grounding_count = len(grounding_scores)
        grounding_scores = {
            item_id: score
            for item_id, score in grounding_scores.items()
            if re.fullmatch(r"W\d+", item_id)
        }
        if len(grounding_scores) < raw_grounding_count:
            logger.warning(
                f"[Evidence] Dropped {raw_grounding_count - len(grounding_scores)} invalid grounding item_ids"
            )
        if not grounding_scores:
            logger.error("[Evidence] No valid item_ids after filtering for grounding")
            return None, {
                "judge_failed": True,
                "failed_dims": ["grounding"],
                "individual_reasons": {"grounding": "no_valid_item_ids_after_filtering"},
            }
        # Normalize grounding scores (1-5 -> 0-1, reward for quality)
        grounding_per_item_normalized = {item_id: (score - 1) / 4.0 for item_id, score in grounding_scores.items()}

    # Compute mean scores for combined scalar (only for active dimensions)
    dim_scores = {}
    if tech_per_item_normalized:
        dim_scores["technical_depth"] = sum(tech_per_item_normalized.values()) / len(tech_per_item_normalized)
    if ground_per_item_normalized:
        dim_scores["outline_grounding"] = sum(ground_per_item_normalized.values()) / len(ground_per_item_normalized)
    if factual_simple_per_item:
        dim_scores["factual_simple"] = sum(factual_simple_per_item.values()) / len(factual_simple_per_item)
    if grounding_per_item_normalized:
        dim_scores["grounding"] = sum(grounding_per_item_normalized.values()) / len(grounding_per_item_normalized)

    # Get active dimension weights and normalize them
    active_weights = {
        dim: DIMENSION_WEIGHTS[dim]
        for dim in ACTIVE_DIMENSIONS
        if dim in dim_scores  # Only include dimensions that succeeded
    }
    total_weight = sum(active_weights.values())
    if total_weight == 0:
        logger.error("[Evidence] No valid dimension scores, cannot compute trajectory quality")
        return None, {
            "judge_failed": True,
            "failed_dims": list(ACTIVE_DIMENSIONS),
            "individual_reasons": {"all": "no_valid_scores_from_any_dimension"},
        }

    # Normalize weights to sum to 1.0
    dim_weights = {dim: w / total_weight for dim, w in active_weights.items()}

    # Combined trajectory quality score:
    # Separate positive-quality dimensions from penalty-only dimensions
    positive_dims = ["technical_depth", "outline_grounding", "grounding"]
    positive_weight_sum = sum(dim_weights.get(dim, 0) for dim in positive_dims if dim in dim_scores)

    if positive_weight_sum > 0:
        # Compute positive quality base from technical_depth + outline_grounding
        positive_quality_base = sum(
            dim_scores[dim] * dim_weights[dim]
            for dim in positive_dims
            if dim in dim_scores
        ) / positive_weight_sum
    else:
        # No positive dimensions active, use neutral baseline
        positive_quality_base = 0.5

    # Add factual_simple penalty if active
    factual_penalty = 0.0
    if "factual_simple" in dim_scores:
        factual_penalty = dim_scores["factual_simple"] * dim_weights["factual_simple"]

    trajectory_quality = positive_quality_base + factual_penalty

    # Apply dimension weights to per-item scores for credit assignment.
    # This ensures dimensions are averaged (not summed) when multiple are active.
    # Each dimension gets its normalized weight (e.g., 0.5 each for 2 active dims),
    # so the total contribution from all dimensions stays within [0, 1] range.
    tech_per_item_weighted = {
        item_id: score * dim_weights.get("technical_depth", 1.0)
        for item_id, score in tech_per_item_normalized.items()
    } if tech_per_item_normalized else {}

    ground_per_item_weighted = {
        item_id: score * dim_weights.get("outline_grounding", 1.0)
        for item_id, score in ground_per_item_normalized.items()
    } if ground_per_item_normalized else {}

    factual_simple_per_item_weighted = {
        item_id: score * dim_weights.get("factual_simple", 1.0)
        for item_id, score in factual_simple_per_item.items()
    } if factual_simple_per_item else {}

    grounding_per_item_weighted = {
        item_id: score * dim_weights.get("grounding", 1.0)
        for item_id, score in grounding_per_item_normalized.items()
    } if grounding_per_item_normalized else {}

    # Build details dict with only active dimensions
    details = {
        # Evidence-based scores (for credit assignment) - now weighted by dimension
        "technical_depth_per_item": tech_per_item_weighted,
        "outline_grounding_per_item": ground_per_item_weighted,
        "factual_simple_per_item": factual_simple_per_item_weighted,
        "grounding_per_item": grounding_per_item_weighted,

        # Combined scores (for logging)
        "trajectory_quality": trajectory_quality,
        "positive_quality_base": positive_quality_base,
        "factual_penalty": factual_penalty,
        "dim_scores": dim_scores,
        "dim_weights": dim_weights,
        "active_dimensions": list(ACTIVE_DIMENSIONS),

        # Raw data
        "raw_scores": {},
        "individual_reasons": {},

        # Flag for workflow to detect evidence-based mode
        "evidence_based": evidence_based,
        "n_steps_summarised": len(step_snapshots),
    }

    # Add raw scores and reasons only for active dimensions
    if tech_scores is not None:
        details["raw_scores"]["technical_depth"] = tech_scores
        details["individual_reasons"]["technical_depth"] = tech_reason
    if ground_scores is not None:
        details["raw_scores"]["outline_grounding"] = ground_scores
        details["individual_reasons"]["outline_grounding"] = ground_reason
    if factual_simple_scores is not None:
        details["raw_scores"]["factual_simple"] = factual_simple_scores
        details["individual_reasons"]["factual_simple"] = factual_simple_reason
    if grounding_scores is not None:
        details["raw_scores"]["grounding"] = grounding_scores
        details["individual_reasons"]["grounding"] = grounding_reason

    return trajectory_quality, details
