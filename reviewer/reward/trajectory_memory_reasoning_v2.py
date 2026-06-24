"""Trajectory-based memory reasoning reward V2.

Evaluates the full sequence of agent steps with IMPROVED rubric that directly
predicts final review quality. V2 focuses on CONTENT QUALITY (specificity,
technical depth, factual accuracy) rather than just process quality.

combined = trajectory_quality (no pending penalty in V2)

- trajectory_quality: LLM judge (1-5 on 5 dimensions -> weighted 0-1)
  - factual_correctness (0.25): memory matches observations
  - claim_specificity (0.25): cites equations/tables/figures
  - technical_depth (0.25): identifies technical issues not surface obs
  - cross_verification (0.15): investigation rigor
  - outline_grounding (0.10): outline reflects memory findings
"""

import asyncio
import json
import logging
import re
import warnings
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory summarisation helpers
# ---------------------------------------------------------------------------

def _trunc(text: str, limit: int = 120) -> str:
    """Truncate text to *limit* chars, adding ellipsis if needed."""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _build_step_summary(step_idx: int, prev_snap: Dict, curr_snap: Dict) -> str:
    """Build a content-rich summary for a single step by diffing two snapshots.

    Includes actual claim/question/note text so the judge can assess quality,
    not just IDs and counts.  Also includes a truncated observation snippet
    so the judge can verify memory entries are grounded in what was read.
    """
    parts = [f"Step {step_idx}:"]

    # --- Observation context: what the agent saw this step ---
    obs_ctx = curr_snap.get("_obs_ctx", {})
    action_name = obs_ctx.get("action_name", "")
    if action_name == "read_section":
        section_name = obs_ctx.get("section_name", "?")
        parts.append(f"  Action: read_section({section_name})")
    elif action_name == "search_paper":
        query = obs_ctx.get("query", "?")
        parts.append(f"  Action: search_paper(\"{query}\")")
    elif action_name == "research":
        parts.append(f"  Action: research")
    elif action_name == "finish":
        parts.append(f"  Action: finish")
    elif action_name:
        parts.append(f"  Action: {action_name}")

    # Show the full observation so the judge can verify grounding accurately.
    obs_snippet = obs_ctx.get("observation_snippet", "")
    if obs_snippet:
        parts.append(f"  Observed: {obs_snippet.strip()}")

    # --- Memory operations ---
    mem_ops = []

    # New claims
    prev_claim_ids = {c.get("id") for c in prev_snap.get("claims", [])}
    new_claims = [c for c in curr_snap.get("claims", []) if c.get("id") not in prev_claim_ids]
    for c in new_claims:
        mem_ops.append(f"    +Claim {c.get('id', '?')} (§{c.get('section', '?')}): {_trunc(c.get('text', ''), 150)}")

    # Claim status changes
    prev_claims = {c.get("id"): c for c in prev_snap.get("claims", [])}
    for c in curr_snap.get("claims", []):
        cid = c.get("id")
        if cid in prev_claims:
            old_status = prev_claims[cid].get("status")
            new_status = c.get("status")
            if old_status != new_status:
                reason = _trunc(c.get("verifier_reason") or "", 200)
                reason_str = f" — {reason}" if reason else ""
                mem_ops.append(f"    Claim {cid}: {old_status} → {new_status}{reason_str}")

    # New questions
    prev_q_ids = {q.get("id") for q in prev_snap.get("questions", [])}
    new_qs = [q for q in curr_snap.get("questions", []) if q.get("id") not in prev_q_ids]
    for q in new_qs:
        mem_ops.append(f"    +Question {q.get('id', '?')} (§{q.get('source_section', '?')}): {_trunc(q.get('question', ''), 150)}")

    # Question status changes
    prev_qs = {q.get("id"): q for q in prev_snap.get("questions", [])}
    for q in curr_snap.get("questions", []):
        qid = q.get("id")
        if qid in prev_qs:
            old_status = prev_qs[qid].get("status")
            new_status = q.get("status")
            if old_status != new_status:
                answer = _trunc(q.get("answer") or "", 200)
                answer_str = f" — {answer}" if answer else ""
                mem_ops.append(f"    Question {qid}: {old_status} → {new_status}{answer_str}")

    # New notes
    prev_note_ids = {n.get("id") for n in prev_snap.get("notes", [])}
    new_notes = [n for n in curr_snap.get("notes", []) if n.get("id") not in prev_note_ids]
    for n in new_notes:
        tags = n.get("tag", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        mem_ops.append(f"    +Note {n.get('id', '?')} (§{n.get('section', '?')}){tag_str}: {_trunc(n.get('text', ''), 150)}")

    # Outline changes
    prev_outline = prev_snap.get("review_outline", {})
    curr_outline = curr_snap.get("review_outline", {})
    for section_key in ("strengths", "weaknesses", "questions"):
        prev_items = prev_outline.get(section_key, [])
        curr_items = curr_outline.get(section_key, [])
        if len(curr_items) > len(prev_items):
            for item in curr_items[len(prev_items):]:
                text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
                refs = []
                if isinstance(item, dict):
                    refs += item.get("related_claims", [])
                    refs += item.get("related_questions", [])
                    refs += item.get("related_notes", [])
                ref_str = f" [{', '.join(refs)}]" if refs else ""
                mem_ops.append(f"    +Outline {section_key}: {_trunc(text, 150)}{ref_str}")

    # Summary change
    prev_summary = (prev_outline.get("summary") or "").strip()
    curr_summary = (curr_outline.get("summary") or "").strip()
    if curr_summary and curr_summary != prev_summary:
        mem_ops.append(f"    +Outline summary: {_trunc(curr_summary, 150)}")

    # Score change
    prev_score = prev_outline.get("overall_score")
    curr_score = curr_outline.get("overall_score")
    if curr_score is not None and curr_score != prev_score:
        mem_ops.append(f"    +Outline score: {curr_score}")

    if mem_ops:
        parts.append("  Memory_ops:")
        parts.extend(mem_ops)

    # Only return if something happened
    if len(parts) == 1:
        parts.append("  (no memory changes)")

    return "\n".join(parts)


def _build_trajectory_summary(step_snapshots: List[Dict], max_tokens: int = 55000) -> str:
    """Build a content-rich per-step trajectory summary from step snapshots.

    Each step shows:
    - What action the agent took and what it observed (full observation text)
    - What memory operations happened (claims, questions, notes, outline)
    including actual text so the judge can assess quality and grounding.

    Keeps first 4 and last 4 steps if over budget, truncates middle.
    Budget set to 120K tokens to accommodate full observations (~50-100K tokens
    for 15-25 steps).
    """
    if not step_snapshots:
        return "(no step snapshots available)"
    if len(step_snapshots) == 1:
        return "(single step — no trajectory to analyse)"

    summaries = []
    empty_snap = {"claims": [], "questions": [], "notes": [], "review_outline": {}, "section_visits": {}}
    for i in range(len(step_snapshots)):
        prev = step_snapshots[i - 1] if i > 0 else empty_snap
        summaries.append(_build_step_summary(i, prev, step_snapshots[i]))

    # Estimate tokens (~4 chars per token)
    total_chars = sum(len(s) for s in summaries)
    char_budget = max_tokens * 4

    if total_chars <= char_budget:
        return "\n".join(summaries)

    # Keep first 4 and last 4, truncate middle
    n_keep = min(4, len(summaries) // 2)
    head = summaries[:n_keep]
    tail = summaries[-n_keep:] if n_keep > 0 else []
    n_skipped = len(summaries) - len(head) - len(tail)
    return "\n".join(head + [f"  ... ({n_skipped} middle steps omitted) ..."] + tail)


# ---------------------------------------------------------------------------
# LLM judge prompt
# ---------------------------------------------------------------------------

# Import V2 prompts
from .trajectory_memory_reasoning_v2_prompt import (
    TRAJECTORY_JUDGE_SYSTEM_PROMPT_V2 as TRAJECTORY_JUDGE_SYSTEM_PROMPT,
    TRAJECTORY_JUDGE_USER_PROMPT_V2 as TRAJECTORY_JUDGE_USER_PROMPT,
    DIMS_V2 as DIMS,
    DIMENSION_WEIGHTS_V2,
)

# V1 prompts are not used in V2 - see trajectory_memory_reasoning_v2_prompt.py


def _parse_trajectory_judge_response(response: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Parse the trajectory judge response.

    Returns (scores_dict, reasons_dict) with keys for each dimension.
    Scores are clamped to [1, 5]. On parse failure returns all-1 scores.
    """
    _default_scores = {d: 1 for d in DIMS}
    _default_reasons = {d: "" for d in DIMS}

    content = response.strip()
    json_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()

    for attempt in [content, re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', content)]:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                scores = {
                    dim: max(1, min(5, int(parsed.get(dim, 1))))
                    for dim in DIMS
                }
                reasons = {
                    dim: str(parsed.get(f"reason_{dim}", ""))
                    for dim in DIMS
                }
                return scores, reasons
        except (json.JSONDecodeError, ValueError):
            pass

    warnings.warn(f"Failed to parse trajectory judge response: {content[:200]}")
    return _default_scores, _default_reasons


def _parse_scirm_response(response: str) -> Tuple[int, str]:
    """Parse SciRM-style output with <reasoning> and <score> tags.

    Returns:
        (score, reason) with score clamped to [1, 5]
    """
    content = response.strip()

    # Extract reasoning - try complete tags first
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        # Fallback: extract content after opening tag if closing tag is missing (incomplete response)
        incomplete_match = re.search(r"<reasoning>\s*(.*?)(?:<score>|$)", content, re.DOTALL | re.IGNORECASE)
        if incomplete_match:
            reasoning = incomplete_match.group(1).strip()
            if reasoning:
                logger.warning(f"Incomplete <reasoning> tag detected, extracted partial content: {reasoning[:100]}...")
        else:
            reasoning = ""

    # Extract score
    score_match = re.search(r"<score>\s*(\d+)\s*</score>", content, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(5, score))  # Clamp to [1, 5]
        return score, reasoning

    # No fallback - return None to signal parse failure
    warnings.warn(f"Failed to parse SciRM response: {content[:200]}")
    return None, "parse_error"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

async def compute_trajectory_memory_reasoning_reward_async(
    log_snapshot: Dict,
    llm_judge_fn,
    step_snapshots: List[Dict],
    format: str = "scirm",
) -> Tuple[float, Dict]:
    """Compute trajectory-based memory reasoning reward.

    combined = trajectory_quality (no pending penalty in V2)

    Where:
      - trajectory_quality: LLM judge on 5 dimensions (1-5 -> 0-1 each),
        weighted average.

    Args:
        log_snapshot: Final log snapshot dict.
        llm_judge_fn: Async callable(system_prompt, user_prompt) -> str.
        step_snapshots: List of per-step log snapshot dicts.
        format: "scirm" (default) or "json" for output format.

    Returns:
        (combined_score, details_dict).
    """
    # Build trajectory summary
    trajectory_summary = _build_trajectory_summary(step_snapshots)

    # Extract final state stats
    claims = log_snapshot.get("claims", [])
    questions = log_snapshot.get("questions", [])
    notes = log_snapshot.get("notes", [])
    outline = log_snapshot.get("review_outline", {})
    section_visits = log_snapshot.get("section_visits", {})

    # Select format-specific prompts and parser
    if format == "scirm":
        from .trajectory_memory_reasoning_v2_prompt_scirm import (
            TRAJECTORY_JUDGE_SYSTEM_PROMPT_SCIRM,
            TRAJECTORY_JUDGE_USER_PROMPT_SCIRM,
            DIMS_SCIRM,
            DIMENSION_WEIGHTS_SCIRM,
            get_scirm_dimension_prompt,
        )

        system_prompt = TRAJECTORY_JUDGE_SYSTEM_PROMPT_SCIRM
        dims = DIMS_SCIRM
        dim_weights = DIMENSION_WEIGHTS_SCIRM

        # SciRM evaluates each dimension separately - prepare all tasks for parallel execution
        async def evaluate_dimension(dim):
            """Evaluate a single dimension and return (dim, score, reason)."""
            query, criteria, examples = get_scirm_dimension_prompt(dim)

            user_prompt = TRAJECTORY_JUDGE_USER_PROMPT_SCIRM.format(
                query=query,
                criteria=criteria,
                examples=examples,
                trajectory_summary=trajectory_summary,
                n_steps=len(step_snapshots),
                n_claims=len(claims),
                n_supported=sum(1 for c in claims if c.get("status") == "supported"),
                n_weak=sum(1 for c in claims if c.get("status") == "weak"),
                n_pending=sum(1 for c in claims if c.get("status") == "to_be_verified"),
                n_questions=len(questions),
                n_resolved=sum(1 for q in questions if q.get("status") == "resolved"),
                n_open=sum(1 for q in questions if q.get("status") == "open"),
                n_notes=len(notes),
                n_strengths=len(outline.get("strengths", [])),
                n_weaknesses=len(outline.get("weaknesses", [])),
                sections_visited=', '.join(list(section_visits.keys())[:10]) if section_visits else "(none)",
            )

            # logger.info(f"[SciRM] Evaluating {dim}, prompt length: {len(system_prompt) + len(user_prompt)} chars")
            try:
                response = await llm_judge_fn(system_prompt, user_prompt)
            except Exception as e:
                logger.error(f"[SciRM] LLM call failed for {dim}: {e}")
                return dim, None, f"llm_error: {e}"

            if response is None:
                logger.error(f"[SciRM] LLM returned None for {dim}")
                return dim, None, "llm_returned_none"

            # logger.info(f"[SciRM] Raw response for {dim} (length={len(response)}): {response[:500]}")
            score, reason = _parse_scirm_response(response)
            if not reason:
                logger.warning(f"[SciRM] Empty reasoning for {dim}. Full response: {response[:1000]}")
            return dim, score, reason

        # Execute all dimension evaluations in parallel
        results = await asyncio.gather(*[evaluate_dimension(dim) for dim in dims])

        # Collect results
        dim_scores_raw = {}
        dim_reasons = {}
        for dim, score, reason in results:
            dim_scores_raw[dim] = score
            dim_reasons[dim] = reason

        # Check if any dimension failed
        if any(s is None for s in dim_scores_raw.values()):
            failed_dims = [d for d, s in dim_scores_raw.items() if s is None]
            logger.error(f"[SciRM] Judge failed for dims {failed_dims}, skipping instance")
            return None, {
                "judge_failed": True,
                "failed_dims": failed_dims,
                "individual_reasons": dim_reasons
            }

        scores = dim_scores_raw

    else:  # format == "json"
        system_prompt = TRAJECTORY_JUDGE_SYSTEM_PROMPT
        dims = DIMS
        dim_weights = DIMENSION_WEIGHTS_V2

        user_prompt = TRAJECTORY_JUDGE_USER_PROMPT.format(
            n_steps=len(step_snapshots),
            trajectory_summary=trajectory_summary,
            n_claims=len(claims),
            n_supported=sum(1 for c in claims if c.get("status") == "supported"),
            n_weak=sum(1 for c in claims if c.get("status") == "weak"),
            n_invalid=sum(1 for c in claims if c.get("status") == "invalid"),
            n_pending=sum(1 for c in claims if c.get("status") == "to_be_verified"),
            n_questions=len(questions),
            n_resolved=sum(1 for q in questions if q.get("status") == "resolved"),
            n_partial=sum(1 for q in questions if q.get("status") == "partially_answered"),
            n_open=sum(1 for q in questions if q.get("status") == "open"),
            n_notes=len(notes),
            n_strengths=len(outline.get("strengths", [])),
            n_weaknesses=len(outline.get("weaknesses", [])),
            sections_visited=list(section_visits.keys()) if section_visits else "(none)",
        )

        response = await llm_judge_fn(system_prompt, user_prompt)
        if response is None:
            logger.warning("[TrajV2] LLM judge returned None, using default scores")
            scores = {dim: 1 for dim in dims}
            dim_reasons = {dim: "LLM returned None - possible API error" for dim in dims}
        else:
            scores, dim_reasons = _parse_trajectory_judge_response(response)

    # Normalise each dimension 1-5 -> 0-1, then weighted average
    dim_scores = {dim: (scores[dim] - 1) / 4.0 for dim in dims}
    quality_score = sum(dim_scores[d] * dim_weights[d] for d in dims)

    # V2: No pending penalty (factual_correctness dimension already checks grounding)
    combined = quality_score

    details = {
        "trajectory_quality": quality_score,
        "raw_scores": scores,
        "dim_scores": dim_scores,
        "dim_weights": dim_weights,
        "individual_reasons": dim_reasons,
        "n_steps_summarised": len(step_snapshots),
    }

    return combined, details
