"""Trajectory-based memory reasoning reward.

Evaluates the full sequence of agent steps (not just the final snapshot)
to assess investigation depth, trajectory consistency, and active memory
usage via a single LLM-as-a-judge call on a clean trajectory summary.

combined = trajectory_quality + pending_penalty

- trajectory_quality: LLM judge (1-5 on 3 dimensions -> weighted 0-1)
- pending_penalty: reused from memory_reasoning.py
"""

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

    # Show a snippet of what the agent actually observed (section content,
    # search results, etc.) so the judge can check grounding.
    obs_snippet = obs_ctx.get("observation_snippet", "")
    if obs_snippet:
        parts.append(f"  Observed: {_trunc(obs_snippet, 300)}")

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


def _build_trajectory_summary(step_snapshots: List[Dict], max_tokens: int = 6000) -> str:
    """Build a content-rich per-step trajectory summary from step snapshots.

    Each step shows:
    - What action the agent took and what it observed (truncated snippet)
    - What memory operations happened (claims, questions, notes, outline)
    including actual text so the judge can assess quality and grounding.

    Keeps first 4 and last 4 steps if over budget, truncates middle.
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

TRAJECTORY_JUDGE_SYSTEM_PROMPT = """You are judging the quality of a paper reviewer agent's investigation trajectory.

You are given a step-by-step summary of what the agent did. Each step shows:
- **Action**: what the agent did (read_section, search_paper, research, finish)
- **Observed**: a snippet of what the agent actually saw (section content, search results)
- **Memory_ops**: claims logged, questions raised, notes taken, status updates, outline additions — with their actual text content

Use the "Observed" snippets to verify that the agent's memory entries are **grounded** in what it actually read. Claims, notes, and questions should reflect real content from the paper, not fabricated or hallucinated statements.

Score each dimension from 1 to 5. Be strict and responsible.

investigation_depth (1-5):
Did the agent genuinely cross-reference claims against evidence in different sections?

KEY DISTINCTIONS — apply these strictly:
- Sequential reading is NOT cross-referencing. If the agent reads sections in order (intro → method → experiments → conclusion) and resolves questions/claims from the immediately next section, that is sequential discovery, not cross-referencing. Score 2.
- Searches that produce NO status updates do not count as cross-referencing. Only count searches/reads that actually led to a claim or question status change.
- Resolving a question at the very next step after logging it is immediate resolution, not incremental investigation. This is expected from sequential reading and does not merit a score above 2-3.
- For score 4+, the agent must go BACK to re-examine earlier sections with new hypotheses, or update claims based on evidence found several steps later in a non-obvious section.

1 = Agent read 1-2 sections and immediately formed conclusions; no cross-referencing
2 = Agent read several sections sequentially; claims/questions resolved from the immediately next section(s); or all verification at finish; searches that produced no status updates
3 = Agent read most sections; some claims were updated based on evidence from non-adjacent sections (not the immediately next one); at least one status update reflects genuine cross-referencing
4 = Agent systematically cross-referenced: went back to re-read earlier sections or searched for specific evidence to verify claims; status updates happened across multiple non-adjacent steps; verifier_reasons cite evidence from sections far from where the claim originated
5 = Thorough: claims were iteratively re-evaluated as new evidence emerged; the agent revised earlier judgments based on later findings; multiple rounds of verification visible in the trajectory

trajectory_consistency (1-5):
Any contradictions, hallucinations, or misalignment between what the agent found and what it concluded?

CHECK THESE SPECIFIC FAILURE MODES:
- Do outline weakness/strength items reference content NOT found in any Observed snippet? If so, they may be hallucinated — score down.
- Are there claims left pending (to_be_verified) that the outline treats as confirmed strengths or weaknesses? That is a consistency gap.
- Does the agent write notes saying "this addresses Q1" but never formally update Q1's status? That is an internal inconsistency.
- Do outline items contradict claim statuses (e.g., listing something as a strength when the related claim is marked weak)?

1 = Major contradictions: claims/notes contradict observations; outline references hallucinated content
2 = Some inconsistencies: outline items conflict with claim statuses; some entries not grounded in observations
3 = Generally consistent with minor gaps; most entries grounded; pending claims are not treated as resolved in the outline
4 = Consistent: outline aligns with claim statuses; entries well-grounded in observations; no ungrounded outline items
5 = Highly consistent: every outline item traceable to specific observations; claim statuses fully aligned with outline; no contradictions at all

memory_quality (1-5):
Did the agent actively maintain memory throughout — updating claims incrementally? Or log-and-forget?

CHECK THESE SPECIFIC PATTERNS:
- If the agent logged zero claims, memory_quality CANNOT exceed 2 — claims are the primary tracking mechanism.
- If claims were logged but most remain to_be_verified at the end, that is log-and-forget (score 2).
- Outline entries all added in 1-2 consecutive steps at the end = batch writing, not incremental construction (caps score at 3).
- For score 4+, the agent must have updated claim/question statuses across 3+ distinct non-adjacent steps AND built the outline incrementally (not in a batch).

1 = Barely used memory; few entries, no updates
2 = Log-and-forget: claims/questions logged but left pending; or bulk status updates all at finish; or outline written in one batch at the end
3 = Mixed: some status updates during investigation, but outline mostly batch-written at the end; OR good incremental updates but many claims/questions left unresolved
4 = Actively maintained: claim statuses updated across 3+ distinct steps spread through the trajectory; questions resolved incrementally with evidence; outline built gradually (entries added at 3+ different steps)
5 = Active reasoning across full trajectory: claims re-evaluated multiple times as understanding evolved; questions investigated with answers citing specific cross-referenced evidence; outline built incrementally and revised based on new findings

Output JSON only: {"investigation_depth": <1-5>, "trajectory_consistency": <1-5>, "memory_quality": <1-5>, "reason": "<one sentence>"}"""

TRAJECTORY_JUDGE_USER_PROMPT = """## Trajectory Summary ({n_steps} steps)

{trajectory_summary}

## Final State
- Claims: {n_claims} total ({n_supported} supported, {n_weak} weak, {n_invalid} invalid, {n_pending} pending)
- Questions: {n_questions} total ({n_resolved} resolved, {n_partial} partial, {n_open} open)
- Notes: {n_notes} total
- Outline: {n_strengths} strengths, {n_weaknesses} weaknesses
- Sections visited: {sections_visited}

---

```json
{{"investigation_depth": <1-5>, "trajectory_consistency": <1-5>, "memory_quality": <1-5>, "reason": "<one sentence>"}}
```"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

DIMS = ("investigation_depth", "trajectory_consistency", "memory_quality")


def _parse_trajectory_judge_response(response: str) -> Tuple[Dict[str, int], str]:
    """Parse the trajectory judge response.

    Returns (scores_dict, reason) with keys for each dimension clamped to [1, 5].
    On parse failure returns all-1 scores.
    """
    _default = {d: 1 for d in DIMS}

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
                reason = str(parsed.get("reason", ""))
                return scores, reason
        except (json.JSONDecodeError, ValueError):
            pass

    warnings.warn(f"Failed to parse trajectory judge response: {content[:200]}")
    return _default, "parse_error"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

async def compute_trajectory_memory_reasoning_reward_async(
    log_snapshot: Dict,
    llm_judge_fn,
    step_snapshots: List[Dict],
) -> Tuple[float, Dict]:
    """Compute trajectory-based memory reasoning reward.

    combined = trajectory_quality + pending_penalty

    Where:
      - trajectory_quality: LLM judge on 3 dimensions (1-5 -> 0-1 each),
        weighted average.
      - pending_penalty: -(n_pending / n_claims), in [-1, 0].

    Args:
        log_snapshot: Final log snapshot dict.
        llm_judge_fn: Async callable(system_prompt, user_prompt) -> str.
        step_snapshots: List of per-step log snapshot dicts.

    Returns:
        (combined_score, details_dict).
    """
    from reviewer.reward.memory_reasoning import compute_pending_penalty

    # Build trajectory summary
    trajectory_summary = _build_trajectory_summary(step_snapshots)

    # Extract final state stats
    claims = log_snapshot.get("claims", [])
    questions = log_snapshot.get("questions", [])
    notes = log_snapshot.get("notes", [])
    outline = log_snapshot.get("review_outline", {})
    section_visits = log_snapshot.get("section_visits", {})

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

    response = await llm_judge_fn(TRAJECTORY_JUDGE_SYSTEM_PROMPT, user_prompt)
    scores, reason = _parse_trajectory_judge_response(response)

    # Normalise each dimension 1-5 -> 0-1, then weighted average
    dim_weights = {
        "investigation_depth": 0.4,
        "trajectory_consistency": 0.3,
        "memory_quality": 0.3,
    }
    dim_scores = {dim: (scores[dim] - 1) / 4.0 for dim in DIMS}
    quality_score = sum(dim_scores[d] * dim_weights[d] for d in DIMS)

    # Pending penalty (reused from memory_reasoning.py)
    penalty, penalty_details = compute_pending_penalty(log_snapshot)
    combined = quality_score + penalty

    return combined, {
        "trajectory_quality": quality_score,
        "raw_scores": scores,
        "dim_scores": dim_scores,
        "dim_weights": dim_weights,
        "reason": reason,
        "pending_penalty": penalty,
        "pending_penalty_details": penalty_details,
        "n_steps_summarised": len(step_snapshots),
    }
