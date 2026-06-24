"""Memory reasoning reward: teaches the agent to use memory for genuine reasoning.

combined = memory_quality + pending_penalty

- memory_quality: LLM judge (1–5 → 0–1) on claim depth, note quality,
  question investigation, and whether critical findings appear in the outline.
- pending_penalty: -(n_pending / n_claims), in [-1, 0]. Pure negative signal
  for leaving claims as to_be_verified — no reward, only penalty.
"""

import json
import logging
import re
import warnings
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt for memory quality judge (single crisp 1–5 judgment)
# Covers: claim depth, evidence specificity, note quality, and reflection of
# critical findings in the outline — no separate reflection component needed.
# ---------------------------------------------------------------------------

MEMORY_QUALITY_JUDGE_SYSTEM_PROMPT = """You are judging the quality of a paper reviewer's working memory log.

Score each dimension from 1 to 5:

claims (1–5):
The primary signal is verifier_reason quality — does the agent actually investigate or just rubber-stamp?
1 = verifier_reasons are absent, one-line, or generic ("The paper states this"); claims restate the abstract
2 = Some verifier_reasons cite sections, but most are superficial or copy the claim text
3 = Most verifier_reasons cite specific sections beyond abstract/intro with substantive detail
4 = Most verifier_reasons cross-reference 2+ sections or cite tables/figures/equations; at least one weak/invalid claim with well-reasoned evidence
5 = Verifier_reasons demonstrate systematic cross-checking across sections; all weak/invalid findings are clearly reflected in outline weaknesses
NOTE: Claims that are supported can score 3–4 if verifier_reasons are substantive. Weak/invalid claims are a bonus signal of deeper investigation.

questions (1–5):
The primary signal is investigation depth — did the agent probe and resolve concerns rigorously?
1 = No questions, or all trivial (e.g. "what dataset was used?") with no follow-up
2 = Questions raised but answers are vague, missing, or don't cite specific evidence
3 = Questions probe real concerns; answers cite specific sections with substantive detail
4 = Questions are substantive; answers reference experiments/tables/results; any open/partial gaps appear as outline weaknesses
5 = Questions demonstrate deep engagement; all unresolved concerns are explicitly reflected in outline weaknesses
NOTE: Questions that are all-resolved can score 3–4 if answers are substantive and cite specific evidence. Leaving strategic open questions is a bonus signal.

notes (1–5):
Notes are a working observation log recorded during reading — the primary signal is whether they capture the agent's genuine observations and evolving interpretations.
1 = Notes are boilerplate template text (e.g. "Starting Phase 1") or verbatim copying of paper sentences with no interpretation
2 = Notes mostly transcribe paper content without interpretation; agent is restating what sections say rather than observing
3 = Notes capture the agent's observations and interpretations: flagging things to investigate ("The paper claims X but doesn't specify Y"), noting what sections reveal about potential issues, showing developing understanding
4 = Notes show the agent building understanding across sections: connects observations to potential issues, records evolving interpretations or hypotheses that go beyond restating the text
5 = Notes demonstrate active reasoning during reading: identify non-obvious connections between sections, flag specific technical gaps or inconsistencies as the agent encounters them, show thinking evolving as more evidence is gathered

Use the "Investigation Trajectory" to assess whether claims were verified across multiple steps (high latency = genuine cross-section investigation) or resolved immediately (rubber-stamping). Status changes during investigation are the strongest signal of genuine re-evaluation.
Use the "Supported Claim Sample" to assess verifier_reason depth when no weak/invalid claims exist.
Use the "Critical Findings" to check whether weak/invalid claims and open/partial questions appear in outline weaknesses.
Use the "Notes" to assess whether the agent recorded genuine observations and interpretations during reading. A note that says "Section 3 claims X but doesn't justify Y" is better than one that says "Section 3 presents method Z". Citation in the outline is NOT required — notes that served their purpose during investigation but were not cited are still valuable.

Output JSON only: {"claims": <1–5>, "questions": <1–5>, "notes": <1–5>, "reason": "<one sentence summarising the overall quality>"}"""

MEMORY_QUALITY_JUDGE_USER_PROMPT = """## Investigation Trajectory (how claims and questions evolved — key signal for investigation depth)

{trajectory_text}

## Claims ({n_claims} total | {n_supported} supported, {n_weak} weak, {n_invalid} invalid, {n_pending} pending)

Supported sample (assess verifier_reason depth):
{supported_sample_text}

## Questions ({n_questions} total | {n_open} open, {n_partial} partially answered, {n_resolved} resolved)

{questions_text}

## Notes ({n_notes} total | {n_notes_cited} cited in outline)

Sample:
{notes_text}

Critical findings (weak/invalid claims + open/partial questions — check whether these appear in outline weaknesses):
{critical_findings_text}

## Outline Weaknesses ({n_weaknesses} total)

{weaknesses_text}

## Sections Visited
{sections_visited}

---

```json
{{"claims": <1–5>, "questions": <1–5>, "notes": <1–5>, "reason": "<one sentence>"}}
```"""


# ---------------------------------------------------------------------------
# Helpers for the quality judge prompt
# ---------------------------------------------------------------------------

def _format_claims_for_quality(claims: List[Dict], max_claims: int = 12) -> str:
    if not claims:
        return "(none)"
    lines = []
    for c in claims[:max_claims]:
        status = c.get("status", "?")
        section = c.get("section", "?")
        text = c.get("text", "")
        reason = (c.get("verifier_reason") or "").strip()
        reason_str = f" | reason: {reason[:250]}" if reason else " | reason: (none)"
        lines.append(f"- [{status}] {c.get('id', '?')} (§{section}): {text}{reason_str}")
    if len(claims) > max_claims:
        lines.append(f"  … ({len(claims) - max_claims} more not shown)")
    return "\n".join(lines)


def _format_questions_for_quality(questions: List[Dict]) -> str:
    if not questions:
        return "(none)"
    lines = []
    for q in questions:
        status = q.get("status", "?")
        qtext = q.get("question", q.get("text", ""))
        answer = (q.get("answer") or "").strip()
        answer_str = f" → {answer[:200]}" if answer else ""
        lines.append(f"- [{status}] {q.get('id', '?')}: {qtext}{answer_str}")
    return "\n".join(lines)


def _summarize_investigation_trajectory(step_snapshots: List[Dict]) -> str:
    """Summarize how claims and questions evolved step-by-step.

    Captures:
    - Verification latency: steps between claim first logged and status resolved
    - Status changes: claims re-evaluated and upgraded/downgraded during investigation
    - Question resolution latency: steps between question raised and answered

    A claim verified 10 steps after logging (across multiple sections) signals
    genuine investigation; one logged and immediately marked supported signals
    rubber-stamping.
    """
    if not step_snapshots or len(step_snapshots) < 2:
        return "(trajectory not available)"

    # Track first appearance and resolution for each claim/question by id
    claim_first: Dict[str, int] = {}   # id -> step index when first seen
    claim_resolved: Dict[str, tuple] = {}  # id -> (step, final_status)
    status_changes: List[str] = []
    prev_statuses: Dict[str, str] = {}

    q_first: Dict[str, int] = {}
    q_resolved: Dict[str, int] = {}

    for step_idx, snap in enumerate(step_snapshots):
        for c in snap.get("claims", []):
            cid = c.get("id", "")
            status = c.get("status", "to_be_verified")
            if cid not in claim_first:
                claim_first[cid] = step_idx
            prev = prev_statuses.get(cid)
            if prev is not None and prev != status:
                status_changes.append(f"  {cid}: {prev} → {status} at step {step_idx}")
                if status != "to_be_verified" and cid not in claim_resolved:
                    claim_resolved[cid] = (step_idx, status)
            elif status != "to_be_verified" and cid not in claim_resolved and cid in claim_first:
                claim_resolved[cid] = (step_idx, status)
            prev_statuses[cid] = status

        for q in snap.get("questions", []):
            qid = q.get("id", "")
            if qid not in q_first:
                q_first[qid] = step_idx
            if q.get("status") in ("resolved", "partially_answered") and qid not in q_resolved:
                q_resolved[qid] = step_idx

    lines = []

    # Claim verification latency
    latencies = []
    for cid, (res_step, final_status) in claim_resolved.items():
        start_step = claim_first.get(cid, res_step)
        latency = res_step - start_step
        latencies.append((latency, cid, final_status))

    if latencies:
        latencies.sort(reverse=True)
        avg_lat = sum(l for l, _, _ in latencies) / len(latencies)
        lines.append(f"Claim verification latency (avg {avg_lat:.1f} steps between logging and resolution):")
        for lat, cid, fstatus in latencies[:6]:
            lines.append(f"  {cid}: {lat} steps to resolve [{fstatus}]")

    # Status changes — strongest signal of genuine re-evaluation
    if status_changes:
        lines.append(f"Claim status changes during investigation ({len(status_changes)} total — agent re-evaluated):")
        lines.extend(status_changes[:5])
    else:
        lines.append("Claim status changes: none (all claims resolved on first evaluation)")

    # Question resolution latency
    q_latencies = [q_resolved[qid] - q_first[qid] for qid in q_resolved if qid in q_first]
    if q_latencies:
        avg_qlat = sum(q_latencies) / len(q_latencies)
        lines.append(f"Question resolution: {len(q_latencies)} resolved, avg {avg_qlat:.1f} steps after logging")

    return "\n".join(lines)


def _format_supported_sample(claims: List[Dict], n_sample: int = 10) -> str:
    """Sample supported claims to let the judge assess verifier_reason depth
    even when no weak/invalid claims exist."""
    supported = [c for c in claims if c.get("status") == "supported"]
    if not supported:
        return "(none)"
    # Prefer claims with longer verifier_reasons to surface the best evidence of investigation
    supported_sorted = sorted(supported, key=lambda c: len(c.get("verifier_reason") or ""), reverse=True)
    lines = []
    for c in supported_sorted[:n_sample]:
        reason = (c.get("verifier_reason") or "").strip()
        reason_str = f" | reason: {reason[:250]}" if reason else " | reason: (none — not investigated)"
        lines.append(f"- [supported] (§{c.get('section', '?')}): {c.get('text', '')}{reason_str}")
    return "\n".join(lines)


def _format_critical_findings(claims: List[Dict], questions: List[Dict]) -> str:
    """Show weak/invalid claims and open/partial questions so the judge can check
    whether these critical findings are reflected in the outline weaknesses."""
    lines = []
    for c in claims:
        if c.get("status") in ("weak", "invalid"):
            reason = (c.get("verifier_reason") or "").strip()
            reason_str = f" | reason: {reason[:200]}" if reason else ""
            lines.append(f"- [CLAIM {c.get('status', '').upper()}] {c.get('text', '')}{reason_str}")
    for q in questions:
        if q.get("status") in ("open", "partially_answered"):
            answer = (q.get("answer") or "").strip()
            answer_str = f" → {answer[:150]}" if answer else ""
            lines.append(f"- [QUESTION {q.get('status', '').upper()}] {q.get('question', q.get('text', ''))}{answer_str}")
    return "\n".join(lines) if lines else "(none — all claims supported and all questions resolved; assess depth via Supported Claim Sample above)"


def _sample_notes_for_quality(notes: List[Dict], n_sample: int = 20) -> str:
    """Return a spread sample of notes across the trajectory."""
    if not notes:
        return "(none)"
    if len(notes) <= n_sample:
        sampled = notes
    else:
        step = max(1, len(notes) // n_sample)
        sampled = [notes[i] for i in range(0, len(notes), step)][:n_sample]
    lines = []
    for n in sampled:
        text = n.get("text", "")
        section = n.get("section", "?")
        tags = n.get("tag", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"- (§{section}){tag_str}: {text[:120]}")
    return "\n".join(lines)


def _count_notes_cited_in_outline(notes: List[Dict], outline: Dict) -> int:
    """Count how many notes are referenced in any outline item."""
    cited_ids = set()
    for section in ("strengths", "weaknesses", "questions"):
        for item in outline.get(section, []):
            if isinstance(item, dict):
                for nid in item.get("related_notes", []):
                    cited_ids.add(nid)
    note_ids = {n.get("id") for n in notes}
    return len(cited_ids & note_ids)


def _format_weaknesses_for_quality(weaknesses: List[Dict]) -> str:
    if not weaknesses:
        return "(none)"
    lines = []
    for i, w in enumerate(weaknesses, 1):
        text = w.get("text", str(w)) if isinstance(w, dict) else str(w)
        refs = []
        if isinstance(w, dict):
            refs += w.get("related_claims", [])
            refs += w.get("related_questions", [])
            refs += w.get("related_notes", [])
        ref_str = f" [{', '.join(refs)}]" if refs else " [no memory refs]"
        lines.append(f"{i}. {text}{ref_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_pending_penalty(log_snapshot: Dict) -> Tuple[float, Dict]:
    """Compute a negative penalty for claims left as to_be_verified.

    penalty = -(n_pending / n_claims)

    Returns a value in [-1, 0]:
    - No pending claims → 0.0  (no penalty)
    - All claims pending → -1.0 (maximum penalty)
    - No claims at all  → 0.0  (no penalty; quality judge handles empty case)

    This is subtracted from the quality score, so the combined reward can go negative.
    """
    claims = log_snapshot.get("claims", [])
    n_claims = len(claims)
    if n_claims == 0:
        return 0.0, {"n_claims": 0, "n_pending": 0, "penalty": 0.0}

    n_pending = sum(1 for c in claims if c.get("status", "to_be_verified") == "to_be_verified")
    penalty = -(n_pending / n_claims) if n_pending > 0 else 0.0
    return penalty, {"n_claims": n_claims, "n_pending": n_pending, "penalty": penalty}


async def compute_memory_quality_reward_async(
    log_snapshot: Dict,
    llm_judge_fn,
    step_snapshots: List[Dict] = None,
) -> Tuple[float, Dict]:
    """LLM judge: rate memory log substantiveness on a 1–5 scale.

    Evaluates:
    - Claim depth and evidence specificity (verifier_reason quality)
    - Question investigation quality
    - Note informativeness vs boilerplate
    - Whether critical findings (weak/invalid claims, open questions)
      are reflected in the outline weaknesses

    Score normalised to [0, 1].
    """
    claims = log_snapshot.get("claims", [])
    questions = log_snapshot.get("questions", [])
    notes = log_snapshot.get("notes", [])
    outline = log_snapshot.get("review_outline", {})
    weaknesses = outline.get("weaknesses", [])
    section_visits = log_snapshot.get("section_visits", {})

    user_prompt = MEMORY_QUALITY_JUDGE_USER_PROMPT.format(
        n_claims=len(claims),
        n_supported=sum(1 for c in claims if c.get("status") == "supported"),
        n_weak=sum(1 for c in claims if c.get("status") == "weak"),
        n_invalid=sum(1 for c in claims if c.get("status") == "invalid"),
        n_pending=sum(1 for c in claims if c.get("status") == "to_be_verified"),
        trajectory_text=_summarize_investigation_trajectory(step_snapshots or []),
        supported_sample_text=_format_supported_sample(claims),
        critical_findings_text=_format_critical_findings(claims, questions),
        n_questions=len(questions),
        n_open=sum(1 for q in questions if q.get("status") == "open"),
        n_partial=sum(1 for q in questions if q.get("status") == "partially_answered"),
        n_resolved=sum(1 for q in questions if q.get("status") == "resolved"),
        questions_text=_format_questions_for_quality(questions),
        n_notes=len(notes),
        n_notes_cited=_count_notes_cited_in_outline(notes, outline),
        notes_text=_sample_notes_for_quality(notes),
        n_weaknesses=len(weaknesses),
        weaknesses_text=_format_weaknesses_for_quality(weaknesses),
        sections_visited=list(section_visits.keys()) if section_visits else "(none)",
    )

    response = await llm_judge_fn(MEMORY_QUALITY_JUDGE_SYSTEM_PROMPT, user_prompt)
    scores, reason = _parse_quality_response(response)

    # Normalise each dimension 1–5 → 0–1, then weighted average.
    # claims and questions capture critical findings (verification depth + gap
    # identification reflected in weaknesses) and are weighted higher than notes,
    # which is a supporting breadth signal.
    dim_scores = {dim: (scores[dim] - 1) / 4.0 for dim in ("claims", "questions", "notes")}
    dim_weights = {"claims": 0.4, "questions": 0.4, "notes": 0.2}
    quality_score = sum(dim_scores[d] * dim_weights[d] for d in dim_scores)

    return quality_score, {
        "raw_scores": scores,
        "dim_scores": dim_scores,
        "dim_weights": dim_weights,
        "reason": reason,
    }


def _parse_quality_response(response: str) -> Tuple[Dict[str, int], str]:
    """Parse the memory quality judge response.

    Returns (scores_dict, reason) where scores_dict has keys
    'claims', 'questions', 'notes' each clamped to [1, 5].
    On parse failure returns all-1 scores.
    """
    _default = {"claims": 1, "questions": 1, "notes": 1}

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
                    for dim in ("claims", "questions", "notes")
                }
                reason = str(parsed.get("reason", ""))
                return scores, reason
        except (json.JSONDecodeError, ValueError):
            pass

    warnings.warn(f"Failed to parse memory quality judge response: {content[:200]}")
    return _default, "parse_error"


async def compute_memory_reasoning_reward_async(
    log_snapshot: Dict,
    llm_judge_fn,
    step_snapshots: List[Dict] = None,
) -> Tuple[float, Dict]:
    """Compute the combined memory reasoning reward.

    combined = memory_quality + pending_penalty

    Where:
      - memory_quality: LLM judge (1–5 → 0–1) covering claim depth, note quality,
        question investigation, and whether critical findings are in the outline.
      - pending_penalty: -(n_pending / n_claims), in [-1, 0]. Pure negative signal
        for leaving claims unverified — no reward, only penalty.

    The combined score can be negative when many claims are left pending.

    Args:
        log_snapshot: Full log snapshot dict.
        llm_judge_fn: Async callable(system_prompt, user_prompt) -> str.

    Returns:
        (combined_score, details_dict).
    """
    quality, quality_details = await compute_memory_quality_reward_async(log_snapshot, llm_judge_fn, step_snapshots=step_snapshots)
    penalty, penalty_details = compute_pending_penalty(log_snapshot)

    combined = quality + penalty

    return combined, {
        "memory_quality": quality,
        "memory_quality_details": quality_details,
        "pending_penalty": penalty,
        "pending_penalty_details": penalty_details,
    }
