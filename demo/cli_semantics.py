"""Argv-level semantics of review_cli.py, ported without pydantic.

The demo builder must reproduce exactly what the CLI did to the review log, so
that replaying a recorded transcript yields the same review_log.json that was
written at the time. Rather than import the skill, this module re-implements the
handful of behaviours that matter. Importing was rejected because:

  * review_cli.py is not importable as a package member — it does a flat
    ``from reviewer_memory import ...`` after mutating sys.path;
  * it requires pydantic, and a demo should build on a bare python3;
  * ``load_log`` calls ``sys.exit(1)`` on a missing file;
  * comparing raw ``json.load`` dicts is a strictly stronger equality assertion
    than comparing two pydantic-normalised models.

Every port below cites the source file and line range it was taken from.
``CONFERENCE_SCALES`` is *loaded* rather than copied, because a scale changing
upstream should not silently diverge here.

Source of truth:
    ProReviewer-Skill/proreviewer/review_cli.py
    ProReviewer-Skill/proreviewer/reviewer_memory.py
"""

import ast
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Checked against the live skill files; a mismatch means the port may be stale.
# review_cli.py moved to cc2e4d19 when the `step` subcommand was added. `step`
# writes steps.jsonl and never touches review_log.json, so no log semantics
# ported here changed — and `step` is deliberately absent from
# shellparse.SUBCOMMANDS so the replay does not treat it as a mutation.
PORTED_FROM_SHA256 = {
    "review_cli.py": "cc2e4d19",
    "reviewer_memory.py": "d163d2c1",
}

_SKILL_DIRS = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "ProReviewer-Skill", "proreviewer"),
    os.path.expanduser("~/.claude/skills/proreviewer"),
)

# Fallback used when reviewer_memory.py cannot be read or is no longer a literal.
_FALLBACK_SCALES: Dict[str, Dict[str, Any]] = {
    "iclr": {
        "name": "ICLR 2026",
        "scores": [0, 2, 4, 6, 8, 10],
        "labels": {
            10: "Strong accept — should be highlighted at the conference",
            8: "Accept — good submission",
            6: "Marginally above the acceptance threshold",
            4: "Marginally below the acceptance threshold",
            2: "Reject — not good enough",
            0: "Strong reject",
        },
    },
}
DEFAULT_CONFERENCE = "iclr"


def find_skill_dir() -> Optional[str]:
    """Locate the skill source directory, preferring the in-repo copy."""
    for path in _SKILL_DIRS:
        if os.path.isfile(os.path.join(path, "reviewer_memory.py")):
            return path
    return None


def load_conference_scales(skill_dir: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Extract CONFERENCE_SCALES and DEFAULT_CONFERENCE without importing.

    reviewer_memory.py imports pydantic at module level, so it cannot be
    imported in a stdlib-only build. Both names are module-level literals, so
    ast.literal_eval recovers them exactly. Handles Assign and AnnAssign because
    CONFERENCE_SCALES carries a type annotation.
    """
    skill_dir = skill_dir or find_skill_dir()
    if not skill_dir:
        logger.warning("skill sources not found; using the vendored ICLR scale only")
        return _FALLBACK_SCALES, DEFAULT_CONFERENCE

    path = os.path.join(skill_dir, "reviewer_memory.py")
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (OSError, SyntaxError) as exc:
        logger.warning("could not parse %s (%s); using the vendored scale", path, exc)
        return _FALLBACK_SCALES, DEFAULT_CONFERENCE

    found: Dict[str, Any] = {}
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        for name in targets:
            if name in ("CONFERENCE_SCALES", "DEFAULT_CONFERENCE") and node.value is not None:
                try:
                    found[name] = ast.literal_eval(node.value)
                except ValueError:
                    logger.warning("%s in %s is no longer a literal", name, path)

    scales = found.get("CONFERENCE_SCALES") or _FALLBACK_SCALES
    default = found.get("DEFAULT_CONFERENCE") or DEFAULT_CONFERENCE
    return scales, default


def check_port_drift(skill_dir: Optional[str] = None) -> List[str]:
    """Warn if the skill sources changed since these semantics were ported."""
    skill_dir = skill_dir or find_skill_dir()
    if not skill_dir:
        return []
    warnings: List[str] = []
    for filename, expected in PORTED_FROM_SHA256.items():
        path = os.path.join(skill_dir, filename)
        try:
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]
        except OSError:
            continue
        if digest != expected:
            warnings.append(
                f"{filename} changed since demo/cli_semantics.py was written "
                f"(sha256 {digest}, expected {expected}); re-verify the ported semantics"
            )
    return warnings


# --------------------------------------------------------------------------
# Ported helpers
# --------------------------------------------------------------------------

def split_list(value: Optional[str]) -> List[str]:
    """review_cli.py:86-90.

    Load-bearing and lossy: it splits on commas with no quoting, which is why
    C1's --issues prose was shredded into four fragments in the recorded log.
    Reproducing the shredding is required for exact replay.
    """
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def item_refs(item: Dict[str, Any]) -> List[str]:
    """review_cli.py:197-199. Evidence IDs in tag order: claims, questions, notes."""
    return (list(item.get("related_claims", []))
            + list(item.get("related_questions", []))
            + list(item.get("related_notes", [])))


def format_item_with_links(item: Dict[str, Any]) -> str:
    """review_cli.py:202-208. Note the literal outer brackets around the links."""
    refs = item_refs(item)
    if not refs:
        return item["text"]
    links = ", ".join(f"[{r}](#{r.lower()})" for r in refs)
    return f"{item['text']} [{links}]"


def ref_sort_key(ref_id: str) -> Tuple[int, int, str]:
    """review_cli.py:211-217. Orders C1..Cn, then Q1..Qn, then N1..Nn."""
    order = {"C": 0, "Q": 1, "N": 2}
    match = re.match(r"([CQN])(\d+)$", ref_id)
    if not match:
        return (3, 0, ref_id)
    return (order[match.group(1)], int(match.group(2)), ref_id)


def format_score_with_scale(score: Any, conference: str, scales: Dict[str, Any]) -> str:
    """reviewer_memory.py:116-127."""
    scale = scales.get(conference)
    if not scale:
        return str(score)
    label = scale.get("labels", {}).get(score)
    if label:
        return f"{score}/{max(scale['scores'])} ({scale['name']}: {label})"
    return f"{score}/{max(scale['scores'])} ({scale['name']})"


# --------------------------------------------------------------------------
# Model defaults — reviewer_memory.py:130-334
# --------------------------------------------------------------------------

def new_claim(claim_id: str, text: str, section: str, claim_type: str,
              issues: Optional[List[str]]) -> Dict[str, Any]:
    return {
        "id": claim_id,
        "text": text,
        "section": section,
        "type": claim_type,
        "status": "to_be_verified",
        "issues": issues or [],
        "cross_references": [],
        "verifier_reason": None,
    }


def new_question(question_id: str, question: str, source_section: str,
                 question_type: str, related_claims: Optional[List[str]]) -> Dict[str, Any]:
    return {
        "id": question_id,
        "question": question,
        "source_section": source_section,
        "status": "open",
        "type": question_type or "clarification",
        "answer": None,
        "answer_sections": [],
        "related_claims": related_claims or [],
    }


def new_note(note_id: str, text: str, section: str, tag: Optional[List[str]]) -> Dict[str, Any]:
    return {"id": note_id, "text": text, "section": section, "tag": tag or []}


def new_outline_item(text: str, claims: List[str], questions: List[str],
                     notes: List[str]) -> Dict[str, Any]:
    return {
        "text": text,
        "related_claims": claims or [],
        "related_questions": questions or [],
        "related_notes": notes or [],
    }


def new_outline(conference: str) -> Dict[str, Any]:
    return {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "overall_score": None,
        "conference": conference,
    }


def coerce_score(raw: str) -> Any:
    """review_cli.py:497-499 — float, then narrowed to int when whole."""
    value = float(raw)
    return int(value) if value == int(value) else value


# --------------------------------------------------------------------------
# review.md rendering — review_cli.py:220-302
# --------------------------------------------------------------------------

_SECTION_TITLES = [
    ("strengths", "Strengths"),
    ("weaknesses", "Weaknesses"),
    ("questions", "Questions for Authors"),
]


def format_evidence_section(log: Dict[str, Any]) -> List[str]:
    """review_cli.py:220-258. Only referenced IDs are emitted, C then Q then N."""
    outline = log["review_outline"]
    referenced = set()
    for key, _ in _SECTION_TITLES:
        for item in outline.get(key, []):
            referenced.update(item_refs(item))
    if not referenced:
        return []

    claims = {c["id"]: c for c in log["claims"]}
    questions = {q["id"]: q for q in log["questions"]}
    notes = {n["id"]: n for n in log["notes"]}

    lines = ["## Evidence", ""]
    for ref_id in sorted(referenced, key=ref_sort_key):
        anchor = f'<a id="{ref_id.lower()}"></a>'
        if ref_id in claims:
            claim = claims[ref_id]
            lines.append(f'{anchor}**{ref_id}** · claim ({claim["status"]}) · {claim["section"]}')
            lines.append(f'> {claim["text"]}')
            if claim.get("verifier_reason"):
                lines.append(f'>\n> *Verification:* {claim["verifier_reason"]}')
        elif ref_id in questions:
            question = questions[ref_id]
            lines.append(
                f'{anchor}**{ref_id}** · question ({question["status"]}) · {question["source_section"]}'
            )
            lines.append(f'> {question["question"]}')
            if question.get("answer"):
                sections = question.get("answer_sections") or []
                suffix = f" ({', '.join(sections)})" if sections else ""
                lines.append(f'>\n> *Answer:* {question["answer"]}{suffix}')
        elif ref_id in notes:
            note = notes[ref_id]
            tags = f' · {", ".join(note["tag"])}' if note.get("tag") else ""
            lines.append(f'{anchor}**{ref_id}** · note · {note["section"]}{tags}')
            lines.append(f'> {note["text"]}')
        lines.append("")
    return lines


def format_review(log: Dict[str, Any], scales: Dict[str, Any],
                  include_evidence: bool = True) -> str:
    """review_cli.py:261-302. Sections are emitted only when non-empty."""
    outline = log["review_outline"]
    lines: List[str] = []

    if outline.get("summary"):
        lines += ["## Summary", "", outline["summary"], ""]

    for key, title in _SECTION_TITLES:
        items = outline.get(key, [])
        if not items:
            continue
        lines += [f"## {title}", ""]
        for item in items:
            rendered = format_item_with_links(item) if include_evidence else item["text"]
            lines.append(f"- {rendered}")
        lines.append("")

    if outline.get("overall_score") is not None:
        lines += [
            "## Overall Score",
            "",
            format_score_with_scale(outline["overall_score"], log["conference"], scales),
            "",
        ]

    if include_evidence:
        evidence = format_evidence_section(log)
        if evidence:
            lines += evidence

    return "\n".join(lines).rstrip() + "\n"
