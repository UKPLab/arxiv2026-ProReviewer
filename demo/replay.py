"""Replay recorded review_cli calls into a step-by-step review log history.

The demo's central claim is that what it animates is what the agent actually
did. That claim is enforced here: replaying the transcript must reproduce the
session's review_log.json exactly, or the build fails.

The output is a patch journal, not a sequence of snapshots. Forty-six snapshots
of a 53 KB log would be 2.4 MB; the journal is ~65 KB and its final fold *is*
review_log.json, so the log need not be shipped separately. All CLI semantics
run once here, in Python, where they are verified — the browser only replays two
opcodes (add, set) and so cannot drift.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from . import cli_semantics as cli
from .shellparse import parse_blob, suspicious_values
from .transcript import ToolEvent

logger = logging.getLogger(__name__)

# Outline question items are namespaced RQ* so they cannot collide with log
# questions Q* in the bidirectional evidence map.
_OUTLINE_PREFIX = {"strengths": "S", "weaknesses": "W", "questions": "RQ"}

_CONFIRMATION = re.compile(
    r"^(?:Added (?P<added>[CQN]\d+)"
    r"|Added (?P<section>strengths|weaknesses|questions)"
    r"|Updated (?P<updated>C\d+)"
    r"|Resolved (?P<resolved>Q\d+)"
    r"|Set (?P<set>summary|overall_score)"
    r"|(?P<error>Error:)"
    r"|(?P<skipped>Skipped:))",
    re.MULTILINE,
)

_MUTATORS = {"add_claim", "add_question", "add_note", "update_claim",
             "resolve_question", "outline", "set_conference", "init"}


class ReplayMismatch(RuntimeError):
    """The replayed log does not match the session's recorded review_log.json."""


class Add:
    """Create an entry. `fields` is the complete initial state, defaults included.

    The field dict is copied on construction. The caller hands in the same dict
    it appends to the snapshot, and later Set ops mutate that dict in place;
    without the copy an entry would report its *final* state as the state it was
    born with, which is precisely what the replay draws at the birth step.
    """

    __slots__ = ("kind", "id", "list_name", "fields")

    def __init__(self, kind: str, entry_id: str, fields: Dict[str, Any],
                 list_name: Optional[str] = None) -> None:
        self.kind = kind
        self.id = entry_id
        self.fields = dict(fields)
        self.list_name = list_name

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"k": self.kind, "id": self.id, "f": self.fields}
        if self.list_name:
            out["l"] = self.list_name
        return out


class Set:
    """Assign one field. `id` is an entry id, or "__outline__" for outline fields."""

    __slots__ = ("id", "field", "value")

    def __init__(self, entry_id: str, field: str, value: Any) -> None:
        self.id = entry_id
        self.field = field
        self.value = value

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "p": self.field, "v": self.value}


class Step:
    """One tool call: what the agent did, and what it changed."""

    __slots__ = ("index", "tool", "status", "phase", "label", "narration",
                 "read", "adds", "sets", "touched", "error_note", "ops",
                 "command", "timestamp")

    def __init__(self, index: int) -> None:
        self.index = index
        self.tool = "Other"
        self.status = "ok"
        self.phase = "setup"
        self.label = ""
        self.narration = ""
        self.timestamp = ""
        self.command = ""
        self.read: Optional[Dict[str, Any]] = None
        self.adds: List[Add] = []
        self.sets: List[Set] = []
        self.touched: List[str] = []
        self.error_note: Optional[str] = None
        self.ops: List[str] = []

    @property
    def mutations(self) -> int:
        return len(self.adds) + len(self.sets)


class LogSnapshot:
    """Materialised log state. Built only in Python, never serialised per step."""

    def __init__(self, conference: str) -> None:
        self.conference = conference
        self.claims: List[Dict[str, Any]] = []
        self.questions: List[Dict[str, Any]] = []
        self.notes: List[Dict[str, Any]] = []
        self.outline = cli.new_outline(conference)

    def _find(self, bucket: List[Dict[str, Any]], entry_id: str) -> Optional[Dict[str, Any]]:
        return next((e for e in bucket if e["id"] == entry_id), None)

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        for bucket in (self.claims, self.questions, self.notes):
            found = self._find(bucket, entry_id)
            if found is not None:
                return found
        return None

    def reset(self, conference: str) -> None:
        """`init` starts a fresh log (review_cli.py cmd_init)."""
        self.conference = conference
        self.claims = []
        self.questions = []
        self.notes = []
        self.outline = cli.new_outline(conference)

    def as_review_log_json(self) -> Dict[str, Any]:
        return {
            "claims": self.claims,
            "questions": self.questions,
            "notes": self.notes,
            "conference": self.conference,
            "review_outline": self.outline,
        }


def _apply(snapshot: LogSnapshot, step: Step, parsed, scales: Dict[str, Any]) -> None:
    """Translate one parsed CLI invocation into patch ops against the snapshot."""
    sub = parsed.sub
    flags = parsed.flags

    if sub == "init":
        conference = parsed.get("conference") or cli.DEFAULT_CONFERENCE
        snapshot.reset(conference.lower())
        return

    if sub == "add_claim":
        entry_id = f"C{len(snapshot.claims) + 1}"
        fields = cli.new_claim(
            entry_id,
            parsed.get("text") or "",
            parsed.get("section") or "",
            parsed.get("claim_type") or "",
            cli.split_list(parsed.get("issues")),
        )
        snapshot.claims.append(fields)
        step.adds.append(Add("claim", entry_id, fields))

    elif sub == "add_question":
        entry_id = f"Q{len(snapshot.questions) + 1}"
        fields = cli.new_question(
            entry_id,
            parsed.get("text") or "",
            parsed.get("section") or "",
            parsed.get("type") or "clarification",
            cli.split_list(parsed.get("related_claims")),
        )
        snapshot.questions.append(fields)
        step.adds.append(Add("question", entry_id, fields))

    elif sub == "add_note":
        entry_id = f"N{len(snapshot.notes) + 1}"
        fields = cli.new_note(
            entry_id,
            parsed.get("text") or "",
            parsed.get("section") or "",
            cli.split_list(parsed.get("tag")),
        )
        snapshot.notes.append(fields)
        step.adds.append(Add("note", entry_id, fields))

    elif sub == "update_claim":
        entry_id = parsed.get("id") or ""
        claim = snapshot.get(entry_id)
        if claim is None:
            # The real CLI exits non-zero here, so the block would have errored.
            logger.warning("update_claim referenced unknown %s", entry_id)
            return
        status = parsed.get("status")
        if status:
            claim["status"] = status
            step.sets.append(Set(entry_id, "status", status))
        reason = parsed.get("reason")
        if reason:
            claim["verifier_reason"] = reason
            step.sets.append(Set(entry_id, "verifier_reason", reason))
        cross_refs = cli.split_list(parsed.get("cross_refs"))
        if cross_refs:
            claim["cross_references"] = cross_refs
            step.sets.append(Set(entry_id, "cross_references", cross_refs))

    elif sub == "resolve_question":
        entry_id = parsed.get("id") or ""
        question = snapshot.get(entry_id)
        if question is None:
            logger.warning("resolve_question referenced unknown %s", entry_id)
            return
        answer = parsed.get("answer") or ""
        sections = cli.split_list(parsed.get("sections"))
        status = parsed.get("status") or "resolved"
        question["answer"] = answer
        question["answer_sections"] = sections
        question["status"] = status
        step.sets.append(Set(entry_id, "answer", answer))
        step.sets.append(Set(entry_id, "answer_sections", sections))
        step.sets.append(Set(entry_id, "status", status))

    elif sub == "outline":
        section = parsed.get("section") or ""
        content = parsed.get("content")
        if section == "overall_score":
            score = cli.coerce_score(content)
            snapshot.outline["overall_score"] = score
            snapshot.outline["conference"] = snapshot.conference
            step.sets.append(Set("__outline__", "overall_score", score))
        elif section == "summary":
            snapshot.outline["summary"] = content or ""
            step.sets.append(Set("__outline__", "summary", content or ""))
        elif section in _OUTLINE_PREFIX:
            item = cli.new_outline_item(
                content or "",
                cli.split_list(parsed.get("claims")),
                cli.split_list(parsed.get("questions")),
                cli.split_list(parsed.get("notes")),
            )
            snapshot.outline[section].append(item)
            entry_id = f"{_OUTLINE_PREFIX[section]}{len(snapshot.outline[section])}"
            step.adds.append(Add("point", entry_id, item, list_name=section))

    elif sub == "set_conference":
        positional = flags.get("_positional") or []
        conference = (positional[0] if positional else "").lower()
        if conference not in scales:
            return
        snapshot.conference = conference
        snapshot.outline["conference"] = conference
        score = snapshot.outline.get("overall_score")
        if score is not None and score not in scales[conference]["scores"]:
            snapshot.outline["overall_score"] = None
            step.sets.append(Set("__outline__", "overall_score", None))


def _read_ref(event: ToolEvent) -> Optional[Dict[str, Any]]:
    """Describe what a Read call showed, preferring the archived file record."""
    if event.tool != "Read":
        return None
    file_record = event.file_result or {}
    path = file_record.get("filePath") or event.input.get("file_path") or ""
    start = file_record.get("startLine") or event.input.get("offset") or 1
    n_lines = file_record.get("numLines") or event.input.get("limit") or 0
    return {
        "file": path.rsplit("/", 1)[-1],
        "start": int(start),
        "lines": int(n_lines),
        "total": int(file_record.get("totalLines") or 0),
        "text": file_record.get("content") or "",
    }


def _label(event: ToolEvent, ops: List[str], read: Optional[Dict[str, Any]]) -> str:
    if read:
        end = read["start"] + read["lines"] - 1 if read["lines"] else read["start"]
        return f'Read {read["file"]} lines {read["start"]}–{end}'
    if ops:
        counts: Dict[str, int] = {}
        for op in ops:
            counts[op] = counts.get(op, 0) + 1
        return ", ".join(f"{op} x{n}" if n > 1 else op for op, n in counts.items())
    return event.description or (event.command.splitlines() or [""])[0][:70]


def build_steps(events: List[ToolEvent], scales: Dict[str, Any],
                conference: str = cli.DEFAULT_CONFERENCE) -> Tuple[List[Step], LogSnapshot]:
    """Fold the event stream into a patch journal plus the final log state."""
    snapshot = LogSnapshot(conference)
    steps: List[Step] = []
    seen_outline = False

    for event in events:
        step = Step(len(steps))
        step.tool = event.tool if event.tool in ("Bash", "Read") else "Other"
        step.narration = event.narration
        step.timestamp = event.timestamp[11:19] if len(event.timestamp) >= 19 else ""
        step.command = event.command
        attempted = _read_ref(event)

        if not event.paired:
            step.status = "unknown"
        elif event.is_error:
            rejected = "doesn't want to proceed" in (event.result_text or "")
            step.status = "rejected" if rejected else "error"
            step.error_note = (event.result_text or "").strip()[:240]

        # A Read that failed returned no text, so the agent did not have that
        # passage in front of it. This run contains one: a paper.md read from
        # the wrong directory, which would otherwise be drawn as a real excerpt
        # and counted as coverage. The attempt still names the step.
        step.read = attempted if step.status == "ok" else None

        if step.status == "ok" and event.tool == "Bash":
            for parsed in parse_blob(event.command):
                for warning in suspicious_values(parsed):
                    logger.warning("step %d: %s", step.index, warning)
                step.ops.append(parsed.sub)
                _apply(snapshot, step, parsed, scales)
        elif step.status != "ok" and event.tool == "Bash":
            # Record what it would have done, for the UI, without applying it.
            step.ops = [p.sub for p in parse_blob(event.command)]

        if any(op == "outline" for op in step.ops):
            seen_outline = True
        if seen_outline:
            step.phase = "outline"
        elif any(op in _MUTATORS for op in step.ops) or step.read:
            step.phase = "investigate"

        step.label = _label(event, step.ops, attempted)
        # `touched` means log entries only. Outline items are reported separately
        # so the log pane does not claim to have changed while the review is
        # being written, and so nothing tries to highlight an id that is not in
        # the log pane at all.
        step.touched = [a.id for a in step.adds if a.kind != "point"] + [
            s.id for s in step.sets if s.id != "__outline__"
        ]
        # Preserve order while removing duplicates from repeated Set ops.
        step.touched = list(dict.fromkeys(step.touched))
        steps.append(step)

    return steps, snapshot


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def _diff_paths(replayed: Any, gold: Any, path: str = "", out: Optional[List[str]] = None,
                limit: int = 12) -> List[str]:
    """Report differing paths between two JSON structures, most specific first."""
    out = out if out is not None else []
    if len(out) >= limit:
        return out
    if isinstance(replayed, dict) and isinstance(gold, dict):
        for key in sorted(set(replayed) | set(gold)):
            if key not in replayed:
                out.append(f"{path}.{key}: missing in replay")
            elif key not in gold:
                out.append(f"{path}.{key}: unexpected in replay")
            else:
                _diff_paths(replayed[key], gold[key], f"{path}.{key}", out, limit)
    elif isinstance(replayed, list) and isinstance(gold, list):
        if len(replayed) != len(gold):
            out.append(f"{path}: length {len(replayed)} != {len(gold)}")
        for i in range(min(len(replayed), len(gold))):
            _diff_paths(replayed[i], gold[i], f"{path}[{i}]", out, limit)
    elif replayed != gold:
        out.append(f"{path}: replay={_short(replayed)} gold={_short(gold)}")
    return out


def _short(value: Any, width: int = 70) -> str:
    text = repr(value)
    return text if len(text) <= width else text[:width] + "…"


def verify_final_state(snapshot: LogSnapshot, gold: Dict[str, Any]) -> List[str]:
    """Hard gate: the replayed log must equal the recorded review_log.json."""
    replayed = snapshot.as_review_log_json()
    if replayed == gold:
        return []
    return _diff_paths(replayed, gold, "log")


def verify_confirmations(steps: List[Step], events: List[ToolEvent]) -> List[str]:
    """Hard gate: the CLI's own stdout must agree with what the reducer minted.

    `is_error` only reports the exit status of a whole Bash block, so a command
    that failed midway through one is invisible to it. The CLI prints a line per
    successful mutation, which gives an independent channel: if the reducer
    minted ids the CLI never confirmed (or vice versa), they disagree here.
    """
    confirmed_adds: List[str] = []
    confirmed_updates = confirmed_resolves = 0
    for event in events:
        if event.tool != "Bash" or not event.paired or event.is_error:
            continue
        for match in _CONFIRMATION.finditer(event.stdout or ""):
            if match.group("added"):
                confirmed_adds.append(match.group("added"))
            elif match.group("updated"):
                confirmed_updates += 1
            elif match.group("resolved"):
                confirmed_resolves += 1

    replayed_adds = [a.id for s in steps for a in s.adds if a.kind != "point"]
    replayed_updates = sum(
        1 for s in steps if s.status == "ok" for op in s.ops if op == "update_claim"
    )
    replayed_resolves = sum(
        1 for s in steps if s.status == "ok" for op in s.ops if op == "resolve_question"
    )

    problems: List[str] = []
    if confirmed_adds and confirmed_adds != replayed_adds:
        problems.append(
            f"entry ids: CLI confirmed {len(confirmed_adds)} adds, replay minted "
            f"{len(replayed_adds)}; first divergence at "
            f"{next((i for i, (a, b) in enumerate(zip(confirmed_adds, replayed_adds)) if a != b), 'end')}"
        )
    if confirmed_resolves and confirmed_resolves != replayed_resolves:
        problems.append(
            f"resolve_question: CLI confirmed {confirmed_resolves}, replay applied {replayed_resolves}"
        )
    if confirmed_updates and confirmed_updates != replayed_updates:
        problems.append(
            f"update_claim: CLI confirmed {confirmed_updates}, replay applied {replayed_updates}"
        )
    return problems


def verify_review_md(snapshot: LogSnapshot, review_md: Optional[str],
                     scales: Dict[str, Any]) -> List[str]:
    """Warn-level: review.md may legitimately differ (e.g. --no-evidence)."""
    if review_md is None:
        return []
    rendered = cli.format_review(snapshot.as_review_log_json(), scales)
    if rendered.strip() == review_md.strip():
        return []
    return [
        "regenerated review.md differs from the one on disk "
        f"({len(rendered)} vs {len(review_md)} chars); the on-disk copy will be used"
    ]


def verify_paper_identity(steps: List[Step], paper_lines: List[str]) -> List[str]:
    """Warn-level: archived Read excerpts should match the shipped paper.md."""
    problems: List[str] = []
    for step in steps:
        read = step.read
        if not read or not read.get("text") or read["file"] != "paper.md":
            continue
        if read["total"] and read["total"] != len(paper_lines):
            problems.append(
                f"step {step.index}: paper.md had {read['total']} lines during the run "
                f"but {len(paper_lines)} on disk"
            )
            break
        start = read["start"] - 1
        expected = "\n".join(paper_lines[start:start + read["lines"]])
        archived = read["text"]
        if archived.strip() and expected.strip() and not _excerpt_matches(archived, expected):
            problems.append(f"step {step.index}: archived excerpt does not match paper.md")
            break
    return problems


def _excerpt_matches(archived: str, expected: str) -> bool:
    """Compare ignoring the Read tool's line-number gutter and trailing space."""
    def normalise(text: str) -> List[str]:
        out = []
        for line in text.splitlines():
            out.append(re.sub(r"^\s*\d+\t", "", line).rstrip())
        while out and not out[-1]:
            out.pop()
        return out
    return normalise(archived) == normalise(expected)
