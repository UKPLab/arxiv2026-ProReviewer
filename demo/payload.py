"""Turn a replayed session into the compact JSON the page consumes.

Two derivations carry most of the meaning:

*Revisions.* An entry is not constant over time — Q4's answer at step 30 is a
different paragraph from its answer at step 33. Each entry therefore carries a
list of revisions, one per step that changed it, and the page shows the newest
revision at or before the current step. Only four entries in the reference
session have more than one, so this costs very little.

*Notable moments.* The most interesting thing in a recorded run is the agent
disagreeing with itself. Those revisions are classified as retractions (a
finding withdrawn) or reinforcements (a finding strengthened) and surfaced as
first-class navigation, because otherwise they are buried mid-scrub.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from . import cli_semantics as cli
from . import crossref
from .replay import Step
from .session import Session, heading_for_line, paper_headings, paper_title

# A revision whose text opens with one of these announces what kind of change it
# is. The agent writes them at character 0 of an 800-character paragraph, where
# nobody will see them; hoisting them out is free signal.
RETRACT_TAGS = frozenset({"CORRECTED", "RETRACTED", "WITHDRAWN", "REVERSED"})
REINFORCE_TAGS = frozenset({"CONFIRMED", "REFINED", "STRENGTHENED", "UPGRADED"})

_LEADING_TAG = re.compile(r"^([A-Z][A-Z]{3,})\b[ ,:.—-]*")

_KIND_OF = {"C": "claim", "Q": "question", "N": "note"}

_ADD_OP = {"claim": "add_claim", "question": "add_question", "note": "add_note"}
_UPDATE_OP = {"C": "update_claim", "Q": "resolve_question", "N": "update_note"}

# What "unsettled" means, and it is the loop's whole control flow: the agent
# keeps going while anything here is non-empty, and may only write the review
# once it is. Notes carry no status and are never on the agenda.
_UNSETTLED = {"C": frozenset({"to_be_verified"}),
              "Q": frozenset({"open", "partially_answered"})}

# The agent states its target before reading — "Iteration 2 — targeting **Q1**,
# **C1/C3/C4**" — so the ids it names are the log entries that sent it there.
_NAMED_ID = re.compile(r"\b([CQN]\d+)\b")


def _change_tag(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = _LEADING_TAG.match(text.strip())
    if not match:
        return None
    tag = match.group(1)
    return tag if tag in RETRACT_TAGS or tag in REINFORCE_TAGS else None


def _strip_tag(text: Optional[str], tag: Optional[str]) -> str:
    if not text or not tag:
        return text or ""
    return _LEADING_TAG.sub("", text.strip(), count=1)


def _entry_kind(entry_id: str) -> str:
    return _KIND_OF.get(entry_id[:1], "claim")


def build_entries(steps: List[Step]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reconstruct per-entry revision histories and the notable-change list.

    Walks the patch journal forward, materialising each entry's state so that a
    revision records the full field set as of that step rather than a delta the
    page would have to merge.
    """
    order: List[str] = []
    entries: Dict[str, Dict[str, Any]] = {}
    state: Dict[str, Dict[str, Any]] = {}
    notable: List[Dict[str, Any]] = []

    for step in steps:
        for add in step.adds:
            if add.kind == "point":
                continue
            state[add.id] = dict(add.fields)
            order.append(add.id)
            entries[add.id] = {
                "id": add.id,
                "kind": add.kind,
                "type": add.fields.get("type", ""),
                "text": add.fields.get("text") or add.fields.get("question") or "",
                "section": add.fields.get("section") or add.fields.get("source_section") or "",
                "issues": add.fields.get("issues", []),
                "tag": add.fields.get("tag", []),
                "born": step.index,
                "revs": [{
                    "step": step.index,
                    "status": add.fields.get("status", ""),
                }],
            }

        touched_this_step: Dict[str, List] = {}
        for op in step.sets:
            if op.id == "__outline__" or op.id not in state:
                continue
            state[op.id][op.field] = op.value
            touched_this_step.setdefault(op.id, []).append(op.field)

        for entry_id, fields in touched_this_step.items():
            entry = entries[entry_id]
            current = state[entry_id]
            previous = entry["revs"][-1]
            body = current.get("answer") if entry["kind"] == "question" else current.get("verifier_reason")
            tag = _change_tag(body)
            rev = {
                "step": step.index,
                "status": current.get("status", ""),
                "body": _strip_tag(body, tag) if tag else (body or ""),
                "srcs": current.get("answer_sections") or current.get("cross_references") or [],
            }
            if tag:
                rev["tag"] = tag

            # A revision that supersedes an already-settled answer is where the
            # agent changed its mind; distinguish withdrawing from strengthening.
            settled_before = previous.get("status") not in ("to_be_verified", "open", "")
            if settled_before:
                kind = None
                label = None
                if tag in RETRACT_TAGS:
                    kind, label = "retraction", "finding withdrawn"
                elif tag in REINFORCE_TAGS:
                    kind, label = "reinforce", "finding strengthened"
                elif previous.get("status") == "weak" and current.get("status") == "supported":
                    kind, label = "retraction", "criticism withdrawn"
                if kind:
                    rev["change"] = kind
                    rev["changeLabel"] = label
                    rev["supersedes"] = len(entry["revs"]) - 1
                    notable.append({
                        "step": step.index,
                        "kind": kind,
                        "id": entry_id,
                        "label": label,
                    })
            entry["revs"].append(rev)

    return [entries[i] for i in order], notable


def entries_from_log(log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build entries straight from review_log.json, with no revision history.

    Used for sessions that have no archived transcript. The log records only the
    final state of each entry, so each gets a single revision and `born` is -1,
    meaning "present from the start" — there is no timeline to place it on.
    """
    entries: List[Dict[str, Any]] = []
    for claim in log.get("claims", []):
        entries.append({
            "id": claim["id"], "kind": "claim", "type": claim.get("type", ""),
            "text": claim.get("text", ""), "section": claim.get("section", ""),
            "issues": claim.get("issues", []), "tag": [], "born": -1,
            "revs": [{
                "step": -1,
                "status": claim.get("status", ""),
                "body": claim.get("verifier_reason") or "",
                "srcs": claim.get("cross_references", []),
            }],
        })
    for question in log.get("questions", []):
        entries.append({
            "id": question["id"], "kind": "question", "type": question.get("type", ""),
            "text": question.get("question", ""),
            "section": question.get("source_section", ""),
            "issues": [], "tag": [], "born": -1,
            "revs": [{
                "step": -1,
                "status": question.get("status", ""),
                "body": question.get("answer") or "",
                "srcs": question.get("answer_sections", []),
            }],
        })
    for note in log.get("notes", []):
        entries.append({
            "id": note["id"], "kind": "note", "type": "",
            "text": note.get("text", ""), "section": note.get("section", ""),
            "issues": [], "tag": note.get("tag", []), "born": -1,
            "revs": [{"step": -1, "status": "", "body": "", "srcs": []}],
        })
    return entries


def build_review(log: Dict[str, Any], steps: List[Step]) -> Dict[str, Any]:
    """Review points with their evidence citations and the step they appeared."""
    born: Dict[str, int] = {}
    for step in steps:
        for add in step.adds:
            if add.kind == "point":
                born[add.id] = step.index

    sections = [("strengths", "S"), ("weaknesses", "W"), ("questions", "RQ")]
    points: List[Dict[str, Any]] = []
    for name, prefix in sections:
        for i, item in enumerate(log["review_outline"].get(name, []), 1):
            point_id = f"{prefix}{i}"
            points.append({
                "id": point_id,
                "section": name,
                "text": item["text"],
                "cites": cli.item_refs(item),
                # -1 = present from the start, for sessions with no timeline.
                "born": born.get(point_id, -1),
            })

    summary_step = -1
    for step in steps:
        if any(op.id == "__outline__" and op.field == "summary" for op in step.sets):
            summary_step = step.index
    return {
        "summary": log["review_outline"].get("summary", ""),
        "summaryBorn": summary_step,
        "points": points,
    }


def build_evidence_index(entries: List[Dict[str, Any]],
                         review: Dict[str, Any]) -> Dict[str, Any]:
    """Both directions of the citation graph, plus the entries nothing cites."""
    entry_to_points: Dict[str, List[str]] = {e["id"]: [] for e in entries}
    point_to_entries: Dict[str, List[str]] = {}
    for point in review["points"]:
        point_to_entries[point["id"]] = list(point["cites"])
        for ref in point["cites"]:
            if ref in entry_to_points:
                entry_to_points[ref].append(point["id"])
    orphans = [e["id"] for e in entries if not entry_to_points[e["id"]]]
    return {
        "entryToPoints": entry_to_points,
        "pointToEntries": point_to_entries,
        "orphans": orphans,
    }


def build_paper(session: Session, steps: List[Step]) -> Optional[Dict[str, Any]]:
    """Inline the paper plus the coverage the run actually achieved.

    The whole text is inlined rather than only the excerpts read: the excerpts
    are 82% of it, and what the agent *never opened* is itself worth showing.
    """
    if not session.has_paper:
        return None
    headings = paper_headings(session.paper_lines)

    spans: List[Tuple[int, int]] = []
    for step in steps:
        read = step.read
        if read and read["file"] == "paper.md" and read["lines"]:
            spans.append((read["start"], read["start"] + read["lines"] - 1))

    merged: List[List[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    total = len(session.paper_lines)
    covered = sum(min(end, total) - start + 1 for start, end in merged)
    gaps: List[List[int]] = []
    cursor = 1
    for start, end in merged:
        if start > cursor:
            gaps.append([cursor, start - 1])
        cursor = max(cursor, end + 1)
    if cursor <= total:
        gaps.append([cursor, total])

    return {
        "text": session.paper,
        "totalLines": total,
        "headings": headings,
        "read": merged,
        "gaps": gaps,
        "coverage": round(covered / total, 4) if total else 0,
    }


# Absolute paths from the machine the run happened on. The page shows the real
# shell commands, so without this a published build carries the author's home
# directory and the session UUIDs of their scratch space.
_REDACT = (
    (re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+"), "~"),
    (re.compile(r"/private/tmp/claude-\d+/[^\s\"']*|/tmp/claude-\d+/[^\s\"']*"), "$TMP"),
)


def redact(text: Optional[str]) -> str:
    """Strip machine-specific paths from anything the page displays.

    Display only: the replay itself runs on the raw transcript in Python, so
    redacting here cannot affect what is verified against review_log.json.
    """
    out = text or ""
    for pattern, replacement in _REDACT:
        out = pattern.sub(replacement, out)
    return out


def _clip(text: Any, width: int = 180) -> str:
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def build_writes(step: Step, status_of: Dict[str, str]) -> List[Dict[str, Any]]:
    """The log operations one step performed, in the order the CLI ran them.

    This is the ledger the page shows directly under the paper text that produced
    it: `add_claim → C13`, `update_claim C5: to_be_verified → weak`. A Set op
    records only the new value, so the previous status comes from `status_of`,
    a map the caller walks forward alongside the steps.
    """
    writes: List[Dict[str, Any]] = []

    for add in step.adds:
        if add.kind == "point":
            writes.append({
                "v": "point",
                "op": "outline",
                "id": add.id,
                "list": add.list_name or "",
                "text": _clip(add.fields.get("text", "")),
                "srcs": cli.item_refs(add.fields),
            })
            continue
        status_of[add.id] = add.fields.get("status", "")
        section = add.fields.get("section") or add.fields.get("source_section") or ""
        writes.append({
            "v": "add",
            "op": _ADD_OP.get(add.kind, "add"),
            "id": add.id,
            "to": add.fields.get("status", ""),
            "text": _clip(add.fields.get("text") or add.fields.get("question") or ""),
            "srcs": [section] if section else [],
        })

    # One CLI call writes several fields; they belong on one row, not three.
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for op in step.sets:
        if op.id == "__outline__":
            continue
        if op.id not in grouped:
            grouped[op.id] = {}
            order.append(op.id)
        grouped[op.id][op.field] = op.value

    for entry_id in order:
        fields = grouped[entry_id]
        before = status_of.get(entry_id, "")
        after = fields.get("status", before)
        status_of[entry_id] = after
        body = fields.get("answer") if entry_id[:1] == "Q" else fields.get("verifier_reason")
        writes.append({
            "v": "update",
            "op": _UPDATE_OP.get(entry_id[:1], "update"),
            "id": entry_id,
            "from": before,
            "to": after,
            "text": _clip(body or ""),
            "srcs": list(fields.get("answer_sections") or fields.get("cross_references") or []),
        })

    for op in step.sets:
        if op.id != "__outline__":
            continue
        # The score's value belongs with the review it summarises, not in a step
        # ledger, so only the field name is reported for it.
        writes.append({
            "v": "outline",
            "op": "outline",
            "id": op.field,
            "text": _clip(op.value) if op.field == "summary" else "",
            "srcs": [],
        })
    return writes


def unsettled(status_of: Dict[str, str]) -> List[List[str]]:
    """The open agenda: every entry still waiting for a verdict, in birth order."""
    return [
        [entry_id, status]
        for entry_id, status in status_of.items()
        if status in _UNSETTLED.get(entry_id[:1], ())
    ]


def named_targets(narration: str, known: Dict[str, str]) -> List[str]:
    """Entry ids the agent named in the narration that precedes a step.

    Restricted to entries that already exist, so a coincidental token cannot
    invent a target, and de-duplicated while preserving the order it wrote them.
    """
    out: List[str] = []
    for entry_id in _NAMED_ID.findall(narration or ""):
        if entry_id in known and entry_id not in out:
            out.append(entry_id)
    return out


def select_visible(steps: List[Step]) -> List[Step]:
    """Drop the steps that are about the harness rather than the paper.

    A recorded session opens with environment work — inspecting the input,
    installing dependencies, creating a venv, `init` — and contains calls the
    operator declined, which the agent simply re-issued a moment later. None of
    it is the review, and starting the replay there means a reader spends the
    first quarter of the run watching a virtualenv get built.

    Two rules, both conservative:

      * skip forward to the first step that successfully reads the paper or
        writes to the log;
      * drop any step that never ran and therefore changed nothing.

    Both are safe by the same invariant, asserted below: a dropped step has no
    adds and no sets, so nothing that happened to the log can vanish with it.
    The full transcript is still replayed and still verified against
    review_log.json — this only decides what is worth *showing*.
    """
    start = 0
    for i, step in enumerate(steps):
        if step.status == "ok" and (step.read or step.adds or step.sets):
            start = i
            break

    kept: List[Step] = []
    for step in steps[start:]:
        if step.status != "ok" and not step.adds and not step.sets:
            continue
        kept.append(step)

    dropped = [s for s in steps if s not in kept]
    mutating = [s.index for s in dropped if s.adds or s.sets]
    if mutating:  # pragma: no cover - guards a bug, not a data case
        raise AssertionError(
            f"refusing to hide steps that changed the log: {mutating}"
        )

    # Renumber so the timeline is contiguous. Everything downstream — entry
    # birth steps, `src` back-pointers, notable moments — is derived from
    # step.index after this point, so it stays consistent by construction.
    for new_index, step in enumerate(kept):
        step.index = new_index
    return kept


def build_steps_payload(steps: List[Step], paper: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    headings = paper["headings"] if paper else []
    out: List[Dict[str, Any]] = []
    status_of: Dict[str, str] = {}
    last_read: Optional[int] = None
    for step in steps:
        record: Dict[str, Any] = {
            "n": step.index,
            "tool": step.tool,
            "status": step.status,
            "phase": step.phase,
            "label": redact(step.label),
            "mut": step.mutations,
            "touched": step.touched,
        }
        if step.timestamp:
            record["t"] = step.timestamp
        if step.narration:
            record["say"] = redact(step.narration)
        if step.ops:
            record["ops"] = step.ops
        if step.error_note:
            record["err"] = redact(step.error_note)
        if step.command and step.tool == "Bash":
            record["cmd"] = redact(step.command)
        # The decision inputs, captured before this step's writes land: what the
        # log still owed an answer, and which of those the agent said it was
        # going after. This is the loop the whole method rests on — the open
        # entries choose where it reads next — and without it the page shows
        # reading and writing with nothing directing either.
        open_before = unsettled(status_of)
        targets = named_targets(step.narration, status_of)
        if open_before:
            record["open"] = open_before
        if targets:
            record["targets"] = targets
        if "show" in step.ops:
            record["consult"] = 1

        writes = build_writes(step, status_of)
        if writes:
            record["writes"] = writes

        open_after = {entry_id for entry_id, _ in unsettled(status_of)}
        settles = [i for i, _ in open_before if i not in open_after]
        if settles:
            record["settles"] = settles
        record["left"] = len(open_after)
        if step.read:
            read = dict(step.read)
            read.pop("text", None)
            if headings and read["file"] == "paper.md":
                title = heading_for_line(headings, read["start"])
                if title:
                    read["heading"] = title
            record["read"] = read
            last_read = step.index
        elif last_read is not None:
            # Which read fed this step. A step that writes to the log almost
            # never reads in the same call, so without this pointer the page
            # shows an entry appearing with the paper text that produced it
            # already scrolled away — which is the one link it exists to make
            # visible.
            record["src"] = last_read
        point_adds = [a.id for a in step.adds if a.kind == "point"]
        if point_adds:
            record["points"] = point_adds
        if any(op.id == "__outline__" for op in step.sets):
            record["outline"] = [
                {"p": op.field, "v": op.value} for op in step.sets if op.id == "__outline__"
            ]
        out.append(record)
    return out


def build_session_payload(session: Session, steps: List[Step], scales: Dict[str, Any],
                          review_md: str, warnings: List[str]) -> Dict[str, Any]:
    """Assemble everything the page needs for one session."""
    # Trim the harness noise before anything is derived, so every index the
    # page uses refers to the timeline the page actually shows. Verification
    # has already run against the full transcript in builder.py.
    recorded = len(steps)
    steps = select_visible(steps)
    hidden = recorded - len(steps)

    # Without a transcript there is no history to reconstruct, so fall back to
    # the log's final state; otherwise the log pane would render empty for every
    # session that was not recorded.
    if steps:
        entries, notable = build_entries(steps)
    else:
        entries, notable = entries_from_log(session.log), []
    review = build_review(session.log, steps)
    evidence = build_evidence_index(entries, review)
    paper = build_paper(session, steps)

    # The citations an entry carries are the only thing tying it to the passage
    # that settled it, so resolve them to real lines and let the page draw the
    # link. Anything that does not resolve is dropped, never guessed.
    links = 0
    if paper:
        index = crossref.build_index(paper["headings"], session.paper_lines)
        links = crossref.link_entries(entries, index)
        paper["index"] = [{"t": key, "n": line} for key, line in sorted(index.items())]

    outline = session.log["review_outline"]
    score = outline.get("overall_score")
    counts = {
        "claims": len(session.log["claims"]),
        "questions": len(session.log["questions"]),
        "notes": len(session.log["notes"]),
        "points": len(review["points"]),
        "steps": len(steps),
        # Kept so the page can say how much of the transcript it is not showing
        # rather than quietly presenting a trimmed run as the whole thing.
        "hiddenSteps": hidden,
        "recordedSteps": recorded,
        "retractions": sum(1 for n in notable if n["kind"] == "retraction"),
        "reinforcements": sum(1 for n in notable if n["kind"] == "reinforce"),
        "orphans": len(evidence["orphans"]),
        "links": links,
    }

    status_counts: Dict[str, int] = {}
    for claim in session.log["claims"]:
        status_counts[claim["status"]] = status_counts.get(claim["status"], 0) + 1
    for question in session.log["questions"]:
        status_counts[question["status"]] = status_counts.get(question["status"], 0) + 1

    return {
        "id": session.paper_id,
        "slug": session.slug,
        "title": paper_title(session),
        "short": session.meta.get("short") or paper_title(session).split(":")[0].strip(),
        "venue": session.meta.get("venue", ""),
        # The two selectors key on these: which paper, and which model reviewed it.
        "model": session.model or session.meta.get("model", "") or "unknown",
        "authors": session.meta.get("authors", []),
        "conference": session.log.get("conference", ""),
        # The score is deliberately not exposed as a headline field; it reaches
        # the page only inside the rendered review, where it has its context.
        "scoreLine": (
            cli.format_score_with_scale(score, session.log["conference"], scales)
            if score is not None else ""
        ),
        "counts": counts,
        "statusCounts": status_counts,
        "entries": entries,
        "steps": build_steps_payload(steps, paper),
        "review": review,
        "reviewMarkdown": review_md,
        "evidence": evidence,
        "notable": notable,
        "paper": paper,
        "hasRun": bool(steps),
        "warnings": warnings,
    }
