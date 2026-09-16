"""Assemble recorded sessions into one self-contained interactive HTML page.

    from demo import build_demo
    build_demo(["review_0A4Uf88pog"], out="demo/index.html")

Everything is inlined: no CDN, no fetch, no build tooling. The page works when
opened from the filesystem.

The build refuses to emit a page whose replay disagrees with the session's
recorded review_log.json. A demo that confidently animates something the agent
did not do would be worse than no demo, so that check is a hard gate rather
than a warning.
"""

import base64
import html
import json
import logging
import os
from typing import Any, Dict, List, Optional

from . import cli_semantics as cli
from . import crossref
from .payload import build_session_payload
from .replay import (
    build_steps,
    verify_confirmations,
    verify_final_state,
    verify_paper_identity,
    verify_review_md,
)
from .session import SessionError, load_session
from .transcript import read_events, read_model

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
PROJECT_FILE = os.path.join(HERE, "project.json")
DEFAULT_OUT = os.path.join("demo", "index.html")
DEFAULT_TITLE = "ProReviewer — how a review gets built"
MAX_BYTES = 8_000_000


class DemoBuildError(RuntimeError):
    """The demo could not be built."""


class ReplayMismatch(DemoBuildError):
    """A session's replay does not reproduce its recorded review log."""


def _load_project() -> Dict[str, Any]:
    """Project metadata for the landing page.

    Kept in demo/project.json rather than in code so that links, authors and the
    citation can be corrected without touching the builder — and so that nothing
    on the page is a URL this module invented.
    """
    try:
        with open(PROJECT_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logger.warning("%s not found; the overview will be mostly empty", PROJECT_FILE)
        return {}
    except json.JSONDecodeError as exc:
        raise DemoBuildError(f"{PROJECT_FILE} is not valid JSON: {exc}") from exc


def _asset(name: str) -> str:
    path = os.path.join(ASSETS, name)
    try:
        return open(path, encoding="utf-8").read()
    except OSError as exc:
        raise DemoBuildError(f"missing asset {path}: {exc}") from exc


def _plain(text: str) -> str:
    """Drop the emphasis markers project.json carries for the page's own copy.

    Fields like `subtitle` are rendered through inlineMd in the browser so a
    phrase can be highlighted, but the same string also becomes the <title>,
    where `**` would show up literally in the browser tab.
    """
    return text.replace("**", "").replace("`", "")


def _figure_data_uri(name: str = "method.png") -> str:
    """Inline the method figure, or return "" if it is not there.

    The page is one self-contained file that has to work over file://, so the
    figure travels as a data URI rather than a second request. It is optional:
    without it the figure block hides itself rather than rendering a broken
    image, and the build says so.
    """
    path = os.path.join(ASSETS, name)
    if not os.path.isfile(path):
        logger.warning("%s not found; the method figure will be omitted", path)
        return ""
    with open(path, "rb") as handle:
        blob = handle.read()
    kind = "svg+xml" if name.endswith(".svg") else name.rsplit(".", 1)[-1]
    logger.info("inlined %s (%d KB as base64)", name, len(blob) * 4 // 3 // 1024)
    return "data:image/%s;base64,%s" % (kind, base64.b64encode(blob).decode("ascii"))


def _embed_json(payload: Dict[str, Any]) -> str:
    """Serialise for a <script type="application/json"> island.

    Both sequences below would otherwise terminate the script element or open a
    comment; escaping them keeps the document parseable without touching the
    JSON semantics.
    """
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--")


def _build_one(path: str, scales: Dict[str, Any], on_replay_mismatch: str) -> Optional[Dict[str, Any]]:
    """Load, replay, verify and package a single session."""
    try:
        session = load_session(path)
    except SessionError as exc:
        logger.error("%s", exc)
        return None

    warnings: List[str] = list(session.degraded)
    steps = []
    events = []

    if session.has_trajectory:
        session.model = read_model(session.trajectory_path) or ""
        events = read_events(session.trajectory_path)
        steps, snapshot = build_steps(events, scales, session.log.get("conference", "iclr"))

        mismatch = verify_final_state(snapshot, session.log)
        confirmations = verify_confirmations(steps, events)
        hard = mismatch + confirmations

        if hard:
            detail = "\n  ".join(hard)
            message = (
                f"{session.slug}: the replayed run does not reproduce {session.slug}/"
                f"review_log.json:\n  {detail}"
            )
            if on_replay_mismatch == "raise":
                raise ReplayMismatch(message)
            logger.warning("%s", message)
            if on_replay_mismatch == "skip_trajectory":
                warnings.append(
                    "the recorded run could not be replayed faithfully, so it is not shown"
                )
                steps = []
            else:
                warnings.append("the recorded run did not verify; it is shown unverified")
        else:
            logger.info("%s: replay verified against review_log.json", session.slug)

        if steps:
            warnings += verify_review_md(snapshot, session.review_md, scales)
            if session.has_paper:
                warnings += verify_paper_identity(steps, session.paper_lines)

    review_md = session.review_md or cli.format_review(session.log, scales)
    payload = build_session_payload(session, steps, scales, review_md, warnings)
    hidden = payload["counts"].get("hiddenSteps", 0)
    logger.info(
        "%s: %d steps, %d entries, %d review points%s",
        session.slug, payload["counts"]["steps"], len(payload["entries"]),
        len(payload["review"]["points"]),
        "" if steps else " (no run to replay)",
    )
    if hidden:
        # Never silent: the replay verified against all N steps but shows fewer,
        # and the difference should be a logged number rather than a discovery.
        logger.info(
            "%s: hiding %d setup/no-op step(s) of %d recorded; all %d replayed "
            "for verification", session.slug, hidden,
            payload["counts"]["recordedSteps"], payload["counts"]["recordedSteps"],
        )

    # Report the citation resolution rate rather than letting a regression in
    # the reference format pass as "this run simply had no citations". The page
    # draws a link for each one, so silence here would be silence about a
    # feature quietly disappearing.
    cited = sum(
        1 for entry in payload["entries"]
        if entry.get("section") or any(rev.get("srcs") for rev in entry["revs"])
    )
    if cited:
        resolved = sum(
            1 for entry in payload["entries"]
            if entry.get("links") or any(rev.get("links") for rev in entry["revs"])
        )
        logger.info(
            "%s: %d/%d entries linked to the paper (%d links)",
            session.slug, resolved, cited, payload["counts"]["links"],
        )
        # A low rate usually is not a bug in the resolver: papers whose headings
        # survive conversion unnumbered ("## Introduction", not "## 1
        # Introduction") cannot be matched by the §N the agent cites, and
        # numbering them by position would guess wrong wherever the conversion
        # also flattened a subsection. Say which it is, so the number is
        # explained rather than mysterious.
        paper_block = payload.get("paper") or {}
        headings = paper_block.get("headings") or []
        if headings and resolved < cited:
            numbered = sum(1 for h in headings if crossref._heading_key(h["title"]))
            if numbered < len(headings) * 0.5:
                logger.info(
                    "%s: %d of %d headings carry no section number, so §N "
                    "citations cannot be resolved; those stay plain text",
                    session.slug, len(headings) - numbered, len(headings),
                )
        if resolved < cited * 0.5:
            # Appended to the payload directly: `warnings` was handed to the
            # payload builder above, and mutating it afterwards would depend on
            # the two being the same list object.
            payload["warnings"].append(
                f"only {resolved} of {cited} entries could be linked to paper.md; "
                "the citation format may have changed"
            )
    return payload


def build_demo(
    sessions: List[str],
    out: str = DEFAULT_OUT,
    *,
    on_replay_mismatch: str = "raise",
    title: str = DEFAULT_TITLE,
) -> str:
    """Build the demo page from explicit session directories.

    Args:
        sessions: paths to review_<paper>/ directories, in display order.
        out: path of the HTML file to write.
        on_replay_mismatch: "raise" (default) aborts the build; "skip_trajectory"
            drops that session's replay but keeps its log and review; "warn"
            ships it flagged as unverified.
        title: document title.

    Returns:
        The path written.
    """
    if not sessions:
        raise DemoBuildError("no sessions given")
    if on_replay_mismatch not in ("raise", "skip_trajectory", "warn"):
        raise DemoBuildError(f"unknown on_replay_mismatch: {on_replay_mismatch!r}")

    scales, _ = cli.load_conference_scales()
    for warning in cli.check_port_drift():
        logger.warning("%s", warning)

    built = [p for p in (_build_one(s, scales, on_replay_mismatch) for s in sessions) if p]
    if not built:
        raise DemoBuildError("no usable sessions; nothing to build")

    project = _load_project()
    payload = {
        "schema": 1,
        "project": project,
        "sessions": built,
        "build": {
            "sessions": [
                {"slug": s["slug"], "hasRun": s["hasRun"], "warnings": s["warnings"]}
                for s in built
            ],
        },
    }

    page = (
        _asset("template.html")
        .replace("{{TITLE}}", html.escape(
            title if title != DEFAULT_TITLE or not project.get("name")
            else "%s — %s" % (project["name"], _plain(project.get("subtitle", "")))))
        .replace("{{CSS}}", _asset("demo.css"))
        .replace("{{JS}}", _asset("demo.js"))
        .replace("{{FIGURE}}", _figure_data_uri())
        .replace("{{DATA}}", _embed_json(payload))
    )

    if len(page.encode("utf-8")) > MAX_BYTES:
        raise DemoBuildError(
            f"page is {len(page.encode('utf-8')) // 1024} KB, over the {MAX_BYTES // 1024} KB guard"
        )

    directory = os.path.dirname(os.path.abspath(out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(page)

    logger.info("wrote %s (%d KB, %d session(s))",
                out, len(page.encode("utf-8")) // 1024, len(built))
    return out
