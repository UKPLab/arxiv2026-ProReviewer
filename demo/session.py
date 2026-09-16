"""Load one recorded review session from disk.

A session directory is whatever `review_cli.py` wrote plus, optionally, an
archived transcript:

    review_<paper>/
        review_log.json     required — without it there is no session
        paper.md            optional — read excerpts and section links
        review.md           optional — regenerable from the log
        trajectory.jsonl    optional — without it there is no "Watch a run"
        meta.json           optional — {"title": ..., "model": ...} labels

meta.json exists because converted papers usually carry no title: paper.md
commonly starts at "## Abstract", so there is nothing to read a title from.

Anything missing degrades a specific part of the demo rather than failing the
build, and every degradation is recorded so it can be surfaced in the UI instead
of silently disappearing.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LOG_FILE = "review_log.json"
PAPER_FILE = "paper.md"
REVIEW_FILE = "review.md"
TRAJECTORY_FILE = "trajectory.jsonl"
META_FILE = "meta.json"

# "## 3 Verina : Data Format" or "### A.5 Implementation..." in the converted paper.
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class SessionError(Exception):
    """The directory cannot be used as a session at all."""


class Session:
    """A loaded session plus a record of what it is missing."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self.slug = os.path.basename(self.path.rstrip(os.sep))
        self.paper_id = self.slug[len("review_"):] if self.slug.startswith("review_") else self.slug
        self.log: Dict[str, Any] = {}
        self.paper: Optional[str] = None
        self.paper_lines: List[str] = []
        self.review_md: Optional[str] = None
        self.trajectory_path: Optional[str] = None
        self.meta: Dict[str, Any] = {}
        self.model: str = ""
        self.degraded: List[str] = []

    @property
    def has_trajectory(self) -> bool:
        return self.trajectory_path is not None

    @property
    def has_paper(self) -> bool:
        return self.paper is not None

    def _read(self, name: str) -> Optional[str]:
        path = os.path.join(self.path, name)
        if not os.path.isfile(path):
            return None
        try:
            return open(path, encoding="utf-8").read()
        except OSError as exc:
            logger.warning("%s: could not read %s (%s)", self.slug, name, exc)
            return None


def load_session(path: str) -> Session:
    """Load a session directory, raising only when the review log is unusable."""
    session = Session(path)
    if not os.path.isdir(session.path):
        raise SessionError(f"{path} is not a directory")

    raw = session._read(LOG_FILE)
    if raw is None:
        raise SessionError(f"{path} has no {LOG_FILE}")
    try:
        session.log = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SessionError(f"{path}/{LOG_FILE} is not valid JSON: {exc}") from exc

    for key in ("claims", "questions", "notes", "review_outline"):
        if key not in session.log:
            raise SessionError(f"{path}/{LOG_FILE} is missing '{key}'")
    session.log.setdefault("conference", "iclr")

    session.model = ""
    raw_meta = session._read(META_FILE)
    if raw_meta:
        try:
            session.meta = json.loads(raw_meta)
        except json.JSONDecodeError as exc:
            logger.warning("%s: ignoring malformed %s (%s)", session.slug, META_FILE, exc)

    session.model = session.meta.get("model", "") or session.model
    session.paper = session._read(PAPER_FILE)
    if session.paper is None:
        session.degraded.append("paper.md is missing; read excerpts will be unavailable")
    else:
        session.paper_lines = session.paper.split("\n")

    session.review_md = session._read(REVIEW_FILE)
    if session.review_md is None:
        session.degraded.append("review.md is missing; it will be regenerated from the log")

    trajectory = os.path.join(session.path, TRAJECTORY_FILE)
    if os.path.isfile(trajectory):
        session.trajectory_path = trajectory
    else:
        session.degraded.append(
            "trajectory.jsonl is missing; this session has no recorded run to replay"
        )

    for reason in session.degraded:
        logger.warning("%s: %s", session.slug, reason)
    return session


def paper_headings(lines: List[str]) -> List[Dict[str, Any]]:
    """Index markdown headings so a read range can be named by its section."""
    out: List[Dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        match = _HEADING.match(line)
        if match:
            out.append({
                "line": number,
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
            })
    return out


def heading_for_line(headings: List[Dict[str, Any]], line: int) -> Optional[str]:
    """The nearest heading at or above `line`."""
    title = None
    for heading in headings:
        if heading["line"] <= line:
            title = heading["title"]
        else:
            break
    return title


def paper_title(session: Session) -> str:
    """Display title for the session.

    Converted papers frequently have no title at all — paper.md often begins at
    "## Abstract" — so the first heading is not a usable source: it yields
    "1 Introduction". Prefer an explicit meta.json, accept a genuine H1, and
    otherwise show the paper id rather than something confidently wrong.
    """
    title = (session.meta.get("title") or "").strip()
    if title:
        return title
    for line in session.paper_lines[:60]:
        match = _HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return _strip_emphasis(match.group(2).strip())
    return session.paper_id


def _strip_emphasis(text: str) -> str:
    """Drop markdown emphasis from a heading.

    Converted papers often render the title as `# **Title**`, and the markers
    would otherwise show up literally in a tab label.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"<sup>(.+?)</sup>", r"\1", text)
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()
