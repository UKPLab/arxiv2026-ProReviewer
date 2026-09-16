"""Split a session transcript that reviewed several papers into one per review.

A single agent session often reviews several papers back to back, leaving one
transcript that covers all of them. The demo's replay assumes one transcript per
review log, so this cuts the shared transcript into per-review `trajectory.jsonl`
files that the existing pipeline can verify and animate unchanged.

The cut is by *contiguous block*, not by every mention: a review of paper X is
the run of records from the first command naming `review_X` up to the first
command naming the next review. Later mentions — copying results around,
listing directories once everything is finished — are housekeeping that happens
to name a directory, and attributing them to that review would put foreign steps
in its timeline.

Nothing here decides whether a split is *correct*. That is settled downstream:
build_demo.py replays each extracted transcript and refuses to publish one that
does not reproduce its own review_log.json.

Usage:
    python demo/split_transcript.py <transcript.jsonl> --out-root .
    python demo/split_transcript.py <transcript.jsonl> --out-root . --dry-run
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# review_0A4Uf88pog — an OpenReview-style id, as the CLI's output directory.
_SESSION_DIR = re.compile(r"review_([A-Za-z0-9_-]{6,})")

# Words that follow "review_" in the log format itself rather than naming a
# directory. Without this, `review_outline` is proposed as a session.
_NOT_A_SESSION = frozenset({"outline", "log", "logs", "output", "outputs"})

TRAJECTORY_FILE = "trajectory.jsonl"


def _mentions(record: Dict[str, Any]) -> List[str]:
    """Session slugs named by the tool calls in one assistant record, in order."""
    found: List[str] = []
    if record.get("type") != "assistant":
        return found
    for block in record.get("message", {}).get("content", []):
        if block.get("type") != "tool_use":
            continue
        payload = block.get("input") or {}
        haystack = " ".join(
            str(payload.get(key, "")) for key in
            ("command", "file_path", "path", "pattern", "description")
        )
        for match in _SESSION_DIR.finditer(haystack):
            if match.group(1).lower() in _NOT_A_SESSION:
                continue
            slug = "review_" + match.group(1)
            if slug not in found:
                found.append(slug)
    return found


def _read(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("%s:%d is not valid JSON; skipped", path, number)
    return records


def find_blocks(records: List[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    """Locate each review's contiguous block as (slug, start, end-exclusive).

    Only the *first* block for a slug is kept. A session that returns to a
    directory later is tidying up, not reviewing, and those records belong to
    no review.
    """
    starts: List[Tuple[int, str]] = []
    seen = set()
    for index, record in enumerate(records):
        for slug in _mentions(record):
            if slug not in seen:
                seen.add(slug)
                starts.append((index, slug))
            break            # the first slug a record names owns that record

    blocks: List[Tuple[str, int, int]] = []
    for position, (start, slug) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(records)
        blocks.append((slug, start, end))
    return blocks


def _pairs_forward(records: List[Dict[str, Any]], start: int, end: int) -> List[Dict[str, Any]]:
    """The block, plus any tool_result that lands after it.

    A block boundary can fall between a tool call and its result. Without the
    result the replay treats the call as unpaired and skips its effects, which
    would silently drop real log mutations — so trailing results are pulled in.
    """
    chunk = records[start:end]
    pending = set()
    for record in chunk:
        if record.get("type") != "assistant":
            continue
        for block in record.get("message", {}).get("content", []):
            if block.get("type") == "tool_use":
                pending.add(block.get("id"))
    for record in chunk:
        if record.get("type") != "user":
            continue
        content = record.get("message", {}).get("content")
        if isinstance(content, list):
            for block in content:
                pending.discard(block.get("tool_use_id"))

    if not pending:
        return chunk
    extra = []
    for record in records[end:]:
        if record.get("type") != "user":
            continue
        content = record.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        ids = {b.get("tool_use_id") for b in content if isinstance(b, dict)}
        if ids & pending:
            extra.append(record)
            pending -= ids
        if not pending:
            break
    if extra:
        logger.info("  carried %d trailing tool_result(s) across the boundary", len(extra))
    return chunk + extra


def split(transcript: str, out_root: str, dry_run: bool = False,
          only: Optional[List[str]] = None, force: bool = False) -> List[str]:
    """Write one trajectory.jsonl per review found in `transcript`."""
    records = _read(transcript)
    blocks = find_blocks(records)
    if not blocks:
        logger.error("no review_* directories are named anywhere in %s", transcript)
        return []

    written: List[str] = []
    for slug, start, end in blocks:
        target = os.path.join(out_root, slug)
        if only and slug not in only:
            logger.info("%-26s skipped (not requested)", slug)
            continue
        if not os.path.isdir(target):
            logger.warning("%-26s no such directory; skipped", slug)
            continue
        if not os.path.isfile(os.path.join(target, "review_log.json")):
            logger.warning("%-26s has no review_log.json; skipped", slug)
            continue

        # A session that reviewed paper A and later merely *copied* results for
        # paper B produces a short, meaningless block for B. Writing that over
        # B's real archived transcript would destroy it, so an existing file is
        # never replaced silently.
        existing = os.path.join(target, TRAJECTORY_FILE)
        if os.path.isfile(existing) and not force:
            logger.warning(
                "%-26s already has %s (%d KB); skipped — pass --force to replace",
                slug, TRAJECTORY_FILE, os.path.getsize(existing) // 1024)
            continue

        chunk = _pairs_forward(records, start, end)
        calls = sum(
            1 for r in chunk if r.get("type") == "assistant"
            for b in r.get("message", {}).get("content", [])
            if b.get("type") == "tool_use"
        )
        out = os.path.join(target, TRAJECTORY_FILE)
        logger.info("%-26s records %4d-%-4d  %3d tool calls -> %s",
                    slug, start, end - 1, calls, out)
        if dry_run:
            continue
        with open(out, "w", encoding="utf-8") as handle:
            for record in chunk:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        written.append(out)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split a multi-review session transcript into per-review trajectories.")
    parser.add_argument("transcript", help="the session .jsonl to split")
    parser.add_argument("--out-root", default=".",
                        help="directory holding the review_* folders (default: .)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="write only these session slugs")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the blocks without writing anything")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing trajectory.jsonl (destructive)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not os.path.isfile(args.transcript):
        logger.error("%s is not a file", args.transcript)
        return 1
    written = split(args.transcript, args.out_root, args.dry_run, args.only, args.force)
    if not args.dry_run:
        logger.info("wrote %d trajectory file(s)", len(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
