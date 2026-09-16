"""Read a Claude Code session transcript into an ordered stream of tool events.

trajectory.jsonl is a raw session log with mixed record types. What the demo
needs from it is: the tool calls in order, whether each one actually succeeded,
what it printed, and the assistant's own narration leading up to it.

Two details are easy to get wrong and both are handled here:

  * ``is_error`` reflects the exit status of an entire Bash block. A command
    that failed *inside* a multi-line block is invisible to it, which is why
    replay.py additionally cross-checks the CLI's printed confirmations.
  * ``toolUseResult`` carries structured output (stdout/stderr for Bash, a file
    record for Read) and is preferred over the flattened ``content`` string.
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class ToolEvent:
    """One tool invocation, paired with its result."""

    __slots__ = ("index", "tool", "tool_use_id", "input", "timestamp",
                 "is_error", "stdout", "stderr", "result_text", "file_result",
                 "narration")

    def __init__(self, index: int, tool: str, tool_use_id: str,
                 tool_input: Dict[str, Any], timestamp: str, narration: str) -> None:
        self.index = index
        self.tool = tool
        self.tool_use_id = tool_use_id
        self.input = tool_input
        self.timestamp = timestamp
        self.narration = narration
        self.is_error: Optional[bool] = None
        self.stdout: str = ""
        self.stderr: str = ""
        self.result_text: str = ""
        self.file_result: Optional[Dict[str, Any]] = None

    @property
    def paired(self) -> bool:
        """False when the session ended before the result was recorded."""
        return self.is_error is not None

    @property
    def command(self) -> str:
        return self.input.get("command", "") if self.tool == "Bash" else ""

    @property
    def description(self) -> str:
        return self.input.get("description", "") or ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ToolEvent({self.index}, {self.tool}, is_error={self.is_error})"


def _iter_records(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s:%d is not valid JSON; skipped", path, line_no)


def _flatten_content(content: Any) -> str:
    """tool_result.content is a str in some versions and a block list in others."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return ""


def read_events(path: str) -> List[ToolEvent]:
    """Parse a transcript into tool events in chronological order.

    Sidechain records (subagent transcripts) are excluded: their tool calls
    belong to a different agent and would inject foreign mutations.
    """
    events: List[ToolEvent] = []
    by_id: Dict[str, ToolEvent] = {}
    pending_narration: List[str] = []

    for record in _iter_records(path):
        kind = record.get("type")
        if record.get("isSidechain") or record.get("isMeta"):
            continue

        if kind == "assistant":
            for block in record.get("message", {}).get("content", []):
                block_type = block.get("type")
                if block_type == "text":
                    text = (block.get("text") or "").strip()
                    if text:
                        pending_narration.append(text)
                elif block_type == "tool_use":
                    event = ToolEvent(
                        index=len(events),
                        tool=block.get("name", ""),
                        tool_use_id=block.get("id", ""),
                        tool_input=block.get("input", {}) or {},
                        timestamp=record.get("timestamp", "") or "",
                        narration="\n\n".join(pending_narration),
                    )
                    pending_narration = []
                    events.append(event)
                    by_id[event.tool_use_id] = event

        elif kind == "user":
            content = record.get("message", {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") != "tool_result":
                    continue
                event = by_id.get(block.get("tool_use_id"))
                if event is None:
                    continue
                event.is_error = bool(block.get("is_error", False))
                event.result_text = _flatten_content(block.get("content"))
                structured = record.get("toolUseResult")
                if isinstance(structured, dict):
                    event.stdout = structured.get("stdout", "") or ""
                    event.stderr = structured.get("stderr", "") or ""
                    file_record = structured.get("file")
                    if isinstance(file_record, dict):
                        event.file_result = file_record

    unpaired = [e for e in events if not e.paired]
    if unpaired:
        logger.warning(
            "%d tool call(s) have no recorded result; their effects are unknown "
            "and will be skipped", len(unpaired)
        )
    return events


def read_model(path: str) -> Optional[str]:
    """The model that produced the run, as recorded in the transcript.

    Taken from the transcript rather than configured, so the label on the page
    is what actually ran. Returns the most frequent value, since a session can
    in principle switch models partway through.
    """
    counts: Dict[str, int] = {}
    for record in _iter_records(path):
        if record.get("type") != "assistant" or record.get("isSidechain"):
            continue
        model = record.get("message", {}).get("model")
        if model:
            counts[model] = counts.get(model, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def successful(events: List[ToolEvent]) -> List[ToolEvent]:
    """Events that actually ran to completion.

    An unpaired event is excluded because we cannot know whether it landed;
    including it would silently invent state, and excluding it makes the replay
    assertion fail loudly instead.
    """
    return [e for e in events if e.paired and not e.is_error]
