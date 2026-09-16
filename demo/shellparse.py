"""Parse review_cli.py invocations out of recorded shell commands.

This is the riskiest module in the demo builder: a subtle parsing bug produces a
demo that is wrong but entirely plausible. Three behaviours are load-bearing and
each is verified by demo/tests/test_replay.py:

1. Commands in a transcript are newline-separated. ``shlex.split`` treats "\\n"
   as ordinary whitespace, so splitting the whole blob at once silently merges
   adjacent commands (94 invocations collapse to 30). Split on *unquoted*
   newlines, ";", "&&", "||" and "|" first.

2. ``shlex`` does not unescape "\\$" inside double quotes; bash does. Without
   normalisation a recorded ``"\\$894.38"`` parses to the literal ``\\$894.38``
   while the review log holds ``$894.38``.

3. The subcommand cannot be found by matching on "review_cli.py": recorded
   commands invoke it through shell variables, e.g.
   ``$RV $CLI --dir $D add_claim --text "..."``. Scan tokens left to right,
   consuming ``--flag value`` pairs, until a token matches a known subcommand.
"""

from typing import Dict, List, Optional, Tuple

# Subcommands of ProReviewer-Skill/proreviewer/review_cli.py (the `commands`
# dispatch dict). Anything outside this set is not a log mutation.
SUBCOMMANDS = frozenset({
    "init",
    "add_claim",
    "update_claim",
    "add_question",
    "resolve_question",
    "add_note",
    "outline",
    "set_conference",
    "show",
    "export",
    "finalize",
})

# Flags that take no value.
BOOLEAN_FLAGS = frozenset({"force", "detailed", "no-evidence", "no_evidence"})

# Operators that terminate a command when they appear outside quotes. Longest
# first so "&&" is matched before "&".
_OPERATORS: Tuple[str, ...] = ("&&", "||", ";;", "\n", ";", "|")


class ParsedCommand:
    """One review_cli invocation: its subcommand and its flags."""

    __slots__ = ("sub", "flags", "raw")

    def __init__(self, sub: str, flags: Dict[str, object], raw: str) -> None:
        self.sub = sub
        self.flags = flags
        self.raw = raw

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        value = self.flags.get(name, default)
        return value if isinstance(value, str) or value is None else default

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ParsedCommand({self.sub!r}, {self.flags!r})"


def split_commands(blob: str) -> List[str]:
    """Split a shell blob into individual commands on unquoted operators.

    Quote state is tracked so that operators inside "..." or '...' — which occur
    constantly in recorded --text and --answer values — do not split a command.
    """
    parts: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    i = 0
    n = len(blob)

    while i < n:
        ch = blob[i]

        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                # Preserve the escape pair intact; unescaping happens later.
                buf.append(blob[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if ch == "\\" and i + 1 < n:
            # Unquoted line continuation joins the next line.
            if blob[i + 1] == "\n":
                i += 2
                continue
            buf.append(ch)
            buf.append(blob[i + 1])
            i += 2
            continue

        matched = next((op for op in _OPERATORS if blob.startswith(op, i)), None)
        if matched:
            parts.append("".join(buf))
            buf = []
            i += len(matched)
            continue

        buf.append(ch)
        i += 1

    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


# Inside double quotes bash unescapes only these; every other backslash is
# literal. Single quotes unescape nothing at all, which is why escapes must be
# resolved while scanning rather than applied to the finished token.
_DQ_ESCAPABLE = frozenset({"$", "`", '"', "\\"})


def tokenize(command: str) -> List[str]:
    """Split one command into argv, applying bash's quoting rules.

    Implemented directly rather than via shlex because shlex's escapedquotes
    handling does not match bash for "\\$" and "\\`" (see module docstring).
    """
    tokens: List[str] = []
    buf: List[str] = []
    started = False
    quote: Optional[str] = None
    i = 0
    n = len(command)

    while i < n:
        ch = command[i]

        if quote == "'":
            # Single quotes are fully literal: no escapes at all.
            if ch == "'":
                quote = None
            else:
                buf.append(ch)
            i += 1
            continue

        if quote == '"':
            if ch == "\\" and i + 1 < n:
                nxt = command[i + 1]
                if nxt == "\n":          # line continuation: both chars vanish
                    i += 2
                    continue
                if nxt in _DQ_ESCAPABLE:
                    buf.append(nxt)
                    i += 2
                    continue
                buf.append(ch)           # backslash is literal before anything else
                i += 1
                continue
            if ch == '"':
                quote = None
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            started = True
            i += 1
            continue

        if ch == "\\" and i + 1 < n:
            buf.append(command[i + 1])
            started = True
            i += 2
            continue

        if ch.isspace():
            if started:
                tokens.append("".join(buf))
                buf = []
                started = False
            i += 1
            continue

        buf.append(ch)
        started = True
        i += 1

    if started:
        tokens.append("".join(buf))
    return tokens


def parse_invocation(command: str) -> Optional[ParsedCommand]:
    """Return the review_cli invocation in `command`, or None if there is none.

    Tokens before the subcommand (interpreter, script path, --dir) are skipped.
    A --flag's value is only consumed when it does not itself look like a flag,
    so boolean flags such as --force parse correctly.
    """
    tokens = tokenize(command)
    if not tokens:
        return None

    idx: Optional[int] = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            name = token[2:]
            if "=" in name:
                i += 1
                continue
            if name not in BOOLEAN_FLAGS and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 2
                continue
            i += 1
            continue
        if token in SUBCOMMANDS:
            idx = i
            break
        i += 1

    if idx is None:
        return None

    flags: Dict[str, object] = {}
    positional: List[str] = []
    j = idx + 1
    while j < len(tokens):
        token = tokens[j]
        if token.startswith("--"):
            name = token[2:]
            if "=" in name:
                key, _, value = name.partition("=")
                flags[key.replace("-", "_")] = value
                j += 1
                continue
            key = name.replace("-", "_")
            if name in BOOLEAN_FLAGS or j + 1 >= len(tokens) or tokens[j + 1].startswith("--"):
                flags[key] = True
                j += 1
                continue
            flags[key] = tokens[j + 1]
            j += 2
            continue
        positional.append(token)
        j += 1

    if positional:
        flags["_positional"] = positional
    return ParsedCommand(tokens[idx], flags, command)


def parse_blob(blob: str) -> List[ParsedCommand]:
    """Extract every review_cli invocation from a recorded shell command blob."""
    found: List[ParsedCommand] = []
    for command in split_commands(blob):
        parsed = parse_invocation(command)
        if parsed is not None:
            found.append(parsed)
    return found


def suspicious_values(parsed: ParsedCommand) -> List[str]:
    """Flag argument values that bash would have expanded but we took literally.

    Recorded commands only use variables in command position ($RV/$CLI/$D), so a
    "$" or backtick surviving into a *value* means the replay may diverge from
    what actually ran. The caller warns; the replay assertion would fail anyway,
    but this names the cause.
    """
    hits: List[str] = []
    for key, value in parsed.flags.items():
        if not isinstance(value, str):
            continue
        if "`" in value:
            hits.append(f"--{key} contains a backtick")
            continue
        for pos, ch in enumerate(value):
            if ch != "$" or pos + 1 >= len(value):
                continue
            if value[pos + 1].isalpha() or value[pos + 1] in "_{(":
                hits.append(f"--{key} contains an unexpanded variable reference")
                break
    return hits
