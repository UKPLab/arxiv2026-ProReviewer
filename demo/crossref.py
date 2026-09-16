"""Resolve the section references in a log entry to lines in the paper.

Entries cite where their evidence lives — ``§3.2``, ``Table 5``, ``§A.5``,
``Appendix C`` — as free text written by the agent. Those strings are the only
thing connecting a claim to the passage that settled it, and the page draws that
connection as a link. So the resolution happens here, in Python, against the
paper actually shipped, and anything that does not resolve is dropped rather
than guessed: a link that scrolls to the wrong section is worse than no link.

Two reference vocabularies appear in practice, both handled:

  * Headings, which carry their own number — ``## 3.2 Benchmark Construction``
    is matched by ``§3.2``, and ``## Appendix B ...`` by both ``§B`` and
    ``Appendix B``.
  * Float captions, which are body text — ``**Table 5: ...**`` on its own line.

Compound strings (``§4.1,Tables 7-8``, ``§Abstract vs §B``, ``§1 Table 1``) are
split into atoms and resolved individually, because the agent writes them as
prose and each part points somewhere different.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# "3.2 Benchmark Construction" / "A.5 Implementation" / "Appendix B Additional"
_HEADING_NUMBER = re.compile(r"^(?:Appendix\s+)?([A-Z]|\d+)((?:\.\d+)*)\b")

# One citable atom. Ordered: the float forms must win before the bare-number
# form gets a chance to read "5" out of "Table 5".
_ATOM = re.compile(
    r"(?P<range>Tables?\s+(?P<rfrom>\d+)\s*[-–—]\s*(?P<rto>\d+))"
    r"|(?P<float>(?P<fkind>Table|Figure)\s+(?P<fnum>\d+))"
    r"|(?P<appendix>Appendix\s+(?P<anum>[A-Z]))"
    # Must carry its own optional §, or "§Abstract" is matched by the section
    # rule below as "§A" and links to Appendix A — a confidently wrong target.
    r"|(?P<abstract>§?\s*Abstract)"
    r"|(?P<section>§\s*(?P<snum>[A-Z]|\d+)(?P<ssub>(?:\.\d+)*))"
    r"|(?P<bare>\b(?P<bnum>\d+(?:\.\d+)+)\b)",
    re.IGNORECASE,
)


def _heading_key(title: str) -> Optional[str]:
    """The number a heading declares, normalised: '3.2', 'A', 'B', '1'."""
    match = _HEADING_NUMBER.match(title.strip())
    if not match:
        return None
    return (match.group(1) + match.group(2)).upper()


def build_index(headings: List[Dict[str, Any]],
                paper_lines: List[str]) -> Dict[str, int]:
    """Map every citable label in the paper to the line it starts on.

    First occurrence wins: a heading is defined where it is declared, and a
    table caption is where the table is, not where it is later discussed.
    """
    index: Dict[str, int] = {}

    def put(key: str, line: int) -> None:
        index.setdefault(key.upper(), line)

    for heading in headings:
        title = heading["title"].strip()
        key = _heading_key(title)
        if key:
            put("S:" + key, heading["line"])
        if title.lower().startswith("abstract"):
            put("S:ABSTRACT", heading["line"])

    for number, line in enumerate(paper_lines, 1):
        stripped = line.strip().lstrip("*").strip()
        # Two caption shapes survive conversion: "**Table 5: ...**" and the
        # doubled "Figure: Figure 12: ...". Both are the float's real location.
        match = re.match(
            r"^(?:(?:Table|Figure)\s*:\s*)?(Table|Figure)\s+(\d+)\s*[:.]",
            stripped, re.IGNORECASE,
        )
        if match:
            put("F:" + match.group(1).upper() + match.group(2), number)
    return index


def resolve(ref: str, index: Dict[str, int]) -> List[Tuple[str, int]]:
    """Every target a reference string names, as (label, line) in citation order.

    Unresolvable atoms are dropped silently — the agent cites things that are
    genuinely not headings ("the discussion in §5 vs the abstract"), and the
    caller renders only what came back.
    """
    out: List[Tuple[str, int]] = []
    seen = set()

    def add(label: str, key: str) -> None:
        line = index.get(key.upper())
        if line is None or label in seen:
            return
        seen.add(label)
        out.append((label, line))

    for match in _ATOM.finditer(ref or ""):
        if match.group("range"):
            # "Tables 7-8" is two links, not one; the reader means both.
            first, last = int(match.group("rfrom")), int(match.group("rto"))
            if last - first > 12:          # a page range, not a float range
                continue
            for n in range(first, last + 1):
                add("Table %d" % n, "F:TABLE%d" % n)
        elif match.group("float"):
            kind = match.group("fkind").title()
            add("%s %s" % (kind, match.group("fnum")),
                "F:%s%s" % (kind.upper(), match.group("fnum")))
        elif match.group("appendix"):
            add("Appendix " + match.group("anum").upper(), "S:" + match.group("anum"))
        elif match.group("abstract"):
            add("Abstract", "S:ABSTRACT")
        elif match.group("section"):
            key = (match.group("snum") + (match.group("ssub") or "")).upper()
            add("§" + key, "S:" + key)
        elif match.group("bare"):
            # Only dotted numbers, so a stray "189 programs" cannot become §189.
            key = match.group("bnum").upper()
            add("§" + key, "S:" + key)
    return out


def link_entries(entries: List[Dict[str, Any]], index: Dict[str, int]) -> int:
    """Attach resolved paper targets to each entry revision, in place.

    Returns how many links were made, so the build can report the resolution
    rate rather than let a silent regression in the reference format pass as
    "this run simply had no citations".
    """
    made = 0
    for entry in entries:
        targets = resolve(entry.get("section", ""), index)
        if targets:
            entry["links"] = [{"t": label, "n": line} for label, line in targets]
            made += len(targets)
        for rev in entry.get("revs", []):
            found: List[Tuple[str, int]] = []
            for src in rev.get("srcs", []):
                for pair in resolve(src, index):
                    if pair not in found:
                        found.append(pair)
            if found:
                rev["links"] = [{"t": label, "n": line} for label, line in found]
                made += len(found)
    return made
