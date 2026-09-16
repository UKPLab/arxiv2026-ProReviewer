"""The link from a log entry to the passage it cites.

Every one of these links is an assertion that a specific line of the paper is
where a specific claim was settled. A link that lands on the wrong section is a
false statement about the paper made in the paper's own words, and it is the
kind that reads as authoritative — so resolution is verified here against the
real document rather than trusted.

Run:
    python demo/tests/test_crossref.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demo import crossref                      # noqa: E402
from demo.session import load_session, paper_headings   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSION = os.path.join(REPO, "review_0A4Uf88pog")

_cache = {}


def fixture():
    if "f" not in _cache:
        session = load_session(SESSION)
        lines = session.paper_lines
        _cache["f"] = (crossref.build_index(paper_headings(lines), lines), lines)
    return _cache["f"]


def test_section_refs_land_on_their_own_heading():
    index, lines = fixture()
    expected = {
        "§3.2": "3.2 Benchmark Construction",
        "§4.1": "4.1 Foundational Tasks",
        "§A.5": "A.5 Implementation",
        "§5": "5 Experimental Evaluation",
        "§B": "Appendix B",
        "Appendix C": "Appendix C",
    }
    for ref, prefix in expected.items():
        found = crossref.resolve(ref, index)
        assert found, f"{ref} did not resolve"
        line = lines[found[0][1] - 1].lstrip("#").strip()
        assert line.startswith(prefix), f"{ref} -> line {found[0][1]}: {line[:60]!r}"


def test_abstract_is_not_mistaken_for_appendix_a():
    """'§Abstract' must not be read as '§A'.

    The section rule matches a § followed by a letter, so without an explicit
    Abstract rule ordered ahead of it, '§Abstract' resolves to Appendix A — a
    confidently wrong target 556 lines from the truth. This regressed once.
    """
    index, lines = fixture()
    for ref in ("§Abstract", "§Abstract,§1", "§Abstract vs §B"):
        found = crossref.resolve(ref, index)
        assert found, f"{ref} did not resolve"
        label, line = found[0]
        assert label == "Abstract", f"{ref} -> {label}"
        assert lines[line - 1].lstrip("#").strip().lower().startswith("abstract")


def test_float_refs_land_on_their_caption():
    index, lines = fixture()
    for n in range(1, 9):
        found = crossref.resolve("Table %d" % n, index)
        assert found, f"Table {n} did not resolve"
        assert ("Table %d:" % n) in lines[found[0][1] - 1]
    # Figures survive conversion as "Figure: Figure 12: ..." rather than a
    # bolded caption, which the naive caption pattern missed entirely.
    found = crossref.resolve("Figure 12", index)
    assert found and "Figure 12:" in lines[found[0][1] - 1]


def test_compound_refs_resolve_every_part():
    index, _ = fixture()
    cases = {
        "§1 Table 1": ["§1", "Table 1"],
        "§3.1,§3.2": ["§3.1", "§3.2"],
        "§4.1,Tables 7-8": ["§4.1", "Table 7", "Table 8"],
        "Tables 6-8": ["Table 6", "Table 7", "Table 8"],
        "§B,Table 6": ["§B", "Table 6"],
    }
    for ref, expected in cases.items():
        assert [label for label, _ in crossref.resolve(ref, index)] == expected, ref


def test_unresolvable_references_are_dropped_not_guessed():
    index, _ = fixture()
    # No Figure 18 exists in this paper; the §-less prose around it must not
    # produce a link to something merely nearby.
    labels = [label for label, _ in crossref.resolve("Appendix C Figure 18", index)]
    assert labels == ["Appendix C"], labels
    assert crossref.resolve("Table 99", index) == []
    assert crossref.resolve("the discussion", index) == []
    # A bare integer is not a section: "189 programs" must not become §189.
    assert crossref.resolve("189 programs were curated", index) == []


def test_every_link_points_inside_the_paper():
    index, lines = fixture()
    for key, line in index.items():
        assert 1 <= line <= len(lines), f"{key} -> {line}, paper has {len(lines)}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
