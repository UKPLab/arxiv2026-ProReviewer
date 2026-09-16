"""Invariants of the payload the page animates.

The page's time travel is a lookup: at step n an entry shows its newest revision
at or before n. These tests assert the properties that lookup depends on, in
Python, against the real session — the equivalent JS is five lines, so the risk
lives in the data, not the traversal. The Add-aliasing bug that made every claim
appear to be born already verified would have been caught here.

Run:
    python -m pytest demo/tests/test_payload.py -q
    python demo/tests/test_payload.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demo import cli_semantics as cli      # noqa: E402
from demo.builder import _build_one        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSION = os.path.join(REPO, "review_0A4Uf88pog")

_cache = {}


def payload():
    if "p" not in _cache:
        scales, _ = cli.load_conference_scales()
        _cache["p"] = _build_one(SESSION, scales, "raise")
    return _cache["p"]


def rev_index_at(entry, step):
    """Mirror of revIndexAt() in demo.js."""
    idx = -1
    for i, rev in enumerate(entry["revs"]):
        if rev["step"] <= step:
            idx = i
        else:
            break
    return idx


def test_revisions_are_chronological():
    for entry in payload()["entries"]:
        steps = [r["step"] for r in entry["revs"]]
        assert steps == sorted(steps), f"{entry['id']} revisions out of order"
        assert steps[0] == entry["born"], f"{entry['id']} first revision is not its birth"


def test_nothing_is_visible_before_it_exists():
    data = payload()
    for entry in data["entries"]:
        assert rev_index_at(entry, entry["born"] - 1) == -1
        assert rev_index_at(entry, entry["born"]) == 0
    for point in data["review"]["points"]:
        assert point["born"] >= 0, f"{point['id']} was never added by any step"


def test_final_step_matches_the_review_log():
    """Folding to the last step must equal the log that shipped."""
    data = payload()
    gold = json.load(open(os.path.join(SESSION, "review_log.json"), encoding="utf-8"))
    final = len(data["steps"]) - 1
    expected = {c["id"]: c["status"] for c in gold["claims"]}
    expected.update({q["id"]: q["status"] for q in gold["questions"]})

    for entry in data["entries"]:
        if entry["kind"] == "note":
            continue
        rev = entry["revs"][rev_index_at(entry, final)]
        assert rev["status"] == expected[entry["id"]], (
            f"{entry['id']} folds to {rev['status']}, log says {expected[entry['id']]}"
        )


def test_every_touched_id_exists_and_is_born_by_then():
    """A step cannot highlight an entry that does not yet exist."""
    data = payload()
    born = {e["id"]: e["born"] for e in data["entries"]}
    for step in data["steps"]:
        for entry_id in step.get("touched", []):
            assert entry_id in born, f"step {step['n']} touches unknown {entry_id}"
            assert born[entry_id] <= step["n"], (
                f"step {step['n']} touches {entry_id}, born at {born[entry_id]}"
            )


def test_failed_steps_change_nothing():
    for step in payload()["steps"]:
        if step["status"] != "ok":
            assert step["mut"] == 0 and not step["touched"]


def test_review_is_written_only_after_the_investigation():
    """The review pane staying empty for most of the run is the point."""
    data = payload()
    first_point = min(p["born"] for p in data["review"]["points"])
    last_entry = max(e["revs"][-1]["step"] for e in data["entries"])
    assert first_point > last_entry, (
        "review points appear before the log finished settling: "
        f"first point at {first_point}, last log change at {last_entry}"
    )
    assert first_point > len(data["steps"]) * 0.7


def test_the_agenda_drains_before_the_review_is_written():
    """The loop's exit condition, which the page now states outright.

    The agent may only write the review once every claim has a verdict and every
    question is answered. If a review point were ever born while something was
    still unsettled, the page's central claim about the method would be false.
    """
    data = payload()
    left_at = {s["n"]: s["left"] for s in data["steps"]}
    assert max(left_at.values()) > 0, "expected the run to have an open agenda at some point"
    assert left_at[len(data["steps"]) - 1] == 0, "the run ended with entries unsettled"
    for point in data["review"]["points"]:
        assert left_at[point["born"]] == 0, (
            f"{point['id']} was written at step {point['born']} with "
            f"{left_at[point['born']]} entries still unsettled"
        )


def test_the_agenda_only_names_entries_that_exist():
    """Nothing on the agenda may be invented, and nothing struck off unlisted."""
    data = payload()
    born = {e["id"]: e["born"] for e in data["entries"]}
    for step in data["steps"]:
        listed = set()
        for entry_id, status in step.get("open", []):
            assert entry_id in born, f"step {step['n']} lists unknown {entry_id}"
            assert born[entry_id] <= step["n"], f"step {step['n']} lists unborn {entry_id}"
            listed.add(entry_id)
        # Targets are parsed out of free-form narration, so this is the check
        # that keeps a stray token from becoming a fabricated intention.
        for entry_id in step.get("targets", []):
            assert entry_id in born and born[entry_id] <= step["n"], (
                f"step {step['n']} claims to target {entry_id}, which does not exist yet"
            )
        for entry_id in step.get("settles", []):
            assert entry_id in listed, (
                f"step {step['n']} strikes off {entry_id}, which it never listed as open"
            )


def test_hidden_steps_never_touched_the_log():
    """Trimming the setup must not be able to lose anything that happened.

    The page shows fewer steps than were recorded. That is only honest while
    every hidden step is inert — if one ever mutated the log, entries would
    appear in the replay from nowhere, and the run would be a fiction.
    """
    from demo import cli_semantics as cli
    from demo.payload import select_visible
    from demo.replay import build_steps
    from demo.session import load_session
    from demo.transcript import read_events

    scales, _ = cli.load_conference_scales()
    session = load_session(SESSION)
    # read_events, not successful(): the builder keeps failed calls so they can
    # be shown, and they are exactly what the trim is deciding about.
    events = read_events(session.trajectory_path)
    steps, _ = build_steps(events, scales, session.log["conference"])

    recorded = len(steps)
    kept = {id(s) for s in select_visible(list(steps))}
    hidden = [s for s in steps if id(s) not in kept]

    assert hidden, "expected this run to contain setup worth hiding"
    for step in hidden:
        assert not step.adds and not step.sets, (
            f"hidden step would lose {len(step.adds)} add(s) and {len(step.sets)} set(s)"
        )
    assert recorded - len(hidden) == payload()["counts"]["steps"]
    assert payload()["counts"]["recordedSteps"] == recorded


def test_the_run_opens_on_the_paper_not_the_setup():
    """The first thing a reader sees must be review work."""
    data = payload()
    first = data["steps"][0]
    assert first["status"] == "ok", first
    assert first.get("read") or first["mut"], (
        f"the run opens on {first['label']!r}, which neither reads nor writes"
    )
    # Nothing that never ran survives into the visible timeline.
    for step in data["steps"]:
        if step["status"] != "ok":
            assert step["mut"], f"step {step['n']} never ran and changed nothing"


def test_citations_resolve_to_real_entries():
    data = payload()
    known = {e["id"] for e in data["entries"]}
    for point in data["review"]["points"]:
        for ref in point["cites"]:
            assert ref in known, f"{point['id']} cites unknown {ref}"


def test_evidence_graph_is_symmetric():
    data = payload()
    forward = data["evidence"]["pointToEntries"]
    reverse = data["evidence"]["entryToPoints"]
    for point_id, entries in forward.items():
        for entry_id in entries:
            assert point_id in reverse[entry_id]
    for entry_id, points in reverse.items():
        for point_id in points:
            assert entry_id in forward[point_id]
    orphans = {e for e, p in reverse.items() if not p}
    assert orphans == set(data["evidence"]["orphans"])


def test_notable_moments_point_at_real_revisions():
    data = payload()
    by_id = {e["id"]: e for e in data["entries"]}
    assert data["notable"], "expected the run to contain self-corrections"
    for item in data["notable"]:
        entry = by_id[item["id"]]
        match = [r for r in entry["revs"] if r["step"] == item["step"]]
        assert match, f"{item['id']} has no revision at step {item['step']}"
        assert match[0].get("change") == item["kind"]
        assert match[0].get("supersedes") is not None


def test_score_is_not_exposed_as_a_headline_field():
    """The score reaches the page only inside the rendered review."""
    data = payload()
    assert "overall_score" not in data
    assert "score" not in data
    for key in ("counts", "statusCounts"):
        assert "score" not in json.dumps(data[key])


def test_paper_coverage_spans_are_sane():
    paper = payload()["paper"]
    assert paper["read"], "expected the agent to have read something"
    for start, end in paper["read"]:
        assert 1 <= start <= end <= paper["totalLines"]
    for i in range(1, len(paper["read"])):
        assert paper["read"][i][0] > paper["read"][i - 1][1] + 1, "spans not merged"
    assert 0 < paper["coverage"] <= 1


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
