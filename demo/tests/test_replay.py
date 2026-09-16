"""Golden test: replaying the recorded transcript must reproduce review_log.json.

This is the assertion the whole demo rests on. If it fails, the "Watch a run"
animation would be showing something the agent did not do.

Run:
    python -m pytest demo/tests/test_replay.py -q
    python demo/tests/test_replay.py          # also works without pytest
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from demo import cli_semantics as cli           # noqa: E402
from demo.replay import (                        # noqa: E402
    build_steps,
    verify_confirmations,
    verify_final_state,
    verify_review_md,
)
from demo.shellparse import parse_blob, split_commands, tokenize  # noqa: E402
from demo.transcript import read_events          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSION = os.path.join(REPO, "review_0A4Uf88pog")


def _load():
    events = read_events(os.path.join(SESSION, "trajectory.jsonl"))
    scales, _ = cli.load_conference_scales()
    steps, snapshot = build_steps(events, scales)
    gold = json.load(open(os.path.join(SESSION, "review_log.json"), encoding="utf-8"))
    return events, steps, snapshot, gold, scales


def test_split_commands_respects_quotes():
    blob = 'a --text "one && two\nthree" \nb --text \'x;y\''
    assert split_commands(blob) == ['a --text "one && two\nthree"', "b --text 'x;y'"]


def test_tokenize_unescapes_dollar_like_bash():
    # shlex would yield a literal backslash here; bash does not.
    assert tokenize('x --reason "costs \\$894.38 total"')[-1] == "costs $894.38 total"


def test_tokenize_single_quotes_are_literal():
    assert tokenize("x --t 'a \\$b'")[-1] == "a \\$b"


def test_subcommand_found_through_shell_variables():
    parsed = parse_blob('$RV $CLI --dir $D add_claim --text "hi" --section "§1"')
    assert len(parsed) == 1
    assert parsed[0].sub == "add_claim"
    assert parsed[0].get("text") == "hi"
    assert parsed[0].get("section") == "§1"


def test_newline_separated_commands_are_not_merged():
    blob = (
        '$RV $CLI --dir $D add_claim --text "a" --section "§1"\n'
        '$RV $CLI --dir $D add_claim --text "b" --section "§2"'
    )
    assert [p.sub for p in parse_blob(blob)] == ["add_claim", "add_claim"]


def test_replay_reproduces_review_log_exactly():
    """The load-bearing assertion."""
    _, _, snapshot, gold, _ = _load()
    problems = verify_final_state(snapshot, gold)
    assert not problems, "replay diverged from review_log.json:\n  " + "\n  ".join(problems)


def test_replay_matches_cli_confirmations():
    """Independent channel: the CLI's own stdout must agree with the reducer."""
    events, steps, _, _, _ = _load()
    problems = verify_confirmations(steps, events)
    assert not problems, "\n  ".join(problems)


def test_errored_calls_do_not_mutate_state():
    """The rejected batch of 7 add_claim calls must leave no trace."""
    _, steps, snapshot, gold, _ = _load()
    rejected = [s for s in steps if s.status in ("error", "rejected")]
    assert rejected, "expected the recorded run to contain failed calls"
    assert all(s.mutations == 0 for s in rejected)
    assert len(snapshot.claims) == len(gold["claims"])


def test_failed_reads_are_not_presented_as_text_the_agent_read():
    """A Read that errored returned nothing, so it must not become an excerpt.

    This run contains one: paper.md read from the wrong directory at step 11.
    Keeping it would draw line 1 of the paper as if the agent had it in front of
    it, and count it as coverage — the page's read band asserts exactly that.
    """
    _, steps, _, _, _ = _load()
    failed_reads = [s for s in steps if s.tool == "Read" and s.status != "ok"]
    assert failed_reads, "expected the recorded run to contain a failed Read"
    for step in failed_reads:
        assert step.read is None, f"step {step.index} exposes a failed read"
        assert step.label, f"step {step.index} lost its label along with its read"


def test_review_md_round_trips():
    _, _, snapshot, _, scales = _load()
    review_md = open(os.path.join(SESSION, "review.md"), encoding="utf-8").read()
    assert not verify_review_md(snapshot, review_md, scales)


def test_adds_record_birth_state_not_final_state():
    """An entry's Add must capture how it was born, not how it ended up.

    The snapshot dict handed to Add is mutated in place by later Set ops, so a
    missing copy silently rewrites history: every claim would appear to have
    been born already verified.
    """
    _, steps, _, _, _ = _load()
    born = {a.id: a.fields for s in steps for a in s.adds if a.kind == "claim"}
    assert born, "expected claims in the recorded run"
    assert all(f["status"] == "to_be_verified" for f in born.values())
    assert all(f["verifier_reason"] is None for f in born.values())

    questions = {a.id: a.fields for s in steps for a in s.adds if a.kind == "question"}
    assert all(f["status"] == "open" and f["answer"] is None for f in questions.values())


def test_revised_entries_keep_their_full_history():
    """C5, Q4, Q7 and Q10 were each written twice; both versions must survive."""
    from demo.payload import build_entries

    _, steps, _, _, _ = _load()
    entries, notable = build_entries(steps)
    by_id = {e["id"]: e for e in entries}

    for entry_id in ("C5", "Q4", "Q7", "Q10"):
        assert len(by_id[entry_id]["revs"]) == 3, f"{entry_id} lost a revision"

    kinds = {n["id"]: n["kind"] for n in notable}
    assert kinds["Q4"] == "retraction"    # answer withdrawn after §A.3
    assert kinds["C5"] == "retraction"    # weak -> supported, criticism withdrawn
    assert kinds["Q7"] == "reinforce"     # CONFIRMED by direct evidence
    assert kinds["Q10"] == "reinforce"    # REFINED with a corrected denominator


def test_orphan_entries_are_detected():
    """Entries no review point cites are the demo's headline finding."""
    from demo.payload import build_entries, build_evidence_index, build_review

    _, steps, _, gold, _ = _load()
    entries, _ = build_entries(steps)
    index = build_evidence_index(entries, build_review(gold, steps))
    assert sorted(index["orphans"]) == ["C12", "N2", "Q2"]


def test_journal_is_smaller_than_snapshots():
    """The patch journal must stay far below the cost of per-step snapshots."""
    _, steps, snapshot, _, _ = _load()
    journal = json.dumps(
        [{"a": [a.to_json() for a in s.adds], "s": [x.to_json() for x in s.sets]} for s in steps],
        ensure_ascii=False, separators=(",", ":"),
    )
    one_snapshot = len(json.dumps(snapshot.as_review_log_json(), ensure_ascii=False))
    assert len(journal) < one_snapshot * 3


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
