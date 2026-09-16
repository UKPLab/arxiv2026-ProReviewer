# demo

Builds a single self-contained HTML page from recorded ProReviewer sessions.
No CDN, no fetch, no npm — open the output straight from the filesystem.

```bash
python build_demo.py review_0A4Uf88pog
python build_demo.py review_0A4Uf88pog review_0ACUx9pMWJ --out demo/index.html
```

Sessions are named explicitly, so nothing is published by accident.

## What the page shows

Two tabs: **overview**, the project page, and **watch_a_run**, the recorded run.

Pick a paper and a model, then scrub. The layout is the method's own shape —
**the paper on the right is the environment, the review log on the left is the
state**, and a step is a move between them:

| Where | What it is |
|---|---|
| **Left · the review log** | Every entry that exists at this step. The ones this step touched are flagged *changed here*; the ones the agent named before reading are flagged *went looking for this*. The header carries the unsettled count, which is the loop's control flow. |
| **Middle · the step** | Sitting between the two things it connects, because that is what a step is. A large arrow points at the column about to change, over one of `fetched from the paper`, `wrote back to the log`, `wrote the review, out of the log`. Below it the commands as chips — `update_claim C5 · ? → ~ weak` — and the agent's own narration. |
| **Right · the paper** | The whole paper, all 838 lines, standing still. The passage the step is acting on is marked; the lines the run never opened stay faint. A second tab holds the review once it starts being written. |
| **The wires** | Drawn *through* the middle column, never across it: on a fetch, one wire from each entry into the channel and a single arrow out to the passage; on a write, the passage in and an arrow out to each entry it produced. |

Reading the three columns left to right is reading the step: this state, this
action, that evidence. The log is what the agent is *building*; the paper is
what it *consults* — which is why the log accumulates, the paper never changes,
and the thing between them is an action rather than a divider.

### The links are resolved, not guessed

Entries cite where their evidence lives — `§3.2`, `Table 5`, `§A.5` — as free
text. `crossref.py` resolves those strings against the shipped `paper.md` at
build time, and every reference that resolves becomes a button that jumps the
right column to that line and marks the block. In the reference run all 32
entries link, 112 links in total, and the build logs the rate so a change in
citation format shows up as a number rather than as links quietly vanishing.

Anything that does not resolve stays plain text. A link that scrolls to the
wrong section is a false claim about the paper made in the paper's own words,
so `test_crossref.py` checks every target lands on the thing it names —
including that `§Abstract` is not read as `§A` and sent to Appendix A, which it
was on the first attempt.

In the reference run the open list drains to zero at step 21 and the first line
of the review is written at step 28. That ordering is the method's exit
condition, so it is a test
(`test_the_agenda_drains_before_the_review_is_written`) rather than a caption.

### The replay starts at the paper, not at the venv

A recorded session opens with environment work — inspecting the input,
installing pydantic, creating a venv, `init` — and contains calls the operator
declined and the agent re-issued a moment later. `select_visible()` skips
forward to the first step that successfully reads the paper or writes to the
log, and drops anything that never ran. In the reference session that hides 12
of 46 steps, and the timeline is renumbered so every index the page uses refers
to what it actually shows.

This is only honest because of one invariant, asserted in the code and tested in
`test_hidden_steps_never_touched_the_log`: **a hidden step has no adds and no
sets.** Nothing that happened to the log can disappear with it. The full
transcript is still replayed and still verified against `review_log.json` — the
trim decides what is worth showing, never what happened. The page says how many
steps it is not showing, under the rail.

Three rules keep the chain from asserting more than the data supports. A `Read`
that errored is not shown as an excerpt and is not counted as coverage — this
run contains one, `paper.md` opened from the wrong directory. The carried
excerpt is only carried onto steps that wrote log entries; while the agent is
writing the review it is working from the log, and band 2 says so. And a target
is only shown when the agent named that entry itself and the entry already
exists, so free-form narration cannot invent an intention.

## The claim this rests on

Replaying the `review_cli` calls in `trajectory.jsonl` — skipping those whose
recorded result is an error — reproduces the session's `review_log.json`
exactly. The build asserts this and **refuses to emit a page when it fails**,
because animating a run the agent did not perform is worse than showing no run.
`--on-mismatch skip_trajectory` degrades that session to its log and review
instead; `--on-mismatch warn` ships it flagged.

Two further checks run per session: the CLI's own printed confirmations must
agree with the ids the replay minted (this catches a command that failed *inside*
a multi-line shell block, which the transcript's error flag does not report), and
archived read excerpts must match the shipped `paper.md`.

## Session directory

```
review_<paper>/
  review_log.json    required
  trajectory.jsonl   optional — without it, no "Watch a run" for that session
  paper.md           optional — read excerpts and the coverage ribbon
  review.md          optional — regenerated from the log if absent
  meta.json          optional — {"title": ..., "authors": [...], "venue": ...}
```

`meta.json` exists because converted papers usually have no title: `paper.md`
commonly begins at `## Abstract`, so reading the first heading yields
`1 Introduction`. Without it the session is labelled by its paper id rather than
by something confidently wrong.

Anything missing degrades one part of the page and is reported in the build log
and in the page's own footer, rather than disappearing silently.

## Layout

| File | Role |
|---|---|
| `shellparse.py` | Shell splitting and argv parsing. The riskiest part — a bug here yields a wrong but plausible demo — so it is isolated and directly tested. |
| `cli_semantics.py` | `review_cli` semantics ported without pydantic, each citing its source lines. `CONFERENCE_SCALES` is AST-loaded rather than copied so it cannot drift. |
| `transcript.py` | `trajectory.jsonl` → tool events, paired with their results. |
| `replay.py` | Events → patch journal + final state, plus the verifications. |
| `session.py`, `payload.py` | Loading, and the JSON the page consumes. |
| `builder.py` | Orchestration and asset inlining. |
| `assets/` | `template.html`, `demo.css`, `demo.js`, inlined at build time. |

The page ships a patch journal, not per-step snapshots: 46 snapshots of this log
would be 2.4 MB, the journal is ~65 KB, and its final fold *is* `review_log.json`.

## Tests

```bash
python demo/tests/test_replay.py     # parser + exact replay against review_log.json
python demo/tests/test_payload.py    # time-travel invariants of the shipped data
osascript -l JavaScript demo/tests/dom_shim.js   # boots demo.js and drives it
```

The last one runs the real `demo.js` against a small DOM shim under
JavaScriptCore — it boots the page, scrubs the run, and exercises the evidence
explorer, so a runtime error surfaces here rather than in front of a reader. It
needs macOS; the first two are plain Python and are the ones that gate the build.

## A note on the score

The reviewer's overall score appears only inside the rendered review. It is the
one output with no evidence links — every other point cites the entries it came
from — so putting it in the header would make the least grounded artifact the
most prominent thing on a page arguing the opposite. Session tabs carry the
run's shape (claims, questions, retractions) instead.
