---
name: proreviewer
description: Review scientific papers using the ProReviewer evidence-based review method. Use when asked to review a paper, critique a manuscript, write a peer review, or evaluate research quality. Produces structured reviews grounded in specific paper evidence via a ReviewLog of claims, questions, and notes.
argument-hint: "[paper-path]"
allowed-tools: Bash(python *) Read Grep
---
# ProReviewer — Evidence-Based Paper Review

## Overview

ProReviewer is a structured method for reviewing scientific papers. Instead of reading and then writing impressions, you **proactively investigate**: extract author claims, raise questions, gather evidence, verify or refute claims, and build every review point from that evidence.

**Core principle:** Every strength, weakness, and question in the final review must trace back to specific evidence found in the paper.

## Workflow

Use `${CLAUDE_SKILL_DIR}/review_cli.py` to maintain a persistent ReviewLog. The ReviewLog is your **working state**: after every step you consult it and let its open entries decide your next action. Do **not** read the paper linearly front-to-back and then batch-log impressions — that reduces the log to post-hoc documentation. Instead, unverified claims and open questions determine *where you read next*.

**IMPORTANT:** The `init` command prints an output directory path (e.g. `review_my_paper/`). You MUST pass `--dir <that_path>` to every subsequent command. Store the path in a variable for reuse.

1. **Initialize** — convert the paper and create the output directory. Use `--conference` to set the rating scale.
   ```bash
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf                      # default: ICLR scale
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf --conference icml     # ICML 1-6 scale
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf --conference neurips  # NeurIPS 1-6 scale
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf --conference acl      # ACL/ARR 1-5 scale (0.5 steps)
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf --conference emnlp    # EMNLP/ARR 1-5 scale (0.5 steps)
   ```
   This creates `review_<paper_name>/` containing `paper.md` and `review_log.json`. Note the output directory path from the output — you need it for all subsequent commands.

   Accepts `.pdf`, `.tex`, `.md`, or a **directory** containing a LaTeX project:
   ```bash
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.tex            # single .tex file
   python ${CLAUDE_SKILL_DIR}/review_cli.py init /path/to/tex-folder  # directory → auto-finds main.tex
   ```
   For `.tex` files, `\input{...}` and `\include{...}` are recursively resolved so the full paper source is inlined into `paper.md`. When a directory is given, the CLI looks for `main.tex`, `paper.tex`, or `manuscript.tex` (in that order), or uses the sole `.tex` file if only one exists.
2. **Investigation loop** — repeat until every entry is settled. Each iteration:
   1. **Consult the log** — run `show` (or recall the current state) and list what is unsettled: `to_be_verified`/flagged claims and `open` questions.
   2. **Pick a target** — choose the highest-priority unsettled entry and decide where its evidence should live (e.g., "C2 claims a 3x speedup → check the results table in §4", "Q1 asks about baselines → Grep for baseline names"). State which entry you are investigating before reading. **If the log is empty** (first iteration), the target is the paper's central claims: read the abstract, introduction, and contribution statements, then log them — claims stay `to_be_verified` and questions stay `open`, since you have not seen the evidence yet.
   3. **Gather evidence** — Read the targeted section/table/appendix, or Grep `<output_dir>/paper.md` for specific terms. Prefer targeted jumps over sequential reading; read a new section in full only when it is itself the target.
   4. **Update the log** — resolve or update the targeted entries with the evidence found (`update_claim`, `resolve_question`), citing sections in `--cross_refs`/`--sections`.
   5. **Log new discoveries** — earlier entries are a starting point, not the full agenda. Every chunk you read is also *new material*: it contains its own author claims (method design choices, ablation conclusions, reward/loss definitions) and raises its own suspicions. Log these as new entries even though you came for something else — deep-paper issues (e.g., a flawed reward design in a method section, a mismatch between an ablation table and its narrative) are usually invisible from the abstract and only enter the log this way.
3. **Exit condition** — the loop ends only when every claim has a final status (`supported`/`weak`/`invalid`) and every question is `resolved` or confirmed unanswerable (a genuine absence, which maps to a weakness). Run `show` to verify. If any section of the paper was never visited, check it before exiting — it may contain evidence that overturns earlier statuses.
4. **Finalize** — build the outline from the settled entries and finalize. This exports `review.md`, prints the review content, and shows the output folder path.
   ```bash
   python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> finalize
   ```
   The output folder contains three artifacts:
   - `paper.md` — converted paper
   - `review_log.json` — investigation log (claims, questions, notes)
   - `review.md` — final review; each point carries clickable evidence links (e.g. `[C3]`) that jump to an Evidence appendix expanding the underlying log records. Pass `--no-evidence` to `finalize`/`export` for a plain review without links.

## The ReviewLog

The `ReviewLog` (from `reviewer_memory.py`) is your persistent investigation record. It has three entry types plus the review outline.

### Claims (C1, C2, ...)

Statements made by the **authors** that you extract and verify. Each carries a verification status:

| Status | Symbol | Meaning |
|---|---|---|
| `to_be_verified` | ? | Not yet checked (initial state) |
| `supported` | ✓ | Evidence validates this claim |
| `weak` | ~ | Overstated, under-qualified, or insufficiently evidenced |
| `invalid` | ✗ | Contradicted by the paper's own evidence |

**Log a claim when:** You encounter an author assertion — results claims, novelty claims, scope claims, comparison claims. Immediately flag issues on anything vague, overstated, or needing verification.

### Questions (Q1, Q2, ...)

Points of **uncertainty or suspicion** you want to investigate. Each has a resolution status:

| Status | Symbol | Meaning |
|---|---|---|
| `open` | ? | Not yet answered |
| `partially_answered` | ~ | Some info found but incomplete |
| `resolved` | ✓ | Fully answered |

**Log a question when:** You have a testable uncertainty you can resolve by reading further — "Does the paper compare to X?", "Is dropout rate reported?", "Does Table 3 support the claim in §2?" Never assert an absence as fact; question it first, then resolve.

### Notes (N1, N2, ...)

**Your own** observations, impressions, plans, and thoughts.

**Log a note when:** Recording first impressions, investigation priorities, presentation quality observations, or any thought worth tracking.

### Choosing the right type

| You are recording... | Use |
|---|---|
| Something the authors wrote or asserted | **Claim** |
| Something you want to investigate or find out | **Question** |
| Something missing from the paper | **Question** (investigate if truly absent) or **Note** |
| Your own thought, plan, or observation | **Note** |

### Operating the log

You maintain the log through `${CLAUDE_SKILL_DIR}/review_cli.py`. State persists in `review_log.json` between calls. **Always pass `--dir <output_dir>`** to every command after init.

**1. Log** — add entries as you read.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> add_claim --text "Our method achieves 3x speedup" --section "§Abstract" --claim_type "empirical" --issues "3x over which baseline?"
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> add_question --text "Does the paper compare against DPO?" --section "§1" --related_claims "C1"
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> add_note --text "Clear motivation — real bottleneck" --section "§1" --tag "methodology"
```

**2. Update** — modify entries when you find new evidence.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> update_claim --id C1 --status weak --reason "Table 3 shows 2.8x, not 3x" --cross_refs "§4.1,Appendix A1"
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> resolve_question --id Q1 --answer "Yes, Table 2 includes DPO" --sections "§4.2"
```

Every claim must reach a final status. If a question has no answer, keep it open — that absence maps to a weakness. Cross-reference as you read: when new evidence relates to an earlier entry, update it.

**Investigation discipline:** logging and resolving are separate acts. An entry is added when something *needs* checking and updated when evidence is *found* — normally in a later iteration, after a targeted read of a different part of the paper (a claim in §1 is verified against the tables in §4, not against §1 itself). Only add-and-resolve in the same step when the current chunk genuinely contains the evidence. If you find yourself resolving every entry immediately after adding it, you are documenting impressions, not investigating.

**3. Outline** — build the review from resolved entries. Add **one point per call**, and cite evidence only via the `--claims`/`--questions`/`--notes` args. Never write tag IDs (C1, Q2, N3...) inside `--content` — at finalize they are rendered automatically as clickable links after each point. Do incorporate the *substance* of the tagged records (numbers, sections, findings) into the content text.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> outline --section strengths --content "Well-motivated contribution..." --claims "C1" --notes "N1"
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> outline --section weaknesses --content "Speedup overstated..." --claims "C1"
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> outline --section summary --content "This paper proposes..."
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> outline --section overall_score --content 6
```

The overall score is validated against the conference scale set at init. Invalid scores are rejected with an error showing the valid values.

**4. Change conference** — switch the rating scale after init (clears the score if it's invalid for the new scale).

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> set_conference neurips
```

**5. View** — check log state at any time.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> show            # brief: status counts + recent entries
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> show --detailed  # full: all entries with reasoning + outline
```

**6. Finalize** — export and print the final review.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py --dir <output_dir> finalize    # exports review.md, prints review + output folder
```

## Conference Rating Scales

Scales are sourced from official reviewer guidelines (fetched 2026-07-02):

| Conference | Scale | Source |
|---|---|---|
| `iclr` | {0, 2, 4, 6, 8, 10} | [arxiv:2511.15462](https://arxiv.org/pdf/2511.15462); labels adapted from prior ICLR scales |
| `icml` | 1–6 | [icml.cc/Conferences/2026/ReviewerInstructions](https://icml.cc/Conferences/2026/ReviewerInstructions) |
| `neurips` | 1–6 | [neurips.cc/Conferences/2026/MainTrackHandbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) |
| `acl` | 1–5 (0.5 steps) | [aclrollingreview.org/reviewform](https://aclrollingreview.org/reviewform) (ARR, Feb 2025+) |
| `emnlp` | 1–5 (0.5 steps) | [aclrollingreview.org/reviewform](https://aclrollingreview.org/reviewform) (ARR, Feb 2025+) |

Run `set_conference <name>` to see the full label descriptions for any scale.

## Writing Quality Standards

Every review point must score well on five dimensions. If any dimension is weak, revise before including.

### 1. Analytical Depth

Engage with technical content (methodology, proofs, experimental design) and explain **why** the issue matters — what breaks, what the consequences are.

- Bad: "The evaluation is limited." (surface observation, no technical engagement)
- Bad: "The proof assumes L-smoothness but uses ReLU." (technical but no consequence)
- Good: "Theorem 2's convergence proof requires L-smoothness (Assumption 3), but ReLU in Eq. 4 is non-smooth at zero, invalidating the O(1/T) rate claimed in Eq. 7." (technical + consequence)

### 2. Grounding Specificity

Reference specific paper elements — Eq 3, Table 2, Fig 4(b), §3.2 — so authors can pinpoint exactly what you mean. Then detail what is wrong.

- Bad: "The experiments could be more thorough."
- Good: "Table 2 evaluates only on CIFAR-10 and ImageNet. The data augmentation strategy in §3.1 may not transfer to domains with scarce labeled data (e.g., medical imaging)."

### 3. Actionability

State what the paper does, what the expectation is, and what to do about it. Authors should know exactly how to address the point.

- Bad: "More baselines are needed."
- Good: "Table 2 compares against methods from 2020-2022. Adding recent baselines (LoRA-DPO, ORPO) from 2024 would validate the claimed improvements against current SOTA."

### 4. Verifiability

Support your assessments with evidence — cite exact numbers, quote the paper, reference specific results. Don't make claims you can't back up.

- Bad: "The improvement is marginal."
- Good: "The improvement is 0.3 BLEU on WMT14 (Table 2), within the baseline's standard deviation of 0.5 reported in Table A3."

### 5. Helpfulness

Each point should help authors improve their paper. Ask: would reading this point give them a clear path to a stronger submission?

## Common Mistakes

1. **Ungrounded review points** — writing a weakness without tracing it to specific evidence in your log. Every point needs evidence tags.
2. **Unresolved log entries** — leaving claims as "to_be_verified" or questions as "open." Resolve everything before finalizing.
3. **Inconsistent statuses** — writing a weakness that challenges a claim but leaving that claim as "supported."
4. **Claims about absences** — logging "The paper does not compare to X" as a claim. Use a question ("Does the paper compare to X?") and resolve it.
5. **Duplicate points** — raising the same concern in different wording across multiple weaknesses.
6. **Generic statements** — "the method has limitations" or "more experiments are needed" without specifying which.
7. **Questions that restate weaknesses** — "Questions for Authors" should ask things they can clarify, not restate what weaknesses already say.
8. **Post-hoc logging** — reading the entire paper first, then batch-logging entries and resolving them immediately from memory. The log must drive the reading order: log early, then investigate targeted sections to settle entries.
9. **Ignoring the log state** — never running `show` and never letting open entries determine the next read. If the next section you read is always just "the next section," you are not investigating.

## Reference

- ReviewLog data structures: `reviewer_memory.py`
