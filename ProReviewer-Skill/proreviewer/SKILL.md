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

Use `${CLAUDE_SKILL_DIR}/review_cli.py` to maintain a persistent ReviewLog. The review runs as a read-then-operate loop:

1. **Initialize** — convert the paper and create the log. Use `--conference` to set the rating scale.
   ```bash
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf                      # default: ICLR scale
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf --conference icml     # ICML 1-6 scale
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf --conference neurips  # NeurIPS 1-6 scale
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf --conference acl      # ACL/ARR 1-5 scale (0.5 steps)
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.pdf --conference emnlp    # EMNLP/ARR 1-5 scale (0.5 steps)
   ```
   Accepts `.pdf`, `.tex`, `.md`, or a **directory** containing a LaTeX project:
   ```bash
   python ${CLAUDE_SKILL_DIR}/review_cli.py init paper.tex            # single .tex file
   python ${CLAUDE_SKILL_DIR}/review_cli.py init /path/to/tex-folder  # directory → auto-finds main.tex
   ```
   For `.tex` files, `\input{...}` and `\include{...}` are recursively resolved so the full paper source is inlined into `paper.md`. When a directory is given, the CLI looks for `main.tex`, `paper.tex`, or `manuscript.tex` (in that order), or uses the sole `.tex` file if only one exists.
2. **Read** — read `paper.md` using your Read tool. Read it in chunks (by section or page range).
3. **Operate the log** — after each read, run CLI commands to add claims/questions/notes or update existing entries.
4. **Search** — use your Grep tool on `paper.md` to find specific terms.
5. **Repeat** — read the next part and operate the log again. Cross-reference new evidence against earlier entries.
6. **Finalize** — once all entries are resolved, build the outline and show the final state.
   ```bash
   python ${CLAUDE_SKILL_DIR}/review_cli.py show --detailed
   ```

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

You maintain the log through `${CLAUDE_SKILL_DIR}/review_cli.py`. State persists in `review_log.json` between calls.

**1. Log** — add entries as you read.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py add_claim --text "Our method achieves 3x speedup" --section "§Abstract" --claim_type "empirical" --issues "3x over which baseline?"
python ${CLAUDE_SKILL_DIR}/review_cli.py add_question --text "Does the paper compare against DPO?" --section "§1" --related_claims "C1"
python ${CLAUDE_SKILL_DIR}/review_cli.py add_note --text "Clear motivation — real bottleneck" --section "§1" --tag "methodology"
```

**2. Update** — modify entries when you find new evidence.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py update_claim --id C1 --status weak --reason "Table 3 shows 2.8x, not 3x" --cross_refs "§4.1,Appendix A1"
python ${CLAUDE_SKILL_DIR}/review_cli.py resolve_question --id Q1 --answer "Yes, Table 2 includes DPO" --sections "§4.2"
```

Every claim must reach a final status. If a question has no answer, keep it open — that absence maps to a weakness. Cross-reference as you read: when new evidence relates to an earlier entry, update it.

**3. Outline** — build the review from resolved entries. Each item must cite evidence tags.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py outline --section strengths --content "Well-motivated contribution..." --claims "C1" --notes "N1"
python ${CLAUDE_SKILL_DIR}/review_cli.py outline --section weaknesses --content "Speedup overstated..." --claims "C1"
python ${CLAUDE_SKILL_DIR}/review_cli.py outline --section summary --content "This paper proposes..."
python ${CLAUDE_SKILL_DIR}/review_cli.py outline --section overall_score --content 6
```

The overall score is validated against the conference scale set at init. Invalid scores are rejected with an error showing the valid values.

**4. Change conference** — switch the rating scale after init (clears the score if it's invalid for the new scale).

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py set_conference neurips
```

**5. View** — check log state at any time.

```bash
python ${CLAUDE_SKILL_DIR}/review_cli.py show            # brief: status counts + recent entries
python ${CLAUDE_SKILL_DIR}/review_cli.py show --detailed  # full: all entries with reasoning + outline
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

## Reference

- ReviewLog data structures: `reviewer_memory.py`

## Platform Notes

This skill uses `${CLAUDE_SKILL_DIR}` to reference the directory containing these files. If your platform does not support this variable, replace it with the absolute path to the directory where `review_cli.py` and `reviewer_memory.py` are installed.

Typical install locations:

| Platform | Path |
|---|---|
| Claude Code (global) | `~/.claude/skills/proreviewer/` |
| Claude Code (project) | `.claude/skills/proreviewer/` |
| Codex CLI | `~/.agents/skills/proreviewer/` |
| Gemini CLI | `~/.gemini/skills/proreviewer/` |

<!-- For agents that do not expand ${CLAUDE_SKILL_DIR}: resolve it to the
     directory containing this SKILL.md file, then call review_cli.py from
     that directory. All three Python files sit side-by-side. -->
