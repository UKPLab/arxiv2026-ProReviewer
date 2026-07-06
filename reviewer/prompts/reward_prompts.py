"""LLM judge prompts for reward calculation.

This module contains prompts for reward components:
1. Recall: Evaluating coverage of human review points
2. Actionable: Evaluating actionability of feedback
3. Grounded: Evaluating evidence grounding
"""

# ============================================================================
# Recall Scenario Prompts
# ============================================================================

RECALL_JUDGE_SYSTEM_PROMPT = """You are an expert peer reviewer evaluating review coverage.

Your task is to determine which points from human reviewers are covered in a generated review by a model. You will be given a list of key points extracted from multiple human reviews, and you need to assess whether each point is also covered by the generated review.

## Coverage Levels

- **full**: the generated review expresses the **same core idea AND comparable specificity** (e.g. names the same missing baseline, cites the same equation/table, identifies the same flaw). Different terminology is fine (e.g. "co-evolution loop" covers "iterative caption→reward→retrain pipeline").
- **partial**: the generated review raises the **same general concern** but is **less specific** (e.g. human says "missing comparison to SAIL, Self-RAG, RQ-RAG" → model says "baselines are insufficient"; or human points to a specific equation error → model notes the proof is unconvincing without pinpointing the equation).
- **not_covered**: the generated review does not address this concern at all.

## Type Matching Rule

A [Weakness] human point can ONLY be covered by a weakness (W*) in the generated review.
A [Strength] human point can ONLY be covered by a strength (S*) in the generated review.
Never cross-match: a weakness cannot be covered by a strength, and vice versa."""

RECALL_JUDGE_USER_PROMPT = """## Human Review Points

Below are {num_points} key points extracted from human reviewers.

{human_points}

---

## Generated Review

{generated_review}

---

## Evaluation Task

For EACH human review point above, evaluate whether it has been covered in the generated review.

## Output Format

```json
[
  {{
    "point_id": 1,
    "coverage": "full" | "partial" | "not_covered",
    "evidence": "A single id (e.g. W1 or S2) of the matching point in the generated review, or null if not covered. [Weakness] points must map to W* ids only, [Strength] points must map to S* ids only.",
    "reasoning": "Brief explanation (1 sentence)"
  }},
  ...
]
```
Provide coverage assessment for ALL {num_points} points."""


# Per-point recall prompt (one LLM call per clustered point)
RECALL_JUDGE_POINT_SYSTEM_PROMPT = """Judge whether a human review point is covered by a generated review.
A [{point_type}] point can ONLY match a {expected_id_type}. Respond with JSON only.

## Coverage levels
- "full": the generated review expresses the **same core idea AND comparable specificity** (e.g. names the same missing baseline, cites the same equation/table, identifies the same flaw).
- "partial": the generated review raises the **same general concern** but is **less specific** (e.g. human says "missing comparison to SAIL, Self-RAG, RQ-RAG" → model says "baselines are insufficient"; or human points to a specific equation error → model notes the proof is unconvincing without pinpointing the equation).
- "not_covered": the generated review does not address this concern at all."""

RECALL_JUDGE_POINT_USER_PROMPT = """Human point [{point_type}]: {point_text}

Generated review:
{generated_review}

Rate the coverage of the human point above. Apply the definitions strictly.

```json
{{"coverage": "full" or "partial" or "not_covered", "matched_id": "matching id (e.g. W1) or null", "justification": "1 sentence — state exactly which part of the matched review entry covers the human point, or why no entry does."}}
```"""


# ============================================================================
# Actionable Scenario Prompts
# ============================================================================

ACTIONABLE_JUDGE_SYSTEM_PROMPT = """You are an expert peer reviewer evaluating the actionability of review feedback.

Your task is to rate how actionable the feedback is for helping authors improve their work on a 1-5 scale.

Actionable feedback provides concrete, specific suggestions that authors can implement. It goes beyond vague criticism to offer constructive guidance with clear next steps."""

ACTIONABLE_JUDGE_USER_PROMPT = """## Paper Information

**Title:** {paper_title}

**Abstract:** {paper_abstract}

---

## Review Weaknesses Section

{weaknesses_section}

---

## Rating Task

Rate the actionability of the weaknesses/feedback on a 1-5 scale:

### Rating Scale

**5 - Highly Actionable**: Specific, concrete suggestions with examples
  - Example: "Add ablation study comparing method X vs Y on dataset Z, as done in Smith et al. (2023)"
  - Example: "Expand Section 3.2 to include proofs for Theorem 1 and Theorem 2"

**4 - Actionable**: Clear guidance for improvement with specific areas
  - Example: "The related work section should discuss recent transformer-based approaches"
  - Example: "Include statistical significance tests for Table 2 results"

**3 - Moderately Actionable**: Some concrete feedback mixed with general statements
  - Example: "The evaluation could be more comprehensive" (somewhat vague, but points to evaluation)
  - Example: "Consider adding more baselines" (actionable direction but not specific)

**2 - Vague**: General criticisms without specific guidance
  - Example: "The writing needs improvement"
  - Example: "The contribution is limited"

**1 - Not Actionable**: Purely critical without guidance or too vague to act on
  - Example: "This paper is not good enough"
  - Example: "Poor quality"

---

## Output Format

Provide your evaluation in the following JSON format:
```json
{{
  "score": 1-5,
  "reasoning": "Explanation for the score (2-3 sentences)",
  "actionable_examples": ["List 2-3 specific actionable items from the review"],
  "vague_examples": ["List 1-2 vague statements if present, or empty list"]
}}
```"""


# ============================================================================
# Grounded Scenario Prompts
# ============================================================================

GROUNDED_JUDGE_SYSTEM_PROMPT = """You are an expert peer reviewer evaluating how well a review is grounded in actual paper content.

Your task is to rate the evidence grounding of a review on a 1-5 scale.

Well-grounded reviews cite specific sections, equations, figures, tables, or experimental results from the paper. Ungrounded reviews make generic or unsupported statements without evidence from the paper."""

GROUNDED_JUDGE_USER_PROMPT = """## Paper Information

**Title:** {paper_title}

**Abstract:** {paper_abstract}

---

## Review Content

**Summary:**
{summary_section}

**Strengths:**
{strengths_section}

**Weaknesses:**
{weaknesses_section}

---

## Rating Task

Rate how well the review is grounded in the paper content on a 1-5 scale:

### Rating Scale

**5 - Excellent Grounding**: Specific citations and evidence throughout
  - Example: "Table 2 shows 92.3% accuracy on MNLI, outperforming BERT by 3.1%"
  - Example: "Equation 5 assumes linearity, which may not hold for non-convex losses (Section 3.2)"
  - Example: "Figure 3 demonstrates the attention patterns capture syntactic dependencies"

**4 - Good Grounding**: Clear evidence from the paper for most claims
  - Example: "The results section shows improvements on 3/5 benchmarks"
  - Example: "The method uses self-attention as described in Section 2"
  - Example: "The experiments cover multiple domains but lack error analysis"

**3 - Adequate Grounding**: Some references to paper content mixed with generic statements
  - Example: "The paper presents experimental results" (mentions experiments but not specific)
  - Example: "The approach is novel" (claim without evidence)

**2 - Weak Grounding**: Mostly generic statements with minimal paper-specific details
  - Example: "The method is interesting"
  - Example: "Good experimental setup"
  - Example: "The paper makes contributions"

**1 - Ungrounded**: Vague claims without any connection to paper content
  - Example: "This is a good/bad paper"
  - Example: "The work needs improvement"
  - Example: "Interesting topic"

---

## Output Format

Provide your evaluation in the following JSON format:
```json
{{
  "score": 1-5,
  "reasoning": "Explanation for the score (2-3 sentences)",
  "grounded_examples": ["List 2-3 well-grounded statements with specific evidence"],
  "ungrounded_examples": ["List 1-2 ungrounded/vague statements if present, or empty list"]
}}
```"""


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_JUDGE_MODEL = "utility-score"
