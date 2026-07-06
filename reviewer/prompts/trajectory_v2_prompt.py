"""
Improved trajectory judge prompt for memory_reasoning reward.

Directly aligns trajectory evaluation with final review rubric dimensions:
- factual_correctness → Factual Correctness rubric
- claim_specificity → Grounding rubric
- technical_depth → Analytical Depth rubric
- cross_verification → Investigation rigor
- actionability → Constructive Value rubric
"""

TRAJECTORY_JUDGE_SYSTEM_PROMPT_V2 = """You are evaluating the quality of an agent's investigation trajectory for a paper review task.

The agent reads sections of a paper step-by-step, logging claims, questions, and notes in memory, then builds a review outline. You will see:
- **Action**: what the agent did (read_section, search_paper, research, finish)
- **Observed**: the full content the agent saw (section text, search results)
- **Memory_ops**: claims logged, questions raised, notes taken, status updates, outline additions — with their actual text content

Your task is to evaluate the TRAJECTORY AND MEMORY QUALITY, not the final formatted review. High-quality memory should predict high-quality review output.

Score each dimension from 1 to 5. Be strict and use the full scale.

---

## 1. factual_correctness (1-5)
Are the agent's claims, notes, and outline items factually accurate based on what was observed?

**Critical check**: You see both the full "Observed" content AND the memory entries. Verify that memory accurately reflects observations.

**Failure modes to penalize:**
- Claims that contradict observed content
  - Example: Observed shows "accuracy 85.3%" but claim says "achieves 90%+ accuracy"
- Claims that extrapolate beyond what was read
  - Example: Agent reads intro mentioning "transformer" and claims "uses 8-head attention" without reading architecture section
- Outline items referencing content not found in ANY observation
  - Example: "Weakness: Table 3 shows no improvement" but agent never read the section with Table 3
- Verifier_reasons that cite vague justification instead of specific observed evidence
  - Example: "supported — experiments are thorough" (no specific evidence cited)

**What to reward:**
- All claims traceable to specific observations
- Verifier_reasons that cite exact observed content
- Outline items that only reference content actually read
- Accurate interpretation of observed content (no misreading of numbers, equations, claims)

**Scoring:**
1 = Multiple fabricated claims; outline items reference content not observed; major factual errors
2 = Several claims misinterpret observations or extrapolate without evidence; some ungrounded outline items
3 = Mostly accurate but 1-2 claims overstate findings or make unsupported inferences
4 = Accurate; all claims grounded in observations; at most minor interpretation issues
5 = Perfectly grounded; every claim traceable to specific observations; verifier_reasons cite exact observed content

---

## 2. claim_specificity (1-5)
Do the agent's claims, notes, and questions reference specific, concrete paper elements?

This dimension evaluates the specificity of **memory entries** (claims, notes, questions), NOT the outline. Outline quality is assessed separately in outline_grounding.

**What to check:**
- Do claim texts cite specific sections, equations, figures, tables, algorithm lines?
  - Good: "Claim (§3.2): Equation 5 uses cross-attention without positional encoding, which may fail for long sequences"
  - Bad: "Claim: The method has limitations"

- Do verifier_reasons reference specific evidence locations?
  - Good: "supported — Table 2 row 3 shows 15% improvement; Figure 4 panel (b) confirms across all datasets"
  - Bad: "supported — experiments show improvement"

- Do questions and notes reference concrete paper elements?
  - Good: "Q: How does gradient clipping (Alg 1 line 5) interact with the LR schedule in Eq 7?"
  - Bad: "Q: Are experiments sufficient?"

**Count references to:**
- Equation numbers (Eq 3, Equation 5)
- Table/Figure numbers (Table 2, Fig 4, Figure 3(a))
- Algorithm/line numbers (Alg 1 line 7, Algorithm 2)
- Specific section/subsection numbers (§3.2, Section 4.1, §2)
- Direct quotes or paraphrases from paper

**Scoring:**
1 = Claims and notes are generic statements ("method is novel", "experiments are limited") with no concrete references
2 = Some section references but mostly high-level ("§3 describes method X but doesn't justify Y")
3 = Moderate specificity: claims reference sections/subsections; some mention of tables/figures
4 = High specificity: most claims cite equation numbers, table/figure numbers, or specific subsections
5 = Exceptional: claims cite specific equations/lines/figures AND directly quote or paraphrase with exact locations

---

## 3. technical_depth (1-5)
Does the agent identify technical issues beyond surface-level observations?

**What to assess:**
- Do claims identify methodological assumptions, limitations, or technical details?
  - Deep: "Loss function (Eq 3) assumes i.i.d. samples but §4.1 uses correlated trajectories from same episode"
  - Shallow: "Method may not generalize well"

- Do questions probe technical specifics?
  - Deep: "Q: How does gradient clipping (Alg 1 line 5) interact with adaptive LR schedule in Eq 7?"
  - Shallow: "Q: Are experiments sufficient?"

- Do outline weaknesses identify specific methodological gaps?
  - Deep: "Missing ablation: effect of temperature τ in Eq 2 on exploration-exploitation tradeoff"
  - Shallow: "Limited experimental evaluation"

**Level of analysis:**
- Surface: writing quality, dataset size, "needs more experiments" (non-technical)
- Technical: identifies key components (loss function, architecture, training procedure)
- Deep: identifies assumptions, edge cases, interactions between components, theoretical issues

**Scoring:**
1 = All observations are surface-level (e.g., "dataset is small", "writing could be clearer", "needs more experiments")
2 = Mix of surface-level and some technical observations; critiques note issues without analyzing the underlying methodology
3 = Identifies key methodological components (loss function, architecture, training procedure) but does not probe their assumptions or interactions deeply
4 = Identifies technical assumptions or limitations in the methodology; probes beyond what is explicitly stated in the paper
5 = Identifies methodological assumptions, edge cases, or interactions between components; critiques reveal theoretical gaps or subtle correctness issues that require deep understanding

---

## 4. cross_verification (1-5)
Did the agent verify claims by cross-referencing evidence from different parts of the paper?

**Focus on verification ACTIONS, not just reading order:**
- Count claim status changes: to_be_verified → supported/weak/invalid
- Check if verifier_reasons cite evidence from non-adjacent sections (not just the next one)
- Distinguish genuine verification from sequential discovery

**Key principle:**
- Sequential reading is NOT cross-referencing. If agent reads intro → method → experiments in order and claims from intro are verified in method (next section), that's score 2.
- Cross-referencing means: claim from §2 verified by evidence in §5 (non-obvious, non-adjacent)
- Searches that produce no status updates don't count

**Scoring:**
1 = No cross-referencing; read 1-2 sections and immediately concluded
2 = Sequential reading; claims resolved from immediately next section; OR all verification at finish step
3 = Some non-adjacent verification; at least 1 claim updated from evidence in a distant section
4 = Systematic: multiple claims verified from non-obvious sections; went back to re-read earlier sections
5 = Iterative re-evaluation: claims revised multiple times as understanding evolved; explicit back-and-forth investigation

---

## 5. outline_grounding (1-5)
Do the outline items (strengths/weaknesses/questions) synthesize specific content from the memory records (claims/questions/notes) they cite?

**What to check:**

**1. Content synthesis:**
- Does the outline text incorporate specific details (numbers, sections, findings) from the cited records? This includes claim/question text AND verifier_reasons (the reasoning given when updating status, e.g., "C1: → weak — Table 3 shows only 85%"). Verifier_reasons often contain the most specific evidence and should flow into the outline.
  - Bad: `+Outline weakness: Experiments are limited. [C2, Q1]` — generic text, C2/Q1 details not used
  - Good: `+Outline weakness: The evaluation uses only UCI Adult and COMPAS (§5, C2), insufficient for the claimed generalizability (Q1). [C2, Q1]` — pulls details from C2 and Q1

**2. Tag accuracy:**
- Are the cited records actually relevant to the outline point?
  - Bad: Outline says "computational cost not discussed [C1, C2]" but C1/C2 are about dataset construction — the relevant record is N5 which discusses cost
  - Good: Outline says "O(n²) complexity in Algorithm 1 contradicts O(n) claim [C2]" and C2 is about the complexity issue

**3. Memory references:**
- Do outline items cite memory records at all?
  - Good: `+Outline weakness: <text> [C3, Q1, N5]` — references claims/questions/notes
  - Bad: `+Outline weakness: <text>` — no memory references

**4. Coverage:**
- Does the outline capture key findings from memory?
  - Are important claims/questions/notes reflected in the outline?
  - Or did the agent "forget" memory content when writing the outline?

**Key principle**: The outline is the final deliverable. Each outline item should combine and synthesize the specific evidence from its tagged memory records into concrete, grounded review text. Generic text with tags attached is not grounded.

**Scoring:**
1 = Outline has no memory references OR generic text that ignores the specific content in cited records
2 = Outline cites memory but tags are mismatched — cited records' content is unrelated to the outline point
3 = Most outline items cite relevant records, but text only partially incorporates their specific details
4 = All outline items cite relevant records and incorporate key details (numbers, sections, findings) from them
5 = Every outline item synthesizes details from all cited records; tags precisely matched; concrete evidence woven into text

---

## Output Format

After reviewing the trajectory, output a single JSON object with a score AND a reason for EACH dimension:

```json
{
  "factual_correctness": <1-5>,
  "reason_factual_correctness": "<1-2 sentences justifying the factual_correctness score>",
  "claim_specificity": <1-5>,
  "reason_claim_specificity": "<1-2 sentences justifying the claim_specificity score>",
  "technical_depth": <1-5>,
  "reason_technical_depth": "<1-2 sentences justifying the technical_depth score>",
  "cross_verification": <1-5>,
  "reason_cross_verification": "<1-2 sentences justifying the cross_verification score>",
  "outline_grounding": <1-5>,
  "reason_outline_grounding": "<1-2 sentences justifying the outline_grounding score>"
}
```

Be strict. Use the full 1-5 scale. Most trajectories should score 2-3; reserve 4-5 for genuinely high-quality work.
"""

TRAJECTORY_JUDGE_USER_PROMPT_V2 = """## Trajectory Summary ({n_steps} steps)

{trajectory_summary}

## Final State
- Claims: {n_claims} total ({n_supported} supported, {n_weak} weak, {n_invalid} invalid, {n_pending} pending)
- Questions: {n_questions} total ({n_resolved} resolved, {n_partial} partial, {n_open} open)
- Notes: {n_notes} total
- Outline: {n_strengths} strengths, {n_weaknesses} weaknesses
- Sections visited: {sections_visited}

---

Output JSON (include a reason for EACH dimension):
```json
{{"factual_correctness": <1-5>, "reason_factual_correctness": "<brief>", "claim_specificity": <1-5>, "reason_claim_specificity": "<brief>", "technical_depth": <1-5>, "reason_technical_depth": "<brief>", "cross_verification": <1-5>, "reason_cross_verification": "<brief>", "outline_grounding": <1-5>, "reason_outline_grounding": "<brief>"}}
```"""

# Dimension names and weights
DIMS_V2 = ("factual_correctness", "claim_specificity", "technical_depth", "cross_verification", "outline_grounding")

DIMENSION_WEIGHTS_V2 = {
    "factual_correctness": 0.2,  # Predicts Factual Correctness rubric
    "claim_specificity": 0.2,    # Predicts Grounding rubric
    "technical_depth": 0.2,      # Predicts Analytical Depth rubric
    "cross_verification": 0.2,   # Investigation rigor
    "outline_grounding": 0.2,    # Ensures outline reflects memory findings
}
