"""Evidence-based prompts for trajectory memory reasoning evaluation.

This variant produces per-item scores instead of trajectory-level scores:
- factual_correctness: List of steps with hallucinations (per-step penalties)
- technical_depth: Score for each weakness only (per-item)
- outline_grounding: Score for each strength and weakness only (per-item)

Output format uses XML tags with JSON arrays for structured parsing.
Score/data comes FIRST, then reasoning.
"""

from typing import Tuple

# System prompt
TRAJECTORY_JUDGE_SYSTEM_PROMPT_EVIDENCE = """\
You are an evaluator of agent investigation trajectories for paper review tasks. You will receive a trajectory showing how an agent investigated a paper step-by-step, along with the final review outline. You should evaluate based on the given criteria and output structured results in XML tags with JSON content. Always output the score/data first, then reasoning.
"""

# Dimensions
DIMS_EVIDENCE = [
    "factual_correctness",
    "technical_depth",
    "outline_grounding",
    "grounding",
]

# Weights (equal for all 3)
DIMENSION_WEIGHTS_EVIDENCE = {
    "factual_correctness": 0.25,
    "technical_depth": 0.25,
    "outline_grounding": 0.25,
    "grounding": 0.25,
}


def get_evidence_dimension_prompt(dimension: str) -> Tuple[str, str, str]:
    """Get [QUERY], [CRITERIA], and [EXAMPLES] for a dimension.

    Returns:
        (query, criteria, examples)
    """

    if dimension == "factual_correctness":
        query = (
            "[QUERY]: Identify which specific steps in the trajectory contain "
            "factual errors or hallucinations. A hallucination is when the agent's "
            "memory entries (claims, notes, outline items) fabricate, misquote, or "
            "contradict the observations provided in the trajectory.\n\n"
        )

        criteria = (
            "[CRITERIA]: For each step, check whether the memory operations "
            "(claims, notes, status changes) are grounded in the \"Observed\" field. "
            "The \"Observed\" field shows the full content the agent read at each step. "
            "Memory operations should not contradict observations, extrapolate beyond "
            "what was read, or reference content not found in any observation. When a "
            "claim's status is changed (e.g. to_be_verified → supported), the "
            "verifier_reason should cite evidence visible in the Observed content.\n\n"
            "Major hallucinations: fabricating content never seen in any Observed "
            "content, misquoting metrics/numbers, or major factual contradictions.\n"
            "Minor hallucinations: overstating findings, extrapolating beyond what "
            "was read, or vague verifier_reasons that don't cite specific observed "
            "content.\n\n"
            "Output format (hallucinations list first, then reasoning):\n"
            "<hallucinations>\n"
            "[{\"step\": 2, \"evidence_id\": \"C1\", \"severity\": \"major\", "
            "\"description\": \"Fabricates architectural details not in observation\"},\n"
            " {\"step\": 5, \"evidence_id\": \"C2\", \"severity\": \"minor\", "
            "\"description\": \"Verifier reason is vague\"}]\n"
            "</hallucinations>\n"
            "<reasoning>Brief explanation of what errors were found</reasoning>\n\n"
            "If no hallucinations found, output empty array: []\n"
            "severity: \"major\" or \"minor\"\n\n"
        )

        examples = (
            "<START OF EXAMPLE>\n\n"
            "TRAJECTORY EXCERPT:\n"
            "Step 2:\n"
            "  Action: read_section(3 methodology)\n"
            "  Observed: We propose a graph neural network (GNN) that operates on "
            "molecular structures. Our architecture uses message-passing layers...\n"
            "  Memory_ops:\n"
            "    +Claim C1 (§3 methodology): The model uses a 6-layer GNN with "
            "residual connections and layer normalization.\n"
            "    +Claim C2 (§3 methodology): The method achieves 94% accuracy "
            "on the benchmark.\n"
            "Step 3:\n"
            "  Action: read_section(4 experiments)\n"
            "  Observed: Table 1 reports an AUC of 0.82 on MoleculeNet...\n"
            "  Memory_ops:\n"
            "    Claim C2: to_be_verified → supported — Experiments confirm strong "
            "results.\n"
            "\nEVALUATION:\n\n"
            "<hallucinations>\n"
            "[{\"step\": 2, \"evidence_id\": \"C1\", \"severity\": \"major\", "
            "\"description\": \"Fabricates architectural details (6-layer, residual, "
            "normalization) not present in observation\"},\n"
            " {\"step\": 2, \"evidence_id\": \"C2\", \"severity\": \"major\", "
            "\"description\": \"Claims 94% accuracy but not in observation\"},\n"
            " {\"step\": 3, \"evidence_id\": \"C2\", \"severity\": \"minor\", "
            "\"description\": \"Vague verifier reason, should cite AUC 0.82\"}]\n"
            "</hallucinations>\n"
            "<reasoning>Step 2: C1 fabricates specific architectural details (6-layer, "
            "residual connections, layer normalization) not present in the Observed "
            "snippet, which only mentions \"graph neural network\" and \"message-passing "
            "layers\". C2 claims \"94% accuracy\" but the observation shows \"AUC of 0.82\" "
            "— a different metric and value. Step 3: The verifier_reason for C2's status "
            "change is vague (\"experiments confirm strong results\") rather than citing "
            "the actual AUC number. Multiple fabrications across claims.</reasoning>\n\n"
            "<END OF EXAMPLE>\n\n"
        )

    elif dimension == "technical_depth":
        query = (
            "[QUERY]: Evaluate the analytical depth of each weakness in "
            "the outline. Analytical depth measures whether the weakness engages "
            "with the paper's technical content and analyzes why the issue matters. "
            "Focus only on weaknesses (W1, W2, ...).\n\n"
        )

        criteria = (
            "[CRITERIA]: This aspect has two components:\n"
            "1. **Technical Engagement**: Whether the critique engages with the paper's "
            "methodology, algorithms, proofs, or experimental design choices (as opposed "
            "to commenting on scope, presentation, or completeness).\n"
            "2. **Analytical Reasoning**: Whether the critique explains why the identified "
            "issue is problematic (e.g., what breaks, what the consequences are, or how it "
            "affects validity).\n\n"
            "It's more important for the critique to be technical (engaging with actual "
            "methodology) than to provide reasoning about why an issue matters.\n\n"
            "Scoring rubric:\n"
            "5: **Technical and Reasoned** - The critique engages with specific technical "
            "content (methodology, algorithms, proofs, experimental design) AND explains why "
            "the issue is problematic (consequences, failure modes, what breaks).\n"
            "4: **Technical but Unreasoned** - The critique engages with specific technical "
            "content but states the issue without fully explaining its consequences or why "
            "it matters.\n"
            "3: **Non-technical but Reasoned** - The critique does not engage with specific "
            "technical content, but provides reasoning about why the identified gap matters.\n"
            "2: **Non-technical and Unreasoned** - The critique neither engages with specific "
            "technical content nor explains reasoning about consequences.\n"
            "1: **No Substance** - Pure surface observation requiring no domain expertise "
            "(e.g., \"Writing could be clearer\", \"Limited novelty\", \"More experiments needed\").\n\n"
            "Output format (item_scores list first, then reasoning):\n"
            "<item_scores>\n"
            "[{\"item_id\": \"W1\", \"score\": 1, \"reason\": \"...\"},\n"
            " {\"item_id\": \"W2\", \"score\": 3, \"reason\": \"...\"}]\n"
            "</item_scores>\n"
            "<reasoning>Brief overall assessment</reasoning>\n\n"
            "item_id: \"W1\", \"W2\", ... for weaknesses only\n"
            "Do NOT score strengths (S1, S2, ...) or questions (Q1, Q2, ...)\n\n"
        )

        examples = (
            "<START OF EXAMPLE>\n\n"
            "FINAL OUTLINE:\n"
            "### Weaknesses\n"
            "W1. Theorem 2's convergence proof requires L-smoothness (Assumption 3), but the ReLU activation in Eq. 4 is non-smooth at zero, invalidating the O(1/T) rate claimed in Eq. 7.\n"
            "W2. Evaluating only on CIFAR-10 and ImageNet limits generalizability because the data augmentation strategy may not transfer to domains with scarce labeled data.\n"
            "W3. The paper needs more experiments and clearer writing. The model seems computationally expensive and may be inefficient in practice.\n"
            "W4. The convergence proof assumes L-smoothness but the loss function uses ReLU which is non-smooth.\n"
            "\nEVALUATION:\n\n"
            "<item_scores>\n"
            "[{\"item_id\": \"W1\", \"score\": 5, \"reason\": \"Technical and Reasoned: engages with specific theorem/equation AND explains the consequence (invalidates convergence rate)\"},\n"
            " {\"item_id\": \"W2\", \"score\": 3, \"reason\": \"Non-technical but Reasoned: does not engage with technical content but reasons about why limited evaluation matters for transfer\"},\n"
            " {\"item_id\": \"W3\", \"score\": 1, \"reason\": \"No Substance: pure surface observations requiring no domain expertise\"},\n"
            " {\"item_id\": \"W4\", \"score\": 4, \"reason\": \"Technical but Unreasoned: engages with convergence proof and smoothness assumption but doesn't explain the consequence\"}]\n"
            "</item_scores>\n"
            "<reasoning>W1 engages with specific technical content "
            "(Theorem 2, Assumption 3, Eq. 4, Eq. 7) and explains why the issue invalidates the main "
            "claim (score 5). W2 doesn't engage with technical content "
            "but reasons about why the gap matters (score 3). W3 is a surface observation (score 1). "
            "W4 identifies a technical assumption mismatch "
            "but doesn't explain the consequence (score 4).</reasoning>\n\n"
            "<END OF EXAMPLE>\n\n"
        )

    elif dimension == "outline_grounding":
        query = (
            "[QUERY]: Evaluate whether each strength and weakness synthesizes specific "
            "content from the memory records it cites. The outline text should incorporate "
            "concrete details (numbers, sections, findings) from the tagged records, "
            "not just attach tags to generic text. Focus only on strengths (S1, S2, ...) and weaknesses (W1, W2, ...).\n\n"
        )

        criteria = (
            "[CRITERIA]: Outline grounding means each outline item incorporates "
            "specific details from its cited memory records into the text. The "
            "tags ([C1], [Q2], [N3]) indicate which records support the point — "
            "the outline text should pull concrete evidence from those records. "
            "This includes claim/question text AND verifier_reasons (the reasoning "
            "given when updating status, e.g. \"C1: -> weak -- Table 3 shows only "
            "85%\"). Verifier_reasons often contain the most specific evidence and "
            "should flow into the outline. "
            "Key failure modes include: (a) generic outline text with tags attached "
            "but no details from the cited records incorporated, (b) mismatched "
            "tags where the cited record's content is unrelated to the outline "
            "point, (c) outline items with no tags at all. Scoring rubric is as "
            "follows:\n"
            "1: Outline has no memory references OR generic text with no details "
            "from cited records (e.g. \"experiments are limited [C2]\" when C2 "
            "contains specific dataset names and numbers).\n"
            "2: Outline cites memory but tags are mismatched — cited records' "
            "content is unrelated to the outline point (e.g. computational cost "
            "weakness cites dataset construction claim [C1]).\n"
            "3: Most outline items cite relevant records, but the text only "
            "partially incorporates their content — some specific details from "
            "the records are missing.\n"
            "4: All outline items cite relevant records and incorporate their "
            "key details (numbers, sections, findings) into the text.\n"
            "5: Every outline item synthesizes details from all cited records; "
            "text includes specific numbers, sections, and findings from each "
            "tagged record; tags are precisely matched.\n\n"
            "Output format (item_scores list first, then reasoning):\n"
            "<item_scores>\n"
            "[{\"item_id\": \"S1\", \"score\": 4, \"reason\": \"Grounded with specific cited evidence\"},\n"
            " {\"item_id\": \"W1\", \"score\": 1, \"reason\": \"Generic text, no "
            "details from C2\"},\n"
            " {\"item_id\": \"W2\", \"score\": 5, \"reason\": \"Incorporates exact "
            "theorem and assumption from C3\"}]\n"
            "</item_scores>\n"
            "<reasoning>Brief overall assessment</reasoning>\n\n"
            "item_id: \"S1\", \"S2\", ... for strengths; \"W1\", \"W2\", ... for weaknesses\n\n"
        )

        examples = (
            "<START OF EXAMPLE>\n\n"
            "MEMORY (final state):\n"
            "  Claim C1 (status=supported): Method achieves 85.3% accuracy on "
            "CIFAR-10 (Table 2).\n"
            "  Claim C2 (status=weak): Paper claims O(n) complexity but §3.1 "
            "suggests O(n²).\n"
            "  Claim C3 (status=supported, verifier_reason=\"§3.1 claims O(n) but "
            "Algorithm 1 line 7 has a nested loop over all pairs, giving O(n²)\"): "
            "Complexity claim contradicted.\n"
            "  Question Q1 (status=resolved): How does the method handle variable-"
            "length inputs? — §3.3 uses padding with masking.\n"
            "\nFINAL OUTLINE:\n"
            "### Strengths\n"
            "S1. The method shows competitive in-domain performance, reaching 85.3% accuracy on CIFAR-10 (Table 2), which supports the paper's effectiveness claim on standard benchmarks. [C1]\n"
            "### Weaknesses\n"
            "W1. The method has some computational limitations. [C2]\n"
            "W2. Algorithm 1 line 7 has a nested loop over all pairs, giving O(n²) "
            "complexity contradicting the O(n) claim in §3.1. [C3]\n"
            "\nEVALUATION:\n\n"
            "<item_scores>\n"
            "[{\"item_id\": \"S1\", \"score\": 4, \"reason\": \"Includes concrete metric and dataset from C1\"},\n"
            " {\"item_id\": \"W1\", \"score\": 1, \"reason\": \"Generic text about "
            "limitations, doesn't incorporate C2's specific complexity details\"},\n"
            " {\"item_id\": \"W2\", \"score\": 5, \"reason\": \"Synthesizes C3's verifier_reason "
            "exact details (Algorithm 1 line 7, nested loop, O(n²) vs O(n), §3.1)\"}]\n"
            "</item_scores>\n"
            "<reasoning>S1 is grounded by concrete evidence from C1 (85.3% on CIFAR-10). "
            "W1 says \"computational limitations\" but doesn't incorporate C2's "
            "specific complexity contradiction (O(n) vs O(n²) in §3.1). The text is generic "
            "with tag attached. W2 incorporates C3's verifier_reason details exactly "
            "(Algorithm 1 line 7, nested loop, O(n²) contradicting §3.1's O(n)).</reasoning>\n\n"
            "<END OF EXAMPLE>\n\n"
        )

    elif dimension == "grounding":
        query = (
            "[QUERY]: Evaluate whether each weakness references specific parts of "
            "the paper and clearly specifies what is wrong or missing. You will be "
            "given the paper content to verify the references. "
            "Focus only on weaknesses (W1, W2, ...).\n\n"
        )

        criteria = (
            "[CRITERIA]: Grounding is evaluated on two components:\n"
            "1. **Grounding**: Can authors identify the specific part of the paper "
            "being addressed? (via sections, tables, figures, equations, or unique elements)\n"
            "2. **Specificity**: Does the comment clearly detail what is wrong or "
            "missing in the referenced part?\n\n"
            "It's more important for the comment to be grounded than to be specific.\n\n"
            "Scoring rubric is as follows:\n"
            "5: **Fully Grounded and Specific** - The comment explicitly mentions "
            "which part of the paper it addresses (via sections, tables, figures, "
            "equations, or unique elements), and clearly specifies what needs to be "
            "addressed in that part.\n"
            "4: **Fully Grounded and Under-Specific** - The comment explicitly "
            "mentions which part of the paper it addresses, but does not clearly "
            "specify what needs to be addressed.\n"
            "3: **Weakly Grounded and Specific** - Authors cannot confidently "
            "determine which part is addressed, but the comment clearly specifies "
            "what needs to be addressed.\n"
            "2: **Weakly Grounded and Not Specific** - Authors cannot confidently "
            "determine which part is addressed, and the comment does not specify "
            "what needs to be addressed.\n"
            "1: **Not Grounded** - The comment is not grounded at all; it does not "
            "identify a specific area in the paper and is highly unspecific.\n\n"
            "Output format (item_scores list first, then reasoning):\n"
            "<item_scores>\n"
            "[{\"item_id\": \"W1\", \"score\": 5, \"reason\": \"References Table 3 in §5.1 and specifies missing comparison\"},\n"
            " {\"item_id\": \"W2\", \"score\": 1, \"reason\": \"No paper reference, vague criticism\"}]\n"
            "</item_scores>\n"
            "<reasoning>Brief overall assessment</reasoning>\n\n"
            "item_id: \"W1\", \"W2\", ... for weaknesses only\n"
            "Do NOT score strengths (S1, S2, ...) or questions (Q1, Q2, ...)\n\n"
        )

        examples = (
            "<START OF EXAMPLE>\n\n"
            "### Weaknesses\n"
            "W1. The paper has limited novelty and the experiments are insufficient.\n"
            "W2. In Table 3 (§5.1), the proposed method is only compared against 2 baselines from 2019, "
            "missing recent strong baselines like MethodX (2023) and MethodY (2024) that report higher accuracy on the same benchmark.\n"
            "W3. The convergence proof assumes bounded gradients but the paper does not discuss this.\n"
            "\nEVALUATION:\n\n"
            "<item_scores>\n"
            "[{\"item_id\": \"W1\", \"score\": 1, \"reason\": \"No paper reference, no specifics on what novelty is lacking or which experiments are insufficient\"},\n"
            " {\"item_id\": \"W2\", \"score\": 5, \"reason\": \"References Table 3 and §5.1, names specific missing baselines with years, clear what to fix\"},\n"
            " {\"item_id\": \"W3\", \"score\": 3, \"reason\": \"No specific section/equation reference for the convergence proof, but clearly specifies the issue (bounded gradient assumption)\"}]\n"
            "</item_scores>\n"
            "<reasoning>W1 is entirely ungrounded — no section, table, or specific critique. "
            "W2 is fully grounded with Table 3/§5.1 reference and specific about missing baselines. "
            "W3 specifies the technical issue clearly but doesn't point to where in the paper the proof is.</reasoning>\n\n"
            "<END OF EXAMPLE>\n\n"
        )

    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    return query, criteria, examples


# User prompt template
TRAJECTORY_JUDGE_USER_PROMPT_EVIDENCE = """{query}{criteria}{examples}[ANSWER]:

{trajectory_summary}

## Final Review Outline

### Strengths
{outline_strengths}

### Weaknesses
{outline_weaknesses}

## Statistics

- Total steps: {n_steps}
- Claims: {n_claims} ({n_supported} supported, {n_weak} weak, {n_pending} pending)
- Memory questions: {n_questions} ({n_resolved} resolved, {n_open} open)
- Notes: {n_notes}
- Sections visited: {sections_visited}

---

Note: For technical_depth, only weaknesses are evaluated.
For outline_grounding, strengths and weaknesses are evaluated against memory records.
"""


# Technical-depth prompt template (with paper content for verifying technical engagement)
TECHNICAL_DEPTH_USER_PROMPT_EVIDENCE = """{query}{criteria}{examples}[ANSWER]:

## Paper Content (for verifying technical engagement)

{paper_content}

## Final Review Outline

### Weaknesses
{outline_weaknesses}

---

Note: For technical_depth, only weaknesses are evaluated.
Use the paper content above to verify whether weaknesses engage with the paper's actual methodology, algorithms, proofs, or experimental design.
"""


# Grounding prompt template (needs paper content to verify references)
GROUNDING_USER_PROMPT_EVIDENCE = """{query}{criteria}{examples}[ANSWER]:

## Paper Content (for reference verification)

{paper_content}

## Final Review Outline

### Weaknesses
{outline_weaknesses}

---

Note: For grounding, only weaknesses are evaluated.
Use the paper content above to verify whether weaknesses reference specific parts of the paper.
"""
