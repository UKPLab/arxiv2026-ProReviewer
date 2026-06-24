"""SciRM-format prompts for trajectory memory reasoning evaluation.

Uses the SciRM model's input/output format:
- Input: [QUERY] + [CRITERIA] + [EXAMPLES] + [ANSWER]
- Output: <reasoning>...</reasoning><score>...</score>

Evaluates the same 5 dimensions as V2 but adapted to SciRM's evaluation framework.

Each dimension uses 3 examples demonstrating scores 1, 3, and 5.
Examples use the actual trajectory format (Step/Action/Observed/Memory_ops) so the judge
sees the same structure it will evaluate.
"""

from typing import Tuple

# System prompt for SciRM format
TRAJECTORY_JUDGE_SYSTEM_PROMPT_SCIRM = """\
You are an evaluator of agent investigation trajectories for paper review tasks. You will receive a trajectory showing how an agent investigated a paper step-by-step, along with criteria explaining the specific evaluation aspect and the scoring rubric. You should evaluate the trajectory quality based on the given criteria. First output your score enclosed between <score> and </score>. Inside <score> provide only the numeric score and nothing else. Then, output your reasoning enclosed between <reasoning> and </reasoning>.
"""
#  The reasoning should be in 3~5 sentences.
# which precisely reflects the evaluation

# Dimensions (same as V2)
DIMS_SCIRM = [
    "factual_correctness",
    "claim_specificity",
    "technical_depth",
    "cross_verification",
    "grounding",  # Replaces outline_grounding; uses paper content + per-weakness rubric from evaluation_prompt.py
]

DIMENSION_WEIGHTS_SCIRM = {
    "factual_correctness": 0.20,
    "claim_specificity": 0.20,
    "technical_depth": 0.20,
    "cross_verification": 0.20,
    "grounding": 0.20,
}


def get_scirm_dimension_prompt(dimension: str) -> Tuple[str, str, str]:
    """Get [QUERY], [CRITERIA], and [EXAMPLES] for a dimension.

    Returns:
        (query, criteria, examples)
    """

    if dimension == "factual_correctness":
        query = (
            "[QUERY]: Evaluate whether the agent's memory entries (claims, notes, "
            "outline items) are factually accurate and grounded in the observations "
            "provided in the trajectory. The agent reads paper sections step-by-step "
            "and logs memory operations. You need to verify that memory accurately "
            "reflects what was observed.\n\n"
        )

        criteria = (
            "[CRITERIA]: Factual correctness means all claims, notes, and outline "
            "items are traceable to specific observations. The \"Observed\" field "
            "shows the full content the agent read at each step. Memory operations "
            "should not contradict observations, extrapolate beyond what was read, "
            "or reference content not found in any observation. When a claim's status "
            "is changed (e.g. to_be_verified → supported), the verifier_reason should "
            "cite evidence visible in the Observed content. Scoring rubric is as "
            "follows:\n"
            "1: Multiple fabricated claims or outline items that reference content "
            "never seen in any Observed content; major factual contradictions.\n"
            "2: Several claims misinterpret observations or extrapolate beyond what "
            "was read; verifier_reasons cite content not visible in observations.\n"
            "3: Mostly accurate but 1-2 claims overstate findings or include details "
            "not directly visible in observations.\n"
            "4: All claims grounded in observations; verifier_reasons cite specific "
            "observed content; at most minor interpretation issues.\n"
            "5: Perfectly grounded; every claim and status change traceable to exact "
            "text in Observed snippets; verifier_reasons quote or closely paraphrase "
            "observed content.\n\n"
        )

        examples = (
            "<START OF EXAMPLE 1>\n\n"
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
            "<score>1</score>\n\n"
            "<reasoning>C1 fabricates specific architectural details (6-layer, "
            "residual connections, layer normalization) that are not present in the "
            "Observed snippet, which only mentions \"graph neural network\" and "
            "\"message-passing layers\". C2 claims \"94% accuracy\" but the observation "
            "shows \"AUC of 0.82\" — a different metric and value. The verifier_reason "
            "for C2's status change is vague (\"experiments confirm strong results\") "
            "rather than citing the actual AUC number. Multiple fabrications across "
            "claims.</reasoning>\n\n"
            "<END OF EXAMPLE 1>\n\n\n"

            # "<START OF EXAMPLE 2>\n\n"
            # "TRAJECTORY EXCERPT:\n"
            # "Step 1:\n"
            # "  Action: read_section(abstract)\n"
            # "  Observed: We present a unified framework for fairness evaluation "
            # "using sparsity measures. Our approach aligns with existing criteria "
            # "such as statistical parity and equalized odds, and we validate it "
            # "on multiple datasets.\n"
            # "  Memory_ops:\n"
            # "    +Claim C1 (§abstract): The paper proposes a unified sparsity-based "
            # "framework that aligns with existing fairness criteria.\n"
            # "    +Claim C2 (§abstract): The framework is validated through experiments "
            # "on 8 benchmark datasets with 12 bias mitigation methods.\n"
            # "Step 5:\n"
            # "  Action: read_section(5 experiments)\n"
            # "  Observed: We apply the PQ Index on several datasets including UCI "
            # "Adult, COMPAS, and LawSchool. We compare against Reweight, FairRR, "
            # "and Reduction...\n"
            # "  Memory_ops:\n"
            # "    Claim C2: to_be_verified → supported — Experiments section confirms "
            # "multiple datasets and mitigation methods.\n"
            # "\nEVALUATION:\n\n"
            # "<reasoning>C1 is well-grounded — the abstract explicitly mentions "
            # "a unified framework aligning with existing criteria. C2 claims \"8 "
            # "benchmark datasets with 12 bias mitigation methods\" but the observation "
            # "only lists 3 datasets (Adult, COMPAS, LawSchool) and 3 methods "
            # "(Reweight, FairRR, Reduction); the specific counts of 8 and 12 are "
            # "not visible in any Observed snippet. The verifier_reason is also "
            # "vague (\"confirms multiple datasets\") without citing the actual "
            # "observed list. This is a case where the claim overstates specifics "
            # "beyond what the observations show. Therefore the evaluation score "
            # "should be 3.</reasoning>\n\n"
            # "<score>3</score>\n\n"
            # "<END OF EXAMPLE 2>\n\n\n"

            # "<START OF EXAMPLE 3>\n\n"
            # "TRAJECTORY EXCERPT:\n"
            # "Step 3:\n"
            # "  Action: read_section(4 experiments)\n"
            # "  Observed: Table 2 shows accuracy of 85.3% on CIFAR-10 and 91.7% "
            # "on SVHN. Our method outperforms the baseline by 2.1% on CIFAR-10...\n"
            # "  Memory_ops:\n"
            # "    +Claim C3 (§4 experiments): Table 2 reports 85.3% accuracy on "
            # "CIFAR-10 and 91.7% on SVHN, outperforming baseline by 2.1% on CIFAR-10.\n"
            # "    Claim C1: to_be_verified → supported — Table 2 shows 85.3% on "
            # "CIFAR-10, confirming the accuracy claim from the abstract.\n"
            # "\nEVALUATION:\n\n"
            # "<reasoning>C3 exactly matches the observed content: the specific "
            # "numbers (85.3%, 91.7%, 2.1%), the table reference, and the datasets "
            # "are all directly present in the Observed snippet. The status change "
            # "for C1 cites the exact table number and figure from the observation. "
            # "Every detail is traceable to observed text. Therefore the evaluation "
            # "score should be 5.</reasoning>\n\n"
            # "<score>5</score>\n\n"
            # "<END OF EXAMPLE 3>\n\n"
        )

    elif dimension == "claim_specificity":
        query = (
            "[QUERY]: Evaluate whether the agent's claims and outline items "
            "reference specific, concrete paper elements rather than making generic "
            "statements like \"method has limitations\" or \"experiments are limited\".\n\n"
        )

        criteria = (
            "[CRITERIA]: Specificity means claims cite concrete paper elements. "
            "Count references to: equation numbers (Eq 3, Equation 5), "
            "table/figure numbers (Table 2, Fig 4, Figure 3(a)), "
            "algorithm/line numbers (Alg 1 line 7, Algorithm 2), "
            "specific section/subsection numbers (§3.2, Section 4.1, §2), "
            "or direct quotes/paraphrases from the paper. Also check whether "
            "verifier_reasons reference specific evidence locations rather than "
            "vague justifications. Scoring rubric is as follows:\n"
            "1: Claims and outline are generic statements (\"method is novel\", "
            "\"experiments are limited\") with no concrete references.\n"
            "2: Some section references but mostly high-level (\"§3 describes "
            "method X but doesn't justify Y\") without citing specific elements "
            "within sections.\n"
            "3: Moderate specificity: claims reference sections/subsections; "
            "some mention of tables/figures.\n"
            "4: High specificity: most claims cite equation numbers, "
            "table/figure numbers, or specific subsections.\n"
            "5: Exceptional: claims cite specific equations/lines/figures AND "
            "outline items directly quote or paraphrase with exact locations.\n\n"
        )

        examples = (
            "<START OF EXAMPLE 1>\n\n"
            "TRAJECTORY EXCERPT:\n"
            "Step 2:\n"
            "  Action: read_section(3 methodology)\n"
            "  Observed: We propose a novel attention mechanism... The loss function "
            "combines cross-entropy with a contrastive term...\n"
            "  Memory_ops:\n"
            "    +Claim C1 (§3 methodology): The method is novel and interesting.\n"
            "    +Outline weakness: Experiments are limited and need more baselines.\n"
            "\nEVALUATION:\n\n"
            "<score>1</score>\n\n"
            "<reasoning>C1 says \"novel and interesting\" without referencing the "
            "attention mechanism or loss function described in the observation. The "
            "outline weakness is entirely generic — \"experiments are limited\" without "
            "specifying what baselines are missing or which experiments are insufficient. "
            "Neither entry cites any concrete paper element.</reasoning>\n\n"
            "<END OF EXAMPLE 1>\n\n\n"

            # "<START OF EXAMPLE 2>\n\n"
            # "TRAJECTORY EXCERPT:\n"
            # "Step 4:\n"
            # "  Action: read_section(4 unifying group fairness with sparsity)\n"
            # "  Observed: We formulate a unified fairness framework where sparsity is "
            # "the inverse of fairness. We replace the Maximum Pairwise Difference "
            # "used in existing metrics with a sparsity measure over w...\n"
            # "  Memory_ops:\n"
            # "    +Claim C2 (§4 unifying group fairness with sparsity): Section 4 "
            # "defines the unified framework by replacing Maximum Pairwise Difference "
            # "with sparsity measures for both classification and regression tasks.\n"
            # "    +Outline strength: Section 4 provides clear definitions for both "
            # "classification and regression tasks. [C2]\n"
            # "    +Outline weakness: The paper does not sufficiently justify why "
            # "sparsity should be the primary lens for fairness evaluation. [Q1, N3]\n"
            # "\nEVALUATION:\n\n"
            # "<reasoning>The claim references a specific section (§4) and names "
            # "concrete concepts (Maximum Pairwise Difference, sparsity measures, "
            # "classification vs regression). The outline items cite memory references "
            # "[C2], [Q1], [N3] and mention specific section content. However, there "
            # "are no references to specific equations, theorems, or figures within "
            # "the section. This is moderate specificity — section-level with named "
            # "concepts but not equation/theorem-level. Therefore the evaluation score "
            # "should be 3.</reasoning>\n\n"
            # "<score>3</score>\n\n"
            # "<END OF EXAMPLE 2>\n\n\n"

            # "<START OF EXAMPLE 3>\n\n"
            # "TRAJECTORY EXCERPT:\n"
            # "Step 3:\n"
            # "  Action: read_section(3 sparsity)\n"
            # "  Observed: Theorem 3.5 shows PQ Index bounds the Gini Index: "
            # "G(w) ≤ PQ(w) ≤ 2G(w). Table 1 compares properties of MPD, Gini, "
            # "and PQ across 6 axioms...\n"
            # "  Memory_ops:\n"
            # "    +Claim C3 (§3 sparsity): Theorem 3.5 establishes bounding "
            # "relationship G(w) ≤ PQ(w) ≤ 2G(w) between Gini Index and PQ Index.\n"
            # "    +Note N4 (§3 sparsity): Table 1 shows PQ satisfies all 6 axioms "
            # "while Gini fails axiom 4 (transfer principle).\n"
            # "    +Outline strength: Theorem 3.5 provides tight bounds between "
            # "Gini and PQ Index (Table 1 confirms PQ satisfies all 6 axioms). "
            # "[C3, N4]\n"
            # "\nEVALUATION:\n\n"
            # "<reasoning>The claim cites a specific theorem number (Theorem 3.5), "
            # "an exact mathematical inequality, and the specific measures involved. "
            # "The note references a specific table (Table 1) with exact details "
            # "(6 axioms, axiom 4 transfer principle). The outline item cites both "
            # "memory records and references both the theorem and table by number. "
            # "This is exceptional specificity with exact locations and content. "
            # "Therefore the evaluation score should be 5.</reasoning>\n\n"
            # "<score>5</score>\n\n"
            # "<END OF EXAMPLE 3>\n\n"
        )

    elif dimension == "technical_depth":
        query = (
            "[QUERY]: Evaluate the analytical depth of the agent's investigation. "
            "Analytical depth measures whether the agent's claims and questions engage "
            "with the paper's technical content and analyze why identified issues matter.\n\n"
        )

        criteria = (
            "[CRITERIA]: Analytical depth has two components:\n"
            "1. **Technical Engagement**: Whether the agent's claims and questions engage "
            "with the paper's methodology, algorithms, proofs, or experimental design "
            "choices (as opposed to commenting on scope, presentation, or completeness).\n"
            "2. **Analytical Reasoning**: Whether the agent explains why identified issues "
            "are problematic (e.g., what breaks, what the consequences are, or how it "
            "affects validity).\n\n"
            "It's more important for the critique to be technical (engaging with actual "
            "methodology) than to provide reasoning about why an issue matters.\n\n"
            "Scoring rubric:\n"
            "5: **Technical and Reasoned** - Claims/questions engage with specific technical "
            "content (methodology, algorithms, proofs, experimental design) AND explain why "
            "issues are problematic (consequences, failure modes, what breaks).\n"
            "4: **Technical but Unreasoned** - Claims/questions engage with specific technical "
            "content but state issues without fully explaining consequences or why they matter.\n"
            "3: **Non-technical but Reasoned** - Claims/questions do not engage with specific "
            "technical content, but provide reasoning about why identified gaps matter.\n"
            "2: **Non-technical and Unreasoned** - Claims/questions neither engage with specific "
            "technical content nor explain reasoning about consequences.\n"
            "1: **No Substance** - All claims/questions are pure surface observations requiring "
            "no domain expertise (e.g., \"dataset is small\", \"writing could be clearer\").\n\n"
        )

        examples = (
            "<START OF EXAMPLE 1>\n\n"
            "TRAJECTORY EXCERPT:\n"
            "Step 2:\n"
            "  Action: read_section(5 experiments)\n"
            "  Observed: We evaluate on CIFAR-10 and SVHN using ResNet-50...\n"
            "  Memory_ops:\n"
            "    +Claim C1 (§5 experiments): The dataset only has two benchmarks "
            "which is limited.\n"
            "    +Outline weakness: More experiments on diverse datasets are "
            "needed. [C1]\n"
            "    +Note N1 (§5 experiments): The paper could benefit from better "
            "writing in the experiments section.\n"
            "\nEVALUATION:\n\n"
            "<score>1</score>\n\n"
            "<reasoning>C1 and the outline weakness are surface observations about "
            "experiment scope (\"limited\", \"more needed\") without engaging with any "
            "technical content or explaining consequences. "
            "N1 is a writing quality comment. None engage with methodology or reason "
            "about why issues matter. No Substance.</reasoning>\n\n"
            "<END OF EXAMPLE 1>\n\n\n"

            # "<START OF EXAMPLE 2>\n\n"
            # "TRAJECTORY EXCERPT:\n"
            # "Step 3:\n"
            # "  Action: read_section(4 unifying group fairness with sparsity)\n"
            # "  Observed: We replace the Maximum Pairwise Difference used in existing "
            # "fairness metrics with a sparsity measure over w, where the length of "
            # "the vector w equals the number of sensitive groups...\n"
            # "  Memory_ops:\n"
            # "    +Question Q1 (§4): Does the paper provide a theoretical justification "
            # "for why sparsity measures are suitable for fairness evaluation, beyond "
            # "the algebraic connection to MPD?\n"
            # "    +Outline weakness: While §4 provides the algebraic framework, it "
            # "does not justify why sparsity should be the primary lens for fairness "
            # "rather than other inequality measures. [Q1]\n"
            # "\nEVALUATION:\n\n"
            # "<reasoning>Q1 goes beyond describing the method to questioning its "
            # "justification — asking whether the algebraic connection to MPD is "
            # "sufficient theoretical grounding. The outline weakness identifies a "
            # "specific conceptual gap (why sparsity over other inequality measures). "
            # "This shows engagement with the methodology and its justification, "
            # "though it does not probe into specific assumptions or edge cases of "
            # "the framework itself. Therefore the evaluation score should be 3."
            # "</reasoning>\n\n"
            # "<score>3</score>\n\n"
            # "<END OF EXAMPLE 2>\n\n\n"

            # "<START OF EXAMPLE 3>\n\n"
            # "TRAJECTORY EXCERPT:\n"
            # "Step 3:\n"
            # "  Action: read_section(3 sparsity)\n"
            # "  Observed: Theorem 3.5 shows PQ(w) bounds Gini: G(w) ≤ PQ(w) ≤ "
            # "2G(w). The proof assumes w ∈ R+^d (non-negative components)...\n"
            # "  Memory_ops:\n"
            # "    +Claim C3 (§3 sparsity): Theorem 3.5 assumes non-negative vector "
            # "components (w ∈ R+^d), but in fairness applications with regression "
            # "tasks the prediction differences can be negative, potentially "
            # "violating this assumption.\n"
            # "    +Question Q3 (§3 sparsity): How does the framework handle "
            # "intersectional fairness where group sizes are highly unequal — does "
            # "the PQ Index's sensitivity to vector dimension d (from Theorem 3.3) "
            # "create a bias toward larger intersectional groups?\n"
            # "\nEVALUATION:\n\n"
            # "<reasoning>C3 identifies a specific assumption (non-negative components) "
            # "in a theorem and connects it to a concrete scenario (regression tasks "
            # "with negative differences) where the assumption may be violated. Q3 "
            # "probes an interaction between a mathematical property (sensitivity to "
            # "dimension d from Theorem 3.3) and a practical scenario (intersectional "
            # "fairness with unequal groups). Both demonstrate deep technical analysis "
            # "that connects theoretical properties to practical failure modes. "
            # "Therefore the evaluation score should be 5.</reasoning>\n\n"
            # "<score>5</score>\n\n"
            # "<END OF EXAMPLE 3>\n\n"
        )

    elif dimension == "cross_verification":
        query = (
            "[QUERY]: Evaluate whether the agent verified claims by "
            "cross-referencing evidence from different parts of the paper. Focus on "
            "verification ACTIONS (claim/question status changes with reasons) rather "
            "than just reading order. Sequential reading with verification from the "
            "immediately next section is not cross-verification.\n\n"
        )

        criteria = (
            "[CRITERIA]: Cross-verification means claims from one section are "
            "verified or revised using evidence from non-adjacent sections. "
            "Sequential reading (intro → method → experiments in order) where claims "
            "are verified from the immediately next section is not genuine "
            "cross-referencing. Check claim status changes (to_be_verified → "
            "supported/weak/invalid) and whether verifier reasons cite evidence "
            "from sections distant from where the claim was originally made. Also "
            "check if claims get revised as understanding evolves (e.g. supported → "
            "weak after reading limitations). Scoring rubric is as follows:\n"
            "1: No cross-referencing; read 1-2 sections and immediately concluded; "
            "OR all claims left unverified.\n"
            "2: Sequential reading; claims resolved from immediately next section; "
            "OR all verification batched at finish step without citing specific "
            "evidence.\n"
            "3: Some non-adjacent verification; at least 1 claim updated using "
            "evidence from a section distant from where it was created.\n"
            "4: Systematic: multiple claims verified from non-obvious sections; "
            "questions answered using evidence gathered across the paper.\n"
            "5: Iterative re-evaluation: claims revised multiple times as "
            "understanding evolved; explicit back-and-forth investigation.\n\n"
        )

        examples = (
            "<START OF EXAMPLE 1>\n\n"
            "TRAJECTORY EXCERPT:\n"
            "Step 0:\n"
            "  Action: read_section(abstract)\n"
            "  Memory_ops:\n"
            "    +Claim C1 (§abstract): Method achieves state-of-the-art performance.\n"
            "Step 1:\n"
            "  Action: finish\n"
            "  Memory_ops:\n"
            "    Claim C1: to_be_verified → supported — The abstract claims SOTA.\n"
            "\nEVALUATION:\n\n"
            "<score>1</score>\n\n"
            "<reasoning>The agent read only the abstract and immediately finished, "
            "marking C1 as supported based solely on the abstract's own claim. "
            "There is no cross-referencing with experiments or any other section. "
            "The verifier_reason merely restates the abstract rather than citing "
            "independent evidence.</reasoning>\n\n"
            "<END OF EXAMPLE 1>\n\n\n"

            # "<START OF EXAMPLE 2>\n\n"
            # "TRAJECTORY EXCERPT:\n"
            # "Step 0:\n"
            # "  Action: read_section(abstract)\n"
            # "  Memory_ops:\n"
            # "    +Claim C1 (§abstract): Framework unifies existing fairness criteria.\n"
            # "    +Claim C2 (§abstract): Experiments validate on multiple datasets.\n"
            # "Step 3:\n"
            # "  Action: read_section(4 unifying group fairness)\n"
            # "  Memory_ops:\n"
            # "    +Question Q1: What specific datasets are used?\n"
            # "Step 5:\n"
            # "  Action: read_section(5 experiments)\n"
            # "  Memory_ops:\n"
            # "    Claim C2: to_be_verified → supported — §5 lists UCI Adult, COMPAS, "
            # "and LawSchool datasets.\n"
            # "    Question Q1: open → resolved — Experiments use 5 classification "
            # "and 2 regression datasets.\n"
            # "Step 6:\n"
            # "  Action: read_section(6 conclusion)\n"
            # "  Memory_ops:\n"
            # "    Claim C1: to_be_verified → supported — §4 and §5 together confirm "
            # "the unification claim.\n"
            # "\nEVALUATION:\n\n"
            # "<reasoning>C2 from the abstract is verified in §5 (experiments), which "
            # "is non-adjacent, and Q1 is also resolved there. C1 is verified at the "
            # "conclusion step by combining evidence from §4 and §5. However, all "
            # "verification flows forward — no claims are revised or re-evaluated, "
            # "and the agent never goes back to re-read earlier sections. This shows "
            # "some non-adjacent verification but not iterative investigation. "
            # "Therefore the evaluation score should be 3.</reasoning>\n\n"
            # "<score>3</score>\n\n"
            # "<END OF EXAMPLE 2>\n\n\n"

            # "<START OF EXAMPLE 3>\n\n"
            # "TRAJECTORY EXCERPT:\n"
            # "Step 0:\n"
            # "  Action: read_section(abstract)\n"
            # "  Memory_ops:\n"
            # "    +Claim C1 (§abstract): Method achieves 90% accuracy.\n"
            # "Step 2:\n"
            # "  Action: read_section(3 methodology)\n"
            # "  Memory_ops:\n"
            # "    +Question Q1: Does the method handle edge cases?\n"
            # "Step 4:\n"
            # "  Action: read_section(5 experiments)\n"
            # "  Memory_ops:\n"
            # "    Claim C1: to_be_verified → weak — Table 3 shows only 85%, not "
            # "90% as claimed in abstract.\n"
            # "Step 5:\n"
            # "  Action: read_section(3 methodology)  [RE-READING]\n"
            # "  Memory_ops:\n"
            # "    Question Q1: open → partially_answered — §3.2 mentions edge cases "
            # "but analysis is incomplete.\n"
            # "Step 6:\n"
            # "  Action: read_section(6 limitations)\n"
            # "  Memory_ops:\n"
            # "    Claim C1: weak → invalid — §6 acknowledges accuracy varies 80-88%, "
            # "confirming the abstract overstates.\n"
            # "\nEVALUATION:\n\n"
            # "<reasoning>C1 undergoes iterative revision: first weakened by Table 3 "
            # "in §5 (step 4), then invalidated by §6 limitations (step 6), using "
            # "evidence from two distant sections. The agent re-reads §3 (step 5) to "
            # "investigate Q1, showing explicit back-and-forth. Claims are revised "
            # "multiple times as understanding evolved. This is iterative "
            # "re-evaluation with cross-referencing across the paper. Therefore the "
            # "evaluation score should be 5.</reasoning>\n\n"
            # "<score>5</score>\n\n"
            # "<END OF EXAMPLE 3>\n\n"
        )

    elif dimension == "grounding":
        query = (
            "[QUERY]: Evaluate whether each weakness in the review references "
            "specific parts of the paper and clearly specifies what is wrong or "
            "missing. You will be given the paper content to verify the references.\n\n"
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
        )

        examples = (
            "<START OF EXAMPLE 1>\n\n"
            "WEAKNESS:\n"
            "The paper has limited novelty and the experiments are insufficient.\n"
            "\nEVALUATION:\n\n"
            "<score>1</score>\n\n"
            "<reasoning>The weakness does not reference any specific part of the "
            "paper — no section, table, figure, or equation is mentioned. It also "
            "does not specify what aspect of novelty is limited or which experiments "
            "are insufficient. Not grounded and not specific.</reasoning>\n\n"
            "<END OF EXAMPLE 1>\n\n\n"
        )

    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    return query, criteria, examples


# User prompt template for SciRM format
TRAJECTORY_JUDGE_USER_PROMPT_SCIRM = """{query}{criteria}{examples}[ANSWER]:

{trajectory_summary}

## Final State Statistics

- Total steps: {n_steps}
- Claims: {n_claims} ({n_supported} supported, {n_weak} weak, {n_pending} pending)
- Questions: {n_questions} ({n_resolved} resolved, {n_open} open)
- Notes: {n_notes}
- Outline: {n_strengths} strengths, {n_weaknesses} weaknesses
- Sections visited: {sections_visited}
"""
