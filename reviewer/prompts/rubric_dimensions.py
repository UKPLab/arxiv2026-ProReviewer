"""Shared rubric dimension definitions and prompt builders.

Single source of truth for rubric dimensions used in:
- Training (per-dimension rubric scoring via RubricEvaluator)
- Evaluation (per-weakness or batched rubric scoring via RubricEvaluator)

Canonical dimension names (snake_case keys):
  technical_depth        — depth of technical engagement + analytical reasoning
  grounding_specificity  — structural references to specific paper parts
  actionability          — actionability of feedback (constructive value)
  verifiability          — evidence supporting the reviewer's claims
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Canonical rubric definitions per dimension
# ---------------------------------------------------------------------------

RUBRIC_DIMENSIONS: Dict[str, Dict[str, str]] = {
    "technical_depth": {
        "definition": (
            "**Analytical Depth**\n"
            "\n"
            "**Definition:** Measures the depth of technical engagement and analytical "
            "reasoning in a review point. This aspect has two components:\n"
            "\n"
            "1. **Technical Engagement**: Whether the critique engages with the paper's "
            "methodology, algorithms, proofs, or experimental design choices (as opposed "
            "to commenting on scope, presentation, or completeness).\n"
            "\n"
            "2. **Analytical Reasoning**: Whether the critique explains why the identified "
            "issue is problematic (e.g., what breaks, what the consequences are, or how it "
            "affects validity).\n"
            "\n"
            "**Importance:** It's more important for the critique to be technical (engaging "
            "with actual methodology) than to provide reasoning about why an issue matters.\n"
            "\n"
            "**Analytical Depth Scale (1-5):**\n"
            "\n"
            "1. **1: No Substance**\n"
            '   - **Definition:** Pure surface observation requiring no domain expertise. '
            'Example: "Writing could be clearer", "Limited novelty", "More experiments needed."\n'
            "\n"
            "2. **2: Non-technical and Unreasoned**\n"
            "   - **Definition:** The critique neither engages with specific technical content "
            "nor explains reasoning about consequences. "
            'Example: "The evaluation only uses 2 datasets, which seems insufficient."\n'
            "\n"
            "3. **3: Non-technical but Reasoned**\n"
            "   - **Definition:** The critique does not engage with specific technical content, "
            "but provides reasoning about why the identified gap matters. "
            'Example: "Evaluating only on CIFAR-10 and ImageNet limits generalizability '
            "because the data augmentation strategy may not transfer to domains with scarce "
            'labeled data."\n'
            "\n"
            "4. **4: Technical but Unreasoned**\n"
            "   - **Definition:** The critique engages with specific technical content but "
            "states the issue without fully explaining its consequences or why it matters. "
            'Example: "The convergence proof assumes L-smoothness but the loss function uses '
            'ReLU which is non-smooth."\n'
            "\n"
            "5. **5: Technical and Reasoned**\n"
            "   - **Definition:** The critique engages with specific technical content "
            "(methodology, algorithms, proofs, experimental design) AND explains why the "
            "issue is problematic (consequences, failure modes, what breaks). "
            "Example: \"Theorem 2's convergence proof requires L-smoothness (Assumption 3), "
            "but the ReLU activation in Eq. 4 is non-smooth at zero, invalidating the O(1/T) "
            'rate claimed in Eq. 7."'
        ),
    },
    "grounding_specificity": {
        "definition": (
            "**Grounding Specificity**\n"
            "\n"
            "**Definition:** Measures how explicitly a review comment refers to a specific "
            "part of the paper and how clearly it identifies the issue with that part. This "
            "helps authors understand what needs revision and why. Grounding specificity has "
            "two key components:\n"
            "\n"
            "1. **Grounding:** How well the authors can identify the specific part of the "
            "paper being addressed.\n"
            "   - **Weak Grounding:** The author can make an educated guess but cannot "
            "precisely identify the referenced part.\n"
            "   - **Full Grounding:** The author can accurately pinpoint the section, table, "
            "figure, or unique aspect being addressed. This can be achieved through:\n"
            "     - Literal mentions of sections, tables, figures, etc.\n"
            "     - Mentions of unique elements of the paper.\n"
            "     - General comments that clearly imply the relevant parts without explicitly "
            "naming them.\n"
            "\n"
            "2. **Specificity:** How clearly the comment details what is wrong or missing in "
            "the referenced part. If external work is mentioned, it also evaluates whether "
            "specific examples are provided.\n"
            "\n"
            "**Importance:** It's more important for the comment to be grounded than to be "
            "specific.\n"
            "\n"
            "**Grounding Specificity Scale (1-5):**\n"
            "\n"
            "1. **Not Grounded**\n"
            "   - **Definition**: This comment is not grounded at all. It does not identify "
            "a specific area in the paper. The comment is highly unspecific.\n"
            "\n"
            "2. **Weakly Grounded and Not Specific**\n"
            "   - **Definition**: The authors cannot confidently determine which part the "
            "comment addresses. Further, the comment does not specify what needs to be "
            "addressed in this part.\n"
            "\n"
            "3. **Weakly Grounded and Specific**\n"
            "   - **Definition**: The authors cannot confidently determine which part the "
            "comment addresses. However, the comment clearly specifies what needs to be "
            "addressed in this part.\n"
            "\n"
            "4. **Fully Grounded and Under-Specific**\n"
            "   - **Definition**: The comment explicitly mentions which part of the paper it "
            "addresses, or it should be obvious to the authors. However, this comment does "
            "not specify what needs to be addressed in this part.\n"
            "\n"
            "5. **Fully Grounded and Specific**\n"
            "   - **Definition**: The comment explicitly mentions which part of the paper it "
            "addresses, and it is obvious to the authors. The comment specifies what needs to "
            "be addressed in this part."
        ),
    },
    "actionability": {
        "definition": (
            "**Constructive Value**\n"
            "\n"
            "**Definition:** Measures the level of actionability in a review point. We "
            "evaluate actionability based on two criteria:\n"
            "\n"
            "1. **Explicit vs. Implicit**:\n"
            "   - **Explicit:** Actions or suggestions that are direct or apparent. Authors "
            "can directly identify modifications they should apply to their draft. "
            "Clarification questions should be treated as explicit statements if they give a "
            "direct action.\n"
            "   - **Implicit:** Actions that need to be inferred from the comment. This "
            "includes missing parts that need to be added. Authors can deduce what needs to "
            "be done after reading the comment.\n"
            "\n"
            "2. **Concrete vs. Vague**:\n"
            "   - **Concrete:** Once the action is identified, the authors know exactly what "
            "needs to be done and how to apply the action.\n"
            "   - **Vague:** After identifying the action, the authors still don't know how "
            "to carry out this action.\n"
            "\n"
            "**Importance:** It's more important for actions to be concrete so that authors "
            "know how to apply them. It's also preferred for actions to be stated directly "
            "rather than inferred.\n"
            "\n"
            "**Constructive Value Scale (1-5):**\n"
            "\n"
            "1. **1: Unactionable**\n"
            "   - **Definition:** The comment lacks meaningful information to help authors "
            "improve the paper. Authors do not know what they should do after reading the "
            "comment.\n"
            "\n"
            "2. **2: Borderline Actionable**\n"
            "   - **Definition:** The comment includes an implicitly stated action or an "
            "action that can be inferred. However, the action itself is vague and lacks "
            "detail on how to apply it.\n"
            "\n"
            "3. **3: Somewhat Actionable**\n"
            "   - **Definition:** The comment explicitly states an action but is vague on "
            "how to execute it.\n"
            "\n"
            "4. **4: Mostly Actionable**\n"
            "   - **Definition:** The comment implicitly states an action but concretely "
            "states how to implement the inferred action.\n"
            "\n"
            "5. **5: Highly Actionable**\n"
            "   - **Definition:** The comment contains an explicit action and concrete "
            "details on how to implement it. Authors know exactly how to apply it."
        ),
    },
    "verifiability": {
        "definition": (
            "**Verifiability**\n"
            "\n"
            "**Definition:** Evaluates whether a review comment contains a claim and, if so, "
            "how well that claim is supported using logical reasoning, common knowledge, or "
            "external references.\n"
            "\n"
            "### **Step 1: Claim Extraction**\n"
            "\n"
            "**Objective:**\n"
            "Determine whether the given text contains a claim (i.e., an opinion, judgment, "
            "or suggestion) or consists solely of factual statements that require no "
            "verification.\n"
            "\n"
            "**Claim Definition:**\n"
            "A statement is considered a claim if it falls into one or more of the following "
            "categories:\n"
            "- **Subjective opinions or disagreements** (e.g., criticism of an experimental "
            "choice).\n"
            "- **Suggestions or requests for changes** (e.g., recommending removal, addition, "
            "or discussion).\n"
            "- **Judgments about the paper** (e.g., stating something is unclear, not "
            "well-written, or lacks detail).\n"
            "- **Deductions or inferred observations** that go beyond merely stating facts.\n"
            "- **Statements requiring justification** to be understood or accepted.\n"
            "\n"
            "\n"
            '**Normal Statements ("No Claim")**\n'
            'A statement is classified as "X" if it:\n'
            "- Describes facts without suggesting changes.\n"
            "- Makes general statements about the paper without an opinion.\n"
            "- Presents objective, verifiable facts that require no justification.\n"
            "- Asks for clarifications or general questions.\n"
            "- States logical statements or directly inferable information.\n"
            '- Makes positive claims (e.g., *"The paper is well-written"*), as these do not '
            "help improve the work.\n"
            "\n"
            "---\n"
            "\n"
            "### **Step 2: Verifiability Verification**\n"
            "\n"
            "**Objective:**\n"
            "Assess how well a claim is verified by examining the reasoning, common knowledge, "
            "or external references provided. The purpose is to ensure that the review comment "
            "helps the authors improve their work.\n"
            "\n"
            "**Verification Methods:**\n"
            "A claim is considered verifiable if supported by one or more of the following:\n"
            "- **Logical reasoning** \u2013 A clear explanation of why the claim is valid.\n"
            "- **Common knowledge** \u2013 Reference to well-accepted practices or standards.\n"
            "- **External references** \u2013 Citation of relevant literature, data, or sources.\n"
            "\n"
            "**Verifiability Scale (1\u20135 & X):**\n"
            "\n"
            "1. **1: Unverifiable**\n"
            "   - The comment contains a claim without any supporting evidence or "
            "justification.\n"
            "2. **2: Borderline Verifiable**\n"
            "   - Some support is provided, but it is vague, insufficient, or difficult to "
            "follow.\n"
            "3. **3: Somewhat Verifiable**\n"
            "   - The claim has some justification but lacks key elements (e.g., examples, "
            "references).\n"
            "4. **4: Mostly Verifiable**\n"
            "   - The claim is well-supported but has minor gaps in explanation or "
            "references.\n"
            "5. **5: Fully Verifiable**\n"
            "   - The claim is thoroughly supported by explicit, sufficient, and robust "
            "evidence, such as:\n"
            "     - Clear reasoning and precise explanations.\n"
            "     - Specific references to external works.\n"
            "     - Logical and unassailable common-sense arguments.\n"
            "6. **X: No Claim**\n"
            "- The comment contains only factual, descriptive statements without claims, "
            "opinions, or suggestions.\n"
            "\n"
            "---\n"
            "\n"
            "### **Instructions for Evaluation:**\n"
            '1. **Extract Claims:** Determine whether the review comment contains a claim or '
            'is a normal statement. If there is no claim, score it as "X"\n'
            "2. **Assess Verifiability:** If a claim exists, score it based on how well it is "
            "justified from 1 to 5."
        ),
    },
    "helpfulness": {
        "definition": (
            "**Helpfulness**\n"
            "\n"
            "**Definition:** Assign a subjective score to reflect the value of the review "
            "comment to the authors. Helpfulness is rated on a scale from 1 to 5, with the "
            "following definitions:\n"
            "\n"
            "1. **1: Not Helpful at All**\n"
            "   - **Definition:** The comment fails to identify meaningful weaknesses or "
            "suggest improvements, leaving the authors with no actionable feedback.\n"
            "\n"
            "2. **2: Barely Helpful**\n"
            "   - **Definition:** The comment identifies a weakness or improvement area but "
            "is vague, lacks clarity, or provides minimal guidance, making it only slightly "
            "beneficial for the authors.\n"
            "\n"
            "3. **3: Somewhat Helpful**\n"
            "   - **Definition:** The comment identifies weaknesses or areas for improvement "
            "but is incomplete or lacks depth. While the authors gain some insights, the "
            "feedback does not fully address their needs for improving the draft.\n"
            "\n"
            "4. **4: Mostly Helpful**\n"
            "   - **Definition:** The comment provides clear and actionable feedback on "
            "weaknesses and areas for improvement, though it could be expanded or refined "
            "to be fully comprehensive and impactful.\n"
            "\n"
            "5. **5: Highly Helpful**\n"
            "   - **Definition:** The comment thoroughly identifies weaknesses and offers "
            "detailed, actionable, and constructive suggestions that empower the authors to "
            "significantly improve their draft."
        ),
    },
}

# Ordered list of canonical dimension names
DIMENSION_NAMES: List[str] = [
    "technical_depth",
    "grounding_specificity",
    "actionability",
    "verifiability",
    "helpfulness",
]


# ---------------------------------------------------------------------------
# Prompt builders for RubricEvaluator
# ---------------------------------------------------------------------------

def build_per_dimension_system_prompt() -> str:
    """Static preamble for per-dimension evaluation (one dim at a time, all weaknesses).

    Used by the ``per_dimension`` strategy in :class:`RubricEvaluator`.
    The LLM is asked to score every weakness on a single rubric dimension
    and return a JSON array.
    """
    return (
        "You are evaluating individual weakness points from a peer review on a "
        "single rubric dimension. Your role is to assess each weakness through "
        "careful, evidence-based analysis.\n\n"
        "You will be given:\n"
        "1. The rubric dimension with its scoring criteria\n"
        "2. The paper content for reference verification\n"
        "3. A numbered list of weaknesses to evaluate\n\n"
        "Score each weakness on the given dimension using the provided scale.\n\n"
        "**Respond directly in JSON format (no additional text):**\n"
        "```json\n"
        '[{"item_id": "W1", "score": <score>, "reason": "<reason>"},\n'
        ' {"item_id": "W2", "score": <score>, "reason": "<reason>"}]\n'
        "```\n\n"
        "This JSON will be automatically parsed, so ensure the format is precise."
    )


def build_per_dimension_user_prompt(
    dim: str,
    weakness_items: List[str],
    paper_content: str,
) -> str:
    """Build the user prompt for per-dimension evaluation.

    Args:
        dim: Canonical dimension name (e.g. ``"technical_depth"``).
        weakness_items: List of weakness text strings.
        paper_content: Full paper text for reference verification.

    Returns:
        Formatted user prompt string.
    """
    rubric = RUBRIC_DIMENSIONS[dim]

    numbered = "\n".join(
        f"W{i+1}. {text}" for i, text in enumerate(weakness_items)
    )

    return (
        f"#### Dimension: {dim} ####\n\n"
        f"{rubric['definition']}\n\n"
        f"#### Paper Text (for reference verification): ####\n"
        f"{paper_content}\n\n"
        f"#### Weaknesses to Evaluate: ####\n"
        f"{numbered}"
    )


def build_per_weakness_system_prompt(dims: List[str], score_only: bool = False) -> str:
    """Eval-style system prompt: all dims in one prompt, JSON output.

    Used by the ``per_weakness`` strategy in :class:`RubricEvaluator`.
    One weakness at a time, scored on all requested dimensions.

    Uses the same prompt style as the utility inference prompt with
    snake_case dimension keys in the JSON output.

    Args:
        dims: List of canonical dimension names to include.
        score_only: If True, only ask for numeric scores (no reasons).

    Returns:
        System prompt string.
    """
    parts = [
        "You are an expert in evaluating peer review comments with respect to "
        "different aspects. These aspects are aimed to maximize the utilization "
        "of the review comments for the authors. The primary purpose of the "
        "review is to help/guide authors in improving their drafts. Keep this "
        "in mind while evaluating the review point. Whenever you encounter a "
        'borderline case, think: "Will this review point help authors improve '
        'their draft?". There is no correlation between the aspect score and '
        "the length of the review point.\n\n"
        "ASPECTS DEFINITIONS:\n\n"
    ]

    for dim_name in dims:
        rubric = RUBRIC_DIMENSIONS[dim_name]
        parts.append(f"Aspect: {dim_name}\n{rubric['definition']}\n\n")
        if dim_name == "verifiability":
            parts.append(
                'Note: When scoring "X", use the string "X" (not a number).\n\n'
            )

    # Output format instructions
    if score_only:
        parts.append(
            "Evaluate the review based on the given definitions of the aspect(s) above. "
            "Output only the numeric score for each aspect in JSON format.\n\n"
            'For each aspect, provide the numeric score (1-5) or "X" if not applicable.\n\n'
            "**Respond directly in JSON format (no additional text):**\n"
            "```json\n{\n"
        )
        for dim_name in dims:
            score_hint = '1-5 or "X"' if dim_name == "verifiability" else "1-5"
            parts.append(f'  "{dim_name}": {score_hint},\n')
    else:
        parts.append(
            "Evaluate the review based on the given definitions of the aspect(s) above. "
            "Generate a rationale and use it to output the score.\n\n"
            "**Respond directly in JSON format (no additional text):**\n"
            "```json\n{\n"
        )
        for dim_name in dims:
            score_hint = '1-5 or "X"' if dim_name == "verifiability" else "1-5"
            parts.append(f'  "{dim_name}_reason": "<reason>",\n')
            parts.append(f'  "{dim_name}": {score_hint},\n')
    # Remove trailing comma from last line
    parts[-1] = parts[-1].rstrip(",\n") + "\n"
    parts.append("}\n```\n\n")
    parts.append(
        "This JSON will be automatically parsed, so ensure the format is precise."
    )

    return "".join(parts)


def build_per_weakness_user_prompt(
    weakness: str, paper_content: str = None
) -> str:
    """Build user prompt for per-weakness evaluation.

    Args:
        weakness: Single weakness text.
        paper_content: Full paper text for reference verification.
            If None, only the review point is included.

    Returns:
        Formatted user prompt string.
    """
    if paper_content:
        return (
            f"The review point is about a paper with the following text:\n"
            f"{paper_content}\n\n"
            f"Review Point: {weakness}"
        )
    return f"Review Point: {weakness}"


def build_batched_system_prompt(dims: List[str], num_weaknesses: int) -> str:
    """System prompt for batched evaluation: all dims, all weaknesses, one call.

    Used by the ``batched`` strategy in :class:`RubricEvaluator`.

    Args:
        dims: List of canonical dimension names to include.
        num_weaknesses: Number of weaknesses (used in output format example).

    Returns:
        System prompt string.
    """
    parts = [
        "You are an expert in evaluating peer review comments with respect to "
        "different aspects. These aspects are aimed to maximize the utilization "
        "of the review comments for the authors. The primary purpose of the "
        "review is to help/guide authors in improving their drafts. Keep this "
        "in mind while evaluating the review point. Whenever you encounter a "
        'borderline case, think: "Will this review point help authors improve '
        'their draft?". There is no correlation between the aspect score and '
        "the length of the review point.\n\n"
        "ASPECTS DEFINITIONS:\n\n"
    ]

    for dim_name in dims:
        rubric = RUBRIC_DIMENSIONS[dim_name]
        parts.append(f"Aspect: {dim_name}\n{rubric['definition']}\n\n")
        if dim_name == "verifiability":
            parts.append(
                'Note: When scoring "X", use the string "X" (not a number).\n\n'
            )

    # Output format
    parts.append(
        "Evaluate the review based on the given definitions of the aspect(s) above. "
        "Generate a rationale and use it to output the score for each weakness.\n\n"
        "**Respond directly in JSON format (no additional text):**\n"
        '```json\n{\n  "weaknesses": [\n    {\n'
    )
    for dim_name in dims:
        score_hint = '1-5 or "X"' if dim_name == "verifiability" else "1-5"
        parts.append(f'      "{dim_name}_reason": "<reason for weakness 1>",\n')
        parts.append(f'      "{dim_name}": {score_hint},\n')
    # Remove trailing comma from last line
    parts[-1] = parts[-1].rstrip(",\n") + "\n"
    parts.append("    },\n")
    parts.append(f"    ... (continue for all {num_weaknesses} weaknesses)\n")
    parts.append("  ]\n}\n```\n\n")
    parts.append(
        "This JSON will be automatically parsed, so ensure the format is precise "
        f"and includes exactly {num_weaknesses} weakness evaluations in the array."
    )

    return "".join(parts)


def build_batched_user_prompt(
    weakness_texts: List[str],
    paper_content: str,
) -> str:
    """Build user prompt for batched evaluation.

    Args:
        weakness_texts: List of weakness text strings.
        paper_content: Full paper text for reference verification.

    Returns:
        Formatted user prompt string.
    """
    weaknesses_list = "\n\n".join(
        f"### Review Point {i+1} ###\n{w}"
        for i, w in enumerate(weakness_texts)
    )
    return (
        f"The review points are about a paper with the following text:\n"
        f"{paper_content}\n\n"
        f"Review Points ({len(weakness_texts)} total):\n"
        f"{weaknesses_list}"
    )
