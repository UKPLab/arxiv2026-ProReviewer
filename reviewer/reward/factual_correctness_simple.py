"""Simple factual correctness: paper + outline items → per-item hallucination scores.

Same interface as technical_depth and outline_grounding — returns per-item scores
that get broadcast back to contributing steps via evidence references.

Input: paper_content, weaknesses, questions
Output: {item_id: score_0_1} for each weakness/question

Usage:
    from reviewer.reward.factual_correctness_simple import (
        compute_factual_correctness_simple_async,
    )

    scores_dict, reasoning = await compute_factual_correctness_simple_async(
        paper_content="...",
        weaknesses=[...],
        questions=[...],
        llm_judge_fn=judge_fn,
    )
    # scores_dict = {"W1": 0.75, "W2": 0.0, "Q1": 1.0, ...}
"""

import asyncio
import json
import logging
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

FACTUAL_SIMPLE_SYSTEM_PROMPT = """\
You are an evaluator of factual correctness in paper reviews. You will receive \
a paper and review outline items (weaknesses and questions). You must score \
each item on factual correctness — whether it accurately represents the paper \
content or contains hallucinations."""

FACTUAL_SIMPLE_QUERY = (
    "[QUERY]: Score each weakness and question for factual correctness. "
    "Factual correctness means the review statements accurately represent "
    "the paper content without fabricating, misquoting, or contradicting "
    "what the paper actually says.\n\n"
)

FACTUAL_SIMPLE_CRITERIA = (
    "[CRITERIA]: Factual correctness checks whether review statements are "
    "grounded in the actual paper content. The paper text is the ground truth.\n\n"
    "What IS hallucination (penalize heavily):\n"
    "- Citing tables, figures, equations, or sections that don't exist\n"
    "- Claiming specific numbers or results not present in the paper\n"
    "- Describing methods or techniques the paper doesn't use\n"
    "- Attributing claims to the paper that it never makes\n\n"
    "What is NOT hallucination (do NOT penalize):\n"
    "- Identifying limitations or missing elements (e.g. 'lacks comparison to X')\n"
    "- Critiquing methodology choices — even if subjective\n"
    "- Asking questions about unclear parts of the paper\n"
    "- Making reasonable inferences clearly framed as interpretation\n\n"
    "Scoring rubric (1-5):\n"
    "5: Fully grounded — every factual claim traceable to specific paper content\n"
    "4: Grounded — all claims match paper content; at most minor interpretation variance\n"
    "3: Mostly grounded — 1 statement overstates or references something not clearly in paper\n"
    "2: Partially hallucinated — references specific content (table, figure, result) not in paper\n"
    "1: Fabricated — multiple claims about paper content that doesn't exist\n\n"
    "Output format (item_scores list first, then reasoning):\n"
    "<item_scores>\n"
    '[{"item_id": "W1", "score": 5, "reason": "Correctly cites Table 2"},\n'
    ' {"item_id": "W2", "score": 1, "reason": "Fabricates architectural details"},\n'
    ' {"item_id": "Q1", "score": 4, "reason": "Valid question grounded in methodology"}]\n'
    "</item_scores>\n"
    "<reasoning>Brief overall assessment</reasoning>\n\n"
    'item_id: "W1", "W2", ... for weaknesses; "Q1", "Q2", ... for questions\n\n'
)

FACTUAL_SIMPLE_EXAMPLES = (
    "<START OF EXAMPLE>\n\n"
    "PAPER EXCERPT:\n"
    "We propose a graph neural network (GNN) that operates on molecular structures. "
    "Our architecture uses message-passing layers. Table 1 reports results on "
    "MoleculeNet with AUC of 0.82. We compare against 3 baselines.\n\n"
    "OUTLINE ITEMS:\n"
    "### Weaknesses\n"
    "W1. The model uses a 6-layer GNN with residual connections and layer "
    "normalization but the paper does not justify this architecture choice.\n"
    "W2. Table 1 shows AUC of 0.82 on MoleculeNet which is only marginally "
    "above the baselines, suggesting limited improvement.\n"
    "W3. The paper only compares against 3 baselines which limits the "
    "generalizability assessment.\n"
    "### Questions\n"
    "Q1. Could the authors provide results on additional molecular benchmarks "
    "beyond MoleculeNet?\n\n"
    "EVALUATION:\n\n"
    "<item_scores>\n"
    '[{"item_id": "W1", "score": 1, "reason": "Fabricates architectural details '
    '(6-layer, residual, normalization) not in paper — paper only says message-passing"},\n'
    ' {"item_id": "W2", "score": 5, "reason": "Correctly references Table 1 and AUC 0.82, '
    'grounded observation about baselines"},\n'
    ' {"item_id": "W3", "score": 5, "reason": "Correctly notes 3 baselines, valid limitation"},\n'
    ' {"item_id": "Q1", "score": 5, "reason": "Valid question for additional experiments, '
    'not hallucination"}]\n'
    "</item_scores>\n"
    "<reasoning>W1 fabricates specific details not present. W2, W3, Q1 are all "
    "grounded in actual paper content or valid critiques/questions.</reasoning>\n\n"
    "<END OF EXAMPLE>\n\n"
)

FACTUAL_SIMPLE_USER_PROMPT = """\
{query}{criteria}{examples}[ANSWER]:

## Paper Content

{paper_content}

## Review Outline Items

### Weaknesses
{outline_weaknesses}

### Questions
{outline_questions}
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_item_scores_response(response: str) -> Tuple[Optional[Dict[str, int]], str]:
    """Parse item scores from <item_scores>...</item_scores> tags.

    Returns:
        (scores_dict, reasoning) where scores_dict maps item_id -> score (1-5)
        scores_dict is None on parse failure
    """
    content = response.strip()

    # Extract reasoning
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        incomplete_match = re.search(r"<reasoning>\s*(.*?)(?:<|$)", content, re.DOTALL | re.IGNORECASE)
        reasoning = incomplete_match.group(1).strip() if incomplete_match else ""

    # Extract item_scores array
    scores_match = re.search(r"<item_scores>(.*?)</item_scores>", content, re.DOTALL | re.IGNORECASE)
    if scores_match:
        scores_content = scores_match.group(1).strip()
    else:
        incomplete_match = re.search(r"<item_scores>\s*(.*?)(?:<reasoning>|$)", content, re.DOTALL | re.IGNORECASE)
        if incomplete_match:
            scores_content = incomplete_match.group(1).strip()
        else:
            warnings.warn(f"Failed to find <item_scores> tag in response: {content[:200]}")
            return None, reasoning or "parse_error"

    # Try to parse JSON
    for attempt in [scores_content, re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', scores_content)]:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                scores_dict = {}
                for entry in parsed:
                    if isinstance(entry, dict) and "item_id" in entry and "score" in entry:
                        item_id = str(entry["item_id"])
                        score = int(entry["score"])
                        score = max(1, min(5, score))  # Clamp to [1, 5]
                        scores_dict[item_id] = score
                return scores_dict, reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    warnings.warn(f"Failed to parse item_scores JSON: {scores_content[:200]}")
    return None, reasoning or "parse_error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_token_count(text: str) -> int:
    """Estimate token count using tiktoken when available, else char heuristic."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text, disallowed_special=())
        return len(tokens)
    except ImportError:
        # Conservative fallback: ~3.5 chars/token for English-like text.
        return int(len(text) / 3.5)


def _format_outline_section(items: list, prefix: str) -> str:
    """Format outline items with item_id prefix (W1, W2, ... or Q1, Q2, ...)."""
    if not items:
        return "(none)"

    lines = []
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            text = item.get("text", str(item))
            refs = []
            refs += item.get("related_claims", [])
            refs += item.get("related_questions", [])
            refs += item.get("related_notes", [])
            ref_str = f" [{', '.join(refs)}]" if refs else ""
            lines.append(f"{prefix}{i}. {text}{ref_str}")
        else:
            lines.append(f"{prefix}{i}. {str(item)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

async def compute_factual_correctness_simple_async(
    paper_content: str,
    weaknesses: List,
    questions: List,
    llm_judge_fn,
) -> Tuple[Optional[Dict[str, float]], str]:
    """Compute per-item factual correctness scores.

    Args:
        paper_content: Full paper text (ground truth).
        weaknesses: List of weakness items (strings or dicts with "text" key).
        questions: List of question items (strings or dicts with "text" key).
        llm_judge_fn: Async callable(system_prompt, user_prompt) -> str.

    Returns:
        (scores_dict, reasoning) where:
        - scores_dict: {item_id: score_1_5} raw scores (1-5) for each W/Q
        - reasoning: judge's reasoning text
        Returns (None, error_msg) on failure.
    """
    if not weaknesses and not questions:
        return None, "No items to evaluate"

    max_paper_tokens = 64_000
    paper_token_count = _estimate_token_count(paper_content)
    if paper_token_count > max_paper_tokens:
        msg = f"paper_too_long: {paper_token_count} tokens > {max_paper_tokens}"
        logger.error(f"[FactualSimple] {msg}")
        return None, msg

    weaknesses_str = _format_outline_section(weaknesses, "W")
    questions_str = _format_outline_section(questions, "Q")

    user_prompt = FACTUAL_SIMPLE_USER_PROMPT.format(
        query=FACTUAL_SIMPLE_QUERY,
        criteria=FACTUAL_SIMPLE_CRITERIA,
        examples=FACTUAL_SIMPLE_EXAMPLES,
        paper_content=paper_content,
        outline_weaknesses=weaknesses_str,
        outline_questions=questions_str,
    )

    try:
        response = await llm_judge_fn(FACTUAL_SIMPLE_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        logger.error(f"[FactualSimple] LLM call failed: {e}")
        return None, f"llm_error: {e}"

    if response is None:
        logger.error("[FactualSimple] LLM returned None")
        return None, "llm_returned_none"

    scores_raw, reasoning = _parse_item_scores_response(response)

    if scores_raw is None:
        logger.error("[FactualSimple] Failed to parse item scores")
        return None, reasoning

    # Return raw 1-5 scores (normalization happens in evidence computation)
    return scores_raw, reasoning


def compute_factual_correctness_simple(
    paper_content: str,
    weaknesses: List,
    questions: List,
    llm_judge_fn,
) -> Tuple[Optional[Dict[str, float]], str]:
    """Sync wrapper for compute_factual_correctness_simple_async."""
    coro = compute_factual_correctness_simple_async(
        paper_content, weaknesses, questions, llm_judge_fn
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
