"""Consolidated entry point for scoring reviews.

Provides a single `score_review()` function usable from both RL training
(via ReviewEnv) and standalone SFT/baseline evaluation.
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from reviewer.reward.calculator import (
    RewardCalculator,
)
from reviewer.reward.components import (
    compute_format_completeness,
    compute_score_difference_reward,
)

logger = logging.getLogger(__name__)

ALL_MODES = {"format", "utility", "score_diff", "recall", "rubric"}  # hallucination folded into per-step syntactic

# Rubric dimensions evaluated per-weakness (Factual Correctness and Overall removed)
RUBRIC_DIMENSIONS = ["Grounding", "Constructive Value", "Analytical Depth", "Verifiability"]

# Coverage grade → numeric score for recall computation
COVERAGE_SCORES = {"full": 1.0, "partial": 0.3, "not_covered": 0.0, "covered": 1.0}


def _extract_rating_from_section(rating_text: str) -> Optional[float]:
    """Extract numeric rating from section text like '5: marginally below acceptance threshold'."""
    if not rating_text:
        return None
    m = re.match(r'(\d+)', rating_text.strip())
    return float(m.group(1)) if m else None


_ORDINAL_SPLIT = re.compile(
    r'(?=\b(?:'
    r'First(?:ly)?,\s|Second(?:ly)?,\s|Third(?:ly)?,\s|Fourth(?:ly)?,\s|'
    r'Fifth(?:ly)?,\s|Sixth(?:ly)?,\s|Seventh(?:ly)?,\s|Eighth(?:ly)?,\s|'
    r'Ninth(?:ly)?,\s|Tenth(?:ly)?,\s|Finally,\s'
    r'))',
    re.IGNORECASE,
)


def _split_weakness_text(text: str) -> List[str]:
    """Split a weakness string into individual points.

    Handles two common formats:
    1. Newline-separated points (e.g. from parsed reviews)
    2. Prose with ordinal markers -- "First, ... Second, ... Finally, ..."
       (e.g. DeepReview output)
    """
    if not text or not text.strip():
        return []

    # Try newline split first
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    # Try ordinal split
    parts = _ORDINAL_SPLIT.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        # Drop short preamble before "First,"
        if len(parts[0]) < 200 and not re.match(r'\b(?:First|Second)', parts[0], re.IGNORECASE):
            parts = parts[1:]
        return parts

    # No split possible — return as single item
    return [text]


MAX_PAPER_TOKENS = 64_000  # Truncate paper content to fit context window


def _truncate_paper(text: str, max_tokens: int = MAX_PAPER_TOKENS) -> str:
    """Truncate paper content to max_tokens using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        # Ensure text is a string
        if not isinstance(text, str):
            logger.error(f"_truncate_paper received non-string type: {type(text)}, value: {text}")
            text = str(text) if text is not None else ""
        tokens = enc.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        truncated = enc.decode(tokens[:max_tokens])
        logger.info(f"Truncated paper from {len(tokens):,} to {max_tokens:,} tokens")
        return truncated + "\n\n[Paper truncated due to length]"
    except ImportError:
        # Fallback: rough char-based truncation (assume ~3.5 chars/token)
        max_chars = int(max_tokens * 3.5)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[Paper truncated due to length]"


async def _compute_rubric_per_weakness(
    weakness_texts: List[str],
    paper_content: str,
    rubric_model: str
) -> Tuple[List[Dict], float, float, float, float, Dict]:
    """Evaluate each weakness individually on Grounding, Constructive Value, Analytical Depth, and Verifiability.

    Returns:
        Tuple of (per_weakness_details, avg_grounding, avg_constructive_value, avg_analytical_depth, avg_verifiability, token_usage)
    """
    import importlib.util, os
    import asyncio
    prompt_path = os.path.join(os.path.dirname(__file__), "../../scripts/evaluation/evaluation_prompt.py")
    spec = importlib.util.spec_from_file_location("evaluation_prompt", os.path.abspath(prompt_path))
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    per_weakness_system_prompt = _mod.per_weakness_system_prompt
    per_weakness_user_prompt = _mod.per_weakness_user_prompt

    from utils.helpers.llm import acall_llm, get_content

    paper_content = _truncate_paper(paper_content)

    rubric_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0, "calls": 0}

    async def evaluate_one_weakness(idx: int, weakness: str):
        """Evaluate a single weakness."""
        messages = [
            {"role": "system", "content": per_weakness_system_prompt},
            {"role": "user", "content": per_weakness_user_prompt.format(
                paper_text=paper_content,
                weakness=weakness
            )},
        ]

        try:
            resp = await acall_llm(rubric_model, messages, temperature=0.0, max_tokens=2048)

            # Track token usage including cached tokens
            if hasattr(resp, 'usage') and resp.usage:
                rubric_token_usage["prompt_tokens"] += getattr(resp.usage, 'prompt_tokens', 0) or 0
                rubric_token_usage["completion_tokens"] += getattr(resp.usage, 'completion_tokens', 0) or 0
                rubric_token_usage["total_tokens"] += getattr(resp.usage, 'total_tokens', 0) or 0
                rubric_token_usage["calls"] += 1
                # Extract cached tokens: DeepSeek uses prompt_cache_hit_tokens at top level,
                # OpenAI uses prompt_tokens_details.cached_tokens
                ds_cached = getattr(resp.usage, 'prompt_cache_hit_tokens', 0) or 0
                if ds_cached:
                    rubric_token_usage["cached_tokens"] += ds_cached
                else:
                    ptd = getattr(resp.usage, 'prompt_tokens_details', None)
                    if ptd:
                        rubric_token_usage["cached_tokens"] += getattr(ptd, 'cached_tokens', 0) or 0

            content = get_content(resp)

            # Parse JSON from response
            import re as _re
            json_match = _re.search(r"```json\s*(.*?)\s*```", content, _re.DOTALL)
            if json_match:
                raw_json = json_match.group(1)
            else:
                raw_json = content

            parsed = json.loads(raw_json)

            # Parse verifiability: handle "X" (no claim) by mapping to 0.0
            raw_verifiability = parsed.get('Verifiability Score', 0)
            verifiability_score = 0.0 if str(raw_verifiability).strip().upper() == 'X' else float(raw_verifiability)

            return {
                'weakness_idx': idx,
                'weakness_text': weakness[:100] + '...' if len(weakness) > 100 else weakness,
                'grounding_score': float(parsed.get('Grounding Score', 0)),
                'grounding_reason': parsed.get('Grounding Reason', ''),
                'constructive_value_score': float(parsed.get('Constructive Value Score', 0)),
                'constructive_value_reason': parsed.get('Constructive Value Reason', ''),
                'analytical_depth_score': float(parsed.get('Analytical Depth Score', 0)),
                'analytical_depth_reason': parsed.get('Analytical Depth Reason', ''),
                'verifiability_score': verifiability_score,
                'verifiability_reason': parsed.get('Verifiability Reason', ''),
            }
        except Exception as e:
            logger.error(f"Failed to evaluate weakness {idx}: {e}")
            return {
                'weakness_idx': idx,
                'weakness_text': weakness[:100] + '...' if len(weakness) > 100 else weakness,
                'grounding_score': 0.0,
                'grounding_reason': f'Error: {str(e)}',
                'constructive_value_score': 0.0,
                'constructive_value_reason': f'Error: {str(e)}',
                'analytical_depth_score': 0.0,
                'analytical_depth_reason': f'Error: {str(e)}',
                'verifiability_score': 0.0,
                'verifiability_reason': f'Error: {str(e)}',
                'error': str(e),
            }

    # Evaluate weaknesses sequentially to maximize prefix cache hits
    results = []
    for idx, w in enumerate(weakness_texts):
        results.append(await evaluate_one_weakness(idx, w))

    # Compute averages
    grounding_scores = [r['grounding_score'] for r in results if r['grounding_score'] > 0]
    constructive_scores = [r['constructive_value_score'] for r in results if r['constructive_value_score'] > 0]
    analytical_scores = [r['analytical_depth_score'] for r in results if r['analytical_depth_score'] > 0]
    verifiability_scores = [r['verifiability_score'] for r in results if r['verifiability_score'] > 0]

    avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0
    avg_constructive = sum(constructive_scores) / len(constructive_scores) if constructive_scores else 0.0
    avg_analytical = sum(analytical_scores) / len(analytical_scores) if analytical_scores else 0.0
    avg_verifiability = sum(verifiability_scores) / len(verifiability_scores) if verifiability_scores else 0.0

    return results, avg_grounding, avg_constructive, avg_analytical, avg_verifiability, rubric_token_usage


async def _compute_rubric_batched_weaknesses(
    weakness_texts: List[str],
    paper_content: str,
    rubric_model: str
) -> Tuple[List[Dict], float, float, float, float, Dict]:
    """Evaluate all weaknesses in a single LLM call (batched).

    This reduces API costs by ~N-fold where N is the number of weaknesses,
    since we only make one call instead of N calls.

    Returns:
        Tuple of (per_weakness_details, avg_grounding, avg_constructive_value, avg_analytical_depth, avg_verifiability, token_usage)
    """
    import importlib.util, os
    prompt_path = os.path.join(os.path.dirname(__file__), "../../scripts/evaluation/evaluation_prompt.py")
    spec = importlib.util.spec_from_file_location("evaluation_prompt", os.path.abspath(prompt_path))
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    batched_weaknesses_system_prompt = _mod.batched_weaknesses_system_prompt
    batched_weaknesses_user_prompt = _mod.batched_weaknesses_user_prompt

    from utils.helpers.llm import acall_llm, get_content

    # Ensure paper_content is a string
    if not isinstance(paper_content, str):
        logger.error(f"paper_content is not a string: type={type(paper_content)}, value={paper_content}")
        paper_content = str(paper_content) if paper_content is not None else ""

    # Ensure all weakness texts are strings
    weakness_texts = [str(w) if not isinstance(w, str) else w for w in weakness_texts]

    paper_content = _truncate_paper(paper_content)

    # Format weaknesses list for the prompt
    weaknesses_list = "\n\n".join([
        f"### Weakness {i+1} ###\n{w}"
        for i, w in enumerate(weakness_texts)
    ])

    messages = [
        {"role": "system", "content": batched_weaknesses_system_prompt},
        {"role": "user", "content": batched_weaknesses_user_prompt.format(
            paper_text=paper_content,
            weaknesses_list=weaknesses_list,
            num_weaknesses=len(weakness_texts)
        )},
    ]

    rubric_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0, "calls": 0}

    try:
        resp = await acall_llm(rubric_model, messages, temperature=0.0, max_tokens=4096)

        # Track token usage
        if hasattr(resp, 'usage') and resp.usage:
            rubric_token_usage["prompt_tokens"] += getattr(resp.usage, 'prompt_tokens', 0) or 0
            rubric_token_usage["completion_tokens"] += getattr(resp.usage, 'completion_tokens', 0) or 0
            rubric_token_usage["total_tokens"] += getattr(resp.usage, 'total_tokens', 0) or 0
            rubric_token_usage["calls"] += 1

            # Extract cached tokens
            ds_cached = getattr(resp.usage, 'prompt_cache_hit_tokens', 0) or 0
            if ds_cached:
                rubric_token_usage["cached_tokens"] += ds_cached
            else:
                ptd = getattr(resp.usage, 'prompt_tokens_details', None)
                if ptd:
                    rubric_token_usage["cached_tokens"] += getattr(ptd, 'cached_tokens', 0) or 0

        content = get_content(resp)

        # Parse JSON from response
        import re as _re
        json_match = _re.search(r"```json\s*(.*?)\s*```", content, _re.DOTALL)
        if json_match:
            raw_json = json_match.group(1)
        else:
            raw_json = content

        parsed = json.loads(raw_json)

        # Extract results for each weakness
        results = []
        weaknesses_data = parsed.get("weaknesses", [])

        for idx, weakness_result in enumerate(weaknesses_data):
            if idx >= len(weakness_texts):
                break

            # Parse verifiability: handle "X" (no claim) by mapping to 0.0
            raw_verifiability = weakness_result.get('Verifiability Score', 0)
            verifiability_score = 0.0 if str(raw_verifiability).strip().upper() == 'X' else float(raw_verifiability)

            results.append({
                'weakness_idx': idx,
                'weakness_text': weakness_texts[idx][:100] + '...' if len(weakness_texts[idx]) > 100 else weakness_texts[idx],
                'grounding_score': float(weakness_result.get('Grounding Score', 0)),
                'grounding_reason': weakness_result.get('Grounding Reason', ''),
                'constructive_value_score': float(weakness_result.get('Constructive Value Score', 0)),
                'constructive_value_reason': weakness_result.get('Constructive Value Reason', ''),
                'analytical_depth_score': float(weakness_result.get('Analytical Depth Score', 0)),
                'analytical_depth_reason': weakness_result.get('Analytical Depth Reason', ''),
                'verifiability_score': verifiability_score,
                'verifiability_reason': weakness_result.get('Verifiability Reason', ''),
            })

        # Handle case where response has fewer results than weaknesses
        while len(results) < len(weakness_texts):
            idx = len(results)
            results.append({
                'weakness_idx': idx,
                'weakness_text': weakness_texts[idx][:100] + '...' if len(weakness_texts[idx]) > 100 else weakness_texts[idx],
                'grounding_score': 0.0,
                'grounding_reason': 'Missing from batched response',
                'constructive_value_score': 0.0,
                'constructive_value_reason': 'Missing from batched response',
                'analytical_depth_score': 0.0,
                'analytical_depth_reason': 'Missing from batched response',
                'verifiability_score': 0.0,
                'verifiability_reason': 'Missing from batched response',
                'error': 'Missing from batched response',
            })

    except Exception as e:
        logger.error(f"Failed to evaluate weaknesses in batch: {e}")
        # Fallback: return zero scores for all weaknesses
        results = []
        for idx, w in enumerate(weakness_texts):
            results.append({
                'weakness_idx': idx,
                'weakness_text': w[:100] + '...' if len(w) > 100 else w,
                'grounding_score': 0.0,
                'grounding_reason': f'Error: {str(e)}',
                'constructive_value_score': 0.0,
                'constructive_value_reason': f'Error: {str(e)}',
                'analytical_depth_score': 0.0,
                'analytical_depth_reason': f'Error: {str(e)}',
                'verifiability_score': 0.0,
                'verifiability_reason': f'Error: {str(e)}',
                'error': str(e),
            })

    # Compute averages
    grounding_scores = [r['grounding_score'] for r in results if r['grounding_score'] > 0]
    constructive_scores = [r['constructive_value_score'] for r in results if r['constructive_value_score'] > 0]
    analytical_scores = [r['analytical_depth_score'] for r in results if r['analytical_depth_score'] > 0]
    verifiability_scores = [r['verifiability_score'] for r in results if r['verifiability_score'] > 0]

    avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0
    avg_constructive = sum(constructive_scores) / len(constructive_scores) if constructive_scores else 0.0
    avg_analytical = sum(analytical_scores) / len(analytical_scores) if analytical_scores else 0.0
    avg_verifiability = sum(verifiability_scores) / len(verifiability_scores) if verifiability_scores else 0.0

    return results, avg_grounding, avg_constructive, avg_analytical, avg_verifiability, rubric_token_usage


async def _compute_rubric(review: Union[str, Dict], paper_content: str, rubric_model: str) -> Tuple[Dict[str, float], Dict]:
    """Compute rubric-based review quality scores via LLM judge.

    Returns:
        Tuple of (scores_dict, raw_response_dict) where scores_dict maps
        dimension names to 1-5 scores.
    """
    import importlib.util, os
    prompt_path = os.path.join(os.path.dirname(__file__), "../../scripts/evaluation/evaluation_prompt.py")
    spec = importlib.util.spec_from_file_location("evaluation_prompt", os.path.abspath(prompt_path))
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    evaluation_system_prompt = _mod.evaluation_system_prompt
    user_prompt = _mod.user_prompt

    from utils.helpers.llm import acall_llm, get_content

    paper_content = _truncate_paper(paper_content)

    # Format review as text if it's a dict
    if isinstance(review, dict):
        parts = []
        if review.get("summary"):
            parts.append(f"## Summary\n{review['summary']}")
        if review.get("strengths"):
            items = review["strengths"]
            if isinstance(items, list):
                items = "\n".join(f"- {s}" for s in items)
            parts.append(f"## Strengths\n{items}")
        if review.get("weaknesses"):
            items = review["weaknesses"]
            if isinstance(items, list):
                items = "\n".join(f"- {w}" for w in items)
            parts.append(f"## Weaknesses\n{items}")
        if review.get("questions"):
            items = review["questions"]
            if isinstance(items, list):
                items = "\n".join(f"- {q}" for q in items)
            parts.append(f"## Questions\n{items}")
        if review.get("overall_score") is not None:
            parts.append(f"## Overall Score\n{review['overall_score']}")
        review_text = "\n\n".join(parts)
    else:
        review_text = review

    messages = [
        {"role": "system", "content": evaluation_system_prompt},
        {"role": "user", "content": user_prompt.format(paper_text=paper_content, review=review_text)},
    ]

    resp = await acall_llm(rubric_model, messages, temperature=0.0, max_tokens=4096)
    content = get_content(resp)

    # Parse JSON from response
    import re as _re
    json_match = _re.search(r"```json\s*(.*?)\s*```", content, _re.DOTALL)
    if json_match:
        raw_json = json_match.group(1)
    else:
        # Try parsing entire content as JSON
        raw_json = content

    parsed = json.loads(raw_json)

    scores = {}
    for dim in RUBRIC_DIMENSIONS:
        score_val = parsed.get(f"{dim} Score")
        if score_val is not None:
            scores[dim] = float(score_val)

    return scores, parsed


def _compute_utility(review: Dict, calculator: RewardCalculator) -> Tuple[float, List[Dict]]:
    """Compute utility reward from weakness points with diversity penalty.

    effective_utility_i = utility_i * diversity_i
    where diversity_i = 1 - max(cosine_sim(w_i, w_j) for j != i)

    Duplicate detection is handled by the diversity penalty itself:
    near-identical weaknesses get diversity ≈ 0, zeroing their utility.

    Returns:
        Tuple of (avg_effective_utility, per-point detail list)
    """
    raw_weakness_texts = review.get("weaknesses", [])
    # Handle string weaknesses (e.g. from baselines that produce plain text)
    if isinstance(raw_weakness_texts, str):
        raw_weakness_texts = _split_weakness_text(raw_weakness_texts)
    # Filter empty entries
    raw_weakness_texts = [t for t in raw_weakness_texts if isinstance(t, str) and t.strip()]
    
    if not raw_weakness_texts:
        logger.info("No weaknesses provided in review; utility reward will be 0.")
        return 0.0, []

    return calculator.compute_utility_sync(raw_weakness_texts)


async def async_score_review(
    review: Union[str, Dict],
    human_avg_score: Optional[float] = None,
    clustered_points: Optional[List[Dict]] = None,
    reward_modes: Set[str] = "full",
    judge_model: str = "utility-score",
    recall_model: Optional[str] = None,
    reward_calculator: Optional[RewardCalculator] = None,
    paper_content: Optional[str] = None,
    rubric_model: Optional[str] = None,
    batch_rubric_weaknesses: bool = False,
) -> Dict:
    """Async version of score_review. Avoids sync-async-sync bouncing and event loop conflicts."""
    if isinstance(review, str):
        from utils.sft.review_parser import parse_complete_review
        parsed = parse_complete_review(review)
        overall = parsed.get("scores", {}).get("overall")
        if overall is None:
            overall = _extract_rating_from_section(parsed["sections"].get("rating", ""))
        review = {
            "summary": parsed["sections"].get("summary", ""),
            "strengths": [c["text"] for c in parsed.get("stength", [])],
            "weaknesses": [i["text"] for i in parsed.get("issues", [])],
            "questions": parsed.get("questions", []),
            "overall_score": overall,
        }

    NEEDS_CALCULATOR = {"utility", "recall"}
    if reward_modes & NEEDS_CALCULATOR:
        if reward_calculator is None:
            reward_calculator = RewardCalculator(judge_model=judge_model, recall_model=recall_model)

    result: Dict = {}

    if "format" in reward_modes:
        result["format"] = compute_format_completeness(review)

    if "utility" in reward_modes:
        raw_weakness_texts = review.get("weaknesses", [])
        if raw_weakness_texts is None:
            raw_weakness_texts = []
        if isinstance(raw_weakness_texts, str):
            raw_weakness_texts = _split_weakness_text(raw_weakness_texts)
        raw_weakness_texts = [t for t in raw_weakness_texts if isinstance(t, str) and t.strip()]

        if raw_weakness_texts:
            utility_score, utility_details = await reward_calculator.compute_utility_async(raw_weakness_texts)
        else:
            utility_score, utility_details = 0.0, []
        result["utility"] = utility_score
        result["utility_details"] = utility_details

    if "score_diff" in reward_modes:
        if human_avg_score is not None:
            score_diff, score_details = compute_score_difference_reward(
                review.get("overall_score"), human_avg_score
            )
        else:
            score_diff, score_details = 0.0, {"warning": "No human_avg_score"}
        result["score_diff"] = score_diff
        result["score_diff_details"] = score_details

    if "recall" in reward_modes:
        cp = clustered_points or []
        cp = [p for p in cp if isinstance(p, dict) and p.get("reviewer_count", 1) >= 2]
        if cp:
            try:
                recall_results = await reward_calculator._compute_recall_per_point_async(review, cp)
            except Exception as e:
                logger.error(f"Recall reward computation failed: {e}")
                raise
            coverage_sum = sum(
                COVERAGE_SCORES.get(p.get("coverage", "not_covered"), 0.0)
                for p in recall_results
            )
            recall_scalar = coverage_sum / len(recall_results) if recall_results else 0.0
        else:
            logger.warning("No valid clustered_points for recall computation, defaulting to 0.0")
            recall_results = []
            recall_scalar = 0.0
        result["recall"] = recall_scalar
        result["recall_results"] = recall_results

    if "rubric" in reward_modes:
        if not paper_content:
            logger.warning("rubric mode requires paper_content; skipping")
        else:
            model = rubric_model or judge_model
            try:
                # Get weaknesses for per-weakness evaluation
                raw_weakness_texts = review.get("weaknesses", [])
                if isinstance(raw_weakness_texts, str):
                    raw_weakness_texts = _split_weakness_text(raw_weakness_texts)
                raw_weakness_texts = [t for t in raw_weakness_texts if isinstance(t, str) and t.strip()]

                # Evaluate each weakness for Grounding, Constructive Value, and Analytical Depth
                if raw_weakness_texts:
                    if batch_rubric_weaknesses:
                        # Batch mode: evaluate all weaknesses in one LLM call
                        per_weakness_details, avg_grounding, avg_constructive, avg_analytical, avg_verifiability, rubric_usage = await _compute_rubric_batched_weaknesses(
                            raw_weakness_texts, paper_content, model
                        )
                    else:
                        # Per-weakness mode: evaluate each weakness separately
                        per_weakness_details, avg_grounding, avg_constructive, avg_analytical, avg_verifiability, rubric_usage = await _compute_rubric_per_weakness(
                            raw_weakness_texts, paper_content, model
                        )
                    rubric_scores = {
                        "Grounding": avg_grounding,
                        "Constructive Value": avg_constructive,
                        "Analytical Depth": avg_analytical,
                        "Verifiability": avg_verifiability,
                    }
                    result["rubric_per_weakness"] = per_weakness_details
                else:
                    logger.warning("No weaknesses found for per-weakness rubric evaluation")
                    rubric_scores = {
                        "Grounding": 0.0,
                        "Constructive Value": 0.0,
                        "Analytical Depth": 0.0,
                        "Verifiability": 0.0,
                    }
                    result["rubric_per_weakness"] = []

                # Use average of the three dimensions as overall rubric score
                result["rubric"] = sum(rubric_scores.values()) / len(rubric_scores) if rubric_scores else 0.0
                result["rubric_scores"] = rubric_scores
            except Exception as e:
                logger.error(f"Rubric scoring failed: {e}")
                result["rubric"] = 0.0
                result["rubric_scores"] = {}
                result["rubric_per_weakness"] = []

    judge_token_usage = {}
    if reward_calculator is not None and reward_calculator.token_usage:
        judge_token_usage.update(reward_calculator.token_usage)
    if "rubric" in reward_modes:
        try:
            if rubric_usage and rubric_usage.get("calls", 0) > 0:
                judge_token_usage[rubric_model or judge_model] = rubric_usage
        except NameError:
            pass  # rubric_usage not defined if rubric had no weaknesses or failed
    if judge_token_usage:
        result["judge_token_usage"] = dict(judge_token_usage)

    return result


def score_review(
    review: Union[str, Dict],
    human_avg_score: Optional[float] = None,
    clustered_points: Optional[List[Dict]] = None,
    reward_modes: Set[str] = "full",
    judge_model: str = "utility-score",
    recall_model: Optional[str] = None,
    reward_calculator: Optional[RewardCalculator] = None,
) -> Dict:
    """Score a review across multiple reward dimensions.

    Syntactic reward is NOT included — it's a trajectory-level (step-by-step)
    metric computed in ReviewEnv.step(), not a property of the finished review.

    Args:
        review: Review dict (with summary, strengths, weaknesses, overall_score keys)
            or raw review text string (will be parsed via parse_complete_review).
        human_avg_score: Average human score for score_diff computation.
        clustered_points: List of clustered human review points for recall.
        reward_modes: Which components to compute. "full" = all 4
            (format, utility, score_diff, recall).
            Can be a set like {"utility", "format"}.
        judge_model: Model name for utility LLM judge calls.
        recall_model: Model name for recall LLM judge (defaults to judge_model).
        reward_calculator: Existing RewardCalculator instance (avoids re-creation).

    Returns:
        Dict with keys per active mode (e.g. "format", "utility", "score_diff",
        "recall") plus "*_details" keys where applicable.
    """
    # Parse string reviews
    if isinstance(review, str):
        from utils.sft.review_parser import parse_complete_review
        parsed = parse_complete_review(review)
        overall = parsed.get("scores", {}).get("overall")
        if overall is None:
            overall = _extract_rating_from_section(parsed["sections"].get("rating", ""))
        # Flatten parsed structure to match the dict format used in RL
        review = {
            "summary": parsed["sections"].get("summary", ""),
            "strengths": [c["text"] for c in parsed.get("stength", [])],
            "weaknesses": [i["text"] for i in parsed.get("issues", [])],
            "questions": parsed.get("questions", []),
            "overall_score": overall,
        }


    # Create calculator if needed
    NEEDS_CALCULATOR = {"utility", "recall"}
    if reward_modes & NEEDS_CALCULATOR:
        if reward_calculator is None:
            reward_calculator = RewardCalculator(judge_model=judge_model, recall_model=recall_model)

    result: Dict = {}

    # Format
    if "format" in reward_modes:
        result["format"] = compute_format_completeness(review)

    # Utility
    if "utility" in reward_modes:
        utility_score, utility_details = _compute_utility(review, reward_calculator)
        result["utility"] = utility_score
        result["utility_details"] = utility_details

    # Score difference
    if "score_diff" in reward_modes:
        if human_avg_score is not None:
            score_diff, score_details = compute_score_difference_reward(
                review.get("overall_score"), human_avg_score
            )
        else:
            score_diff, score_details = 0.0, {"warning": "No human_avg_score"}
        result["score_diff"] = score_diff
        result["score_diff_details"] = score_details

    # Recall
    if "recall" in reward_modes:
        cp = clustered_points or []
        # Keep only points mentioned by 2+ reviewers
        cp = [p for p in cp if p.get("reviewer_count", 1) >= 2]
        if cp:
            try:
                recall_results = reward_calculator.compute_recall_sync(review, cp)
            except Exception as e:
                logger.error(f"Recall reward computation failed: {e}")
                raise
            coverage_sum = sum(
                COVERAGE_SCORES.get(p.get("coverage", "not_covered"), 0.0)
                for p in recall_results
            )
            recall_scalar = coverage_sum / len(recall_results) if recall_results else 0.0
        else:
            raise ValueError("clustered_points is required for recall reward computation")
        result["recall"] = recall_scalar
        result["recall_results"] = recall_results

    if reward_calculator is not None and reward_calculator.token_usage:
        result["judge_token_usage"] = dict(reward_calculator.token_usage)

    return result
