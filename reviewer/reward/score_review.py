"""Consolidated entry point for scoring reviews.

Provides `score_review()` and `async_score_review()` usable from both RL
training (via ReviewEnv) and standalone SFT/baseline evaluation.

Reward modes: format, score_diff, rubric.
"""

import asyncio
import json
import logging
import re
import warnings
from typing import Dict, List, Optional, Set, Tuple, Union

from reviewer.reward.rule_evaluator import (
    compute_format_completeness,
    compute_score_difference_reward,
)
from reviewer.reward.rubric_evaluator import (
    RubricEvaluator,
    TRAINING_CONFIG,
    EVAL_CONFIG,
    REVUTIL_EVAL_CONFIG,
    to_training_format,
)

logger = logging.getLogger(__name__)

ALL_MODES = {"format", "score_diff", "rubric"}

# Default judge model for rubric scoring
DEFAULT_JUDGE_MODEL = "revutil"


# ---------------------------------------------------------------------------
# Text splitting utilities
# ---------------------------------------------------------------------------

_MARKER_PATTERNS = [
    re.compile(r"^(?:[-*•‣▪◦·—–]+)\s+(?P<body>.+)$"),
    re.compile(r"^\(?\d+[)\.:\-]?\s+(?P<body>.+)$"),
    re.compile(r"^\(?[A-Za-z][)\.:]\s+(?P<body>.+)$"),
    re.compile(r"^\(?[IVXLCDM]+[)\.]\s+(?P<body>.+)$", re.IGNORECASE),
]

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'\(\[]?[A-Z0-9])")

_ABBREVIATION_RE = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|fig|sec|al|mr|mrs|ms|dr|prof|dept|inc|co|jr|sr))\.$",
    re.IGNORECASE,
)

_ORDINAL_SPLIT = re.compile(
    r'(?=\b(?:'
    r'First(?:ly)?,\s|Second(?:ly)?,\s|Third(?:ly)?,\s|Fourth(?:ly)?,\s|'
    r'Fifth(?:ly)?,\s|Sixth(?:ly)?,\s|Seventh(?:ly)?,\s|Eighth(?:ly)?,\s|'
    r'Ninth(?:ly)?,\s|Tenth(?:ly)?,\s|Finally,\s'
    r'))',
    re.IGNORECASE,
)


def split_review_text(raw_text: str) -> List[str]:
    """Split review text into individual points (bullets, numbered items, sentences)."""
    if not raw_text:
        return []

    normalized = re.sub(r"\r\n?", "\n", raw_text).strip()
    if not normalized:
        return []

    _ABBREV_WORD_RE = re.compile(r"\b[A-Za-z]{1,6}\Z")

    def _presplit_sub(m: re.Match) -> str:
        before = m.string[:m.start()]
        word_part = before[:-1]
        if _ABBREV_WORD_RE.search(word_part):
            return m.group(0)
        return "\n"

    normalized = re.sub(r"(?<=[.!?;])\s+(?=\(?\d+[)\.:\-]\s)", _presplit_sub, normalized)

    points: List[str] = []
    current: List[str] = []

    def flush_current():
        if current:
            segment = " ".join(current).strip()
            if segment:
                points.append(segment)
            current.clear()

    def strip_marker(line: str):
        for pattern in _MARKER_PATTERNS:
            match = pattern.match(line)
            if match:
                return match.group("body").strip()
        return None

    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_current()
            continue

        payload = strip_marker(stripped)
        if payload is not None:
            flush_current()
            if payload:
                current.append(payload)
            continue

        current.append(stripped)

    flush_current()

    if not points:
        points = [normalized]

    if len(points) <= 1:
        raw_chunks = [
            chunk.strip()
            for chunk in _SENTENCE_BOUNDARY_RE.split(normalized)
            if chunk.strip()
        ]

        merged_chunks: List[str] = []
        for chunk in raw_chunks:
            if merged_chunks and _ABBREVIATION_RE.search(merged_chunks[-1].rstrip()):
                merged_chunks[-1] = f"{merged_chunks[-1]} {chunk}".strip()
            else:
                merged_chunks.append(chunk)

        if len(merged_chunks) > len(points):
            points = merged_chunks

    return points


def _split_weakness_text(text: str) -> List[str]:
    """Split a weakness string into individual points.

    Handles newline-separated points and prose with ordinal markers.
    """
    if not text or not text.strip():
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    parts = _ORDINAL_SPLIT.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        if len(parts[0]) < 200 and not re.match(r'\b(?:First|Second)', parts[0], re.IGNORECASE):
            parts = parts[1:]
        return parts

    return [text]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_rating_from_section(rating_text: str) -> Optional[float]:
    """Extract numeric rating from section text like '5: marginally below ...'."""
    if not rating_text:
        return None
    m = re.match(r'(\d+)', rating_text.strip())
    return float(m.group(1)) if m else None


MAX_PAPER_TOKENS = 64_000


def _truncate_paper(text: str, max_tokens: int = MAX_PAPER_TOKENS) -> str:
    """Truncate paper content to max_tokens using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        if not isinstance(text, str):
            logger.error(f"_truncate_paper received non-string type: {type(text)}")
            text = str(text) if text is not None else ""
        tokens = enc.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        truncated = enc.decode(tokens[:max_tokens])
        logger.info(f"Truncated paper from {len(tokens):,} to {max_tokens:,} tokens")
        return truncated + "\n\n[Paper truncated due to length]"
    except ImportError:
        max_chars = int(max_tokens * 3.5)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[Paper truncated due to length]"


def _extract_weakness_texts(review: Dict) -> List[str]:
    """Extract and normalize weakness texts from a review dict."""
    raw = review.get("weaknesses", [])
    if raw is None:
        raw = []
    if isinstance(raw, str):
        raw = _split_weakness_text(raw)
    return [t for t in raw if isinstance(t, str) and t.strip()]


# ---------------------------------------------------------------------------
# Rubric scoring via RubricEvaluator
# ---------------------------------------------------------------------------

def _select_rubric_config(rubric_model: str, training: bool) -> dict:
    """Select the appropriate rubric config based on model and mode.

    - Training: TRAINING_CONFIG (per_dimension, 2 dims)
    - Eval with revutil: REVUTIL_EVAL_CONFIG (per_weakness, 3 dims, no technical_depth)
    - Eval default: EVAL_CONFIG (per_weakness, 4 dims)
    """
    if training:
        return TRAINING_CONFIG
    if "revutil" in rubric_model.lower():
        return REVUTIL_EVAL_CONFIG
    return EVAL_CONFIG


async def _compute_rubric(
    weakness_texts: List[str],
    paper_content: str,
    rubric_model: str,
    training: bool = False,
) -> Dict:
    """Evaluate weaknesses using RubricEvaluator.

    Returns:
        Dict with rubric, rubric_scores, rubric_per_weakness, and token_usage.
    """
    config = _select_rubric_config(rubric_model, training)
    evaluator = RubricEvaluator(config, model=rubric_model)
    eval_result = await evaluator.evaluate(weakness_texts, paper_content)

    if training:
        # Convert to scalar score for training reward
        score, details = to_training_format(eval_result)
        if score is None:
            return {
                "rubric": 0.0,
                "rubric_scores": {},
                "rubric_per_weakness": eval_result.get("per_weakness", []),
                "rubric_details": details,
                "token_usage": eval_result.get("token_usage", {}),
            }
        return {
            "rubric": score,
            "rubric_scores": eval_result.get("averages", {}),
            "rubric_per_weakness": eval_result.get("per_weakness", []),
            "rubric_details": details,
            "token_usage": eval_result.get("token_usage", {}),
        }
    else:
        # Eval: return per-dimension averages
        return {
            "rubric": eval_result.get("overall", 0.0),
            "rubric_scores": eval_result.get("averages", {}),
            "rubric_per_weakness": eval_result.get("per_weakness", []),
            "token_usage": eval_result.get("token_usage", {}),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def async_score_review(
    review: Union[str, Dict],
    human_avg_score: Optional[float] = None,
    reward_modes: Set[str] = "full",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    paper_content: Optional[str] = None,
    rubric_model: Optional[str] = None,
    training: bool = False,
) -> Dict:
    """Score a review across multiple reward dimensions (async).

    Args:
        review: Review dict or raw text string.
        human_avg_score: Average human score for score_diff.
        reward_modes: Set of modes to compute. "full" = all.
        judge_model: Model for LLM judge calls.
        paper_content: Paper text (required for rubric mode).
        rubric_model: Model for rubric evaluation (defaults to judge_model).
            If "revutil", uses REVUTIL_EVAL_CONFIG (no technical_depth).
        training: If True, use TRAINING_CONFIG for rubric (per_dimension, 2 dims).
                  If False, use EVAL_CONFIG or REVUTIL_EVAL_CONFIG based on model.

    Returns:
        Dict with keys per active mode plus "*_details" where applicable.
    """
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

    result: Dict = {}
    token_usage: Dict = {}

    if "format" in reward_modes:
        result["format"] = compute_format_completeness(review)

    if "score_diff" in reward_modes:
        if human_avg_score is not None:
            score_diff, score_details = compute_score_difference_reward(
                review.get("overall_score"), human_avg_score
            )
        else:
            score_diff, score_details = 0.0, {"warning": "No human_avg_score"}
        result["score_diff"] = score_diff
        result["score_diff_details"] = score_details

    if "rubric" in reward_modes:
        if not paper_content:
            logger.warning("rubric mode requires paper_content; skipping")
        else:
            model = rubric_model or judge_model
            try:
                weakness_texts = _extract_weakness_texts(review)
                if weakness_texts:
                    rubric_result = await _compute_rubric(
                        weakness_texts, paper_content, model, training=training
                    )
                    result["rubric"] = rubric_result["rubric"]
                    result["rubric_scores"] = rubric_result["rubric_scores"]
                    result["rubric_per_weakness"] = rubric_result["rubric_per_weakness"]
                    if training and "rubric_details" in rubric_result:
                        result["rubric_details"] = rubric_result["rubric_details"]
                    rubric_usage = rubric_result.get("token_usage", {})
                    if rubric_usage and rubric_usage.get("calls", 0) > 0:
                        token_usage[model] = rubric_usage
                else:
                    result["rubric"] = 0.0
                    result["rubric_scores"] = {}
                    result["rubric_per_weakness"] = []
            except Exception as e:
                logger.error(f"Rubric scoring failed: {e}")
                result["rubric"] = 0.0
                result["rubric_scores"] = {}
                result["rubric_per_weakness"] = []

    if token_usage:
        result["judge_token_usage"] = token_usage

    return result


def score_review(
    review: Union[str, Dict],
    human_avg_score: Optional[float] = None,
    reward_modes: Set[str] = "full",
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> Dict:
    """Score a review across multiple reward dimensions (sync wrapper).

    Args:
        review: Review dict or raw text string.
        human_avg_score: Average human score for score_diff.
        reward_modes: Set of modes to compute. "full" = all.
        judge_model: Model for LLM judge calls.

    Returns:
        Dict with keys per active mode plus "*_details" where applicable.
    """
    coro = async_score_review(
        review=review,
        human_avg_score=human_avg_score,
        reward_modes=reward_modes,
        judge_model=judge_model,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
