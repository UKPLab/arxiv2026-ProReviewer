"""Unified configurable RubricEvaluator for training and eval.

Builds prompts from ``rubric_dimensions.py`` and supports three strategies:
- ``per_dimension``: one LLM call per dimension, all weaknesses (training)
- ``per_weakness``: one LLM call per weakness, all dimensions (eval)
- ``batched``: one LLM call total, all weaknesses and dimensions (eval, cheap)

Usage::

    evaluator = RubricEvaluator(TRAINING_CONFIG, llm_judge_fn=my_fn)
    result = await evaluator.evaluate(weakness_texts, paper_content)

    evaluator = RubricEvaluator(EVAL_CONFIG, model="gpt-4o")
    result = await evaluator.evaluate(weakness_texts, paper_content)
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional

from reviewer.prompts.rubric_dimensions import (
    build_batched_system_prompt,
    build_batched_user_prompt,
    build_per_dimension_system_prompt,
    build_per_dimension_user_prompt,
    build_per_weakness_system_prompt,
    build_per_weakness_user_prompt,
)
logger = logging.getLogger(__name__)

MAX_PAPER_TOKENS = 64_000


def _truncate_paper(text: str, max_tokens: int = MAX_PAPER_TOKENS) -> str:
    """Truncate paper content to max_tokens using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        if not isinstance(text, str):
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

# ---------------------------------------------------------------------------
# Hardcoded configs
# ---------------------------------------------------------------------------

TRAINING_CONFIG = {
    "dimensions": ["technical_depth", "grounding_specificity"],
    "dimension_weights": {"technical_depth": 0.50, "grounding_specificity": 0.50},
    "strategy": "per_dimension",
    "temperature": 0.0,
    "max_tokens": 8192,
}

EVAL_CONFIG = {
    "dimensions": ["technical_depth", "grounding_specificity", "actionability", "verifiability"],
    "strategy": "per_weakness",
    "temperature": 0.0,
    "max_tokens": 2048,
}

EVAL_BATCHED_CONFIG = {
    "dimensions": ["technical_depth", "grounding_specificity", "actionability", "verifiability"],
    "strategy": "batched",
    "temperature": 0.0,
    "max_tokens": 4096,
}

# RevUtil model doesn't support technical_depth; score-only, no paper context
REVUTIL_EVAL_CONFIG = {
    "dimensions": ["grounding_specificity", "actionability", "verifiability", "helpfulness"],
    "strategy": "per_weakness",
    "score_only": True,
    "include_paper": False,
    "temperature": 0.0,
    "max_tokens": 2048,
}

# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str):
    """Extract and parse JSON from LLM output (handles ```json fencing)."""
    m = _JSON_FENCE_RE.search(text)
    raw = m.group(1) if m else text
    return json.loads(raw)


# ---------------------------------------------------------------------------
# RubricEvaluator
# ---------------------------------------------------------------------------


class RubricEvaluator:
    """Unified rubric evaluator that works in training or eval mode.

    Args:
        config: One of ``TRAINING_CONFIG``, ``EVAL_CONFIG``,
            ``EVAL_BATCHED_CONFIG``, or a custom dict with the same keys.
        model: Model name string — uses ``acall_llm`` (eval path).
        llm_judge_fn: Async callable ``(system, user) -> str`` (training path).

    Exactly one of *model* or *llm_judge_fn* must be provided.
    """

    def __init__(self, config: dict, *, model: Optional[str] = None,
                 llm_judge_fn=None):
        if (model is None) == (llm_judge_fn is None):
            raise ValueError("Exactly one of model or llm_judge_fn is required")
        self.config = config
        self.model = model
        self.llm_judge_fn = llm_judge_fn
        self._token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "calls": 0,
        }

    # ------------------------------------------------------------------
    # LLM calling
    # ------------------------------------------------------------------

    async def _call_llm(self, system: str, user: str) -> str:
        """Route to model-based or fn-based LLM call."""
        if self.model is not None:
            from utils.helpers.llm import acall_llm, get_content

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            resp = await acall_llm(
                self.model,
                messages,
                temperature=self.config.get("temperature", 0.0),
                max_tokens=self.config.get("max_tokens", 4096),
            )

            # Accumulate token usage
            if hasattr(resp, "usage") and resp.usage:
                self._token_usage["prompt_tokens"] += getattr(resp.usage, "prompt_tokens", 0) or 0
                self._token_usage["completion_tokens"] += getattr(resp.usage, "completion_tokens", 0) or 0
                self._token_usage["total_tokens"] += getattr(resp.usage, "total_tokens", 0) or 0
                self._token_usage["calls"] += 1
                # DeepSeek cached tokens
                ds_cached = getattr(resp.usage, "prompt_cache_hit_tokens", 0) or 0
                if ds_cached:
                    self._token_usage["cached_tokens"] += ds_cached
                else:
                    ptd = getattr(resp.usage, "prompt_tokens_details", None)
                    if ptd:
                        self._token_usage["cached_tokens"] += getattr(ptd, "cached_tokens", 0) or 0

            return get_content(resp)
        else:
            return await self.llm_judge_fn(system, user)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        weakness_texts: List[str],
        paper_content: str,
        weakness_ids: Optional[List[str]] = None,
    ) -> dict:
        """Evaluate weaknesses on configured rubric dimensions.

        Args:
            weakness_texts: List of weakness text strings.
            paper_content: Full paper text.
            weakness_ids: Optional custom IDs (default: ``W1``, ``W2``, ...).

        Returns:
            Unified result dict with ``per_weakness``, ``averages``,
            ``overall``, ``dimensions``, ``strategy``, ``token_usage`` keys.
        """
        if weakness_ids is None:
            weakness_ids = [f"W{i+1}" for i in range(len(weakness_texts))]

        paper_content = _truncate_paper(paper_content)

        strategy = self.config["strategy"]
        if strategy == "per_dimension":
            return await self._run_per_dimension(weakness_texts, paper_content, weakness_ids)
        elif strategy == "per_weakness":
            return await self._run_per_weakness(weakness_texts, paper_content, weakness_ids)
        elif strategy == "batched":
            return await self._run_batched(weakness_texts, paper_content, weakness_ids)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ------------------------------------------------------------------
    # Strategy: per_dimension
    # ------------------------------------------------------------------

    async def _run_per_dimension(
        self,
        weakness_texts: List[str],
        paper_content: str,
        weakness_ids: List[str],
    ) -> dict:
        """One LLM call per dimension, all weaknesses in each call."""
        dims = self.config["dimensions"]
        system_prompt = build_per_dimension_system_prompt()

        async def _eval_dim(dim: str):
            user_prompt = build_per_dimension_user_prompt(dim, weakness_texts, paper_content)
            try:
                content = await self._call_llm(system_prompt, user_prompt)
                items = _extract_json(content)
                # Build id -> {score, reason} mapping
                scores = {}
                for item in items:
                    wid = item.get("item_id", "")
                    scores[wid] = {
                        "score": item.get("score", 0),
                        "reason": item.get("reason", ""),
                    }
                return dim, scores
            except Exception as e:
                logger.error(f"Per-dimension eval failed for {dim}: {e}")
                return dim, None

        results = await asyncio.gather(*[_eval_dim(d) for d in dims])

        # Check for failed dimensions — mirrors training pipeline behaviour
        # where any failed dim causes the whole instance to be skipped.
        dim_scores = {dim: scores for dim, scores in results}
        failed_dims = [dim for dim, scores in dim_scores.items() if scores is None]
        if failed_dims:
            logger.error(f"Per-dimension eval failed for dims {failed_dims}, returning failure")
            return {
                "per_weakness": [],
                "averages": {d: 0.0 for d in dims},
                "overall": 0.0,
                "dimensions": list(dims),
                "strategy": "per_dimension",
                "token_usage": dict(self._token_usage),
                "judge_failed": True,
                "failed_dims": failed_dims,
            }

        # Assemble per-weakness output (flat legacy format)
        per_weakness = []
        for i, (wid, wtext) in enumerate(zip(weakness_ids, weakness_texts)):
            entry = {"weakness_idx": i, "weakness_text": wtext}
            for dim in dims:
                scores = dim_scores[dim]
                if wid in scores:
                    entry[f"{dim}_score"] = scores[wid]["score"]
                    entry[f"{dim}_reason"] = scores[wid]["reason"]
                else:
                    entry[f"{dim}_score"] = 0.0
                    entry[f"{dim}_reason"] = "Missing from response"
            per_weakness.append(entry)

        return self._build_result(per_weakness, dims)

    # ------------------------------------------------------------------
    # Strategy: per_weakness
    # ------------------------------------------------------------------

    async def _run_per_weakness(
        self,
        weakness_texts: List[str],
        paper_content: str,
        weakness_ids: List[str],  # unused, kept for interface consistency
    ) -> dict:
        """One LLM call per weakness, all dimensions scored together.

        Sequential to maximize prefix cache hits.
        """
        dims = self.config["dimensions"]
        score_only = self.config.get("score_only", False)
        include_paper = self.config.get("include_paper", True)
        system_prompt = build_per_weakness_system_prompt(dims, score_only=score_only)

        per_weakness = []
        for idx, wtext in enumerate(weakness_texts):
            user_prompt = build_per_weakness_user_prompt(
                wtext, paper_content if include_paper else None
            )
            entry = {"weakness_idx": idx, "weakness_text": wtext}
            try:
                content = await self._call_llm(system_prompt, user_prompt)
                parsed = _extract_json(content)
                for dim in dims:
                    raw_score = parsed.get(dim, 0)
                    # Handle verifiability "X" (no claim) by mapping to 0.0
                    if dim == "verifiability" and str(raw_score).strip().upper() == "X":
                        score = 0.0
                    else:
                        try:
                            score = float(raw_score)
                        except (ValueError, TypeError):
                            score = 0.0
                    entry[f"{dim}_score"] = score
                    if not score_only:
                        entry[f"{dim}_reason"] = parsed.get(f"{dim}_reason", "")
            except Exception as e:
                logger.error(f"Per-weakness eval failed for W{idx+1}: {e}")
                for dim in dims:
                    entry[f"{dim}_score"] = 0.0
                    if not score_only:
                        entry[f"{dim}_reason"] = f"Error: {str(e)}"
                entry["error"] = str(e)
            per_weakness.append(entry)

        return self._build_result(per_weakness, dims)

    # ------------------------------------------------------------------
    # Strategy: batched
    # ------------------------------------------------------------------

    async def _run_batched(
        self,
        weakness_texts: List[str],
        paper_content: str,
        weakness_ids: List[str],  # unused, kept for interface consistency
    ) -> dict:
        """Single LLM call for all weaknesses and all dimensions."""
        dims = self.config["dimensions"]
        system_prompt = build_batched_system_prompt(dims, len(weakness_texts))
        user_prompt = build_batched_user_prompt(weakness_texts, paper_content)

        per_weakness = []
        try:
            content = await self._call_llm(system_prompt, user_prompt)
            parsed = _extract_json(content)
            weaknesses_data = parsed.get("weaknesses", [])

            for idx, wtext in enumerate(weakness_texts):
                entry = {"weakness_idx": idx, "weakness_text": wtext}
                if idx < len(weaknesses_data):
                    wr = weaknesses_data[idx]
                    for dim in dims:
                        raw_score = wr.get(dim, 0)
                        # Handle verifiability "X" (no claim) by mapping to 0.0
                        if dim == "verifiability" and str(raw_score).strip().upper() == "X":
                            score = 0.0
                        else:
                            try:
                                score = float(raw_score)
                            except (ValueError, TypeError):
                                score = 0.0
                        reason = wr.get(f"{dim}_reason", "")
                        entry[f"{dim}_score"] = score
                        entry[f"{dim}_reason"] = reason
                else:
                    for dim in dims:
                        entry[f"{dim}_score"] = 0.0
                        entry[f"{dim}_reason"] = "Missing from batched response"
                    entry["error"] = "Missing from batched response"
                per_weakness.append(entry)

        except Exception as e:
            logger.error(f"Batched eval failed: {e}")
            for idx, wtext in enumerate(weakness_texts):
                entry = {"weakness_idx": idx, "weakness_text": wtext}
                for dim in dims:
                    entry[f"{dim}_score"] = 0.0
                    entry[f"{dim}_reason"] = f"Error: {str(e)}"
                entry["error"] = str(e)
                per_weakness.append(entry)

        return self._build_result(per_weakness, dims)

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    def _build_result(self, per_weakness: List[dict], dims: List[str]) -> dict:
        """Assemble the unified output dict from per-weakness entries."""
        # Compute averages per dimension, skipping failed entries
        averages = {}
        for dim in dims:
            scores = [
                pw[f"{dim}_score"] for pw in per_weakness
                if "error" not in pw
            ]
            averages[dim] = sum(scores) / len(scores) if scores else 0.0

        numeric_avgs = [v for v in averages.values() if v > 0]
        overall = sum(numeric_avgs) / len(numeric_avgs) if numeric_avgs else 0.0

        result = {
            "per_weakness": per_weakness,
            "averages": averages,
            "overall": overall,
            "dimensions": list(dims),
            "strategy": self.config.get("strategy", "unknown"),
            "token_usage": dict(self._token_usage),
        }
        if "dimension_weights" in self.config:
            result["dimension_weights"] = self.config["dimension_weights"]
        return result


# ---------------------------------------------------------------------------
# Training-pipeline conversion
# ---------------------------------------------------------------------------

def to_training_format(eval_result, dimension_weights=None):
    """Convert evaluator output to the training-pipeline ``(score, details)`` tuple.

    Mirrors the normalisation in ``review_quality_trajectory_v2_evidence.py``:
    raw 1-5 scores are mapped to 0-1 and weighted by *dimension_weights*.

    Args:
        eval_result: Dict returned by :meth:`RubricEvaluator.evaluate`.
        dimension_weights: ``{dim: weight}`` dict.  Falls back to
            ``eval_result["dimension_weights"]`` (set by config), then
            equal weights.

    Returns ``(None, {"judge_failed": True, ...})`` when no valid scores remain.
    """
    dims = eval_result.get("dimensions", [])
    if dimension_weights is None:
        dimension_weights = eval_result.get("dimension_weights", {})

    # Normalise weights to sum to 1.0
    raw_w = {d: dimension_weights.get(d, 1.0) for d in dims}
    total = sum(raw_w.values()) or 1.0
    dim_weights = {d: w / total for d, w in raw_w.items()}

    # Collect raw scores per dim (flat format: {dim}_score keys, weakness_idx for IDs)
    raw_scores = {d: {} for d in dims}
    for pw in eval_result.get("per_weakness", []):
        if "error" in pw:
            continue
        wid = f"W{pw['weakness_idx'] + 1}"
        for d in dims:
            s = pw.get(f"{d}_score", 0)
            if isinstance(s, (int, float)):
                raw_scores[d][wid] = float(s)

    # Each dimension must have at least one scored item
    empty = [d for d in dims if not raw_scores[d]]
    if empty:
        return None, {
            "judge_failed": True,
            "failed_dims": empty,
            "individual_reasons": {d: "no_valid_scores" for d in empty},
        }

    # Normalise (1-5 → 0-1), compute dim means, apply weights
    per_item_weighted = {}
    dim_means = {}
    for d in dims:
        normed = {wid: (s - 1) / 4.0 for wid, s in raw_scores[d].items()}
        dim_means[d] = sum(normed.values()) / len(normed)
        per_item_weighted[d] = {wid: s * dim_weights[d] for wid, s in normed.items()}

    # Weighted average across dimensions
    trajectory_quality = sum(dim_means[d] * dim_weights[d] for d in dims)

    details = {
        # Per-item weighted scores
        "technical_depth_per_item": per_item_weighted.get("technical_depth", {}),
        "grounding_specificity_per_item": per_item_weighted.get("grounding_specificity", {}),
        # Scalars / metadata for logging
        "trajectory_quality": trajectory_quality,
        "dim_scores": dim_means,
        "dim_weights": dim_weights,
        "active_dimensions": list(dims),
        "raw_scores": raw_scores,
        "individual_reasons": {},
        "evidence_based": True,
        "token_usage": eval_result.get("token_usage", {}),
    }
    return trajectory_quality, details
