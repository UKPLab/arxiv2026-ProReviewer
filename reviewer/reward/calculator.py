"""Reward calculator for review quality assessment.

This module provides the RewardCalculator class with LLM-calling primitives
for computing utility scores and recall judgments.
"""

import asyncio
import json
import warnings
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Union

from utils.helpers.llm import call_llm, acall_llm, get_content
from reviewer.reward.prompts import (
    RECALL_JUDGE_SYSTEM_PROMPT,
    RECALL_JUDGE_USER_PROMPT,
    RECALL_JUDGE_POINT_SYSTEM_PROMPT,
    RECALL_JUDGE_POINT_USER_PROMPT,
    DEFAULT_JUDGE_MODEL
)
from utils.helpers.logger import logger

# Import review utility prompt builder
from reviewer.core.review_utility_prompt import build_inference_prompt


# ==================== Text Splitting for Review Utility ====================

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


def split_review_text(raw_text: str) -> List[str]:
    """Split review text into individual points (bullets, sentences)."""
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


def is_duplicate_text(text_a: str, text_b: str, threshold: float = 0.7) -> bool:
    """Check if two texts are semantically duplicate using SequenceMatcher."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio() > threshold


class RewardCalculator:
    """Compute multi-dimensional rewards for review evaluation."""

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        recall_model: str = "qwen35-122b",
        embed_model: str = "qwen3-embedding-8b",
        use_cache: bool = True,
        api_key: Optional[str] = None,
        max_concurrent_judge_calls: int = 32,
        **kwargs
    ):
        """Initialize reward calculator.

        Args:
            judge_model: LLM model for utility judge evaluations
            recall_model: LLM model for recall judge (defaults to judge_model)
            embed_model: Config name for embedding model (diversity computation)
            use_cache: Enable caching of LLM responses
            api_key: API key for LLM calls (optional)
            max_concurrent_judge_calls: Max concurrent LLM judge requests to
                avoid overwhelming the vLLM server (default: 32).
        """
        self.judge_model = judge_model
        self.recall_model = recall_model
        self.embed_model = embed_model
        self._embed_client: Optional[object] = None
        self.use_cache = use_cache
        self.api_key = api_key
        self._cache = {} if use_cache else None
        self._judge_semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrent_judge_calls = max_concurrent_judge_calls
        # Per-model token usage accumulators: {model: {prompt, completion, total, calls}}
        self.token_usage: Dict[str, Dict[str, int]] = {}

    def compute_single_point_utility(
        self,
        weakness_text: str,
        aspect_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict]:
        """Compute utility reward for a single weakness point (synchronous).

        Evaluates a single weakness text on 4 aspects:
        - Actionability, Grounding Specificity, Verifiability, Helpfulness

        Args:
            weakness_text: The weakness text to evaluate
            aspect_weights: Custom weights for aspects (default: equal 0.25 each)

        Returns:
            Tuple of (final_score, aspect_details)
        """
        if not weakness_text or not weakness_text.strip():
            raise ValueError("Empty weakness text provided")

        if aspect_weights is None:
            aspect_weights = {a: 0.25 for a in ['actionability', 'grounding_specificity', 'verifiability', 'helpfulness']}

        prompt_messages = build_inference_prompt(
            review_point=weakness_text,
            aspects='all',
            include_paper_text=False,
            generation_type='score_only'
        )
        system_content = prompt_messages[0]['content']
        user_content = prompt_messages[1]['content']

        max_retries = 2
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0 and self.use_cache:
                    self._cache.pop(hash((system_content, user_content)), None)
                response = self._call_llm_judge_sync(system_content, user_content)
                return self._score_point_from_response(response, aspect_weights)
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    logger.warning(f"Single point eval attempt {attempt+1} failed: {e}, retrying...")
        logger.error(f"Single point eval failed after {max_retries+1} attempts: {last_exc}")
        raise last_exc

    async def _evaluate_single_point_async(
        self,
        point_text: str,
        aspect_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict]:
        """Evaluate a single review point on 4 utility aspects (async)."""
        if aspect_weights is None:
            aspect_weights = {a: 0.25 for a in ['actionability', 'grounding_specificity', 'verifiability', 'helpfulness']}

        prompt_messages = build_inference_prompt(
            review_point=point_text,
            aspects='all',
            include_paper_text=False,
            generation_type='score_only'
        )
        system_content = prompt_messages[0]['content']
        user_content = prompt_messages[1]['content']

        max_retries = 2
        last_exc = None
        for attempt in range(max_retries + 1):
            if attempt > 0 and self.use_cache:
                self._cache.pop(hash((system_content, user_content)), None)
            try:
                response = await self._call_llm_judge_async(system_content, user_content)
                return self._score_point_from_response(response, aspect_weights)
            except Exception as e:
                last_exc = e
                logger.warning(f"Single point eval attempt {attempt+1}/{max_retries+1} failed: {e}")
        logger.error(f"Single point eval failed after {max_retries+1} attempts: {last_exc}")
        raise last_exc

    async def _compute_utility_async(
        self,
        unique_points: List[Tuple[int, str]],
        aspect_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, List[Dict]]:
        """Compute utility for all weakness points concurrently (async).

        Args:
            unique_points: List of (outline_idx, text) tuples.
            aspect_weights: Custom weights for aspects (default: equal 0.25 each).

        Returns:
            Tuple of (avg_utility_score, per-point detail list).
        """
        if aspect_weights is None:
            aspect_weights = {a: 0.25 for a in ['actionability', 'grounding_specificity', 'verifiability', 'helpfulness']}

        async def _eval_one(outline_idx: int, text: str):
            try:
                score, details = await self._evaluate_single_point_async(text, aspect_weights)
                return {
                    'outline_idx': outline_idx,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'utility_score': score,
                    'actionability': details['raw_scores'].get('actionability'),
                    'grounding_specificity': details['raw_scores'].get('grounding_specificity'),
                    'verifiability': details['raw_scores'].get('verifiability'),
                    'helpfulness': details['raw_scores'].get('helpfulness'),
                    'actionability_normalized': details['normalized_scores'].get('actionability'),
                    'grounding_specificity_normalized': details['normalized_scores'].get('grounding_specificity'),
                    'verifiability_normalized': details['normalized_scores'].get('verifiability'),
                    'helpfulness_normalized': details['normalized_scores'].get('helpfulness'),
                }
            except Exception as e:
                logger.error(f"Failed to compute utility for weakness (outline {outline_idx}): {e}")
                return {
                    'outline_idx': outline_idx,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'utility_score': 0.0,
                    'actionability': None,
                    'grounding_specificity': None,
                    'verifiability': None,
                    'helpfulness': None,
                    'actionability_normalized': None,
                    'grounding_specificity_normalized': None,
                    'verifiability_normalized': None,
                    'helpfulness_normalized': None,
                    'error': str(e),
                }

        results = await asyncio.gather(*[_eval_one(idx, text) for idx, text in unique_points])
        scores = [r['utility_score'] for r in results]
        avg_utility = sum(scores) / len(scores) if scores else 0.0
        return avg_utility, list(results)

    async def compute_utility_async(

        self,

        weakness_texts: List[str],

        aspect_weights: Optional[Dict[str, float]] = None,

    ) -> Tuple[float, List[Dict]]:

        """Compute pure utility for weakness points (no diversity weighting)."""

        if not weakness_texts:
            return 0.0, []

        all_points = [(idx, text) for idx, text in enumerate(weakness_texts)]
        avg_utility, utility_details = await self._compute_utility_async(all_points, aspect_weights)

        return avg_utility, utility_details

    def compute_diversity_score(self, weakness_texts: List[str]) -> float:
        """Compute a scalar diversity score for a set of weakness points.

        Returns 0.0 for n<=1 (no diversity possible), otherwise the mean
        per-weakness diversity: mean(1 - max_cosine_sim(w_i, w_j for j!=i)).

        This is separate from utility so both can be weighted independently.
        """
        n = len(weakness_texts)
        if n <= 1:
            return 0.0

        per_w_diversity, _ = self.compute_weakness_diversity(weakness_texts)
        return sum(per_w_diversity) / len(per_w_diversity)



    def compute_utility_sync(
        self,
        weakness_texts: List[str],
        aspect_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, List[Dict]]:

        """Compute pure utility, callable from sync context."""

        if not weakness_texts:
            return 0.0, []

        coro = self.compute_utility_async(weakness_texts, aspect_weights)
        try:
            loop = asyncio.get_running_loop()

        except RuntimeError:
            loop = None

        if loop is not None:
            with ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, coro).result()

        else:
            return asyncio.run(coro)


    def compute_utility_sync(
        self,
        unique_points: List[Tuple[int, str]],
        aspect_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, List[Dict]]:
        """Compute utility for all weakness points, callable from sync context."""
        if not unique_points:
            return 0.0, []

        coro = self._compute_utility_async(unique_points, aspect_weights)

        # If already inside an event loop, run in a separate thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            with ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def compute_weakness_diversity(
        self,
        weakness_texts: List[str],
    ) -> Tuple[List[float], Dict]:
        """Compute per-weakness diversity via embedding cosine similarity.

        For each weakness i:
            diversity_i = 1 - max(cosine(w_i, w_j) for j != i)

        Calls the vLLM embedding endpoint (OpenAI-compatible /v1/embeddings).
        Model config is resolved from config.toml via embed_model name.

        Returns:
            Tuple of (per-weakness diversity scores, details dict).
        """
        import numpy as np
        from openai import OpenAI
        from utils.helpers.llm import MODEL_CONFIGS

        n = len(weakness_texts)
        if n <= 1:
            return [1.0] * n, {
                'n_weaknesses': n,
                'per_weakness_max_sim': [0.0] * n,
                'per_weakness_diversity': [1.0] * n,
            }

        # Resolve embed_model config and create/reuse client
        config = MODEL_CONFIGS.get(self.embed_model, {})
        base_url = config.get("base_url", "http://localhost:8000/v1")
        api_key = config.get("api_key", "EMPTY")
        model_name = config.get("model", self.embed_model).removeprefix("openai/")

        if self._embed_client is None:
            self._embed_client = OpenAI(base_url=base_url, api_key=api_key)

        response = self._embed_client.embeddings.create(
            model=model_name,
            input=weakness_texts,
        )

        # Extract embedding vectors sorted by index
        embeddings = np.array([
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ])

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # Cosine similarity matrix
        sim_matrix = embeddings @ embeddings.T

        # Per-weakness: diversity_i = 1 - max(sim to other weaknesses)
        per_w_max_sim = []
        per_w_diversity = []
        for i in range(n):
            other_sims = [float(sim_matrix[i, j]) for j in range(n) if j != i]
            max_sim = max(other_sims)
            per_w_max_sim.append(round(max_sim, 4))
            per_w_diversity.append(round(max(0.0, 1.0 - max_sim), 4))

        return per_w_diversity, {
            'n_weaknesses': n,
            'per_weakness_max_sim': per_w_max_sim,
            'per_weakness_diversity': per_w_diversity,
        }

    async def _compute_recall_async(
        self,
        generated_reviews: Dict,
        clustered_points: List[Dict]
    ) -> List:
        """Compute recall reward using LLM judge (1 call for all points).

        Args:
            generated_reviews: Review dict with strengths/weaknesses lists
            clustered_points: List of clustered points

        Returns:
            List of recall result dicts per point
        """
        # Only evaluate strength and weakness points; questions are not actionable review points
        clustered_points = [cp for cp in clustered_points if isinstance(cp, dict) and cp.get('type', '').lower() in ('strength', 'weakness')]
        if not clustered_points:
            return []

        strengths = generated_reviews.get('strengths', [])
        weaknesses = generated_reviews.get('weaknesses', [])
        if isinstance(strengths, str):
            logger.warning("_compute_recall_async: 'strengths' is a string, not a list — splitting by newlines. Fix the upstream review format.")
            strengths = [l.strip() for l in strengths.splitlines() if l.strip()]
        if isinstance(weaknesses, str):
            logger.warning("_compute_recall_async: 'weaknesses' is a string, not a list — splitting by newlines. Fix the upstream review format.")
            weaknesses = [l.strip() for l in weaknesses.splitlines() if l.strip()]
        generated_weakness_strengths = '# Strengths\n\n' + '\n\n'.join([f"S{i+1}. {point}" for i, point in enumerate(strengths)]) + '\n\n' + '# Weaknesses\n\n' + '\n\n'.join([f"W{i+1}. {point}" for i, point in enumerate(weaknesses)])

        prompt = RECALL_JUDGE_USER_PROMPT.format(
            num_points=len(clustered_points),
            human_points=self._format_points_for_prompt(clustered_points),
            generated_review=generated_weakness_strengths
        )

        response = await self._call_llm_judge_async(
            RECALL_JUDGE_SYSTEM_PROMPT,
            prompt,
            model_override=self.recall_model,
        )
        
        recall_results = self._parse_judge_response(response)

        if not isinstance(recall_results, list):
            warnings.warn("Recall judge did not return a list, using fallback")
            return []

        # Enrich each result with the point's context and normalize coverage
        for r in recall_results:
            pid = r.get("point_id")
            if pid is not None and 1 <= pid <= len(clustered_points):
                cp = clustered_points[pid - 1]
                r["point_text"] = cp.get("text", "")
                r["point_type"] = cp.get("type", "")
            # Normalize coverage: accept legacy "covered" as "full"
            raw_cov = r.get('coverage', 'not_covered')
            if raw_cov == 'covered':
                raw_cov = 'full'
            if raw_cov not in ('full', 'partial', 'not_covered'):
                raw_cov = 'not_covered'
            r['coverage'] = raw_cov

        return recall_results

    def compute_recall_sync(
        self,
        generated_reviews: Dict,
        clustered_points: List[Dict]
    ) -> List:
        """Compute recall per-point using async LLM calls, callable from sync context."""
        if not clustered_points:
            return []

        coro = self._compute_recall_per_point_async(generated_reviews, clustered_points)

        # If already inside an event loop, run in a separate thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            with ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def _score_point_from_response(
        self,
        response: str,
        aspect_weights: Dict[str, float],
    ) -> Tuple[float, Dict]:
        """Parse a judge LLM response for one weakness point and return (score, details)."""
        aspects = ['actionability', 'grounding_specificity', 'verifiability', 'helpfulness']
        label_to_score = {
            'not grounded': 1, 'weakly grounded and not specific': 2,
            'weakly grounded and specific': 3, 'fully grounded and under-specific': 4,
            'fully grounded and specific': 5,
            'unactionable': 1, 'borderline actionable': 2, 'somewhat actionable': 3,
            'mostly actionable': 4, 'highly actionable': 5,
            'unverifiable': 1, 'borderline verifiable': 2, 'somewhat verifiable': 3,
            'mostly verifiable': 4, 'fully verifiable': 5,
            'not helpful at all': 1, 'barely helpful': 2, 'somewhat helpful': 3,
            'mostly helpful': 4, 'highly helpful': 5,
        }
        NOT_APPLICABLE = {'x', 'no claim'}

        result = self._parse_judge_response(response)
        if not result or not isinstance(result, dict):
            logger.error(f"Utility judge returned invalid/empty result. Raw response:\n{response[:1000]}")
            raise ValueError(f"Invalid judge response (parsed to {type(result).__name__}). Raw: {response[:500]}")
        scores = {}
        for aspect in aspects:
            label_key = f'{aspect}_label'
            raw = result.get(label_key) or result.get(aspect)
            if raw is None:
                logger.error(f"Missing aspect '{aspect}' in parsed result: {result}. Raw response:\n{response[:1000]}")
                raise ValueError(f"Missing score for aspect '{aspect}'. Result: {result}")
            if isinstance(raw, str):
                normalized = raw.lower().strip()
                if normalized in NOT_APPLICABLE:
                    continue
                raw = label_to_score.get(normalized, raw)
            scores[aspect] = max(1, min(5, int(raw)))

        if not scores:
            raise ValueError("All aspects were not-applicable; cannot compute score.")

        normalized = {a: (s - 1.0) / 4.0 for a, s in scores.items()}
        final_score = sum(
            normalized[a] * aspect_weights.get(a, 0.25)
            for a in scores
        )
        return final_score, {
            'raw_scores': scores,
            'normalized_scores': normalized,
            'final_score': final_score,
            'aspect_weights': aspect_weights,
        }

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _get_judge_semaphore(self) -> asyncio.Semaphore:
        """Lazy-init semaphore (must be created inside a running event loop)."""
        if self._judge_semaphore is None:
            self._judge_semaphore = asyncio.Semaphore(self._max_concurrent_judge_calls)
        return self._judge_semaphore

    async def _call_llm_judge_async(
        self,
        system_prompt: str,
        user_prompt: str,
        model_override: Optional[str] = None,
        max_tokens: int = 8192,
    ) -> str:
        """Call LLM judge with caching and concurrency limiting (async).

        Args:
            system_prompt: System prompt for the judge
            user_prompt: User prompt for the judge
            model_override: Optional model to use instead of default judge_model
            max_tokens: Maximum tokens to generate (default 8192 for complete responses)
        """
        model = model_override or self.judge_model
        cache_key = hash((system_prompt, user_prompt, model))
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Semaphore to limit concurrent requests to avoid overwhelming vLLM server
        async with self._get_judge_semaphore():
            call_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": max_tokens,
            }
            if self.api_key:
                call_params["api_key"] = self.api_key

            response = await acall_llm(**call_params)
        content = get_content(response)

        # Track token usage per model
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            if model not in self.token_usage:
                self.token_usage[model] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
            self.token_usage[model]["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.token_usage[model]["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            self.token_usage[model]["total_tokens"] += getattr(u, "total_tokens", 0) or 0
            self.token_usage[model]["calls"] += 1

        if self.use_cache:
            self._cache[cache_key] = content

        return content

    def _call_llm_judge_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        model_override: Optional[str] = None,
    ) -> str:
        """Call LLM judge synchronously with caching."""
        model = model_override or self.judge_model
        cache_key = hash((system_prompt, user_prompt, model))
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        call_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }
        if self.api_key:
            call_params["api_key"] = self.api_key

        response = call_llm(**call_params)
        content = get_content(response)

        # Track token usage per model
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            if model not in self.token_usage:
                self.token_usage[model] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
            self.token_usage[model]["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.token_usage[model]["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            self.token_usage[model]["total_tokens"] += getattr(u, "total_tokens", 0) or 0
            self.token_usage[model]["calls"] += 1

        if self.use_cache:
            self._cache[cache_key] = content

        return content

    def _parse_judge_response(self, content: str) -> Union[Dict, List]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        if not content:
            warnings.warn("Judge returned empty/None content")
            return {}
        content = content.strip()

        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif content.startswith("```"):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1]).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Retry with escaped backslashes (LLMs often emit raw LaTeX like \sigma_t)
            try:
                return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', content))
            except json.JSONDecodeError as e:
                warnings.warn(f"Failed to parse judge response: {e}\nContent: {content[:200]}...")
                if content.startswith('['):
                    return []
                return {}

    async def _compute_recall_per_point_async(
        self,
        generated_reviews: Dict,
        clustered_points: List[Dict]
    ) -> List:
        """Compute recall by calling the judge once per point (parallel)."""
        # Only evaluate strength and weakness points; questions are not actionable review points
        clustered_points = [cp for cp in clustered_points if isinstance(cp, dict) and cp.get('type', '').lower() in ('strength', 'weakness')]
        if not clustered_points:
            return []

        strengths = generated_reviews.get('strengths', [])
        weaknesses = generated_reviews.get('weaknesses', [])
        if isinstance(strengths, str):
            strengths = [l.strip() for l in strengths.splitlines() if l.strip()]
        if isinstance(weaknesses, str):
            weaknesses = [l.strip() for l in weaknesses.splitlines() if l.strip()]
        generated_review_text = (
            '# Strengths\n\n' + '\n\n'.join([f"S{i+1}. {p}" for i, p in enumerate(strengths)]) +
            '\n\n# Weaknesses\n\n' + '\n\n'.join([f"W{i+1}. {p}" for i, p in enumerate(weaknesses)])
        )

        async def judge_one(i, cp):
            point_type = cp['type'].title()
            expected_id_type = 'weakness (W*)' if point_type == 'Weakness' else 'strength (S*)'
            sys_prompt = RECALL_JUDGE_POINT_SYSTEM_PROMPT.format(
                point_type=point_type,
                expected_id_type=expected_id_type,
            )
            prompt = RECALL_JUDGE_POINT_USER_PROMPT.format(
                point_type=point_type,
                point_text=cp['text'],
                generated_review=generated_review_text,
            )

            max_retries = 2
            result = None
            for attempt in range(max_retries + 1):
                if attempt > 0 and self.use_cache:
                    # Clear cache so we get a fresh response on retry
                    cache_key = hash((sys_prompt, prompt, self.recall_model))
                    self._cache.pop(cache_key, None)
                response = await self._call_llm_judge_async(
                    sys_prompt, prompt, model_override=self.recall_model
                )
                result = self._parse_judge_response(response)
                if isinstance(result, dict) and 'coverage' in result:
                    break
                logger.warning(
                    f"Recall judge parse failed (attempt {attempt+1}/{max_retries+1}) "
                    f"for point {i+1} [{point_type}]. Raw: {(response or '')[:200]}"
                )
                result = None

            if not isinstance(result, dict):
                logger.error(f"Recall judge failed after {max_retries+1} attempts for point {i+1}, defaulting to not_covered")
                result = {}
            # Normalize coverage value: accept legacy "covered" as "full"
            raw_cov = result.get('coverage', 'not_covered')
            if raw_cov == 'covered':
                raw_cov = 'full'
            if raw_cov not in ('full', 'partial', 'not_covered'):
                raw_cov = 'not_covered'
            result['coverage'] = raw_cov
            result['point_id'] = i + 1
            result['point_text'] = cp.get('text', '')
            result['point_type'] = cp.get('type', '')
            return result

        return await asyncio.gather(*[judge_one(i, cp) for i, cp in enumerate(clustered_points)])

    def _format_points_for_prompt(self, clustered_points: List[Dict]) -> str:
        """Format clustered points for recall judge prompt."""
        lines = []
        for i, point in enumerate(clustered_points, 1):
            point_type = point['type'].title()
            lines.append(
                f"{i}. [{point_type}] {point['text']}\n")
        return '\n\n'.join(lines)


if __name__ == '__main__':
    import asyncio, json, os, time

    EVAL_DIR   = '/pfss/mlde/workspaces/mlde_wsp_Reviewer_R1/Reviewer-R1/outputs/eval/step_770/papers'
    DATA_DIR   = os.path.join(os.path.dirname(__file__), '../../data/test_data')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../outputs/recall_comparison')
    N_PAPERS   = 10

    def get_generated_review(paper: Dict) -> Dict:
        """Extract final generated review from the done trajectory step."""
        for step in reversed(paper['trajectory']):
            if step.get('done'):
                return step['action']['args']['review_data']
        raise ValueError(f"No done step found for paper {paper.get('paper_id')}")

    # Load 10 papers from step_770 + their clustered_points from test_data
    eval_files = sorted(os.listdir(EVAL_DIR))[:N_PAPERS]
    papers = []
    for fname in eval_files:
        pid = fname.replace('.json', '')
        eval_paper = json.load(open(os.path.join(EVAL_DIR, fname)))
        test_paper = json.load(open(os.path.join(DATA_DIR, fname)))
        papers.append({'paper_id': pid, 'eval': eval_paper, 'clustered_points': test_paper['clustered_points']})

    async def run_paper(calc, paper):
        pid = paper['paper_id']
        clustered_points = paper['clustered_points']
        t0 = time.perf_counter()
        results = await calc._compute_recall_per_point_async(
            get_generated_review(paper['eval']), clustered_points
        )
        elapsed = time.perf_counter() - t0
        covered = sum(1 for r in results if r['coverage'] == 'covered')
        print(f"{pid}  {covered}/{len(results)}  {elapsed:.1f}s")
        return {
            'paper_id': pid,
            'recall': covered / len(results) if results else None,
            'covered': covered,
            'total': len(results),
            'latency_s': round(elapsed, 2),
            'points': [
                {
                    'point_id':   r['point_id'],
                    'point_type': r.get('point_type', clustered_points[r['point_id']-1].get('type', '')),
                    'point_text': r.get('point_text', clustered_points[r['point_id']-1].get('text', '')),
                    'coverage':   r['coverage'],
                    'evidence':   r.get('evidence', ''),
                    'reasoning':  r.get('reasoning', ''),
                }
                for r in results
            ],
        }

    async def main():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        calc = RewardCalculator(recall_model='qwen35-397B', use_cache=False)
        sem = asyncio.Semaphore(20)  # cap total concurrent LLM calls across all papers

        # patch judge_one to respect semaphore
        _orig = calc._call_llm_judge_async
        async def _limited(*args, **kwargs):
            async with sem:
                return await _orig(*args, **kwargs)
        calc._call_llm_judge_async = _limited

        t_start = time.perf_counter()
        paper_results = await asyncio.gather(*[run_paper(calc, p) for p in papers])
        total_elapsed = time.perf_counter() - t_start

        all_covered = sum(r['covered'] for r in paper_results if r['recall'] is not None)
        all_total   = sum(r['total']   for r in paper_results if r['recall'] is not None)
        macro = all_covered / all_total

        output = {
            '_summary': {'macro_recall': macro, 'model': 'qwen35-397B',
                         'n_papers': len(paper_results), 'total_latency_s': round(total_elapsed, 2)},
            'papers': {r['paper_id']: r for r in paper_results},
        }

        out = os.path.join(OUTPUT_DIR, f'qwen3.5-397B_per-point_think_{timestamp}.json')
        with open(out, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nmacro recall={macro:.2%}  total={total_elapsed:.1f}s  saved -> {out}")

    asyncio.run(main())
