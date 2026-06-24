"""Embedding-based duplicate checker for ReviewLog entries.

Detects duplicate memory entries using cosine similarity of embeddings.
Used to reject near-duplicate claims, questions, notes, and outline items
at insertion time, providing immediate feedback via mem_error penalty.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingDuplicateChecker:
    """Check for duplicate entries using embedding cosine similarity.

    Maintains a cache of embeddings per entry type (claim, question, note, etc.)
    and compares new entries against existing ones. If similarity exceeds the
    threshold, the entry is rejected.
    """

    def __init__(self, embed_model: str = "qwen3-embedding-8b", threshold: float = 0.85):
        """Initialize the duplicate checker.

        Args:
            embed_model: Model name for embedding (uses MODEL_CONFIGS from utils.helpers.llm)
            threshold: Cosine similarity threshold for duplicate detection (default: 0.85)
        """
        self.embed_model = embed_model
        self.threshold = threshold
        self._client = None
        self._model_name = None
        self._cache: Dict[str, List[np.ndarray]] = {}  # type -> list of embeddings
        self._texts: Dict[str, List[str]] = {}  # type -> list of texts (for error messages)

    def _get_client(self):
        """Lazily initialize the OpenAI embedding client."""
        if self._client is None:
            from openai import OpenAI
            from utils.helpers.llm import MODEL_CONFIGS

            config = MODEL_CONFIGS.get(self.embed_model, {})
            base_url = config.get("base_url", "http://localhost:8000/v1")
            api_key = config.get("api_key", "EMPTY")
            self._model_name = config.get("model", self.embed_model).removeprefix("openai/")
            self._client = OpenAI(base_url=base_url, api_key=api_key)
            logger.info(f"Initialized embedding client for {self._model_name} at {base_url}")
        return self._client

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single text and L2-normalize.

        Args:
            text: Text to embed

        Returns:
            L2-normalized embedding vector
        """
        client = self._get_client()
        response = client.embeddings.create(model=self._model_name, input=[text])
        vec = np.array(response.data[0].embedding)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def check_and_register(
        self,
        new_text: str,
        entry_type: str
    ) -> Optional[Tuple[int, float, str]]:
        """Check if new_text is duplicate of existing entries of same type.

        Computes embedding for new_text and compares against all cached embeddings
        of the same type using cosine similarity. If max similarity >= threshold,
        returns duplicate info. Otherwise, caches the embedding and returns None.

        Args:
            new_text: Text to check for duplication
            entry_type: Type of entry (claim, question, note, strength, weakness, etc.)

        Returns:
            (index, similarity, existing_text_preview) if duplicate found, else None
        """
        vec = self._embed(new_text)
        existing = self._cache.get(entry_type, [])

        if existing:
            # Compute cosine similarities against all existing entries
            sims = np.array([float(vec @ e) for e in existing])
            max_idx = int(np.argmax(sims))
            max_sim = float(sims[max_idx])

            if max_sim >= self.threshold:
                preview = self._texts[entry_type][max_idx][:60]
                return (max_idx, max_sim, preview)

        # Not a duplicate -- cache it
        self._cache.setdefault(entry_type, []).append(vec)
        self._texts.setdefault(entry_type, []).append(new_text)
        return None

    def reset(self):
        """Clear all cached embeddings. Call at start of new episode."""
        self._cache.clear()
        self._texts.clear()
        logger.debug("Duplicate checker cache cleared")
