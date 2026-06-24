"""Cache management for paper downloads.

This module provides functionality to cache downloaded papers, track failures,
and avoid redundant downloads.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class CacheManager:
    """Manager for caching downloaded paper content."""

    def __init__(self, cache_dir: Path):
        """Initialize the cache manager.

        Cache structure:
            cache_dir/
                content/         # Processed paper content (markdown .md files)
                pdfs/            # Downloaded PDFs (intermediate)
                metadata.json    # Cache metadata

        Args:
            cache_dir: Root directory for cache
        """
        self.cache_dir = Path(cache_dir)
        self.content_dir = self.cache_dir / "content"
        self.pdfs_dir = self.cache_dir / "pdfs"
        self.latex_dir = self.cache_dir / "latex"
        self.metadata_file = self.cache_dir / "metadata.json"

        # Create directories
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.latex_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata from disk.

        Returns:
            Dictionary mapping paper_id to metadata
        """
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.debug(f"Loaded cache metadata: {len(metadata)} entries")
            return metadata
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            return {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            # Atomic write using temp file
            temp_file = self.metadata_file.with_suffix('.json.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            temp_file.replace(self.metadata_file)
            logger.debug(f"Saved cache metadata: {len(self.metadata)} entries")
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def get_cached_paper(self, paper_id: str) -> Optional[str]:
        """Get cached paper content if available.

        Args:
            paper_id: Paper ID

        Returns:
            Paper content as string, or None if not cached or failed
        """
        if paper_id not in self.metadata:
            return None

        entry = self.metadata[paper_id]

        # Don't return content for failed papers
        if not entry.get('success', False):
            return None

        # Read content from file
        file_path = entry.get('file_path')
        if not file_path:
            return None

        content_path = Path(file_path)
        if not content_path.exists():
            logger.warning(
                f"Cache metadata points to non-existent file: {file_path}"
            )
            return None

        try:
            content = content_path.read_text(encoding='utf-8', errors='ignore')
            logger.debug(f"Cache hit for {paper_id}")
            return content
        except Exception as e:
            logger.error(f"Error reading cached content for {paper_id}: {e}")
            return None

    def cache_paper(
        self,
        paper_id: str,
        content: str,
        source: str,
        format_type: str = "unknown",
        arxiv_id: Optional[str] = None
    ):
        """Cache paper content with metadata.

        Args:
            paper_id: Paper ID
            content: Paper content (LaTeX or markdown)
            source: Source of content ("arxiv" or "openreview")
            format_type: Content format ("latex" or "markdown")
            arxiv_id: arXiv ID if from arXiv
        """
        # Determine file extension
        ext = ".tex" if format_type == "latex" else ".md"
        content_path = self.content_dir / f"{paper_id}{ext}"

        try:
            # Write content to file
            content_path.write_text(content, encoding='utf-8')

            # Update metadata
            self.metadata[paper_id] = {
                "source": source,
                "arxiv_id": arxiv_id,
                "format": format_type,
                "downloaded_at": datetime.now().isoformat(),
                "file_path": str(content_path),
                "success": True
            }

            self._save_metadata()

            logger.info(
                f"Cached paper {paper_id} from {source} "
                f"({len(content)} chars, {format_type})"
            )

        except Exception as e:
            logger.error(f"Error caching paper {paper_id}: {e}")

    def mark_failed(self, paper_id: str, reason: str, error_type: str = "other"):
        """Mark a paper as failed to avoid retrying.

        Args:
            paper_id: Paper ID
            reason: Reason for failure
            error_type: Type of error ("not_found", "rate_limit", "forbidden",
                       "timeout", "other")
        """
        from pathlib import Path as PathLib

        # Create a consistent failed path for reference
        failed_dir = self.cache_dir / "failed"
        failed_dir.mkdir(parents=True, exist_ok=True)
        failed_path = failed_dir / f"{paper_id}.txt"

        # Write error information to file
        try:
            failed_path.write_text(
                f"Failed to fetch paper {paper_id}\n"
                f"Error type: {error_type}\n"
                f"Reason: {reason}\n"
                f"Attempted at: {datetime.now().isoformat()}\n",
                encoding='utf-8'
            )
        except Exception as e:
            logger.warning(f"Could not write failed marker file for {paper_id}: {e}")

        self.metadata[paper_id] = {
            "success": False,
            "error": reason,
            "error_type": error_type,
            "attempted_at": datetime.now().isoformat(),
            "file_path": str(failed_path)
        }

        self._save_metadata()

        logger.debug(f"Marked {paper_id} as failed: {reason} (type: {error_type})")

    def is_failed(self, paper_id: str) -> bool:
        """Check if a paper previously failed and should not be retried.

        Args:
            paper_id: Paper ID

        Returns:
            True if paper previously failed with a permanent error (should skip retry),
            False if not failed, successful, or failed with rate_limit (should retry)
        """
        if paper_id not in self.metadata:
            return False

        entry = self.metadata[paper_id]
        
        # If successful, not failed
        if entry.get('success', False):
            return False
        
        # If failed, check error type
        error_type = entry.get('error_type', 'other')
        
        # Rate limit errors should be retried, so return False (not failed for retry purposes)
        if error_type == 'rate_limit':
            return False
        
        # Other errors are permanent and should not be retried
        return True

    def get_paper_metadata(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a cached paper.

        Args:
            paper_id: Paper ID

        Returns:
            Metadata dictionary, or None if not cached
        """
        return self.metadata.get(paper_id)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with statistics:
                - total: Total entries in cache
                - successful: Successful downloads
                - failed: Failed downloads
                - arxiv: Papers from arXiv
                - openreview: Papers from OpenReview
        """
        total = len(self.metadata)
        successful = sum(
            1 for entry in self.metadata.values()
            if entry.get('success', False)
        )
        failed = total - successful
        arxiv = sum(
            1 for entry in self.metadata.values()
            if entry.get('source') == 'arxiv'
        )
        openreview = sum(
            1 for entry in self.metadata.values()
            if entry.get('source') == 'openreview'
        )

        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'arxiv': arxiv,
            'openreview': openreview
        }

    def clear_failed(self):
        """Clear failed entries from cache to allow retry."""
        failed_count = 0
        for paper_id in list(self.metadata.keys()):
            if not self.metadata[paper_id].get('success', False):
                del self.metadata[paper_id]
                failed_count += 1

        if failed_count > 0:
            self._save_metadata()
            logger.info(f"Cleared {failed_count} failed entries from cache")
