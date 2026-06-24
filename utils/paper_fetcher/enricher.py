"""Paper enricher for orchestrating the fetching pipeline.

This module coordinates all components to fetch paper content from arXiv
(converted to markdown using arxiv2md) and OpenReview (PDFs converted to
markdown), with caching and concurrent processing.
"""

import asyncio
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from difflib import SequenceMatcher

from tqdm import tqdm

from .arxiv_client import ArxivClient
from .openreview_client import OpenReviewClient
from .pdf_processor import MinerUProcessor
from .cache_manager import CacheManager
from .conference_config import is_arxiv_valid_for_review, get_rebuttal_date


logger = logging.getLogger(__name__)


def _normalize_name(s: str) -> str:
    """Normalize a name token: lowercase, strip diacritics, remove punctuation."""
    # NFD decompose then drop combining (accent) characters
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.lower().replace('.', '').replace(' ', '')


def parse_author_name(name: str) -> Dict[str, str]:
    """Parse an author name into components (first, middle, last).

    Handles formats like:
    - "John Smith" -> {first: "john", middle: "", last: "smith"}
    - "John A. Smith" -> {first: "john", middle: "a", last: "smith"}
    - "John Andrew Smith" -> {first: "john", middle: "andrew", last: "smith"}
    - "J. A. Smith" -> {first: "j", middle: "a", last: "smith"}
    - "Jean-Claude Dubois" -> {first: "jean-claude", middle: "", last: "dubois"}
    - "José García" -> {first: "jose", middle: "", last: "garcia"}

    Args:
        name: Author name string

    Returns:
        Dictionary with 'first', 'middle', and 'last' keys (ASCII-normalized)
    """
    # Remove extra whitespace and punctuation (except periods, hyphens, and apostrophes)
    name = re.sub(r'[^\w\s.\'-]', ' ', name)
    name = ' '.join(name.split())

    # Split into tokens
    tokens = name.split()

    if not tokens:
        return {'first': '', 'middle': '', 'last': ''}

    last_name = tokens[-1]
    first_name = tokens[0]
    middle_names = tokens[1:-1] if len(tokens) > 2 else []
    middle_name = ' '.join(middle_names) if middle_names else ''

    return {
        'first':  _normalize_name(first_name),
        'middle': _normalize_name(middle_name) if middle_name else '',
        'last':   _normalize_name(last_name),
    }


def names_match(name1: str, name2: str) -> bool:
    """Check if two author names match, handling abbreviations.
    
    Matches if:
    - First names match (or one is an initial of the other)
    - Last names match
    - Middle names match (or one is an abbreviation of the other, or one is missing)
    
    Args:
        name1: First author name
        name2: Second author name
        
    Returns:
        True if names match, False otherwise
    """
    parsed1 = parse_author_name(name1)
    parsed2 = parse_author_name(name2)
    
    # Last names must match exactly
    if parsed1['last'] != parsed2['last']:
        return False
    
    # First names must match (or one is an initial of the other)
    first1 = parsed1['first']
    first2 = parsed2['first']
    
    if first1 == first2:
        first_match = True
    elif len(first1) == 1 and first2.startswith(first1):
        # first1 is an initial of first2
        first_match = True
    elif len(first2) == 1 and first1.startswith(first2):
        # first2 is an initial of first1
        first_match = True
    else:
        first_match = False
    
    if not first_match:
        return False
    
    # Middle names: match if:
    # 1. Both are empty
    # 2. One is empty (middle name optional)
    # 3. One is an abbreviation/initial of the other
    # 4. They match exactly
    middle1 = parsed1['middle']
    middle2 = parsed2['middle']
    
    if not middle1 and not middle2:
        # Both empty - match
        return True
    elif not middle1 or not middle2:
        # One is empty - still match (middle name is optional)
        return True
    else:
        # Both have middle names - check if they match
        # Split middle names (could be multiple)
        mids1 = [m.replace('.', '') for m in middle1.split()]
        mids2 = [m.replace('.', '') for m in middle2.split()]
        
        # If different number of middle names, try to match initials
        if len(mids1) == len(mids2):
            # Same number of middle names - check each pair
            for m1, m2 in zip(mids1, mids2):
                if m1 == m2:
                    continue
                elif len(m1) == 1 and m2.startswith(m1):
                    continue  # m1 is initial of m2
                elif len(m2) == 1 and m1.startswith(m2):
                    continue  # m2 is initial of m1
                else:
                    return False
            return True
        else:
            # Different number of middle names - check if one set of initials matches the other
            # Convert to initials
            initials1 = ''.join([m[0] for m in mids1 if m])
            initials2 = ''.join([m[0] for m in mids2 if m])
            
            if initials1 == initials2 or (len(initials1) == 1 and initials2.startswith(initials1)) or (len(initials2) == 1 and initials1.startswith(initials2)):
                return True
            
            # Also check if one set of middle names contains the other as substrings
            mids1_str = ' '.join(mids1)
            mids2_str = ' '.join(mids2)
            if mids1_str in mids2_str or mids2_str in mids1_str:
                return True
            
            return False


def compare_author_lists(authors1: List[str], authors2: List[str]) -> float:
    """Compare two author lists and return overlap ratio (0.0 to 1.0).
    
    Uses name component matching to handle abbreviations and variations.
    
    Args:
        authors1: First list of author names
        authors2: Second list of author names
        
    Returns:
        Overlap ratio between 0.0 and 1.0
    """
    if not authors1 or not authors2:
        return 0.0
    
    # Match each author in list1 with authors in list2
    matched_indices = set()
    matches = 0
    
    for author1 in authors1:
        for idx, author2 in enumerate(authors2):
            if idx in matched_indices:
                continue
            if names_match(author1, author2):
                matches += 1
                matched_indices.add(idx)
                break
    
    # Calculate overlap ratio
    max_len = max(len(authors1), len(authors2))
    if max_len == 0:
        return 0.0
    
    overlap_ratio = matches / max_len
    return overlap_ratio


def clean_title(title: str) -> str:
    """Clean a title for comparison."""
    # Remove LaTeX commands
    title = re.sub(r'\\[vh]space\{[^}]*\}', ' ', title)
    title = re.sub(r'\\raisebox\{[^}]*\}\{[^}]*\}', ' ', title)
    title = re.sub(r'\\includegraphics\[[^\]]*\]\{[^}]*\}', ' ', title)
    title = re.sub(r'\\includegraphics\{[^}]*\}', ' ', title)
    title = re.sub(r'\\\\', ' ', title)

    prev_title = None
    while prev_title != title:
        prev_title = title
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)

    title = re.sub(r'\\[a-zA-Z]+', '', title)
    title = re.sub(r'[^\w\s-]', ' ', title)
    title = ' '.join(title.split())

    return title.strip()


def title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles."""
    clean1 = clean_title(title1).lower()
    clean2 = clean_title(title2).lower()
    return SequenceMatcher(None, clean1, clean2).ratio()


class PaperEnricher:
    """Orchestrates the paper enrichment pipeline."""

    def __init__(
        self,
        arxiv_client: ArxivClient,
        openreview_client: OpenReviewClient,
        pdf_processor: Optional[MinerUProcessor],
        cache_manager: CacheManager,
        conference: str,
        arxiv_only: bool = False,
        fetch_reviews: bool = False,
        max_paper_age_days: int = 180
    ):
        """Initialize the paper enricher.

        Args:
            arxiv_client: Client for arXiv API
            openreview_client: Client for OpenReview API
            pdf_processor: PDF processor (MinerU + PyMuPDF), or None to disable
            cache_manager: Cache manager
            conference: Conference name (e.g., "ICLR 2024")
            arxiv_only: If True, skip OpenReview fallback
            fetch_reviews: If True, fetch reviews from OpenReview (default: False)
            max_paper_age_days: Maximum age of arXiv versions in days from cutoff (default: 180)
        """
        self.arxiv_client = arxiv_client
        self.openreview_client = openreview_client
        self.pdf_processor = pdf_processor
        self.cache_manager = cache_manager
        self.conference = conference
        self.arxiv_only = arxiv_only
        self.fetch_reviews = fetch_reviews
        self.max_paper_age_days = max_paper_age_days

        # Create conference-specific directory
        self.conference_folder = self._sanitize_conference_name(conference)
        self.conference_content_dir = self.cache_manager.content_dir / self.conference_folder
        self.conference_content_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using conference-specific directory: {self.conference_content_dir}")
        if not fetch_reviews:
            logger.info("Review fetching is disabled. Use scripts/add_reviews_to_enriched.py to add reviews separately.")
        if max_paper_age_days == 0:
            logger.info("Paper age restriction is disabled - will accept any version before cutoff date")
        else:
            logger.info(f"Maximum paper age: {max_paper_age_days} days from cutoff date")

    def _sanitize_conference_name(self, conference: str) -> str:
        """Sanitize conference name for use as directory name.

        Args:
            conference: Conference name (e.g., "ICLR 2024")

        Returns:
            Sanitized name suitable for directory (e.g., "iclr2024")
        """
        # Convert to lowercase and remove special characters
        sanitized = re.sub(r'[^\w\s-]', '', conference.lower())
        # Replace spaces with nothing
        sanitized = re.sub(r'\s+', '', sanitized)
        return sanitized

    async def enrich_paper(self, paper_data: Dict) -> Dict:
        """Enrich a single paper with content.

        Algorithm:
        1. Check cache
        2. Try arXiv (search by title, validate date, convert to markdown with arxiv2md)
        3. Fallback to OpenReview (download PDF, convert to markdown)
        4. Cache result

        Args:
            paper_data: Paper metadata dictionary

        Returns:
            Enriched paper dictionary with paper_content field
        """
        paper_id = paper_data.get('id', 'unknown')
        title = paper_data.get('title', '')

        # Check cache first
        cached_meta = self.cache_manager.get_paper_metadata(paper_id)
        if cached_meta is not None and cached_meta.get('success', False):
            cached_path = cached_meta.get('file_path')
            if cached_path:
                logger.info(f"Using cached content for {paper_id}")
                result = {
                    **paper_data,
                    "paper_content": cached_path,
                    "content_source": cached_meta.get('source'),
                    "arxiv_id": cached_meta.get('arxiv_id'),
                    "content_format": cached_meta.get('format'),
                    "content_fetched_at": cached_meta.get('downloaded_at')
                }
                # Optionally fetch reviews even if content is cached (reviews might not be cached)
                if self.fetch_reviews:
                    reviews, review_status = await self._fetch_reviews(paper_id)
                    if reviews is not None:
                        result['reviews'] = reviews
                    result['reviews_fetch_status'] = review_status
                return result

        # Check if previously failed (and should not be retried)
        if cached_meta and not cached_meta.get('success', False):
            error_type = cached_meta.get('error_type', 'other')
            
            # Rate limit errors should be retried, so continue processing
            if error_type == 'rate_limit':
                logger.info(
                    f"Retrying paper {paper_id}: "
                    f"{cached_meta.get('error', 'Unknown error')}"
                )
            # Other permanent errors should be skipped
            else:
                failed_path = cached_meta.get('file_path') if cached_meta.get('file_path') else str(Path(".cache/papers/failed") / paper_id)
                logger.debug(
                    f"Skipping previously failed paper {paper_id}: "
                    f"{cached_meta.get('error', 'Unknown error')} (type: {error_type})"
                )
                return {
                    **paper_data,
                    "paper_content": failed_path,
                    "content_source": None,
                    "fetch_error": cached_meta.get('error', 'Unknown error'),
                    "error_type": error_type,
                    "content_fetched_at": cached_meta.get('attempted_at', datetime.now().isoformat())
                }

        # Try arXiv path
        arxiv_result = await self._try_arxiv(paper_id, paper_data)
        if arxiv_result is not None:
            # Optionally fetch reviews from OpenReview
            if self.fetch_reviews:
                reviews, review_status = await self._fetch_reviews(paper_id)
                if reviews is not None:
                    arxiv_result['reviews'] = reviews
                arxiv_result['reviews_fetch_status'] = review_status
            return {**paper_data, **arxiv_result}

        # Check if arXiv attempt resulted in an error
        arxiv_error_meta = self.cache_manager.get_paper_metadata(paper_id)
        arxiv_error_type = None
        if arxiv_error_meta and not arxiv_error_meta.get('success'):
            arxiv_error_type = arxiv_error_meta.get('error_type')

        # Fallback to OpenReview path (if not arxiv_only)
        if not self.arxiv_only:
            openreview_result = await self._try_openreview(paper_id, title)
            if openreview_result is not None:
                # Optionally fetch reviews from OpenReview
                if self.fetch_reviews:
                    reviews, review_status = await self._fetch_reviews(paper_id)
                    if reviews is not None:
                        openreview_result['reviews'] = reviews
                    openreview_result['reviews_fetch_status'] = review_status
                return {**paper_data, **openreview_result}

        # Both methods failed - determine final error type
        # Check current metadata (may have been updated by OpenReview attempt)
        current_meta = self.cache_manager.get_paper_metadata(paper_id)
        
        if current_meta and not current_meta.get('success'):
            # Already marked as failed
            final_error_type = current_meta.get('error_type', 'not_found')
            final_error_msg = current_meta.get('error', 'Unknown error')
            
            # If arXiv had a more specific error type than OpenReview, prefer it
            if arxiv_error_type and arxiv_error_type != final_error_type:
                # Prefer more specific error types (rate_limit > forbidden > timeout > not_found > other)
                error_priority = {"rate_limit": 4, "forbidden": 3, "timeout": 2, "not_found": 1, "other": 0}
                arxiv_priority = error_priority.get(arxiv_error_type, 0)
                current_priority = error_priority.get(final_error_type, 0)
                
                if arxiv_priority > current_priority:
                    # Update with arXiv error type (more specific)
                    self.cache_manager.mark_failed(paper_id, current_meta.get('error', ''), arxiv_error_type)
                    final_error_type = arxiv_error_type
        else:
            # Not yet marked as failed - determine error type
            error_type = "not_found"
            error_msg = "Not found on arXiv or OpenReview"
            
            if self.arxiv_only:
                error_msg = "Not found on arXiv (OpenReview fallback disabled)"
                error_type = arxiv_error_type or "not_found"
            else:
                # Both failed - check what OpenReview reported
                openreview_error_meta = self.cache_manager.get_paper_metadata(paper_id)
                openreview_error_type = None
                if openreview_error_meta and not openreview_error_meta.get('success'):
                    openreview_error_type = openreview_error_meta.get('error_type')
                
                # Prefer more specific error type
                error_types = [t for t in [arxiv_error_type, openreview_error_type] if t]
                if "rate_limit" in error_types:
                    error_type = "rate_limit"
                    error_msg = "Rate limit error on arXiv or OpenReview"
                elif "forbidden" in error_types:
                    error_type = "forbidden"
                    error_msg = "Access denied on arXiv or OpenReview"
                elif "timeout" in error_types:
                    error_type = "timeout"
                    error_msg = "Timeout error on arXiv or OpenReview"
                elif "other" in error_types:
                    error_type = "other"
                    error_msg = "Error fetching from arXiv or OpenReview"
                else:
                    error_type = "not_found"
                    error_msg = "Not found on arXiv or OpenReview"
            
            # Mark as failed with determined error type
            self.cache_manager.mark_failed(paper_id, error_msg, error_type)
            final_error_type = error_type
            final_error_msg = error_msg
            
            # Update current_meta for path retrieval
            current_meta = self.cache_manager.get_paper_metadata(paper_id)

        failed_path = current_meta.get('file_path') if current_meta else str(Path(".cache/papers/failed") / paper_id)

        logger.error(f"Failed to fetch paper {paper_id}: {final_error_msg} (type: {final_error_type})")

        return {
            **paper_data,
            "paper_content": failed_path,
            "content_source": None,
            "fetch_error": final_error_msg,
            "error_type": final_error_type,
            "content_fetched_at": datetime.now().isoformat()
        }

    async def _try_arxiv(
        self,
        paper_id: str,
        paper_data: Dict
    ) -> Optional[Dict]:
        """Try to fetch paper from arXiv.

        Args:
            paper_id: Paper ID
            paper_data: Paper metadata dictionary (must contain 'title' and optionally 'author')

        Returns:
            Dict with paper_content and metadata, or None if failed
        """
        title = paper_data.get('title', '')
        if not title:
            logger.debug(f"No title for {paper_id}, skipping arXiv")
            return None

        # Extract authors from paper_data (semicolon-separated string)
        author_str = paper_data.get('author', '')
        paper_authors = [a.strip() for a in author_str.split(';') if a.strip()] if author_str else []

        # Search arXiv by title
        results, search_error_type, search_error_msg = self.arxiv_client.search_by_title(title, max_results=3)
        
        if not results:
            logger.debug(f"No arXiv results for '{title}'" + (f": {search_error_msg}" if search_error_msg else ""))
            # Only mark as failed for specific errors (rate_limit, etc.)
            # "not_found" will be handled at enrich_paper level
            if search_error_type and search_error_type != "not_found":
                error_msg = search_error_msg or f"Error searching arXiv for '{title}'"
                # Check if there's already an error type
                existing_meta = self.cache_manager.get_paper_metadata(paper_id)
                if existing_meta and not existing_meta.get('success'):
                    existing_error_type = existing_meta.get('error_type')
                    # If existing error is rate_limit, always update (retrying temporary error)
                    # Otherwise, only update if new error has higher priority
                    if existing_error_type == 'rate_limit':
                        self.cache_manager.mark_failed(paper_id, error_msg, search_error_type)
                    else:
                        error_priority = {"rate_limit": 4, "forbidden": 3, "timeout": 2, "not_found": 1, "other": 0}
                        existing_priority = error_priority.get(existing_error_type, 0)
                        new_priority = error_priority.get(search_error_type, 0)
                        if new_priority > existing_priority:
                            self.cache_manager.mark_failed(paper_id, error_msg, search_error_type)
                else:
                    self.cache_manager.mark_failed(paper_id, error_msg, search_error_type)
            return None

        # Try each result until we find one that passes verification
        best_match = None
        for result in results:
            # Verify title and authors
            arxiv_title = result.title
            arxiv_authors = result.authors

            title_sim = title_similarity(title, arxiv_title)
            authors_overlap = compare_author_lists(paper_authors, arxiv_authors) if paper_authors else 0.0

            # Matching criteria: title_sim >= 0.85 and authors_overlap >= 0.8
            # If no authors in paper_data, only check title
            if paper_authors:
                is_match = title_sim >= 0.85 and authors_overlap >= 0.8
            else:
                # If no authors available, only verify title
                is_match = title_sim >= 0.85
                logger.debug(
                    f"No authors in paper_data for {paper_id}, "
                    f"verifying only title (similarity: {title_sim:.2%})"
                )

            if is_match:
                best_match = result
                logger.info(
                    f"Verified match for {paper_id}: "
                    f"title_similarity={title_sim:.2%}, "
                    f"authors_overlap={authors_overlap:.2%}"
                )
                break
            else:
                logger.debug(
                    f"Match verification failed for {paper_id} with {result.arxiv_id}: "
                    f"title_similarity={title_sim:.2%}, "
                    f"authors_overlap={authors_overlap:.2%}"
                )

        if best_match is None:
            logger.warning(
                f"No verified match found for {paper_id} among {len(results)} arXiv results. "
                f"Tried {len(results)} result(s), none passed verification."
            )
            return None

        # Get rebuttal deadline for this conference
        cutoff_date = get_rebuttal_date(self.conference)

        # Find the latest version posted before the rebuttal deadline
        valid_version = self.arxiv_client.get_version_before_date(
            best_match.arxiv_id,
            cutoff_date,
            max_age_days=self.max_paper_age_days
        )

        if valid_version is None:
            logger.warning(
                f"No versions of arXiv paper {best_match.arxiv_id} "
                f"found before rebuttal deadline "
                f"({cutoff_date.strftime('%Y-%m-%d')}), skipping"
            )
            return None

        # Convert to markdown using arxiv2md
        # Use conference-specific directory for temporary download
        markdown_dir = self.conference_content_dir
        markdown_content, conversion_error_type, conversion_error_msg = self.arxiv_client.download_markdown_with_arxiv2md(
            best_match.arxiv_id,
            markdown_dir,
            version=valid_version
        )

        if markdown_content is None:
            logger.debug(
                f"Failed to convert {best_match.arxiv_id}v{valid_version} to markdown"
                + (f": {conversion_error_msg}" if conversion_error_msg else "")
            )
            # Store specific error type if available (except "not_found" - handled at enrich_paper level)
            if conversion_error_type and conversion_error_type != "not_found":
                error_msg = conversion_error_msg or f"Failed to convert {best_match.arxiv_id}v{valid_version} to markdown"
                # Check if there's already an error type
                existing_meta = self.cache_manager.get_paper_metadata(paper_id)
                if existing_meta and not existing_meta.get('success'):
                    existing_error_type = existing_meta.get('error_type')
                    # If existing error is rate_limit, always update (retrying temporary error)
                    # Otherwise, only update if new error has higher priority
                    if existing_error_type == 'rate_limit':
                        self.cache_manager.mark_failed(paper_id, error_msg, conversion_error_type)
                    else:
                        error_priority = {"rate_limit": 4, "forbidden": 3, "timeout": 2, "not_found": 1, "other": 0}
                        existing_priority = error_priority.get(existing_error_type, 0)
                        new_priority = error_priority.get(conversion_error_type, 0)
                        if new_priority > existing_priority:
                            self.cache_manager.mark_failed(paper_id, error_msg, conversion_error_type)
                    # else: keep existing error type (already set)
                else:
                    self.cache_manager.mark_failed(paper_id, error_msg, conversion_error_type)
            return None

        # Save markdown content to conference-specific file
        arxiv_id_with_version = f"{best_match.arxiv_id}v{valid_version}"
        markdown_path = self.conference_content_dir / f"{paper_id}_arxiv_{arxiv_id_with_version}.md"
        markdown_path.write_text(markdown_content, encoding='utf-8')

        # Construct relative path starting with .cache/ for storage
        # This ensures paths in JSON are always relative, not absolute
        relative_cache_dir = Path(self.cache_manager.cache_dir).relative_to(Path.cwd()) if Path(self.cache_manager.cache_dir).is_absolute() else Path(self.cache_manager.cache_dir)
        markdown_path_str = str(relative_cache_dir / "content" / self.conference_folder / f"{paper_id}_arxiv_{arxiv_id_with_version}.md")

        # Store the path in metadata
        self.cache_manager.metadata[paper_id] = {
            "source": "arxiv",
            "arxiv_id": arxiv_id_with_version,
            "format": "markdown",
            "downloaded_at": datetime.now().isoformat(),
            "file_path": markdown_path_str,
            "success": True
        }
        self.cache_manager._save_metadata()

        logger.info(
            f"Successfully fetched {paper_id} from arXiv "
            f"({arxiv_id_with_version})"
        )

        return {
            "paper_content": markdown_path_str,
            "content_source": "arxiv",
            "arxiv_id": arxiv_id_with_version,
            "arxiv_version": valid_version,
            "content_format": "markdown",
            "content_fetched_at": datetime.now().isoformat()
        }

    async def _try_openreview(
        self,
        paper_id: str,
        title: str
    ) -> Optional[Dict]:
        """Try to fetch paper from OpenReview.

        Args:
            paper_id: Paper ID
            title: Paper title

        Returns:
            Dict with paper_content and metadata, or None if failed
        """
        # Download PDF
        pdf_bytes, error_type, error_msg = self.openreview_client.get_paper_pdf(paper_id)
        if pdf_bytes is None:
            logger.debug(f"Failed to download PDF for {paper_id}" + (f": {error_msg}" if error_msg else ""))
            # Only mark as failed for specific errors (not "not_found" - that will be handled at enrich_paper level)
            # This preserves more specific errors from arXiv (e.g., rate_limit)
            if error_type and error_type != "not_found":
                full_error_msg = error_msg or f"Failed to download PDF for {paper_id}"
                # Check if there's already an error type
                existing_meta = self.cache_manager.get_paper_metadata(paper_id)
                if existing_meta and not existing_meta.get('success'):
                    existing_error_type = existing_meta.get('error_type')
                    # If existing error is rate_limit, always update (retrying temporary error)
                    # Otherwise, prefer more specific error types (rate_limit > forbidden > timeout > not_found > other)
                    if existing_error_type == 'rate_limit':
                        self.cache_manager.mark_failed(paper_id, full_error_msg, error_type)
                    else:
                        error_priority = {"rate_limit": 4, "forbidden": 3, "timeout": 2, "not_found": 1, "other": 0}
                        existing_priority = error_priority.get(existing_error_type, 0)
                        new_priority = error_priority.get(error_type, 0)
                        if new_priority > existing_priority:
                            self.cache_manager.mark_failed(paper_id, full_error_msg, error_type)
                    # else: keep existing error type (already set)
                else:
                    self.cache_manager.mark_failed(paper_id, full_error_msg, error_type)
            return None

        # Save PDF to cache
        pdf_path = self.cache_manager.pdfs_dir / f"{paper_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)

        # Convert PDF to markdown
        if self.pdf_processor is None:
            logger.error("PDF processor is None, cannot convert PDF")
            self.cache_manager.mark_failed(paper_id, "PDF processor is None, cannot convert PDF", "other")
            return None

        markdown_output_dir = self.cache_manager.pdfs_dir / f"{paper_id}_md"
        markdown_content = self.pdf_processor.convert_pdf_to_markdown(
            pdf_path,
            markdown_output_dir
        )

        if markdown_content is None:
            logger.debug(f"Failed to convert PDF for {paper_id}")
            self.cache_manager.mark_failed(paper_id, f"Failed to convert PDF to markdown for {paper_id}", "other")
            return None

        # Write markdown to conference-specific file and store path
        markdown_path = self.conference_content_dir / f"{paper_id}.md"
        markdown_path.write_text(markdown_content, encoding='utf-8')

        # Construct relative path starting with .cache/ for storage
        # This ensures paths in JSON are always relative, not absolute
        relative_cache_dir = Path(self.cache_manager.cache_dir).relative_to(Path.cwd()) if Path(self.cache_manager.cache_dir).is_absolute() else Path(self.cache_manager.cache_dir)
        markdown_path_str = str(relative_cache_dir / "content" / self.conference_folder / f"{paper_id}.md")

        # Store metadata
        self.cache_manager.metadata[paper_id] = {
            "source": "openreview",
            "format": "markdown",
            "downloaded_at": datetime.now().isoformat(),
            "file_path": markdown_path_str,
            "success": True
        }
        self.cache_manager._save_metadata()

        logger.info(f"Successfully fetched {paper_id} from OpenReview")

        return {
            "paper_content": markdown_path_str,
            "content_source": "openreview",
            "content_format": "markdown",
            "content_fetched_at": datetime.now().isoformat()
        }

    async def _fetch_reviews(self, paper_id: str) -> Tuple[Optional[List[Dict]], Dict[str, Any]]:
        """Fetch reviews from OpenReview for a paper.

        Args:
            paper_id: Paper ID (OpenReview ID)

        Returns:
            Tuple of (reviews_list, status_dict)
            - reviews_list: List of review dictionaries, or None if fetch fails
            - status_dict: Dictionary with 'success', 'error_type', and 'error_message' keys
        """
        try:
            reviews, error_type, error_msg = self.openreview_client.get_paper_reviews(paper_id)

            if reviews is None:
                if error_type == "rate_limit":
                    # Rate limit - log but don't fail the whole paper
                    logger.warning(f"Rate limit when fetching reviews for {paper_id}, skipping reviews")
                elif error_type == "not_found":
                    logger.debug(f"No reviews found for {paper_id} (paper might not be on OpenReview)")
                elif error_type == "forbidden":
                    logger.debug(f"Reviews not accessible for {paper_id} (might be private)")
                else:
                    logger.debug(f"Failed to fetch reviews for {paper_id}: {error_msg}")

                return None, {
                    'success': False,
                    'error_type': error_type,
                    'error_message': error_msg
                }

            return reviews, {
                'success': True,
                'count': len(reviews)
            }

        except Exception as e:
            logger.warning(f"Error fetching reviews for {paper_id}: {e}")
            return None, {
                'success': False,
                'error_type': 'other',
                'error_message': str(e)
            }

    async def enrich_dataset(
        self,
        input_json_path: Path,
        output_json_path: Path,
        max_concurrent: int = 5,
        skip_failed: bool = True
    ) -> Dict:
        """Enrich all papers in a dataset file.

        Args:
            input_json_path: Path to input JSON file
            output_json_path: Path to output JSON file
            max_concurrent: Maximum concurrent downloads
            skip_failed: Skip previously failed papers

        Returns:
            Statistics dictionary
        """
        # Load input data
        logger.info(f"Loading papers from {input_json_path}")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)

        if not isinstance(papers, list):
            raise ValueError("Input JSON must be a list of papers")

        # Filter out papers with empty ratings
        original_count = len(papers)
        papers = [p for p in papers if p.get('rating', '') != '']
        skipped_count = original_count - len(papers)
        
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} paper(s) with empty ratings")
        
        logger.info(f"Processing {len(papers)} papers")

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        # Track statistics
        stats_lock = asyncio.Lock()
        cached_count = 0
        downloaded_count = 0
        failed_count = 0

        # Create progress bar
        pbar = tqdm(total=len(papers), desc="Fetching papers", unit="paper")

        async def process_paper(paper_data):
            nonlocal cached_count, downloaded_count, failed_count

            async with semaphore:
                try:
                    # Check if this paper is in cache (success or failure)
                    paper_id = paper_data.get('id', 'unknown')
                    cached_meta = self.cache_manager.get_paper_metadata(paper_id)
                    was_in_cache = cached_meta is not None  # Any cache hit (success or failed)

                    result = await self.enrich_paper(paper_data)

                    # Update counters based on result
                    async with stats_lock:
                        if was_in_cache:
                            # Paper was in cache (either previously succeeded or failed)
                            cached_count += 1
                        elif result.get('fetch_error') is not None or result.get('content_source') is None:
                            # Paper was NOT in cache and just failed
                            failed_count += 1
                        else:
                            # Paper was NOT in cache and just succeeded
                            downloaded_count += 1

                        # Update progress bar description
                        pbar.set_description(
                            f"Fetching papers [Cached: {cached_count}, Downloaded: {downloaded_count}, Failed: {failed_count}]"
                        )

                    return result
                finally:
                    pbar.update(1)

        # Process papers concurrently
        enriched_papers = await asyncio.gather(
            *[process_paper(paper) for paper in papers]
        )

        pbar.close()

        # Filter out failed papers (those with fetch_error)
        successful_papers = [
            p for p in enriched_papers
            if p.get('fetch_error') is None and p.get('content_source') is not None
        ]
        failed_papers = [
            p for p in enriched_papers
            if p.get('fetch_error') is not None or p.get('content_source') is None
        ]
        
        # Log summary of results
        if successful_papers:
            logger.info(f"✓ Successfully fetched {len(successful_papers)} paper(s)")
        if failed_papers:
            logger.warning(f"✗ Failed to fetch {len(failed_papers)} paper(s)")

        # Write output (only successful papers)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(successful_papers, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote {len(successful_papers)} successful papers to {output_json_path}")
        if failed_papers:
            logger.info(f"Skipped {len(failed_papers)} failed papers (not written to output)")

        # Calculate statistics
        total = len(enriched_papers)
        successful = len(successful_papers)
        failed = len(failed_papers)
        arxiv = sum(
            1 for p in enriched_papers
            if p.get('content_source') == 'arxiv'
        )
        openreview = sum(
            1 for p in enriched_papers
            if p.get('content_source') == 'openreview'
        )

        stats = {
            'total': total,
            'successful': successful,
            'failed': failed,
            'arxiv': arxiv,
            'openreview': openreview,
            'success_rate': f"{successful / total * 100:.1f}%" if total > 0 else "0%"
        }

        logger.info(f"Statistics: {stats}")

        return stats
