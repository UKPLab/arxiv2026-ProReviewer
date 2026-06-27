"""OpenReview API client for downloading papers.

This module provides functionality to download PDFs and metadata from
OpenReview using their public API v2.
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class OpenReviewClient:
    """Client for interacting with the OpenReview API v2."""

    BASE_URL = "https://api2.openreview.net"

    def __init__(self, timeout: int = 60, max_retries: int = 3):
        """Initialize the OpenReview client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.timeout = timeout

        # Create session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": "ProReviewer-Paper-Fetcher/1.0"
        })

    def get_paper_pdf(self, paper_id: str) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        """Download PDF for an OpenReview paper.

        Args:
            paper_id: OpenReview paper ID (e.g., "HE9eUQlAvo")

        Returns:
            Tuple of (pdf_bytes, error_type, error_message)
            - pdf_bytes: PDF content as bytes, or None if download fails
            - error_type: "not_found", "forbidden", "rate_limit", "timeout", "other", or None if successful
            - error_message: Error message, or None if successful
        """
        pdf_url = f"{self.BASE_URL}/pdf?id={paper_id}"

        logger.debug(f"Downloading PDF for {paper_id}")

        try:
            response = self.session.get(pdf_url, timeout=self.timeout)
            response.raise_for_status()

            # Verify we got a PDF
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower():
                logger.warning(
                    f"Unexpected content type for {paper_id}: {content_type}"
                )

            logger.info(
                f"Successfully downloaded PDF for {paper_id} "
                f"({len(response.content) / 1024:.1f} KB)"
            )

            return (response.content, None, None)

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(
                    f"Paper {paper_id} not found (404) - "
                    "might not be public or doesn't exist"
                )
                return (None, "not_found", f"Paper {paper_id} not found (404) - might not be public or doesn't exist")
            elif e.response.status_code == 403:
                logger.warning(
                    f"Access denied for {paper_id} (403) - "
                    "paper might be withdrawn or restricted"
                )
                return (None, "forbidden", f"Access denied for {paper_id} (403) - paper might be withdrawn or restricted")
            elif e.response.status_code == 429:
                logger.warning(
                    f"Rate limit hit for {paper_id} (429 Too Many Requests)"
                )
                return (None, "rate_limit", f"Rate limit hit for {paper_id} (429 Too Many Requests)")
            else:
                logger.error(
                    f"HTTP error downloading PDF for {paper_id}: "
                    f"{e.response.status_code}"
                )
                return (None, "other", f"HTTP error {e.response.status_code} downloading PDF")

        except requests.Timeout:
            logger.error(f"Timeout downloading PDF for {paper_id}")
            return (None, "timeout", f"Timeout downloading PDF for {paper_id}")

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                logger.error(f"Rate limit error downloading PDF for {paper_id}: {e}")
                return (None, "rate_limit", f"Rate limit error: {e}")
            else:
                logger.error(f"Error downloading PDF for {paper_id}: {e}")
                return (None, "other", f"Error downloading PDF: {e}")

    def get_paper_metadata(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an OpenReview paper.

        Args:
            paper_id: OpenReview paper ID (e.g., "HE9eUQlAvo")

        Returns:
            Dictionary with paper metadata, or None if request fails
        """
        notes_url = f"{self.BASE_URL}/notes?id={paper_id}"

        logger.debug(f"Fetching metadata for {paper_id}")

        try:
            response = self.session.get(notes_url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # OpenReview API returns a list of notes
            if not data.get('notes'):
                logger.warning(f"No notes found for {paper_id}")
                return None

            note = data['notes'][0]

            # Extract relevant metadata
            content = note.get('content', {})
            metadata = {
                'id': note.get('id'),
                'title': content.get('title', ''),
                'authors': content.get('authors', []),
                'abstract': content.get('abstract', ''),
                'keywords': content.get('keywords', []),
                'venue': content.get('venue', ''),
                'venueid': content.get('venueid', ''),
                'cdate': note.get('cdate'),  # Creation timestamp
                'mdate': note.get('mdate'),  # Modification timestamp
            }

            logger.info(f"Retrieved metadata for {paper_id}: {metadata['title']}")

            return metadata

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Paper {paper_id} not found (404)")
            else:
                logger.error(
                    f"HTTP error fetching metadata for {paper_id}: "
                    f"{e.response.status_code}"
                )
            return None

        except requests.Timeout:
            logger.error(f"Timeout fetching metadata for {paper_id}")
            return None

        except Exception as e:
            logger.error(f"Error fetching metadata for {paper_id}: {e}")
            return None

    def get_paper_reviews(self, paper_id: str) -> Tuple[Optional[list], Optional[str], Optional[str]]:
        """Get reviews for an OpenReview paper.

        Fetches all official reviews for a paper by querying notes with review invitations.

        Args:
            paper_id: OpenReview paper ID (e.g., "HE9eUQlAvo")

        Returns:
            Tuple of (reviews_list, error_type, error_message)
            - reviews_list: List of review dictionaries, or None if fetch fails
            - error_type: "not_found", "forbidden", "rate_limit", "timeout", "other", or None if successful
            - error_message: Error message, or None if successful
        """
        # OpenReview API v2 uses GET requests with query parameters
        reviews_url = f"{self.BASE_URL}/notes"

        logger.debug(f"Fetching reviews for {paper_id}")

        try:
            # Query notes in the forum (paper_id is the forum ID)
            # OpenReview API v2 uses GET with query parameters
            params = {
                'forum': paper_id,
                'details': 'replyCount,invitation,original'
            }

            response = self.session.get(
                reviews_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            notes = data.get('notes', [])
            
            # Filter for Official_Review notes.
            # OpenReview API v2 may return the invitation in either:
            #   - note['invitation']  (older conferences, single string)
            #   - note['invitations'] (newer conferences, list of strings)
            reviews = []
            all_invitations = []

            for note in notes:
                try:
                    # Collect all invitation strings for this note
                    inv_single = note.get('invitation', '') or ''
                    inv_list   = note.get('invitations', []) or []
                    note_invitations = ([inv_single] if inv_single else []) + list(inv_list)
                    all_invitations.extend(note_invitations)

                    content = note.get('content', {})

                    # A note is a reviewer review iff one of its invitations
                    # contains 'Official_Review' (and not 'Meta_Review').
                    is_review = any(
                        'Official_Review' in inv and 'Meta_Review' not in inv
                        for inv in note_invitations
                    )

                    if not is_review:
                        continue

                    # Extract the reviewer's short ID from the signatures.
                    # Signature format: ".../Reviewer_XXXX"
                    reviewer_id = note.get('id')  # fallback to note ID
                    for sig in (note.get('signatures') or []):
                        if 'Reviewer_' in sig:
                            reviewer_id = sig.split('Reviewer_')[-1]
                            break

                    def _c(field):
                        val = content.get(field, '') if isinstance(content, dict) else ''
                        return val.get('value', '') if isinstance(val, dict) else (val or '')

                    review_data = {
                        'id':             reviewer_id,
                        'invitation':     inv_single,
                        'cdate':          note.get('cdate'),
                        'mdate':          note.get('mdate'),
                        'rating':         _c('rating'),
                        'confidence':     _c('confidence'),
                        'soundness':      _c('soundness'),
                        'contribution':   _c('contribution'),
                        'presentation':   _c('presentation'),
                        'summary':        _c('summary'),
                        'strengths':      _c('strengths'),
                        'weaknesses':     _c('weaknesses'),
                        'questions':      _c('questions'),
                        'limitations':    _c('limitations'),
                        'recommendation': _c('recommendation'),
                        'review':         _c('review'),
                        'title':          _c('title'),
                    }
                    reviews.append(review_data)
                except Exception as e:
                    logger.warning(f"Error processing note for {paper_id}: {e}")
                    continue

            # Log if no reviews found but notes exist (for debugging)
            if not reviews and all_invitations:
                unique_invitations = list(set(all_invitations))
                logger.debug(
                    f"No reviews found for {paper_id}. Found {len(notes)} note(s) with invitations: "
                    f"{', '.join(unique_invitations[:5])}{'...' if len(unique_invitations) > 5 else ''}"
                )
            
            logger.info(f"Retrieved {len(reviews)} review(s) for {paper_id}")
            
            return (reviews, None, None)
            
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Paper {paper_id} not found (404) - cannot fetch reviews")
                return (None, "not_found", f"Paper {paper_id} not found (404) - cannot fetch reviews")
            elif e.response.status_code == 403:
                logger.warning(
                    f"Access denied for {paper_id} (403) - "
                    "reviews might not be public or paper is restricted"
                )
                return (None, "forbidden", f"Access denied for {paper_id} (403) - reviews might not be public")
            elif e.response.status_code == 429:
                logger.warning(
                    f"Rate limit hit for {paper_id} (429 Too Many Requests)"
                )
                return (None, "rate_limit", f"Rate limit hit for {paper_id} (429 Too Many Requests)")
            else:
                logger.error(
                    f"HTTP error fetching reviews for {paper_id}: "
                    f"{e.response.status_code}"
                )
                return (None, "other", f"HTTP error {e.response.status_code} fetching reviews")
                
        except requests.Timeout:
            logger.error(f"Timeout fetching reviews for {paper_id}")
            return (None, "timeout", f"Timeout fetching reviews for {paper_id}")
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                logger.error(f"Rate limit error fetching reviews for {paper_id}: {e}")
                return (None, "rate_limit", f"Rate limit error: {e}")
            else:
                logger.error(f"Error fetching reviews for {paper_id}: {e}")
                return (None, "other", f"Error fetching reviews: {e}")
