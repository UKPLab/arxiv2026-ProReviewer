"""arXiv API client for searching and downloading papers.

This module provides functionality to:
- Search arXiv by paper title with fuzzy matching
- Download LaTeX source files (.tar.gz)
- Extract and process LaTeX content
- Rate limit requests to comply with arXiv API guidelines
"""

import re
import time
import tarfile
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import requests

try:
    import arxiv
except ImportError:
    arxiv = None


logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Metadata for an arXiv paper."""

    arxiv_id: str
    title: str
    published_date: datetime
    pdf_url: str
    source_url: str
    authors: List[str]
    summary: str
    version: Optional[int] = None  # arXiv version number (v1, v2, etc.)
    updated_date: Optional[datetime] = None  # Date of this version


class ArxivClient:
    """Client for interacting with the arXiv API."""

    def __init__(self, rate_limit_delay: float = 5.0):
        """Initialize the arXiv client.

        Args:
            rate_limit_delay: Seconds to wait between requests (default: 5.0)
                             arXiv API guidelines recommend < 1 request/second;
                             export.arxiv.org enforces stricter limits.
        """
        if arxiv is None:
            raise ImportError(
                "arxiv package is required. Install with: pip install arxiv"
            )

        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Reviewer-R1-Paper-Fetcher/1.0"
        })

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _clean_title(self, title: str) -> str:
        """Clean a title for search purposes.

        Removes LaTeX commands, special characters, and normalizes whitespace.

        Args:
            title: Raw title string

        Returns:
            Cleaned title string
        """
        # Remove LaTeX commands
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)
        title = re.sub(r'\\[a-zA-Z]+', '', title)

        # Remove special characters but keep alphanumeric and spaces
        title = re.sub(r'[^\w\s-]', ' ', title)

        # Normalize whitespace
        title = ' '.join(title.split())

        return title.strip()

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles.

        Args:
            title1: First title
            title2: Second title

        Returns:
            Similarity score: 1.0 if titles match exactly, 0.0 otherwise
        """
        clean1 = self._clean_title(title1).lower()
        clean2 = self._clean_title(title2).lower()
        return 1.0 if clean1 == clean2 else 0.0

    def search_by_title(
        self,
        title: str,
        max_results: int = 5,
        max_retries: int = 5
    ) -> Tuple[List[ArxivPaper], Optional[str], Optional[str]]:
        """Search arXiv by paper title with retry logic.

        Searches arXiv using the paper title and returns results ranked by
        title similarity.

        Args:
            title: Paper title to search for
            max_results: Maximum number of results to return
            max_retries: Maximum number of retries on rate limit errors

        Returns:
            Tuple of (results_list, error_type, error_message)
            - results_list: List of ArxivPaper objects, or empty list on error
            - error_type: "rate_limit", "not_found", "other", or None if successful
            - error_message: Error message, or None if successful
        """
        self._rate_limit()

        logger.debug(f"Searching arXiv for: {title}")

        # Clean the title for search
        search_query = self._clean_title(title)

        for attempt in range(max_retries):
            try:
                # Search using arxiv library
                search = arxiv.Search(
                    query=f'ti:"{search_query}"',
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                results = []
                for result in search.results():
                    # Extract arxiv ID and version
                    entry_id = result.entry_id.split('/')[-1]
                    arxiv_id = entry_id.split('v')[0] if 'v' in entry_id else entry_id
                    version = int(entry_id.split('v')[1]) if 'v' in entry_id else None

                    paper = ArxivPaper(
                        arxiv_id=arxiv_id,
                        title=result.title,
                        published_date=result.published,  # Date of first version
                        pdf_url=result.pdf_url,
                        source_url=f"https://arxiv.org/e-print/{arxiv_id}",
                        authors=[author.name for author in result.authors],
                        summary=result.summary,
                        version=version,
                        updated_date=result.updated  # Date of latest version
                    )

                    # Calculate title similarity
                    similarity = self._title_similarity(title, result.title)
                    if similarity == 1.0:
                        return ([paper], None, None)
                    results.append(paper)

                # No exact match, return results or empty list
                if results:
                    return (results, None, None)
                else:
                    return ([], "not_found", f"No results found for '{title}'")

            except Exception as e:
                error_msg = str(e)

                # Check if it's a rate limit error (HTTP 429)
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff: 15s, 45s, 135s, 405s
                        backoff_time = 15 * (3 ** attempt)
                        logger.warning(
                            f"Rate limit hit for '{title}'. "
                            f"Retrying in {backoff_time}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(backoff_time)
                        continue
                    else:
                        logger.error(
                            f"Rate limit error after {max_retries} attempts for '{title}': {e}"
                        )
                        return ([], "rate_limit", f"Rate limit error after {max_retries} attempts: {e}")
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(f"Error searching arXiv for '{title}': {e}")
                    return ([], "other", f"Error searching arXiv: {e}")

        return ([], "other", "Unknown error during search")

    def get_version_before_date(
        self,
        arxiv_id: str,
        cutoff_date: datetime,
        max_age_days: int = 180
    ) -> Optional[int]:
        """Find the latest arXiv version posted before a cutoff date.

        The selected version must satisfy two conditions:
        1. It must be posted before the cutoff date (rebuttal deadline)
        2. It must be no more than max_age_days old from the cutoff date

        Args:
            arxiv_id: arXiv paper ID (without version, e.g., "2308.07074")
            cutoff_date: Cutoff date (timezone-aware)
            max_age_days: Maximum age in days from cutoff date (default: 180).
                         Use 0 to disable age check.

        Returns:
            Version number (e.g., 1, 2, 3) or None if no valid version found
        """
        self._rate_limit()

        logger.debug(
            f"Finding version of {arxiv_id} before {cutoff_date.strftime('%Y-%m-%d')}"
        )

        # Make cutoff_date timezone-aware if needed
        if cutoff_date.tzinfo is None:
            from datetime import timezone
            cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)

        # Calculate the minimum date based on max_age_days
        if max_age_days > 0:
            min_date = cutoff_date - timedelta(days=max_age_days)
            logger.debug(
                f"Version must be between {min_date.strftime('%Y-%m-%d')} "
                f"and {cutoff_date.strftime('%Y-%m-%d')} (max age: {max_age_days} days)"
            )
        else:
            # No age restriction, only cutoff date matters
            min_date = None
            logger.debug(
                f"Version must be before {cutoff_date.strftime('%Y-%m-%d')} "
                f"(no age restriction)"
            )

        latest_valid_version = None
        latest_valid_version_date = None

        # Query all versions at once (up to max_versions) for efficiency
        # Most papers have < 10 versions, but we'll check up to 20 to be safe
        max_versions = 20
        
        # Build list of all potential version IDs to query in one request
        version_ids = [f"{arxiv_id}v{i}" for i in range(1, max_versions + 1)]
        
        try:
            # Query all versions in a single API call
            search = arxiv.Search(id_list=version_ids)
            results = list(search.results())
            
            # Sort results by version number (extract from entry_id)
            def get_version_num(entry_id: str) -> int:
                """Extract version number from entry_id like 'http://arxiv.org/abs/2308.07074v2'"""
                parts = entry_id.split('/')[-1]
                if 'v' in parts:
                    return int(parts.split('v')[1])
                return 1
            
            # Sort by version number to process in order
            results.sort(key=lambda r: get_version_num(r.entry_id))
            
            logger.debug(f"Found {len(results)} version(s) for {arxiv_id}")
            
            for result in results:
                version_date = result.updated
                
                # Make version_date timezone-aware if needed
                if version_date.tzinfo is None:
                    from datetime import timezone
                    version_date = version_date.replace(tzinfo=timezone.utc)
                
                # Extract version number
                entry_id = result.entry_id.split('/')[-1]
                if 'v' in entry_id:
                    version_num = int(entry_id.split('v')[1])
                else:
                    version_num = 1
                
                logger.debug(
                    f"  v{version_num}: {version_date.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                if version_date < cutoff_date:
                    # This version is before the cutoff
                    # Check if it's within the age window (if age restriction is enabled)
                    if min_date is None or version_date >= min_date:
                        # This version is valid (before cutoff and within age window if applicable)
                        latest_valid_version = version_num
                        latest_valid_version_date = version_date
                    else:
                        # This version is too old (more than max_age_days before cutoff)
                        logger.debug(
                            f"  v{version_num} is too old "
                            f"({version_date.strftime('%Y-%m-%d')} < {min_date.strftime('%Y-%m-%d')})"
                        )
                else:
                    # This version is after cutoff, we can stop processing
                    # (but we already have all results, so we'll just skip it)
                    logger.debug(
                        f"  v{version_num} posted after cutoff "
                        f"({version_date} >= {cutoff_date})"
                    )
                    
        except Exception as e:
            logger.warning(
                f"Error querying versions for {arxiv_id}: {e}"
            )

        if latest_valid_version:
            if min_date:
                logger.info(
                    f"Selected v{latest_valid_version} of {arxiv_id} "
                    f"(latest before {cutoff_date.strftime('%Y-%m-%d')} "
                    f"and within {max_age_days} days: {latest_valid_version_date.strftime('%Y-%m-%d')})"
                )
            else:
                logger.info(
                    f"Selected v{latest_valid_version} of {arxiv_id} "
                    f"(latest before {cutoff_date.strftime('%Y-%m-%d')}: "
                    f"{latest_valid_version_date.strftime('%Y-%m-%d')})"
                )
        else:
            if min_date:
                logger.warning(
                    f"No versions of {arxiv_id} found before {cutoff_date.strftime('%Y-%m-%d')} "
                    f"and within {max_age_days} days (min date: {min_date.strftime('%Y-%m-%d')})"
                )
            else:
                logger.warning(
                    f"No versions of {arxiv_id} found before {cutoff_date.strftime('%Y-%m-%d')}"
                )

        return latest_valid_version

    def download_latex_source(
        self,
        arxiv_id: str,
        output_dir: Path,
        version: Optional[int] = None
    ) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
        """Download and extract LaTeX source from arXiv.

        Downloads the .tar.gz source file, extracts it, and returns the path
        to the extracted directory containing all LaTeX files.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2401.12345")
            output_dir: Directory to extract files to
            version: Optional version number (e.g., 1, 2, 3). If None, downloads latest.

        Returns:
            Tuple of (path, error_type, error_message)
            - path: Path to extracted directory, or None if download/extraction fails
            - error_type: "forbidden", "not_found", "rate_limit", "timeout", "other", or None if successful
            - error_message: Error message, or None if successful
        """
        self._rate_limit()

        # Construct URL with version if specified
        if version is not None:
            source_url = f"https://arxiv.org/e-print/{arxiv_id}v{version}"
            version_suffix = f"v{version}"
        else:
            source_url = f"https://arxiv.org/e-print/{arxiv_id}"
            version_suffix = ""

        tar_path = output_dir / f"{arxiv_id}{version_suffix}.tar.gz"
        extract_dir = output_dir / f"{arxiv_id}{version_suffix}"

        logger.debug(
            f"Downloading LaTeX source for {arxiv_id}"
            f"{f' (version {version})' if version else ''}"
        )

        try:
            # Download the source tarball
            response = self.session.get(source_url, timeout=60)
            response.raise_for_status()

            # Save to file
            output_dir.mkdir(parents=True, exist_ok=True)
            tar_path.write_bytes(response.content)

            # Extract the tarball
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)

            # Find the main .tex file
            tex_files = list(extract_dir.glob("**/*.tex"))

            if not tex_files:
                logger.warning(f"No .tex files found for {arxiv_id}")
                return (None, "other", f"No .tex files found for {arxiv_id}")

            # Heuristics to find the main .tex file
            main_tex = None

            # Try common names first
            for name in ["main.tex", "paper.tex", "manuscript.tex"]:
                candidates = [f for f in tex_files if f.name == name]
                if candidates:
                    main_tex = candidates[0]
                    break

            # If not found, use the largest .tex file
            if main_tex is None:
                main_tex = max(tex_files, key=lambda f: f.stat().st_size)

            logger.info(f"Using main tex file: {main_tex.name} for {arxiv_id}")

            # Clean up tar file
            tar_path.unlink()

            # Return the path to the extracted directory (contains all files)
            return (extract_dir, None, None)

        except requests.HTTPError as e:
            if e.response.status_code == 403:
                logger.warning(
                    f"Source not available for {arxiv_id} (403 Forbidden)"
                )
                return (None, "forbidden", f"Source not available for {arxiv_id} (403 Forbidden)")
            elif e.response.status_code == 404:
                logger.warning(
                    f"Source not found for {arxiv_id} (404 Not Found)"
                )
                return (None, "not_found", f"Source not found for {arxiv_id} (404 Not Found)")
            elif e.response.status_code == 429:
                logger.warning(
                    f"Rate limit hit for {arxiv_id} (429 Too Many Requests)"
                )
                return (None, "rate_limit", f"Rate limit hit for {arxiv_id} (429 Too Many Requests)")
            else:
                logger.error(
                    f"HTTP error downloading source for {arxiv_id}: {e}"
                )
                return (None, "other", f"HTTP error {e.response.status_code} downloading source: {e}")

        except requests.Timeout:
            logger.error(f"Timeout downloading source for {arxiv_id}")
            return (None, "timeout", f"Timeout downloading source for {arxiv_id}")

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                logger.error(f"Rate limit error downloading source for {arxiv_id}: {e}")
                return (None, "rate_limit", f"Rate limit error: {e}")
            else:
                logger.error(
                    f"Error downloading/extracting source for {arxiv_id}: {e}"
                )
                return (None, "other", f"Error downloading/extracting: {e}")

    def download_markdown_with_arxiv2md(
        self,
        arxiv_id: str,
        output_dir: Path,
        version: Optional[int] = None,
        arxiv2md_path: str = ".venv/bin/arxiv2md",
        timeout: int = 300
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Download and convert arXiv paper to markdown using arxiv2md.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2401.12345")
            output_dir: Directory to save markdown file
            version: Optional version number (e.g., 1, 2, 3). If None, uses latest.
            arxiv2md_path: Path to arxiv2md executable
            timeout: Timeout in seconds for the conversion

        Returns:
            Tuple of (markdown_content, error_type, error_message)
            - markdown_content: Markdown string, or None if conversion fails
            - error_type: "not_found", "timeout", "other", or None if successful
            - error_message: Error message, or None if successful
        """
        self._rate_limit()

        # Construct arxiv ID with version
        if version is not None:
            arxiv_id_with_version = f"{arxiv_id}v{version}"
        else:
            arxiv_id_with_version = arxiv_id

        logger.debug(
            f"Converting {arxiv_id_with_version} to markdown using arxiv2md"
        )

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{arxiv_id_with_version}.md"

        # Check if arxiv2md executable exists
        if not Path(arxiv2md_path).exists():
            error_msg = f"arxiv2md not found at {arxiv2md_path}"
            logger.error(error_msg)
            return (None, "other", error_msg)

        # Call arxiv2md
        cmd = [
            str(arxiv2md_path),
            arxiv_id_with_version,
            "-o", str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )

            # Check exit code
            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else result.stdout[:500]
                logger.warning(
                    f"arxiv2md failed for {arxiv_id_with_version} "
                    f"(exit {result.returncode}): {error_msg}"
                )
                return (None, "other", f"arxiv2md failed (exit {result.returncode}): {error_msg}")

            # Check output file exists
            if not output_path.exists():
                error_msg = "Output file not created"
                logger.warning(
                    f"arxiv2md did not create output file for {arxiv_id_with_version}"
                )
                return (None, "other", error_msg)

            # Check output file size
            file_size = output_path.stat().st_size
            if file_size < 100:
                error_msg = f"Output file too small ({file_size} bytes)"
                logger.warning(
                    f"arxiv2md output too small for {arxiv_id_with_version}: {file_size} bytes"
                )
                return (None, "other", error_msg)

            # Read markdown content
            markdown_content = output_path.read_text(encoding='utf-8')

            # Clean up the temporary file created by arxiv2md
            # (The enricher will save it with the proper filename including paper_id)
            try:
                output_path.unlink()
                logger.debug(f"Cleaned up temporary arxiv2md output: {output_path}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file {output_path}: {e}")

            logger.info(
                f"Successfully converted {arxiv_id_with_version} to markdown "
                f"({file_size} bytes)"
            )

            return (markdown_content, None, None)

        except subprocess.TimeoutExpired:
            error_msg = f"Conversion timeout (>{timeout}s)"
            logger.warning(
                f"arxiv2md timeout for {arxiv_id_with_version} after {timeout}s"
            )
            return (None, "timeout", error_msg)

        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            logger.error(
                f"Error running arxiv2md for {arxiv_id_with_version}: {e}"
            )
            return (None, "other", error_msg)

    def get_paper_metadata(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Get metadata for a specific arXiv paper.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2401.12345")

        Returns:
            ArxivPaper object, or None if not found
        """
        self._rate_limit()

        logger.debug(f"Fetching metadata for {arxiv_id}")

        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())

            # Extract arxiv ID (remove version if present)
            clean_id = result.entry_id.split('/')[-1]
            if 'v' in clean_id:
                clean_id = clean_id.split('v')[0]

            return ArxivPaper(
                arxiv_id=clean_id,
                title=result.title,
                published_date=result.published,
                pdf_url=result.pdf_url,
                source_url=f"https://arxiv.org/e-print/{clean_id}",
                authors=[author.name for author in result.authors],
                summary=result.summary
            )

        except Exception as e:
            logger.error(f"Error fetching metadata for {arxiv_id}: {e}")
            return None
