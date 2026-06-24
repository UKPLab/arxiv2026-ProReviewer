"""Conference-specific configuration for rebuttal deadlines.

This module contains hardcoded rebuttal start dates for ICLR conferences,
used to validate whether arXiv papers were posted before the rebuttal phase.
"""

from datetime import datetime, timezone
from typing import Dict


# Rebuttal start dates for ICLR conferences
# Papers posted to arXiv after these dates should not be used as they might
# incorporate reviewer feedback
CONFERENCE_REBUTTAL_DATES: Dict[str, datetime] = {
    "ICLR 2024": datetime(2023, 11, 10, tzinfo=timezone.utc),  # Rebuttal start date
    "ICLR 2025": datetime(2024, 11, 13, tzinfo=timezone.utc),  # Rebuttal start date
    "ICLR 2026": datetime(2025, 11, 11, tzinfo=timezone.utc),  # Rebuttal start date (estimated)
}


def get_rebuttal_date(conference: str) -> datetime:
    """Get the rebuttal start date for a conference.

    Args:
        conference: Conference name (e.g., "ICLR 2024")

    Returns:
        datetime: The rebuttal start date

    Raises:
        ValueError: If conference is not recognized
    """
    if conference not in CONFERENCE_REBUTTAL_DATES:
        raise ValueError(
            f"Unknown conference: {conference}. "
            f"Valid conferences: {', '.join(CONFERENCE_REBUTTAL_DATES.keys())}"
        )
    return CONFERENCE_REBUTTAL_DATES[conference]


def is_arxiv_valid_for_review(arxiv_published_date: datetime, conference: str) -> bool:
    """Check if an arXiv paper was posted before the rebuttal deadline.

    Papers posted after the rebuttal start date might incorporate reviewer
    feedback and should not be used as the "original" submission.

    Args:
        arxiv_published_date: When the paper was published on arXiv
        conference: Conference name (e.g., "ICLR 2024")

    Returns:
        bool: True if the paper was posted before the rebuttal deadline
    """
    deadline = get_rebuttal_date(conference)

    # Make sure both datetimes are timezone-aware for comparison
    if arxiv_published_date.tzinfo is None:
        arxiv_published_date = arxiv_published_date.replace(tzinfo=timezone.utc)

    return arxiv_published_date < deadline
