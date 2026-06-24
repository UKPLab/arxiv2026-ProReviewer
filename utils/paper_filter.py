"""Paper filtering utilities based on review consistency.

This module provides functions to filter papers based on the consistency
of reviewer ratings (measured by standard deviation).
"""

import statistics
from typing import List, Dict, Any


def parse_ratings(rating_str: str) -> List[float]:
    """Parse semicolon-separated rating string to list of floats.

    Args:
        rating_str: Rating string (e.g., "3;5;6;6;6")

    Returns:
        List of float ratings

    Examples:
        >>> parse_ratings("3;5;6;6;6")
        [3.0, 5.0, 6.0, 6.0, 6.0]
        >>> parse_ratings("6;7;6;7")
        [6.0, 7.0, 6.0, 7.0]
        >>> parse_ratings("")
        []
    """
    # Handle empty or invalid strings
    if not rating_str or rating_str.strip() == "":
        return []

    # Split by semicolon and convert to floats
    try:
        return [float(r.strip()) for r in rating_str.split(";") if r.strip()]
    except ValueError:
        return []


def calculate_rating_std(rating_str: str) -> float:
    """Calculate standard deviation of ratings.

    Args:
        rating_str: Rating string (e.g., "3;5;6;6;6")

    Returns:
        Standard deviation (returns float('inf') for invalid/empty ratings)

    Examples:
        >>> calculate_rating_std("6;6;6;6")
        0.0
        >>> 0.4 < calculate_rating_std("6;7;6;7") < 0.6
        True
        >>> calculate_rating_std("")
        inf
    """
    ratings = parse_ratings(rating_str)

    if len(ratings) < 2:
        # Need at least 2 ratings to calculate std
        return float('inf')

    return statistics.stdev(ratings)


def is_consistent(paper: Dict[str, Any], threshold: float = 1.0) -> bool:
    """Check if a paper has consistent reviewer ratings.

    Args:
        paper: Paper dictionary with 'rating' field
        threshold: Maximum std deviation for consistency (default: 1.0)

    Returns:
        True if rating std <= threshold, False otherwise

    Examples:
        >>> is_consistent({"rating": "6;7;6;7"}, threshold=1.0)
        True
        >>> is_consistent({"rating": "3;8;5;7"}, threshold=1.0)
        False
    """
    rating_str = paper.get("rating", "")
    std = calculate_rating_std(rating_str)
    return std <= threshold


def filter_consistent_papers(
    papers: List[Dict[str, Any]],
    threshold: float = 1.0
) -> List[Dict[str, Any]]:
    """Filter papers to keep only those with consistent ratings.

    Args:
        papers: List of paper dictionaries
        threshold: Maximum std deviation for consistency (default: 1.0)

    Returns:
        Filtered list containing only consistent papers

    Examples:
        >>> papers = [
        ...     {"id": "1", "rating": "6;7;6;7"},
        ...     {"id": "2", "rating": "3;8;5;7"},
        ... ]
        >>> filtered = filter_consistent_papers(papers, threshold=1.0)
        >>> len(filtered)
        1
        >>> filtered[0]["id"]
        '1'
    """
    return [paper for paper in papers if is_consistent(paper, threshold)]


def add_consistency_metrics(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Add rating_std and is_consistent fields to paper.

    Args:
        paper: Paper dictionary

    Returns:
        Paper with added 'rating_std' and 'is_consistent' fields

    Examples:
        >>> paper = {"id": "1", "rating": "6;7;6;7"}
        >>> enriched = add_consistency_metrics(paper)
        >>> 0.4 < enriched["rating_std"] < 0.6
        True
        >>> enriched["is_consistent"]
        True
    """
    rating_std = calculate_rating_std(paper.get("rating", ""))

    # Create a copy to avoid modifying original
    enriched = paper.copy()
    enriched["rating_std"] = rating_std if rating_std != float('inf') else None
    enriched["is_consistent"] = rating_std <= 1.0  # Default threshold

    return enriched
