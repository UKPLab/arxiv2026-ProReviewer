"""Reward component implementations for review quality assessment.

This module contains individual reward component calculations:
- Format compliance (rule-based)
- Score difference penalty (rule-based)
- Weakness diversity (embedding cosine via vLLM)
- Helper functions for consensus weighting
"""

import re
from typing import Dict, List, Tuple

from utils.sft.review_parser import extract_claims_from_strengths, extract_weaknesses_as_issues


def compute_syntactic_reward(
    n_turns: int,
    c_format: int,
    c_correct: int,
    c_error: int,
    has_tool_call: bool,
) -> float:
    """Compute Stage 1 syntactic reward: RStage1 = Itool * (Rformat + Rtool).

    Useful for offline evaluation of trajectories.

    Args:
        n_turns: Total number of turns in the episode
        c_format: Turns where JSON parsing failed or required structure is missing
        c_correct: Tool calls with correct name AND valid args
        c_error: Tool calls with wrong args + unknown names + failed memory ops
        has_tool_call: Whether any non-format-error action was attempted

    Returns:
        Syntactic reward in range [0, 2]
    """
    if n_turns == 0:
        return 0.0

    R_format = (n_turns - c_format) / n_turns

    total_calls = c_correct + c_error
    R_tool = c_correct / total_calls if total_calls > 0 else 0.0

    I_tool = 1.0 if has_tool_call else 0.0

    return I_tool * (R_format + R_tool)


def compute_format_completeness(review: Dict, min_weakness_count: int = 3) -> float:
    """Format completeness reward incorporating field presence and weakness count.

    Combines structural completeness (are all fields present?) with a weakness
    count check (enough weaknesses?).  This replaces the old standalone
    ``count_penalty`` component — fewer reward components, same signal.

    Score breakdown (each worth 0.2, total 0.0–1.0):
      - summary present & ≥50 chars
      - strengths present (≥1)
      - weaknesses present (≥1)
      - weakness count ≥ min_weakness_count
      - overall_score present

    Args:
        review: Review dict with summary, strengths, weaknesses, overall_score.
        min_weakness_count: Target number of weaknesses for full credit (default 3).

    Returns:
        Format completeness score in [0.0, 1.0].
    """
    checks = []

    # 1. Summary present and non-trivial
    summary = review.get('summary', '')
    checks.append(isinstance(summary, str) and len(summary) >= 50)

    # 2. Strengths present
    strengths = review.get('strengths', [])
    if strengths is None:
        strengths = []
    if isinstance(strengths, str):
        strengths = [strengths] if strengths.strip() else []
    checks.append(len(strengths) >= 1)

    # 3. Weaknesses present
    weaknesses = review.get('weaknesses', [])
    if weaknesses is None:
        weaknesses = []
    if isinstance(weaknesses, str):
        weaknesses = [weaknesses] if weaknesses.strip() else []
    checks.append(len(weaknesses) >= 1)

    # 4. Enough weaknesses (replaces standalone count_penalty)
    checks.append(len(weaknesses) >= min_weakness_count)

    # 5. Overall score present
    checks.append(review.get('overall_score') is not None)

    return sum(checks) / len(checks)


def compute_format_reward(generate_review: Dict[list, str]) -> Tuple[float, Dict]:
    """Compute format compliance reward (rule-based, no LLM).

    Validates ICLR review structure using existing parser.

    Args:
        review_text: The review text to validate

    Returns:
        Tuple of (reward score [0.0-1.0], detailed breakdown dict)
    """

    score = 0.0
    details = {}

    # 1. Summary section exists
    summary = generate_review.get('summary', '')
    summary_present = len(summary) >= 50
    details['summary_length'] = len(summary)

    # 2. Strengths section exists
    strengths_count = len(generate_review['strengths'])
    details['strengths_count'] = strengths_count
    strengths_present = strengths_count > 0

    # 3. Weaknesses section exists
    weaknesses_count = len(generate_review['weaknesses'])
    details['weaknesses_count'] = weaknesses_count
    weaknesses_present = weaknesses_count > 0

    # 4. Questions section exists
    questions_count = len(generate_review['questions'])
    details['questions_count'] = questions_count
    questions_present = questions_count > 0

    # 5. Overall score exists (don't need all required ratings present (0.2))
    # required_scores = ['soundness', 'contribution', 'presentation']
    # all_scores_present = all(
    #     score_name in parsed['scores'] for score_name in required_scores
    # )
    overall_score_present =  generate_review.get('overall_score', None) is not None
    # if overall_score_present:
    #     score += 0.2
    details['overall_score'] = generate_review.get('overall_score', None)

    score = (summary_present + strengths_present + weaknesses_present + questions_present + overall_score_present) / 5
    return score, details


def compute_score_difference_reward(
    generated_score: float,
    human_avg_score: float,
    score_range: float = 9.0
) -> Tuple[float, Dict]:
    """Compute overall score difference reward (rule-based, no LLM).

    R = 1 - |y_true - y_pred| / S, clipped to [0, 1].

    Args:
        generated_score: Generated overall score
        human_avg_score: Human average score (float)
        score_range: Maximum possible absolute difference (default 9.0 for a 1-10 scale)

    Returns:
        Tuple of (reward score [0.0-1.0], detailed breakdown dict)
    """
    if not human_avg_score:
        # No human reviews to compare against — return 0 reward instead of crashing
        return 0.0, {
            'score_diff': None,
            'reward': 0.0,
            'human_avg_scores': human_avg_score,
            'generated_score': generated_score,
            'error': 'No human score provided'
        }

    if generated_score is None:
        # No generated score available - return 0 reward
        return 0.0, {
            'score_diff': None,
            'reward': 0.0,
            'human_avg_scores': human_avg_score,
            'generated_score': None,
            'error': 'Generated score is None'
        }

    generated_score = round(generated_score)
    abs_diff = abs(generated_score - human_avg_score)
    reward = max(0.0, 1.0 - abs_diff / score_range)

    details = {
        'score_diff': generated_score - human_avg_score,
        'reward': reward,
        'human_avg_scores': human_avg_score,
        'generated_score': generated_score
    }

    return reward, details


def extract_human_review_points(
    human_reviews: List[Dict]
) -> List[Dict]:
    """Extract claims and issues from human reviews.

    Args:
        human_reviews: List of human review dictionaries

    Returns:
        List of point dictionaries with:
        {
            'text': "point text",
            'type': "strength" | "weakness",
            'reviewer_id': "review_id",
            'source_section': "strengths" | "weaknesses"
        }
    """
    points = []

    for i, review in enumerate(human_reviews):
        reviewer_id = review.get('id', f'reviewer_{i}')

        # Extract strengths
        strengths_text = ''
        if isinstance(review.get('strengths'), dict):
            strengths_text = review['strengths'].get('value', '')
        elif isinstance(review.get('strengths'), str):
            strengths_text = review['strengths']

        if strengths_text:
            claims = extract_claims_from_strengths(strengths_text)
            for claim in claims:
                points.append({
                    'text': claim['text'],
                    'type': 'strength',
                    'reviewer_id': reviewer_id,
                    'source_section': 'strengths'
                })

        # Extract weaknesses
        weaknesses_text = ''
        if isinstance(review.get('weaknesses'), dict):
            weaknesses_text = review['weaknesses'].get('value', '')
        elif isinstance(review.get('weaknesses'), str):
            weaknesses_text = review['weaknesses']

        if weaknesses_text:
            issues = extract_weaknesses_as_issues(weaknesses_text)
            for issue in issues:
                points.append({
                    'text': issue['text'],
                    'type': 'weakness',
                    'reviewer_id': reviewer_id,
                    'source_section': 'weaknesses'
                })

    return points

def extract_keywords(text: str) -> set:
    """Extract keywords from text for similarity matching.

    Args:
        text: Input text

    Returns:
        Set of lowercase keywords
    """
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()

    # Filter out common stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'this',
        'that', 'these', 'those', 'it', 'its'
    }

    keywords = {w for w in words if w not in stopwords and len(w) > 2}
    return keywords


def compute_keyword_similarity(keywords1: set, keywords2: set) -> float:
    """Compute Jaccard similarity between two keyword sets.

    Args:
        keywords1: First keyword set
        keywords2: Second keyword set

    Returns:
        Similarity score [0-1]
    """
    if not keywords1 or not keywords2:
        return 0.0

    intersection = keywords1 & keywords2
    union = keywords1 | keywords2

    return len(intersection) / len(union) if union else 0.0
