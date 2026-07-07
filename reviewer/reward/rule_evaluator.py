"""Rule-based reward evaluators (no LLM calls).

- Format completeness: checks review structure and weakness count
- Score difference: penalises deviation from human average score
"""

from typing import Dict, Tuple


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
