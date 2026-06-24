"""
Review Parser - Extract structured information from review text

This module parses review text into structured components:
- Sections (Summary, Strengths, Weaknesses, Questions)
- Claims extracted from strengths
- Issues extracted from weaknesses
- Score mappings

Used for generating SFT training data from gold reviews.
"""

import re
from typing import Dict, List, Optional, Tuple


def parse_review_sections(review_text: str) -> Dict[str, str]:
    """
    Parse review text into sections.

    Args:
        review_text: Full review text with section headers

    Returns:
        Dictionary mapping section names to content:
        {
            "summary": "...",
            "strengths": "...",
            "weaknesses": "...",
            "questions": "...",
            "additional_comments": "..."
        }
    """
    sections = {}

    # Normalize different header formats to a standard format
    # Convert **Section:** to ## Section
    review_text = re.sub(r'\*\*(\w+(?:\s+\w+)?):\*\*', r'## \1', review_text)

    # Split by section headers (##, #, or simple colons)
    # Look for patterns like:
    # - "## Summary"
    # - "Summary:"
    section_markers = r'(?:^|\n)(?:#{1,3}\s+)?(\w+(?:\s+\w+)?):?\s*\n'

    parts = re.split(section_markers, review_text, flags=re.MULTILINE | re.IGNORECASE)

    # Parse split results (alternates between section name and content)
    current_section = None
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Odd indices (after split) contain section names
        # Even indices contain section content
        if i % 2 == 1:  # Section name
            # Normalize section name
            section_key = part.lower().replace(" ", "_")
            # Map variations to standard names
            if section_key in ["summary", "summary_of_contribution", "overview"]:
                current_section = "summary"
            elif section_key in ["strengths", "strength", "positive"]:
                current_section = "strengths"
            elif section_key in ["weaknesses", "weakness", "limitations", "negative"]:
                current_section = "weaknesses"
            elif section_key in ["questions", "question", "questions_for_authors"]:
                current_section = "questions"
            elif section_key in ["comments", "additional_comments", "notes"]:
                current_section = "additional_comments"
            else:
                current_section = section_key
        elif i % 2 == 0 and current_section:  # Section content
            sections[current_section] = part

    # Fallback: if no structured sections found, treat entire text as summary
    if not sections:
        sections["summary"] = review_text.strip()

    return sections



def extract_claims_from_strengths(strengths_text: str) -> List[Dict[str, str]]:
    """
    Extract individual claims from strengths section.

    Args:
        strengths_text: Strengths section text

    Returns:
        List of claim dictionaries with 'text' field
    """
    if not strengths_text:
        return []

    claims = []
    # Split by bullet points or numbered lists
    lines = re.split(r'[\n\r]+\s*[-•*\d+\.]\s+', strengths_text)

    for line in lines:
        line = line.strip()
        # Remove leading bullet/dash if present
        line = re.sub(r'^[-•*]\s+', '', line)
        if not line or len(line) < 10:
            continue

        claims.append({'text': line})

    return claims


def extract_weaknesses_as_issues(weaknesses_text: str) -> List[Dict[str, str]]:
    """
    Extract individual issues from weaknesses section.

    Args:
        weaknesses_text: Weaknesses section text

    Returns:
        List of issue dictionaries with 'text' field
    """
    if not weaknesses_text:
        return []

    issues = []
    # Split by bullet points or numbered lists
    lines = re.split(r'[\n\r]+\s*[-•*\d+\.]\s+', weaknesses_text)

    for line in lines:
        line = line.strip()
        # Remove leading bullet/dash if present
        line = re.sub(r'^[-•*]\s+', '', line)
        if not line or len(line) < 10:
            continue

        issues.append({'text': line})

    return issues


def extract_questions(questions_text: str) -> List[str]:
    """
    Extract individual questions from questions section.

    Args:
        questions_text: Questions section text

    Returns:
        List of question strings
    """
    if not questions_text:
        return []

    # Split by bullet points, numbered lists, or question marks followed by newlines
    questions = []
    lines = re.split(r'[\n\r]+\s*[-•*\d+\.]\s+', questions_text)

    for line in lines:
        line = line.strip()
        # Remove leading bullet/dash if present (for first item before split pattern)
        line = re.sub(r'^[-•*]\s+', '', line)
        if not line or len(line) < 10:
            continue

        # Ensure it's a question (ends with ? or starts with interrogative)
        if '?' in line or any(line.lower().startswith(w) for w in ['what', 'how', 'why', 'when', 'where', 'which', 'is', 'are', 'can', 'could', 'would', 'does']):
            questions.append(line)

    return questions


def map_scores_to_assessments(scores: Dict[str, float]) -> List[Dict]:
    """
    Map numerical scores to assessment reasoning templates.

    Args:
        scores: Dictionary of numerical scores
        {
            "soundness": 3,
            "contribution": 3,
            "presentation": 4,
            "overall": 6
        }

    Returns:
        List of assessment dictionaries:
        [{
            "aspect": "soundness",
            "score": 3,
            "reasoning_template": "..."
        }, ...]
    """
    assessments = []

    # Score interpretation templates
    templates = {
        "soundness": {
            1: "Critical methodological flaws that invalidate the results",
            2: "Significant methodological issues that weaken the results",
            3: "Generally sound methodology with minor concerns",
            4: "Strong methodology with comprehensive validation",
            5: "Exceptionally rigorous methodology and validation"
        },
        "contribution": {
            1: "Minimal or no contribution to the field",
            2: "Limited contribution with unclear significance",
            3: "Moderate contribution with clear but incremental advances",
            4: "Significant contribution with notable advances",
            5: "Outstanding contribution with major breakthroughs"
        },
        "presentation": {
            1: "Very poor presentation that severely hinders understanding",
            2: "Poor presentation with significant clarity issues",
            3: "Adequate presentation with room for improvement",
            4: "Well-written and clearly presented",
            5: "Exceptionally clear and well-organized presentation"
        }
    }

    for aspect in ["soundness", "contribution", "presentation"]:
        if aspect in scores:
            score = int(round(scores[aspect]))
            score = max(1, min(5, score))  # Clamp to 1-5

            assessments.append({
                "aspect": aspect,
                "score": score,
                "reasoning_template": templates[aspect].get(score, "")
            })

    return assessments


def parse_complete_review(review_text: str, scores: Optional[Dict[str, float]] = None) -> Dict:
    """
    Parse complete review into all structured components.

    Args:
        review_text: Full review text
        scores: Optional dictionary of numerical scores

    Returns:
        Complete parsed review:
        {
            "sections": {...},
            "claims": [...],
            "issues": [...],
            "questions": [...],
            "assessments": [...],
            "scores": {...}
        }
    """
    sections = parse_review_sections(review_text)

    parsed = {
        "sections": sections,
        "stength": extract_claims_from_strengths(sections.get("strengths", "")),
        "issues": extract_weaknesses_as_issues(sections.get("weaknesses", "")),
        "questions": extract_questions(sections.get("questions", "")),
        "scores": scores or {}
    }

    # Map scores to assessments if provided
    if scores:
        parsed["assessments"] = map_scores_to_assessments(scores)
    else:
        parsed["assessments"] = []

    return parsed


# Helper functions

def _infer_claim_type(text: str) -> str:
    """Infer claim type from text content."""
    text_lower = text.lower()

    # Keywords indicating novelty
    novelty_keywords = ["novel", "new", "first", "original", "innovative", "pioneering"]
    if any(kw in text_lower for kw in novelty_keywords):
        return "novelty"

    # Keywords indicating empirical results
    empirical_keywords = ["achieve", "improve", "outperform", "result", "accuracy", "performance", "experiment"]
    if any(kw in text_lower for kw in empirical_keywords):
        return "empirical"

    # Keywords indicating theoretical contribution
    theoretical_keywords = ["proof", "theorem", "theory", "framework", "formalization", "analysis"]
    if any(kw in text_lower for kw in theoretical_keywords):
        return "theoretical"

    # Default
    return "empirical"


def _infer_severity(text: str) -> str:
    """Infer weakness severity from text content."""
    text_lower = text.lower()

    # Keywords indicating major issues
    major_keywords = ["critical", "fundamental", "significant", "major", "serious", "severe", "fatal", "invalidate"]
    if any(kw in text_lower for kw in major_keywords):
        return "major"

    # Keywords indicating minor issues
    minor_keywords = ["minor", "small", "slight", "cosmetic", "typo", "formatting"]
    if any(kw in text_lower for kw in minor_keywords):
        return "minor"

    # Default: moderate severity
    return "moderate"


# Example usage
if __name__ == "__main__":
    # Example review text
    example_review = """## Summary
This paper proposes a novel attention mechanism for transformer models that improves efficiency on long sequences.

## Strengths
- Novel sparse attention pattern that reduces computational complexity
- Strong empirical results on multiple long-sequence benchmarks
- Clear presentation with good visualizations
- Comprehensive ablation studies

## Weaknesses
- Limited theoretical analysis of why the sparse pattern works
- Only evaluated on language modeling tasks, not tested on other domains
- Comparison with recent concurrent work is missing
- Some implementation details are unclear

## Questions
- How does the method perform on extremely long sequences (>10k tokens)?
- What is the memory overhead compared to standard attention?
- Can this be combined with other efficiency techniques?
"""

    scores = {
        "soundness": 3,
        "contribution": 3,
        "presentation": 4,
        "overall": 6
    }

    parsed = parse_complete_review(example_review, scores)

    print("=== Parsed Review ===\n")
    print(f"Claims: {len(parsed['claims'])}")
    for i, claim in enumerate(parsed['claims'], 1):
        print(f"  {i}. [{claim['claim_type']}] {claim['text'][:60]}...")

    print(f"\nIssues: {len(parsed['issues'])}")
    for i, issue in enumerate(parsed['issues'], 1):
        print(f"  {i}. [{issue['severity']}] {issue['text'][:60]}...")

    print(f"\nQuestions: {len(parsed['questions'])}")
    for i, q in enumerate(parsed['questions'], 1):
        print(f"  {i}. {q[:60]}...")

    print(f"\nAssessments: {len(parsed['assessments'])}")
    for assessment in parsed['assessments']:
        print(f"  {assessment['aspect']}: {assessment['score']}/5")
