import asyncio
import json
import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
from utils.helpers.llm import acall_llm
from tqdm import tqdm


CONSENSUS_PROMPT = """You are analyzing peer review comments to extract concrete review points that could be used as training examples for a review agent.

**Paper Title:** {title}
**Total Reviewers:** {total_reviewers}

**Review Comments:**

{formatted_comments}

**Task:** Aggregate review comments into concrete, specific observations. Write each point as if it's a finding from reading the paper - NOT as a meta-summary of what reviewers said.

**Rules:**
1. Include ALL significant points, even if mentioned by only 1 reviewer
2. Calculate consensus_weight as reviewer_count / total_reviewers
3. For each point, extract the specific quote or paraphrase from EACH reviewer who mentioned it

**CRITICAL - Style Guidelines:**
4. Write as CONCRETE OBSERVATIONS, not meta-review summaries:
   - BAD: "The reviewers noted that improvements are marginal"
   - GOOD: "The improvements over baselines are marginal (1-2%), which may not justify the additional computational overhead"

5. Include specific details when available (numbers, method names, section references):
   - BAD: "Missing baseline comparisons"
   - GOOD: "The experiments lack comparison to other SAM variants (ASAM, RSAM, LookBehind-SAM)"

6. Rephrase prior work concerns concretely:
   - BAD: "This is similar to Abbas et al. 2022"
   - GOOD: "The method closely resembles prior SAM+MAML combinations without clear differentiation of the contribution"

7. Do NOT merge points that have different nature:
   - "The selection criteria lack mathematical definition" (methodology)
   - "Consider 'extreme' instead of 'super'" (terminology)
   → These are SEPARATE points

8. Preserve specific technical observations even from single reviewers:
   - "The notation inconsistently uses 'Yij' vs 'Y_{{ij}}'"
   - "The connection to adversarial robustness is not discussed"

**Examples of GOOD points:**
- "The improvements over RTN baseline are marginal (< 1% accuracy), while incurring 27% additional wall-clock time"
- "The method's core mechanism resembles prior SAM+MAML approaches without clear differentiation"

**Examples of BAD points:**
- "Reviewers questioned the novelty of this work" (meta-review tone)
- "There are concerns about experimental validation" (too vague)

**Output Format (JSON):**
{{
  "clustered_points": [
    {{
      "reviewer_count": <number>,
      "total_reviewers": {total_reviewers},
      "text": "Concrete, specific observation (as if from reading the paper)",
      "type": "strength|weakness|question",
      "consensus_weight": <reviewer_count/total_reviewers as float>,
      "supporting_evidence": {{
        "reviewer_id_1": "Direct quote or close paraphrase from this reviewer",
        "reviewer_id_2": "...",
      }}
    }}
  ]
}}

Return ONLY the JSON, no additional text.
"""


async def _extract_points_core(data: Dict, model: str) -> List[Dict]:
    reviews = data.get('reviews', [])
    total_reviewers = len(reviews)

    if total_reviewers < 1:
        return []

    formatted_comments = _format_reviews(reviews)

    prompt = CONSENSUS_PROMPT.format(
        title=data.get('title', 'Unknown'),
        total_reviewers=total_reviewers,
        formatted_comments=formatted_comments
    )

    messages = [
        {"role": "system", "content": "You are a scientific peer review analysis assistant."},
        {"role": "user", "content": prompt}
    ]

    response = await acall_llm(
        model=model,
        messages=messages,
        max_tokens=16384
    )

    response_text = _get_content(response)
    return _parse_llm_response(response_text)


async def extract_clustered_points(input_file: str, model: str = "gpt-5mini", semaphore: asyncio.Semaphore = None) -> None:
    """
    Extract shared comments between reviewers and add to the same file.

    Args:
        input_file: Path to paper JSON file
        model: LLM model to use (default: gpt-5mini)
        semaphore: Optional semaphore to limit concurrency
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    if data.get('clustered_points') or data.get('clustered_points_gpt-5mini'):
        print(f"Skipping {input_file} because clustered_points already exist")
        return

    if semaphore:
        async with semaphore:
            clustered_points = await _extract_points_core(data, model)
    else:
        clustered_points = await _extract_points_core(data, model)

    data['clustered_points_gpt-5mini'] = clustered_points
    _safe_write(input_file, data)
    print(f"Extracted {len(clustered_points)} points from {input_file}")


def _format_reviews(reviews: List[Dict]) -> str:
    """Format review comments for LLM prompt."""
    formatted = []

    for i, review in enumerate(reviews):
        reviewer_id = review.get('id', f'R{i+1}')
        strengths = review.get('strengths', '').strip()
        weaknesses = review.get('weaknesses', '').strip()
        questions = review.get('questions', '').strip()

        review_text = f"Reviewer {reviewer_id}:"

        if strengths:
            review_text += f"\n  Strengths: {strengths}"
        if weaknesses:
            review_text += f"\n  Weaknesses: {weaknesses}"
        if questions:
            review_text += f"\n  Questions: {questions}"

        formatted.append(review_text)

    return "\n\n".join(formatted)


def _get_content(response) -> str:
    """Extract content from LLM response."""
    if hasattr(response, 'choices') and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice, 'message'):
            return choice.message.content
        elif hasattr(choice, 'text'):
            return choice.text
    elif isinstance(response, dict):
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if 'message' in choice:
                return choice['message']['content']
            elif 'text' in choice:
                return choice['text']

    return str(response)


def _parse_llm_response(response_text: str) -> List[Dict]:
    """Parse JSON response from LLM, handling code blocks."""
    if "```json" in response_text:
        match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
    elif "```" in response_text:
        match = re.search(r'```\s*\n(.*?)\n```', response_text, re.DOTALL)
        if match:
            response_text = match.group(1)

    response_text = response_text.strip()

    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            return parsed.get('clustered_points', [])
        elif isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError as e:
        # Try to recover partial results from truncated JSON
        recovered = _recover_partial_json(response_text)
        if recovered:
            print(f"Warning: Truncated JSON response, recovered {len(recovered)} partial points (error: {e})")
            return recovered
        print(f"Warning: Failed to parse LLM response as JSON: {e}")
        print(f"Response text: {response_text[:500]}")
        return []


def _recover_partial_json(response_text: str) -> List[Dict]:
    """Extract complete clustered_points objects from a truncated JSON response."""
    # Find the start of the clustered_points array
    array_start = response_text.find('"clustered_points"')
    if array_start == -1:
        return []
    bracket_pos = response_text.find('[', array_start)
    if bracket_pos == -1:
        return []

    points = []
    pos = bracket_pos + 1
    depth = 0
    obj_start = None

    while pos < len(response_text):
        ch = response_text[pos]
        if ch == '{':
            if depth == 0:
                obj_start = pos
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(response_text[obj_start:pos + 1])
                    points.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == ']' and depth == 0:
            break
        pos += 1

    return points


def _safe_write(input_file: str, data: Dict) -> None:
    """Write JSON data to file safely using temp file + atomic rename."""
    input_path = Path(input_file)
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', dir=input_path.parent)

    try:
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        shutil.move(temp_path, input_file)
    except Exception as e:
        if Path(temp_path).exists():
            os.unlink(temp_path)
        raise e


if __name__ == "__main__":
    input_dir = '/pfss/mlde/workspaces/mlde_wsp_Reviewer_R1/Reviewer-R1/data/test_data'
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    filepaths = [os.path.join(input_dir, f) for f in input_files]

    print(f"Found {len(filepaths)} files, processing with gpt-5mini...\n")

    async def main():
        semaphore = asyncio.Semaphore(10)  # max 20 concurrent API calls
        tasks = [extract_clustered_points(fp, model="gpt-5mini", semaphore=semaphore) for fp in filepaths]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await coro

    asyncio.run(main())
