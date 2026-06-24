"""Simplified evaluation system for comparing review agent performance.

This module provides evaluation framework for two scenarios:
1. Detection: Evaluate error detection accuracy
2. Quality: Evaluate review quality across multiple dimensions
"""

import os
import json
import warnings
import argparse
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm

from reviewer.legacy.judge_prompts import (
    DETECTION_JUDGE_SYSTEM_PROMPT,
    DETECTION_JUDGE_USER_PROMPT,
    QUALITY_JUDGE_SYSTEM_PROMPT,
    QUALITY_JUDGE_USER_PROMPT,
    DEFAULT_JUDGE_MODEL
)
from utils.helpers.llm import call_llm, get_content

try:
    from reviewer.reward.calculator import RewardCalculator
    REWARDS_AVAILABLE = True
except ImportError:
    REWARDS_AVAILABLE = False
    RewardCalculator = None


class EvaluationScenario(Enum):
    """Evaluation scenario types."""
    DETECTION = "detection"
    QUALITY = "quality"


def _infer_error_type_from_path(review_dir: str, error_types: List[str]) -> Optional[str]:
    """Extract error type from directory path.

    Args:
        review_dir: Directory path to check
        error_types: List of possible error types

    Returns:
        Error type if found in path, None otherwise
    """
    path_lower = review_dir.lower()
    for error_type in error_types:
        if error_type.lower() in path_lower:
            return error_type
    return None


class ReviewEvaluator:
    """Simplified evaluator for review quality assessment."""

    def __init__(
        self,
        ground_truth_dir: str,
        scenario: EvaluationScenario,
        review_dir: str,
        agent_type: str,
        model_name: str,
        judge_model: Optional[str] = None,
        error_types: Optional[List[str]] = None,
        compute_rewards: bool = False
    ):
        """Initialize evaluator.

        Args:
            ground_truth_dir: Base directory containing ground truth files
            scenario: Evaluation scenario (DETECTION or QUALITY)
            review_dir: Directory containing review trajectory files
            agent_type: Type of agent (e.g., 'evolving_draft')
            model_name: Name of model (e.g., 'deepseek-reasoner')
            judge_model: LLM judge model (default from judge_prompts.py)
            error_types: List of error types for detection scenario
            compute_rewards: Enable reward calculation (quality scenario only)
        """
        self.ground_truth_dir = ground_truth_dir
        self.scenario = scenario
        self.review_dir = review_dir
        self.agent_type = agent_type
        self.model_name = model_name
        self.judge_model = judge_model or os.getenv("MODEL_NAME", DEFAULT_JUDGE_MODEL)
        self.error_types = error_types or []

        # Infer error type from review directory if not already known
        self.inferred_error_type = None
        if self.scenario == EvaluationScenario.DETECTION and self.error_types:
            self.inferred_error_type = _infer_error_type_from_path(review_dir, error_types)

        self.ground_truth = self._load_ground_truth()

        # Initialize reward calculator if requested
        self.reward_calculator = None
        if compute_rewards:
            if not REWARDS_AVAILABLE:
                warnings.warn("RewardCalculator not available. Install required dependencies.")
            elif self.scenario != EvaluationScenario.QUALITY:
                warnings.warn("Rewards only supported for quality scenario. Ignoring compute_rewards.")
            else:
                self.reward_calculator = RewardCalculator(judge_model=self.judge_model)

    def _load_ground_truth(self) -> Dict[str, Dict]:
        """Load ground truth files.

        Returns:
            Dictionary mapping keys to ground truth data.
            For detection: keys are '{paper_id}:{error_type}'
            For quality: keys are paper IDs
        """
        ground_truth = {}

        if self.scenario == EvaluationScenario.DETECTION:
            # Load from blueprint_{error_type}_picf/label/ structure
            for error_type in self.error_types:
                label_dir = os.path.join(
                    self.ground_truth_dir,
                    f"blueprint_{error_type}_picf",
                    "label"
                )

                if not os.path.exists(label_dir):
                    warnings.warn(f"Label directory not found: {label_dir}")
                    continue

                for filename in os.listdir(label_dir):
                    if not filename.endswith('.json'):
                        continue

                    filepath = os.path.join(label_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        paper_id = filename.replace('.json', '')
                        composite_key = f"{paper_id}:{error_type}"
                        ground_truth[composite_key] = data
                    except Exception as e:
                        warnings.warn(f"Failed to load {filename} from {label_dir}: {e}")

        else:  # QUALITY scenario
            # Load from single directory
            if not os.path.exists(self.ground_truth_dir):
                warnings.warn(f"Ground truth directory not found: {self.ground_truth_dir}")
                return ground_truth

            for filename in os.listdir(self.ground_truth_dir):
                if not filename.endswith('.json'):
                    continue

                filepath = os.path.join(self.ground_truth_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    paper_id = filename.replace('.json', '')
                    ground_truth[paper_id] = data
                except Exception as e:
                    warnings.warn(f"Failed to load {filename}: {e}")

        return ground_truth

    def _extract_review_from_trajectory(self, trajectory: List[Dict]) -> Optional[str]:
        """Extract final review text from trajectory JSON.

        Args:
            trajectory: List of message dictionaries

        Returns:
            Review text or None if not found
        """
        # Strategy 1: Look for tool call with name='write_review'
        for msg in reversed(trajectory):
            if msg.get('role') == 'tool' and msg.get('name') == 'write_review':
                content = msg.get('content', '')
                if content and content != 'N/A':
                    return content

        # Strategy 2: Look for assistant message with 'sections' in tool_calls
        for msg in reversed(trajectory):
            if msg.get('role') == 'assistant':
                tool_calls = msg.get('tool_calls', [])
                if tool_calls:
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            if tool_call.get('function', {}).get('name') == 'write_review':
                                # Extract from arguments
                                args = tool_call.get('function', {}).get('arguments', '')
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except:
                                        pass
                                if isinstance(args, dict):
                                    return json.dumps(args.get('sections', args))

        # Strategy 3: Look in content for review structure
        for msg in reversed(trajectory):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if isinstance(content, str) and len(content) > 100:
                    # Check if it looks like a review
                    if any(keyword in content.lower() for keyword in ['summary', 'strengths', 'weaknesses', 'soundness']):
                        return content

        return None

    def _call_detection_judge(self, review_text: str, ground_truth: Dict) -> Dict:
        """Binary detection evaluation using LLM judge.

        Args:
            review_text: The review to evaluate
            ground_truth: Ground truth containing error information

        Returns:
            Judge evaluation result
        """
        logic_gap = ground_truth.get("logic_gap_summary", "N/A")

        # Escape curly braces in review_text to avoid format string conflicts
        review_text_escaped = review_text.replace('{', '{{').replace('}', '}}')

        prompt = DETECTION_JUDGE_USER_PROMPT.format(
            logic_gap=logic_gap,
            review_text=review_text_escaped
        )

        try:
            response = call_llm(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": DETECTION_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            content = get_content(response).strip()

            # Parse JSON response - handle markdown code blocks
            if "```json" in content:
                # Extract content between ```json and ```
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif content.startswith("```"):
                # Handle generic code block
                lines = content.split('\n')
                # Remove first and last lines (the ``` markers)
                content = '\n'.join(lines[1:-1]).strip()

            result = json.loads(content)

            # Normalize boolean
            if isinstance(result.get('detected'), str):
                result['detected'] = result['detected'].lower() in ['true', 'yes']

            return result

        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse judge response: {e}\nContent preview: {content[:500]}")
            return {
                "detected": False,
                "evidence_from_review": "N/A",
                "reasoning": f"Parse error: {str(e)}"
            }
        except Exception as e:
            warnings.warn(f"Error calling detection judge: {e}")
            return {
                "detected": False,
                "evidence_from_review": "N/A",
                "reasoning": f"Error: {str(e)}"
            }

    def _call_quality_judge(self, review_text: str, paper_info: Dict) -> Dict:
        """Quality evaluation using LLM judge.

        Args:
            review_text: The review to evaluate
            paper_info: Paper information (title, abstract)

        Returns:
            Judge evaluation result with dimension scores
        """
        # Escape curly braces in review_text to avoid format string conflicts
        review_text_escaped = review_text.replace('{', '{{').replace('}', '}}')

        prompt = QUALITY_JUDGE_USER_PROMPT.format(
            paper_title=paper_info.get('title', 'N/A'),
            paper_abstract=paper_info.get('abstract', paper_info.get('summary', 'N/A')),
            review_text=review_text_escaped
        )

        try:
            response = call_llm(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": QUALITY_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            content = get_content(response).strip()

            # Parse JSON response - handle markdown code blocks
            if "```json" in content:
                # Extract content between ```json and ```
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif content.startswith("```"):
                # Handle generic code block
                lines = content.split('\n')
                # Remove first and last lines (the ``` markers)
                content = '\n'.join(lines[1:-1]).strip()

            result = json.loads(content)

            # Ensure all dimensions are present
            dimensions = ['comprehensiveness', 'specificity', 'constructiveness',
                         'accuracy', 'structure', 'overall']
            for dim in dimensions:
                if dim not in result:
                    result[dim] = 3  # Default to middle value

            return result

        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse judge response: {e}")
            return {
                'comprehensiveness': 3,
                'specificity': 3,
                'constructiveness': 3,
                'accuracy': 3,
                'structure': 3,
                'overall': 3,
                'reasoning': f"Parse error: {str(e)}"
            }
        except Exception as e:
            warnings.warn(f"Error calling quality judge: {e}")
            return {
                'comprehensiveness': 3,
                'specificity': 3,
                'constructiveness': 3,
                'accuracy': 3,
                'structure': 3,
                'overall': 3,
                'reasoning': f"Error: {str(e)}"
            }

    def evaluate(self) -> Dict:
        """Evaluate all reviews in the specified directory.

        Returns:
            Dictionary with metadata, aggregate metrics, and per-paper results
        """
        results = []

        if not os.path.exists(self.review_dir):
            warnings.warn(f"Review directory not found: {self.review_dir}")
            return self._create_output({})

        review_files = [
            f for f in os.listdir(self.review_dir)
            if f.endswith('_trajectory.json')
        ]

        if not review_files:
            warnings.warn(f"No trajectory files found in {self.review_dir}")
            return self._create_output({})

        for review_file in tqdm(review_files, desc=f"Evaluating {self.agent_type}/{self.model_name}"):
            try:
                # Load trajectory
                filepath = os.path.join(self.review_dir, review_file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    trajectory = json.load(f)

                # Extract review text
                review_text = self._extract_review_from_trajectory(trajectory)
                if not review_text:
                    warnings.warn(f"Could not extract review from {review_file}")
                    continue

                # Extract paper ID and error type from filename
                # Expected format: {paper_id}_{error_type}_trajectory.json
                # or legacy format: {paper_id}_trajectory.json
                filename_without_suffix = review_file.replace('_trajectory.json', '')

                # Try to split by underscore to get error type
                parts = filename_without_suffix.rsplit('_', 1)
                if len(parts) == 2 and parts[1] in self.error_types:
                    # New format: paper_id_errortype
                    paper_id = parts[0]
                    error_type_from_filename = parts[1]
                else:
                    # Legacy format: just paper_id
                    paper_id = filename_without_suffix
                    error_type_from_filename = None

                # Find ground truth
                ground_truth = None
                error_type_used = None

                if self.scenario == EvaluationScenario.DETECTION:
                    # Priority 1: Use error type from filename
                    if error_type_from_filename:
                        composite_key = f"{paper_id}:{error_type_from_filename}"
                        ground_truth = self.ground_truth.get(composite_key)
                        error_type_used = error_type_from_filename

                    # Priority 2: Try inferred error type from directory path
                    if ground_truth is None and self.inferred_error_type:
                        composite_key = f"{paper_id}:{self.inferred_error_type}"
                        ground_truth = self.ground_truth.get(composite_key)
                        error_type_used = self.inferred_error_type

                    # Priority 3: Try all error types
                    if ground_truth is None:
                        for error_type in self.error_types:
                            composite_key = f"{paper_id}:{error_type}"
                            if composite_key in self.ground_truth:
                                ground_truth = self.ground_truth[composite_key]
                                error_type_used = error_type
                                break
                else:  # QUALITY
                    ground_truth = self.ground_truth.get(paper_id)

                if ground_truth is None:
                    warnings.warn(f"No ground truth found for {paper_id}")
                    continue

                # Call appropriate judge
                if self.scenario == EvaluationScenario.DETECTION:
                    result = self._call_detection_judge(review_text, ground_truth)
                    result['paper_id'] = paper_id
                    result['error_type'] = error_type_used
                else:
                    result = self._call_quality_judge(review_text, ground_truth)
                    result['paper_id'] = paper_id

                    # Compute rewards if enabled
                    if self.reward_calculator:
                        try:
                            # Prepare paper context
                            paper_context = {
                                'title': ground_truth.get('title', 'N/A'),
                                'abstract': ground_truth.get('abstract', ground_truth.get('summary', 'N/A'))
                            }

                            # Get human reviews (may be in different formats)
                            human_reviews = ground_truth.get('reviews', [])
                            if not human_reviews:
                                # Single review in ground_truth itself
                                human_reviews = [ground_truth]

                            # Compute reward
                            reward_result = self.reward_calculator.compute_reward(
                                generated_review=review_text,
                                human_reviews=human_reviews,
                                paper_context=paper_context
                            )

                            # Add reward scores to result
                            result['rewards'] = reward_result.to_dict()
                        except Exception as e:
                            warnings.warn(f"Error computing rewards for {paper_id}: {e}")
                            result['rewards'] = {'error': str(e)}

                results.append(result)

            except Exception as e:
                import traceback
                warnings.warn(f"Error processing {review_file}: {e}\n{traceback.format_exc()}")
                continue

        return self._create_output(results)

    def _compute_detection_metrics(self, results: List[Dict]) -> Dict:
        """Compute detection accuracy metrics.

        Args:
            results: List of detection evaluation results

        Returns:
            Detection metrics dictionary
        """
        if not results:
            return {
                'accuracy': 0.0,
                'total_papers': 0,
                'detected_count': 0
            }

        total = len(results)
        detected = sum(1 for r in results if r.get('detected', False))

        return {
            'accuracy': detected / total if total > 0 else 0.0,
            'total_papers': total,
            'detected_count': detected
        }

    def _compute_quality_metrics(self, results: List[Dict]) -> Dict:
        """Compute mean and std for quality dimensions.

        Args:
            results: List of quality evaluation results

        Returns:
            Quality metrics dictionary
        """
        if not results:
            return {
                'total_papers': 0
            }

        dimensions = ['comprehensiveness', 'specificity', 'constructiveness',
                     'accuracy', 'structure', 'overall']

        metrics = {'total_papers': len(results)}

        for dim in dimensions:
            scores = [r.get(dim, 3) for r in results]
            metrics[f'{dim}_mean'] = float(np.mean(scores))
            metrics[f'{dim}_std'] = float(np.std(scores))

        # Compute reward metrics if available
        results_with_rewards = [r for r in results if 'rewards' in r and 'error' not in r.get('rewards', {})]
        if results_with_rewards:
            reward_components = ['total_reward', 'recall_reward', 'format_reward',
                               'actionable_reward', 'grounded_reward', 'score_difference_reward']

            metrics['total_papers_with_rewards'] = len(results_with_rewards)

            for comp in reward_components:
                scores = [r['rewards'].get(comp, 0.0) for r in results_with_rewards]
                metrics[f'{comp}_mean'] = float(np.mean(scores))
                metrics[f'{comp}_std'] = float(np.std(scores))

        return metrics

    def _create_output(self, results: List[Dict]) -> Dict:
        """Create output JSON structure.

        Args:
            results: List of per-paper results

        Returns:
            Complete output dictionary
        """
        # Compute aggregate metrics
        if self.scenario == EvaluationScenario.DETECTION:
            aggregate_metrics = self._compute_detection_metrics(results)
        else:
            aggregate_metrics = self._compute_quality_metrics(results)

        # Create output structure
        output = {
            "metadata": {
                "scenario": self.scenario.value,
                "agent_type": self.agent_type,
                "model": self.model_name,
                "review_dir": self.review_dir,
                "timestamp": datetime.now().isoformat(),
                "judge_model": self.judge_model
            },
            "aggregate_metrics": aggregate_metrics,
            "per_paper_results": results
        }

        # Add scenario-specific metadata
        if self.scenario == EvaluationScenario.DETECTION:
            output["metadata"]["error_types"] = self.error_types
            output["metadata"]["inferred_error_type"] = self.inferred_error_type

        return output


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate review agent performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detection scenario
  # Note: Trajectory files should be named {paper_id}_{error_type}_trajectory.json
  # e.g., 2023.acl%2023.acl-long.124_conclusion_trajectory.json
  python -m reviewer.evaluation \\
    --scenario detection \\
    --ground-truth-dir datasets/counter-review \\
    --error-types conclusion finding result \\
    --review-dir results/evolving_draft/deepseek-reasoner \\
    --agent-type evolving_draft \\
    --model-name deepseek-reasoner \\
    --output results/detection_results.json

  # Quality scenario
  python -m reviewer.evaluation \\
    --scenario quality \\
    --ground-truth-dir datasets/quality-papers \\
    --review-dir results/evolving_draft/deepseek-reasoner \\
    --agent-type evolving_draft \\
    --model-name deepseek-reasoner \\
    --output results/quality_results.json
        """
    )

    # Required arguments
    parser.add_argument(
        '--scenario',
        choices=['detection', 'quality'],
        required=True,
        help='Evaluation scenario: detection or quality'
    )
    parser.add_argument(
        '--ground-truth-dir',
        required=True,
        help='Base directory containing ground truth files'
    )
    parser.add_argument(
        '--review-dir',
        required=True,
        help='Directory containing review trajectory files'
    )
    parser.add_argument(
        '--agent-type',
        required=True,
        help='Agent type (e.g., evolving_draft)'
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help='Model name (e.g., deepseek-reasoner)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output JSON file path'
    )

    # Optional arguments
    parser.add_argument(
        '--error-types',
        nargs='+',
        help='Error types for detection scenario (e.g., conclusion finding result)'
    )
    parser.add_argument(
        '--judge-model',
        help='LLM judge model (default from judge_prompts.py or MODEL_NAME env var)'
    )
    parser.add_argument(
        '--compute-rewards',
        action='store_true',
        help='Compute reward scores (quality scenario only)'
    )

    args = parser.parse_args()

    # Validate detection scenario requirements
    if args.scenario == 'detection' and not args.error_types:
        parser.error("--error-types is required for detection scenario")

    # Determine scenario
    scenario = (
        EvaluationScenario.DETECTION
        if args.scenario == 'detection'
        else EvaluationScenario.QUALITY
    )

    # Print header
    print("=" * 80)
    print("REVIEW AGENT EVALUATION")
    print("=" * 80)
    print(f"\nScenario: {scenario.value}")
    print(f"Agent: {args.agent_type}")
    print(f"Model: {args.model_name}")
    print(f"Review directory: {args.review_dir}")
    print(f"Ground truth: {args.ground_truth_dir}")
    if args.error_types:
        print(f"Error types: {', '.join(args.error_types)}")
    if args.judge_model:
        print(f"Judge model: {args.judge_model}")

    # Initialize evaluator
    evaluator = ReviewEvaluator(
        ground_truth_dir=args.ground_truth_dir,
        scenario=scenario,
        review_dir=args.review_dir,
        agent_type=args.agent_type,
        model_name=args.model_name,
        judge_model=args.judge_model,
        error_types=args.error_types,
        compute_rewards=args.compute_rewards
    )

    print(f"\nLoaded {len(evaluator.ground_truth)} ground truth files")

    # Run evaluation
    print(f"\nEvaluating...")
    results = evaluator.evaluate()

    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print(f"{scenario.value.upper()} EVALUATION RESULTS")
    print("=" * 80)
    print("\nAggregate Metrics:")
    for key, value in results['aggregate_metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nDetailed results saved to: {args.output}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
