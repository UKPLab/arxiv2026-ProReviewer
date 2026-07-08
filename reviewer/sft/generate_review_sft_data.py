"""Generate SFT training data from human reviews via guided reconstruction.

For each paper, selects all qualified human reviews, runs the teacher LLM
through the real ReviewAgent + ReviewEnv loop with each gold review in its system
prompt, then saves the traces with the standard system prompt for SFT training.
Output files are named {paper_id}_{reviewer_id}.json.

Example usage:
    python -m reviewer.rllm_version.generate_review_sft_data \\
        --data_path data/paper_triplets/iclr2026 \\
        --output_dir outputs/sft_traces \\
        --model Qwen/Qwen3-32B \\
        --api_base http://localhost:8000/v1 \\
        --concurrency 8
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from reviewer.rllm_version.trace_generator import TraceGenerator
from reviewer.reward.score_review import split_review_text
from reviewer.reward.rubric_evaluator import RubricEvaluator, UTILITY_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


MIN_PAPER_CONTENT_LENGTH = 2000  # chars — filters out stubs or parsing failures

def load_papers(data_path: str, max_papers: int = None) -> list[dict]:
    """Load paper JSONs from a directory, applying basic quality filters."""
    papers = []
    for json_file in sorted(Path(data_path).glob("**/*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            content = data.get("markdown", {}).get("content", "")
            if len(content) < MIN_PAPER_CONTENT_LENGTH:
                logger.warning(f"Skipping {json_file.stem}: content too short ({len(content)} chars)")
                continue
            if not data.get("reviews"):
                logger.warning(f"Skipping {json_file.stem}: no reviews")
                continue
            inject_initial_ratings(data)
            papers.append(data)
        except Exception as e:
            logger.warning(f"Skipping {json_file}: {e}")

        if max_papers and len(papers) >= max_papers:
            break

    return papers


async def compute_human_review_utility(
    evaluator: RubricEvaluator,
    human_review: dict,
) -> tuple[float, int]:
    """Score weaknesses of a human review point-by-point.

    Splits the weaknesses text into individual bullet points via split_review_text(),
    scores each point using RubricEvaluator, and returns the average.

    Returns:
        (avg_utility_score, num_points). Score is 0.0 if there are no weakness points.
    """
    weaknesses_text = human_review.get("weaknesses", "")
    points = split_review_text(weaknesses_text)
    if not points:
        return 0.0, 0

    eval_result = await evaluator.evaluate(points, "")
    # Normalize from 1-5 to 0-1: (avg - 1) / 4
    score = max(0.0, (eval_result["overall"] - 1) / 4)
    return score, len(points)


async def process_paper(
    i: int,
    total: int,
    paper: dict,
    generator: TraceGenerator,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
    skip_existing: bool = False,
    evaluator: Optional[RubricEvaluator] = None,
    utility_threshold: Optional[float] = None,
) -> list[bool | None]:
    """Run trace generation for one paper with all qualified reviews.

    Generates one trace per qualified review, saved as <output_dir>/<paper_id>_<reviewer_id>.json.

    Returns a list of results (True/False/None) — one per qualified review.
    """
    paper_id = paper.get("paper_id") or paper.get("id", f"paper_{i}")

    reviews = paper.get("reviews", [])
    qualified_reviews = TraceGenerator.select_qualified_reviews(reviews)
    if not qualified_reviews:
        logger.warning(f"[{i+1}/{total}] {paper_id}: no review passes quality filters, skipping")
        return [None]

    logger.info(f"[{i+1}/{total}] {paper_id}: {len(qualified_reviews)} qualified review(s)")

    results = []
    for review in qualified_reviews:
        reviewer_id = review.get("id", "unknown")
        trace_name = f"{paper_id}_{reviewer_id}"

        if skip_existing and (output_dir / f"{trace_name}.json").exists():
            logger.info(f"[{i+1}/{total}] {trace_name}: already exists, skipping")
            results.append(None)
            continue

        logger.info(
            f"[{i+1}/{total}] {trace_name} — "
            f"rating {review.get('rating', '?')}, "
            f"confidence {review.get('confidence', '?')}"
        )

        # --- Utility pre-filter: score the human review's weaknesses before generating ---
        if evaluator is not None and utility_threshold is not None:
            utility_score, num_points = await compute_human_review_utility(evaluator, review)
            if utility_score < utility_threshold:
                logger.info(
                    f"[{i+1}/{total}] {trace_name}: human review utility {utility_score:.3f} "
                    f"({num_points} pts) < threshold {utility_threshold} — skipping"
                )
                results.append(None)
                continue
            logger.info(
                f"[{i+1}/{total}] {trace_name}: human review utility {utility_score:.3f} "
                f"({num_points} pts) — OK"
            )

        t0 = time.perf_counter()
        async with semaphore:
            result = await generator.generate_trace(paper, review)
        elapsed = time.perf_counter() - t0

        if result is None:
            logger.info(f"[{trace_name}] Failed after {elapsed:.1f}s")
            results.append(None)
            continue

        messages, is_success = result
        trace = {"paper_id": paper_id, "reviewer_id": reviewer_id, "is_success": is_success, "messages": messages}
        out_path = output_dir / f"{trace_name}.json"
        with open(out_path, "w") as f:
            json.dump(trace, f, indent=2)
        logger.info(f"[{trace_name}] Saved trace to {out_path} ({elapsed:.1f}s)")
        results.append(is_success)

    return results


async def async_main(args):
    papers = load_papers(args.data_path, args.max_papers)
    logger.info(f"Loaded {len(papers)} papers from {args.data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key)
    generator = TraceGenerator(
        llm_client=client,
        model_name=args.model,
        max_steps=args.max_steps,
        action_retries=args.action_retries,
    )

    evaluator = None
    if args.utility_threshold is not None:
        judge_model = args.judge_model or "revutil"
        evaluator = RubricEvaluator(UTILITY_CONFIG, model=judge_model)
        logger.info(
            f"Utility pre-filter enabled: threshold={args.utility_threshold}, "
            f"judge_model={judge_model}"
        )

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_paper(
            i, len(papers), paper, generator, semaphore, output_dir,
            args.skip_existing, evaluator, args.utility_threshold
        )
        for i, paper in enumerate(papers)
    ]

    t_start = time.perf_counter()
    paper_results = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - t_start

    # Flatten: each paper returns a list of per-review results
    results = [r for paper_res in paper_results for r in paper_res]

    n_success = sum(1 for r in results if r is True)
    n_incomplete = sum(1 for r in results if r is False)
    n_skip = sum(1 for r in results if r is None)
    logger.info(
        f"Summary: {n_success} succeeded, {n_incomplete} incomplete, "
        f"{n_skip} skipped/failed out of {len(results)} traces ({len(papers)} papers)"
    )
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    if n_success + n_incomplete > 0:
        logger.info(f"Avg time per trace: {total_elapsed/(n_success + n_incomplete):.1f}s")
    logger.info(f"Traces saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT data from gold human reviews")
    parser.add_argument("--data_path", required=True, help="Directory of paper JSONs")
    parser.add_argument("--output_dir", default="sft_traces", help="Directory to save per-paper JSON traces")
    parser.add_argument("--model", required=True, help="Teacher LLM model name")
    parser.add_argument("--api_base", required=True, help="vLLM/OpenAI-compatible endpoint URL")
    parser.add_argument("--api_key", default="EMPTY", help="API key (default: EMPTY)")
    parser.add_argument("--max_steps", type=int, default=30, help="Max agent steps per trace")
    parser.add_argument("--max_papers", type=int, default=None, help="Limit number of papers")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent traces")
    parser.add_argument("--skip_existing", action="store_true", help="Skip papers that already have output files")
    parser.add_argument("--action_retries", type=int, default=3, help="Max retries per step on action/memory errors (default: 3)")
    parser.add_argument(
        "--utility_threshold", type=float, default=None,
        help="Minimum utility score [0-1] for the human review's weaknesses. "
             "Papers below this threshold are skipped before trace generation. Default: no filtering."
    )
    parser.add_argument(
        "--judge_model", type=str, default=None,
        help="LLM judge model for utility scoring (default: DEFAULT_JUDGE_MODEL)."
    )
    parser.add_argument(
        "--judge_api_key", type=str, default="EMPTY",
        help="API key for the judge model (default: EMPTY)."
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
