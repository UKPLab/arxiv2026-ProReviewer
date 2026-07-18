"""Inference and evaluation for the Reviewer-R1 agent.

Three entry points:
  - load_test_data: load paper triplet JSONs
  - run_inference:  run ProReviewer + ReviewEnv loop for one paper
  - score_reviews:  score pre-generated review JSONs using async_score_review

CLI:
  python evaluation.py --mode infer  --model claude-opus-4 --test_data data/test_data --output_dir outputs/reviews
  python evaluation.py --mode score  --reviews_dir outputs/reviews --triplets_dir data/test_data --output_dir outputs/eval
  python evaluation.py --mode run    --model claude-opus-4 --test_data data/test_data --output_dir outputs/eval
"""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from reviewer.core.proreviewer import ProReviewer
from reviewer.core.review_env import ReviewEnv
from utils.helpers.llm import acall_llm, get_content

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load paper triplet JSONs from *data_dir*.

    Each JSON is expected to have at minimum:
      - markdown.content (str)
      - optionally: paper_id, title, scores.rating_avg

    Returns a list of dicts with keys:
      paper_id, paper_content, human_avg_score
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if max_samples is not None:
        files = files[:max_samples]

    data: List[Dict] = []
    for fname in files:
        with open(os.path.join(data_dir, fname)) as f:
            triplet = json.load(f)

        paper_id = triplet.get("paper_id", fname.replace(".json", ""))
        title = triplet.get("title", "")
        content = triplet["markdown"]["content"]
        if title and not content.startswith(f"# {title}") and not content.lower().startswith("title:"):
            content = f"# {title}\n\n{content}"

        data.append({
            "paper_id": paper_id,
            "paper_content": content,
            "human_avg_score": float(triplet.get("scores", {}).get("rating_avg", 0)),
        })

    logger.info("Loaded %d papers from '%s'", len(data), data_dir)
    return data


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _extract_usage(response: Any) -> Dict[str, int]:
    """Extract token usage from an LLM response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0
    if total == 0:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


async def _call_llm_with_retry(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Tuple[Optional[str], Dict[str, int]]:
    """Call LLM via acall_llm with exponential back-off."""
    for attempt in range(max_retries):
        try:
            response = await acall_llm(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return get_content(response), _extract_usage(response)
        except Exception as e:
            wait = 2 ** attempt
            logger.warning("LLM attempt %d/%d failed: %s. Retrying in %ds...", attempt + 1, max_retries, e, wait)
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
    logger.error("LLM call failed after %d attempts", max_retries)
    return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def run_inference(
    paper: Dict,
    model: str,
    max_steps: int = 25,
    nudge_steps: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
) -> Optional[Dict]:
    """Run the multi-turn ProReviewer + ReviewEnv loop for a single paper.

    Args:
        paper: dict with paper_id, paper_content, human_avg_score
        model: LLM model identifier (litellm string, config name, or local path)
        max_steps: maximum agent steps before nudging
        nudge_steps: extra steps after nudge to finish
        temperature: sampling temperature
        max_tokens: max tokens per LLM response
        system_prompt: optional custom system prompt (defaults to built-in)

    Returns:
        Review dict with paper_id, summary, strengths, weaknesses, questions,
        overall_score, n_steps, token_usage, trajectory; or None on failure.
    """
    paper_id = paper["paper_id"]
    task = {
        "paper_id": paper_id,
        "paper_content": paper["paper_content"],
        "human_avg_score": float(paper.get("human_avg_score", 0)),
    }

    env = ReviewEnv(task=task, reward_mode=["format"])
    obs, info = env.reset()

    agent = ProReviewer(
        system_prompt=system_prompt,
    )
    agent.reset()

    info["max_turns"] = max_steps
    info["current_turn"] = 1
    agent.update_from_env(obs, 0, False, info)

    done = False
    total_steps = 0
    total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    trajectory: List[Dict] = []

    for step_idx in range(max_steps):
        response, usage = await _call_llm_with_retry(model, agent.chat_completions, temperature, max_tokens)
        for k in total_usage:
            total_usage[k] += usage[k]

        if response is None:
            logger.warning("[%s] LLM call failed at step %d, stopping early", paper_id, step_idx + 1)
            break

        action = agent.update_from_model(response)
        action_dict = action.action if hasattr(action, "action") else action
        next_obs, reward, done, step_info = env.step(action_dict)
        total_steps += 1

        trajectory.append({
            "step": total_steps,
            "action": step_info.get("action_name", action_dict.get("name", "unknown")),
            "llm_response": response,
            "observation": next_obs.get("action_result", "")[:2000],
            "memory_ops_results": list(agent._last_memory_results) if agent._last_memory_results else [],
        })

        step_info["max_turns"] = max_steps
        step_info["current_turn"] = step_idx + 2
        agent.update_from_env(next_obs, reward, done, step_info)

        if done:
            break

    # Nudge the agent to finish if it hasn't already
    if not done:
        nudge_msg = (
            "You have used all your research steps. You MUST now call the 'finish' action immediately. "
            "Before finishing, add any remaining outline entries (summary, strengths, weaknesses, "
            "questions, overall_score) based on what you have gathered so far. "
            "Do NOT call read_section or search_paper again. Call 'finish' now."
        )
        agent._messages.append({"role": "user", "content": nudge_msg})
        logger.info("[%s] Nudging agent to finish (%d extra steps)", paper_id, nudge_steps)

        for extra_idx in range(nudge_steps):
            response, usage = await _call_llm_with_retry(model, agent.chat_completions, temperature, max_tokens)
            for k in total_usage:
                total_usage[k] += usage[k]

            if response is None:
                break
            action = agent.update_from_model(response)
            action_dict = action.action if hasattr(action, "action") else action
            next_obs, reward, done, step_info = env.step(action_dict)
            total_steps += 1

            trajectory.append({
                "step": total_steps,
                "action": step_info.get("action_name", action_dict.get("name", "unknown")),
                "llm_response": response,
                "observation": next_obs.get("action_result", "")[:2000],
                "memory_ops_results": list(agent._last_memory_results) if agent._last_memory_results else [],
                "nudge": True,
            })

            step_info["max_turns"] = max_steps + nudge_steps
            step_info["current_turn"] = max_steps + extra_idx + 2
            agent.update_from_env(next_obs, reward, done, step_info)
            if done:
                break

    # Extract review
    review = env._finished_review
    if review is None:
        logger.warning("[%s] No finish action; extracting review from agent log", paper_id)
        review = agent.get_review_from_log()

    if review is None:
        logger.warning("[%s] No review produced after %d steps", paper_id, total_steps)
        return None

    result = {
        "paper_id": paper_id,
        "summary": review.get("summary", ""),
        "strengths": review.get("strengths", []),
        "weaknesses": review.get("weaknesses", []),
        "questions": review.get("questions", []),
        "overall_score": review.get("overall_score"),
        "n_steps": total_steps,
        "token_usage": total_usage,
        "trajectory": trajectory,
    }

    logger.info(
        "[%s] done=%s steps=%d score=%s w=%d s=%d tokens=%d",
        paper_id, done, total_steps, result.get("overall_score"),
        len(result.get("weaknesses", [])), len(result.get("strengths", [])),
        total_usage["total_tokens"],
    )
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

async def score_reviews(
    reviews_dir: str,
    triplets_dir: str,
    output_dir: str,
    reward_modes: Set[str],
    rubric_model: str = "revutil",
    batched: bool = False,
    concurrency: int = 8,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Score pre-generated review JSONs using async_score_review.

    Reads review JSONs from *reviews_dir* (each with at least summary,
    strengths, weaknesses, questions, overall_score), joins with triplets
    from *triplets_dir* for ground-truth signals, and writes per-review
    scored results to ``<output_dir>/papers/<paper_id>.json``.

    Args:
        reviews_dir: directory of review JSON files
        triplets_dir: directory of paper triplet JSONs (ground truth)
        output_dir: output directory for scored results
        reward_modes: set of reward modes (e.g. {"format", "rubric", "score_diff"})
        rubric_model: model for rubric evaluation
        concurrency: number of concurrent scoring tasks
        max_samples: limit number of reviews to score

    Returns:
        List of per-review scored result dicts.
    """
    from reviewer.reward.score_review import async_score_review

    out = Path(output_dir)
    papers_dir = out / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    # Load reviews
    review_files = sorted(f for f in os.listdir(reviews_dir) if f.endswith(".json"))
    if max_samples is not None:
        review_files = review_files[:max_samples]

    items: List[Dict] = []
    for fname in review_files:
        with open(os.path.join(reviews_dir, fname)) as f:
            review_data = json.load(f)

        paper_id = fname.replace(".json", "")
        base_id = re.sub(r"_r\d+$", "", paper_id)

        # Load triplet for ground truth
        triplet_path = os.path.join(triplets_dir, f"{base_id}.json")
        human_avg_score = None
        paper_content: Optional[str] = None

        if os.path.exists(triplet_path):
            with open(triplet_path) as f:
                triplet = json.load(f)
            human_avg_score = float(triplet.get("scores", {}).get("rating_avg", 0) or 0)
            paper_content = triplet.get("markdown", {}).get("content")
            title = triplet.get("title", "")
            if paper_content and title and not paper_content.startswith(f"# {title}"):
                paper_content = f"# {title}\n\n{paper_content}"
        else:
            logger.warning("[%s] triplet not found at %s", paper_id, triplet_path)

        # Build review object
        review_obj = {
            "summary": review_data.get("summary", ""),
            "strengths": review_data.get("strengths", []),
            "weaknesses": review_data.get("weaknesses", []),
            "questions": review_data.get("questions", []),
            "overall_score": review_data.get("overall_score"),
        }

        items.append({
            "paper_id": paper_id,
            "review": review_obj,
            "human_avg_score": human_avg_score,
            "paper_content": paper_content,
        })

    logger.info("Loaded %d reviews from '%s'", len(items), reviews_dir)

    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(item: Dict) -> Dict:
        pid = item["paper_id"]
        paper_file = papers_dir / f"{pid}.json"

        if paper_file.exists():
            logger.info("[%s] skipping (already scored)", pid)
            with open(paper_file) as f:
                return json.load(f)

        review = item["review"]
        if not review.get("summary") and not review.get("strengths") and not review.get("weaknesses"):
            failed = {"paper_id": pid, "has_review": False, "error": "Empty review"}
            with open(paper_file, "w") as f:
                json.dump(failed, f, indent=2, default=str)
            return failed

        async with semaphore:
            result = await async_score_review(
                review=review,
                human_avg_score=item.get("human_avg_score"),
                reward_modes=reward_modes,
                rubric_model=rubric_model,
                paper_content=item.get("paper_content"),
                batched=batched,
            )

        # Remap keys to standard names
        scored = {
            "paper_id": pid,
            "has_review": True,
            "format_reward": result.pop("format", None),
            "score_diff_reward": result.pop("score_diff", None),
            "rubric_reward": result.pop("rubric", None),
            "rubric_scores": result.pop("rubric_scores", None),
            **result,
        }

        with open(paper_file, "w") as f:
            json.dump(scored, f, indent=2, default=str)

        scores_str = " ".join(
            f"{k}={v:.4f}" for k, v in scored.items()
            if isinstance(v, (int, float)) and k != "has_review"
        )
        logger.info("[%s]: %s", pid, scores_str)
        return scored

    # Group by base paper for prefix-cache locality
    paper_groups: Dict[str, List[Dict]] = {}
    for item in items:
        base_id = re.sub(r"_r\d+$", "", item["paper_id"])
        paper_groups.setdefault(base_id, []).append(item)

    async def score_group(group: List[Dict]) -> List[Dict]:
        return [await score_one(item) for item in group]

    group_results = await asyncio.gather(*[score_group(g) for g in paper_groups.values()])
    all_results = [r for group in group_results for r in group]

    # Write summary
    summary = _compute_summary(all_results)
    summary_file = out / "score_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Scored %d reviews. Summary: %s", len(all_results), summary_file)
    return all_results


def _compute_summary(results: List[Dict]) -> Dict:
    """Compute aggregate mean/std for standard reward metrics."""
    completed = [r for r in results if r.get("has_review")]
    metrics = ["format_reward", "score_diff_reward", "rubric_reward"]
    summary: Dict[str, Any] = {
        "total": len(results),
        "completed": len(completed),
    }
    for key in metrics:
        vals = [r[key] for r in completed if r.get(key) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
            summary[key] = {"mean": round(mean, 4), "std": round(std, 4), "count": len(vals)}
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / f"eval_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )


async def _cli_infer(args) -> None:
    """CLI handler for --mode infer."""
    output_dir = Path(args.output_dir)
    _setup_logging(output_dir)

    papers = load_test_data(args.test_data, max_samples=args.max_samples)
    sem = asyncio.Semaphore(args.concurrency)

    async def process(paper: Dict) -> None:
        paper_id = paper["paper_id"]
        for run_idx in range(1, args.n_runs + 1):
            out_path = output_dir / f"{paper_id}_r{run_idx}.json"
            if out_path.exists():
                logger.info("[skip] %s_r%d (exists)", paper_id, run_idx)
                continue

            async with sem:
                result = await run_inference(
                    paper, args.model,
                    max_steps=args.max_steps,
                    nudge_steps=args.nudge_steps,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

            if result is None:
                logger.error("[error] %s_r%d failed", paper_id, run_idx)
                continue

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

    await asyncio.gather(*[process(p) for p in papers])
    print(f"\nReviews saved to: {output_dir}")


async def _cli_score(args) -> None:
    """CLI handler for --mode score."""
    output_dir = Path(args.output_dir)
    _setup_logging(output_dir)

    modes = _parse_reward_modes(args.reward_mode)
    results = await score_reviews(
        reviews_dir=args.reviews_dir,
        triplets_dir=args.triplets_dir,
        output_dir=args.output_dir,
        reward_modes=modes,
        rubric_model=args.rubric_model,
        batched=args.batched,
        concurrency=args.concurrency,
        max_samples=args.max_samples,
    )
    _print_summary(results)


async def _cli_run(args) -> None:
    """CLI handler for --mode run (infer + score)."""
    # Infer first
    await _cli_infer(args)

    # Then score
    args.reviews_dir = args.output_dir
    args.triplets_dir = args.test_data
    eval_dir = str(Path(args.output_dir).parent / (Path(args.output_dir).name + "_eval"))
    args.output_dir = eval_dir
    await _cli_score(args)


def _parse_reward_modes(reward_mode_str: str) -> Set[str]:
    FULL_MODES = {"format", "score_diff", "rubric"}
    if reward_mode_str == "full":
        return FULL_MODES
    modes = set(reward_mode_str.split(","))
    invalid = modes - FULL_MODES
    if invalid:
        raise ValueError(f"Invalid reward modes: {invalid}")
    return modes


def _print_summary(results: List[Dict]) -> None:
    completed = [r for r in results if r.get("has_review")]
    print(f"\nScored {len(completed)}/{len(results)} reviews")
    metrics = ["format_reward", "score_diff_reward", "rubric_reward"]
    for key in metrics:
        vals = [r[key] for r in completed if r.get(key) is not None]
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  {key}: {mean:.4f} (n={len(vals)})")


def main():
    parser = argparse.ArgumentParser(
        description="Reviewer-R1 inference and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["infer", "score", "run"], required=True,
                        help="infer: generate reviews; score: score existing reviews; run: both")

    # Inference args
    parser.add_argument("--model", type=str, help="LLM model identifier (required for infer/run)")
    parser.add_argument("--test_data", type=str, default="data/test_data",
                        help="Directory with test paper triplets")
    parser.add_argument("--max_steps", type=int, default=25, help="Max agent steps per review")
    parser.add_argument("--nudge_steps", type=int, default=5, help="Extra steps after nudge")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per LLM response")
    parser.add_argument("--n_runs", type=int, default=4, help="Number of reviews per paper")

    # Scoring args
    parser.add_argument("--reviews_dir", type=str, help="Directory of review JSONs (for score mode)")
    parser.add_argument("--triplets_dir", type=str, default="data/test_data",
                        help="Directory with paper triplet JSONs (ground truth)")
    parser.add_argument("--reward_mode", type=str, default="rubric,format,score_diff",
                        help="Comma-separated reward modes or 'full'")
    parser.add_argument("--rubric_model", type=str, default="revutil",
                        help="Model for rubric evaluation (LLM-as-a-judge)")
    parser.add_argument("--batched", action="store_true",
                        help="Use batched rubric evaluation (single LLM call for all weaknesses)")
    # Common args
    parser.add_argument("--output_dir", type=str, default="outputs/eval", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of papers/reviews")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent tasks")

    args = parser.parse_args()

    # Validate
    if args.mode in ("infer", "run") and not args.model:
        parser.error("--model is required for infer/run mode")
    if args.mode == "score" and not args.reviews_dir:
        parser.error("--reviews_dir is required for score mode")

    if args.mode == "infer":
        asyncio.run(_cli_infer(args))
    elif args.mode == "score":
        asyncio.run(_cli_score(args))
    else:
        asyncio.run(_cli_run(args))


if __name__ == "__main__":
    main()
