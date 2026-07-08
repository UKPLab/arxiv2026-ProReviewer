"""Inference script for Reviewer-R1 with advanced LLMs (Claude Opus, Gemini, GPT-5, etc.).

Runs the full multi-turn agent-environment loop using ReviewAgent + ReviewEnv,
with LLM calls routed through litellm (acall_llm) so any API or local model works.

Usage:
    # Quick test with Claude Opus
    python -m baselines.reviewer_r1_inference.run_inference \
        --model "anthropic/claude-opus-4" \
        --test_data data/test_data \
        --output_dir outputs/baseline_res/reviewer_r1_advanced \
        --max_samples 1 --n_runs 1 --max_steps 25

    # Full run with Gemini
    python -m baselines.reviewer_r1_inference.run_inference \
        --model "gemini/gemini-2.5-pro" \
        --n_runs 4 --max_steps 25 --concurrency 4

    # Config-name from config.toml
    python -m baselines.reviewer_r1_inference.run_inference \
        --model "gpt-5-4" \
        --n_runs 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import argparse
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reviewer.prompts.reviewer_prompts_advanced import REVIEWER_ADVANCED_SYSTEM_PROMPT
from reviewer.core.proreviewer import ProReviewer as ReviewAgent
from reviewer.core.review_env import ReviewEnv
from utils.helpers.llm import acall_llm, get_content

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> Path:
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"inference_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return log_file


def load_test_data(test_data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test papers from a directory of per-paper JSON triplet files."""
    files = sorted(f for f in os.listdir(test_data_dir) if f.endswith(".json"))
    if max_samples is not None:
        files = files[:max_samples]

    data = []
    for fname in files:
        with open(os.path.join(test_data_dir, fname)) as f:
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
            "clustered_points": triplet.get("clustered_points_gpt-5mini", triplet.get("clustered_points", [])),
        })

    logger.info(f"Loaded {len(data)} papers from '{test_data_dir}'")
    return data


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
    model: str, messages: list, args, max_retries: int = 3
) -> Tuple[Optional[str], Dict[str, int]]:
    """Call LLM via acall_llm with exponential backoff.

    Returns:
        Tuple of (response_text, usage_dict). response_text is None on failure.
    """
    for attempt in range(max_retries):
        try:
            response = await acall_llm(
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return get_content(response), _extract_usage(response)
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait}s...")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
    logger.error(f"LLM call failed after {max_retries} attempts")
    return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def run_one(paper: dict, model: str, args) -> Optional[dict]:
    """Run the multi-turn agent-env loop for a single paper and return the result."""
    paper_id = paper["paper_id"]
    task = {
        "paper_id": paper_id,
        "paper_content": paper["paper_content"],
        "human_avg_score": float(paper.get("human_avg_score", 0)),
        "clustered_points": paper.get("clustered_points", []),
    }

    # Create env with minimal reward mode (scoring done separately by eval_baseline.py)
    env = ReviewEnv(task=task, reward_mode=["format"])
    obs, info = env.reset()

    # Create agent with the advanced prompt
    agent = ReviewAgent(
        system_prompt=REVIEWER_ADVANCED_SYSTEM_PROMPT,
    )
    agent.reset()

    # Feed initial observation to agent with turn budget
    info["max_turns"] = args.max_steps
    info["current_turn"] = 1
    agent.update_from_env(obs, 0, False, info)

    # Accumulators
    done = False
    total_steps = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    trajectory = []  # per-step records

    for step_idx in range(args.max_steps):
        response, usage = await _call_llm_with_retry(model, agent.chat_completions, args)
        for k in total_usage:
            total_usage[k] += usage[k]

        if response is None:
            logger.warning(f"[{paper_id}] LLM call failed at step {step_idx + 1}, stopping early")
            break

        action = agent.update_from_model(response)
        action_dict = action.action if hasattr(action, "action") else action
        next_obs, reward, done, step_info = env.step(action_dict)
        total_steps += 1

        # Record trajectory step
        trajectory.append({
            "step": total_steps,
            "action": step_info.get("action_name", action_dict.get("name", "unknown")),
            "llm_response": response,
            "observation": next_obs.get("action_result", "")[:2000],
            "memory_ops_results": list(agent._last_memory_results) if agent._last_memory_results else [],
        })

        # Inject turn budget into info for the agent
        step_info["max_turns"] = args.max_steps
        step_info["current_turn"] = step_idx + 2  # next turn number
        agent.update_from_env(next_obs, reward, done, step_info)

        if done:
            break

    # Nudge: if agent didn't finish, inject a message telling it to wrap up
    if not done:
        nudge_msg = (
            "You have used all your research steps. You MUST now call the 'finish' action immediately. "
            "Before finishing, add any remaining outline entries (summary, strengths, weaknesses, "
            "questions, overall_score) based on what you have gathered so far. "
            "Do NOT call read_section or search_paper again. Call 'finish' now."
        )
        agent._messages.append({"role": "user", "content": nudge_msg})
        logger.info(f"[{paper_id}] Nudging agent to finish ({args.nudge_steps} extra steps)")

        for extra_idx in range(args.nudge_steps):
            response, usage = await _call_llm_with_retry(model, agent.chat_completions, args)
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

            step_info["max_turns"] = args.max_steps + args.nudge_steps
            step_info["current_turn"] = args.max_steps + extra_idx + 2
            agent.update_from_env(next_obs, reward, done, step_info)
            if done:
                break

    # Extract review: prefer env's finished review, fall back to agent log
    review = env._finished_review
    if review is None:
        logger.warning(f"[{paper_id}] No finish action; extracting review from agent log")
        review = agent.get_review_from_log()

    if review is None:
        logger.warning(f"[{paper_id}] No review produced after {total_steps} steps")
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
        f"[{paper_id}] done={done} steps={total_steps} "
        f"score={result.get('overall_score')} "
        f"w={len(result.get('weaknesses', []))} s={len(result.get('strengths', []))} "
        f"tokens={total_usage['total_tokens']} "
        f"(prompt={total_usage['prompt_tokens']}, completion={total_usage['completion_tokens']})"
    )
    return result


async def run_paper_n_times(paper: dict, model: str, args, output_dir: Path, n_runs: int = 4):
    """Generate n_runs reviews for a single paper, skipping existing outputs."""
    paper_id = paper["paper_id"]

    for i in range(1, n_runs + 1):
        out_path = output_dir / f"{paper_id}_r{i}.json"
        if out_path.exists():
            logger.info(f"[skip] {paper_id}_r{i} (already exists)")
            continue

        logger.info(f"[gen] {paper_id}_r{i}")
        result = await run_one(paper, model, args)

        if result is None:
            logger.error(f"[error] {paper_id}_r{i} failed to generate review")
            continue

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate reviews using Reviewer-R1 agent loop with advanced LLMs"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier: litellm string (e.g. 'anthropic/claude-opus-4'), "
                             "config name from config.toml (e.g. 'gpt-5-4'), "
                             "or local vLLM path")
    parser.add_argument("--test_data", type=str, default="data/test_data",
                        help="Directory with test paper triplets")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline_res/reviewer_r1_advanced",
                        help="Output directory for generated reviews")
    parser.add_argument("--n_runs", type=int, default=4,
                        help="Number of reviews to generate per paper")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of papers to process (None = all)")
    parser.add_argument("--max_steps", type=int, default=25,
                        help="Max agent steps per review")
    parser.add_argument("--nudge_steps", type=int, default=5,
                        help="Extra steps after nudging agent to finish")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens per LLM response")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of papers to process concurrently")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    logger.info(f"Starting Reviewer-R1 advanced inference with args: {vars(args)}")

    # Load test data
    papers = load_test_data(args.test_data, max_samples=args.max_samples)

    # Process papers with concurrency control
    sem = asyncio.Semaphore(args.concurrency)

    async def process_with_sem(paper):
        async with sem:
            await run_paper_n_times(paper, args.model, args, output_dir, n_runs=args.n_runs)

    await asyncio.gather(*[process_with_sem(paper) for paper in papers], return_exceptions=True)

    logger.info(f"Inference complete. Results saved to {output_dir}")
    print(f"\nGenerated reviews saved to: {output_dir}")
    print(f"\nTo evaluate, run:")
    print(f"  python scripts/evaluation/eval_baseline.py \\")
    print(f"    --score_reviews {output_dir} \\")
    print(f"    --triplets_dir {args.test_data} \\")
    print(f"    --output_dir {output_dir.parent / (output_dir.name + '_eval')} \\")
    print(f"    --reward_mode rubric,format,score_diff,utility \\")
    print(f"    --judge_model utility-score \\")
    print(f"    --rubric_model deepseek-v4 \\")
    print(f"    --batch_rubric_weaknesses")


if __name__ == "__main__":
    asyncio.run(main())
