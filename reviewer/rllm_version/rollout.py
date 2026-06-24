import asyncio
import json
import argparse
import logging
from functools import partial
from pathlib import Path
from datetime import datetime
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from reviewer.rllm_version.review_env import ReviewEnv
from reviewer.rllm_version.review_agent import ReviewAgent
from reviewer.core.reviewer_prompts_direct import REVIEWER_DIRECT_SYSTEM_PROMPT
from transformers import AutoTokenizer
import tiktoken


def setup_logging():
    """Setup logging to both file and console."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rollout_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return log_file


async def main(direct: bool = False, model_name: str = "Qwen/Qwen3-8B", reward_mode: str = "full"):
    # Setup logging first
    log_file = setup_logging()
    logger = logging.getLogger(__name__)

    if "gpt" in model_name.lower() or "openai" in model_name.lower():
        tokenizer = tiktoken.get_encoding("o200k_base")
        logger.info(f"Using tiktoken encoding 'o200k_base' for model: {model_name}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Using HuggingFace tokenizer for model: {model_name}")

    if direct:
        agent_class = partial(ReviewAgent, system_prompt=REVIEWER_DIRECT_SYSTEM_PROMPT)
    else:
        agent_class = ReviewAgent

    engine = AgentExecutionEngine(
        agent_class=agent_class,
        env_class=ReviewEnv,
        engine_name="openai",
        tokenizer=tokenizer,
        n_parallel_agents=64,
        max_steps=15,
        max_response_length=32768,
        max_prompt_length=32768,
        sampling_params={"temperature": 0.6, "top_p": 0.95, "extra_body": {"top_k": 20}},
        rollout_engine_args={
            # "base_url": "https://api.openai.com/v1/models",
            "base_url": "http://localhost:8000/v1/models",
            "model": model_name
        }
    )

    with open("/pfss/mlde/workspaces/mlde_wsp_Reviewer_R1/Reviewer-R1/data/paper_triplets/iclr2025/gpt5-mini/0Ag8FQ5Rr3.json", "r") as f:
        paper_data = json.load(f)

        tasks = [{
        "task": {
            "paper_id": paper_data.get("paper_id", "paper0"),
            "paper_content": f"# {paper_data['title']}\n\n" + paper_data["markdown"]["content"],
            "human_avg_score": paper_data["scores"]["rating_avg"],
            "clustered_points": paper_data["clustered_points"],
        },
        "reward_mode": reward_mode,
    }]

    if not direct:
        tasks[0]["research_model"] = "gpt-5mini"

    results = await engine.execute_tasks(tasks)
    logger.info(f"Execution completed. Log saved to: {log_file}")
    return results, log_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rollout")
    parser.add_argument("--direct", action="store_true", help="Use direct investigation agent (no research subagent)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model name for the rollout engine")
    parser.add_argument("--reward_mode", type=str, default="syntactic", choices=["syntactic", "utility", "full"],
                        help="Reward mode: 'syntactic' for Stage 1 format-only rewards, 'utility' for weakness utility rewards, 'full' for LLM judge rewards")
    args = parser.parse_args()

    results, log_file = asyncio.run(main(direct=args.direct, model_name=args.model_name, reward_mode=args.reward_mode))

    output_file = "./data/results/rollout_results.json"
    with open(output_file, "w") as f:
        json.dump(results[0].to_dict(), f, indent=4)

    print(f"\n{'='*60}")
    print(f"Rollout complete!")
    print(f"Results saved to: {output_file}")
    print(f"Logs saved to: {log_file}")
    print(f"{'='*60}\n")