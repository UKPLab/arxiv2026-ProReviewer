"""SFT training script for Reviewer-R1 using rLLM AgentSFTTrainer.

Converts JSON traces from outputs/sft_traces to parquet, then runs SFT.

Usage:
    # Single-node, 4 GPUs:
    torchrun --nproc_per_node=4 -m reviewer.rllm_version.train_sft \
        model.partial_pretrain=Qwen/Qwen3-8B \
        data.train_files=outputs/sft_parquet/train.parquet \
        data.val_files=outputs/sft_parquet/val.parquet

    # Or first convert traces, then train:
    python -m reviewer.rllm_version.train_sft --convert_only \
        --traces_dir outputs/sft_traces --output_dir outputs/sft_parquet

    # Convert + train in one go (the shell script does this):
    See train_scripts/train_sft.sh
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

from reviewer.core.reviewer_prompts_direct import REVIEWER_DIRECT_SYSTEM_PROMPT


def _has_empty_review_sections(messages: list[dict]) -> bool:
    """Check if the final review outline has empty summary, strengths, or weaknesses.

    The last message is a user message containing the Review Outline.
    If a section is empty, to_prompt_str() omits its label entirely,
    so we just check whether each label appears in the content.
    """
    last_msg = messages[-1] if messages else None
    if last_msg is None:
        return True

    content = last_msg.get("content", "")
    if not content:
        return True

    # Empty sections are omitted entirely by to_prompt_str(),
    # so a missing label means the section was empty.
    for section in ("\nSummary:", "\nStrengths:", "\nWeaknesses:"):
        if section not in content:
            return True

    return False


def decompose_trace_to_mdp_steps(messages: list[dict], memory_in_first_message: bool = False) -> list[dict]:
    """Decompose a full multi-turn trace into MDP-style per-step examples.

    Mirrors the sliding window used during RL inference (ReviewAgent.update_from_env):
        Step 0: [system, initial_obs] → assistant[0]
        Step k: [system, initial_obs, prev_assistant, cur_observation] → assistant[k]

    Each example has 3 or 5 messages (input) + 1 assistant (output).
    """
    # messages alternate: system, user, assistant, user, assistant, ...
    # indices: system=0, user=1, asst=2, user=3, asst=4, user=5, asst=6, ...
    if len(messages) < 3:
        raise ValueError(f"Expected at least 3 messages (system, obs, asst), got {len(messages)}")

    system_msg = messages[0]   # system prompt
    initial_obs = messages[1]  # paper title + sections

    steps = []

    # Step 0: [system, initial_obs] → assistant[2]
    steps.append({
        "messages": [system_msg, initial_obs, messages[2]]
    })

    # Subsequent steps: [system, initial_obs, prev_assistant, observation] → assistant
    # assistant indices: 2, 4, 6, 8, ...  (i.e., messages[2k+2] for k=0,1,2,...)
    # observation indices: 3, 5, 7, 9, ... (i.e., messages[2k+1] for k=1,2,3,...)
    k = 1
    while True:
        prev_asst_idx = 2 * k    # previous assistant response
        obs_idx = 2 * k + 1      # observation for this step
        asst_idx = 2 * k + 2     # assistant response for this step


        if asst_idx >= len(messages):
            break
        if messages[asst_idx]["role"] != "assistant":
            break
        if memory_in_first_message:
            review_log = ""
            if "<current_log_state>\n===" in messages[obs_idx]["content"]:
                try:
                    review_log = messages[obs_idx]["content"].split("<current_log_state>\n=== Complete Review Log ===")[1].split("</current_log_state>")[0].strip()
                except Exception as e:
                    print(messages[obs_idx]["content"])
                    exit(1)

            first_step_content = initial_obs["content"] + f"\n\nYour current review log is:\n<current_log_state>\n{review_log}\n</current_log_state> \n\n[Turn {k}/30]"
            first_step_msg = {"role": "user", "content": first_step_content}

            # update the observation to only include the new content (strip the log context)
            observation = messages[obs_idx]["content"].split("<current_log_state>\n=== Complete Review Log ===")[0].strip()
            messages[obs_idx]["content"] = "The message above is your previous response. Below is the environment's feedback from your last action.\n\n" + observation

        steps.append({
            "messages": [
                system_msg,
                first_step_msg if memory_in_first_message else initial_obs,
                messages[prev_asst_idx],  # last assistant
                messages[obs_idx],         # current observation
                messages[asst_idx],        # target assistant response
            ]
        })
        
        k += 1

    return steps


def convert_traces_to_parquet(
    traces_dir: str,
    output_dir: str,
    val_ratio: float = 0.05,
    success_only: bool = True,
) -> tuple[str, str]:
    """Convert JSON SFT traces to MDP-decomposed train/val parquet files.

    Each trace is decomposed into per-step examples matching the sliding window
    used during RL inference. Returns (train_path, val_path).
    """
    all_steps = []
    n_traces = 0

    for fname in sorted(os.listdir(traces_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(traces_dir, fname)) as f:
            data = json.load(f)

        if success_only and not data.get("is_success", False):
            continue

        messages = data.get("messages", [])
        if len(messages) < 3:
            raise ValueError(f"Trace {fname} has too few messages ({len(messages)})")

        # Strip to role + content only, replacing teacher system prompt with inference prompt
        clean = []
        for m in messages:
            msg = {"role": m["role"], "content": m.get("content", "")}
            if m["role"] == "system":
                msg["content"] = REVIEWER_DIRECT_SYSTEM_PROMPT
            clean.append(msg)

        # Skip traces with empty summary, strengths, or weaknesses
        if _has_empty_review_sections(clean):
            print(f"Skipping {fname}: empty summary/strengths/weaknesses in final review")
            continue

        paper_id = data.get("paper_id", fname.replace(".json", ""))
        steps = decompose_trace_to_mdp_steps(clean)
        for step in steps:
            step["paper_id"] = paper_id
        all_steps.extend(steps)
        n_traces += 1

    print(f"Loaded {n_traces} traces → {len(all_steps)} MDP steps from {traces_dir}")
    if not all_steps:
        raise ValueError("No valid steps found!")

    # Split by paper_id to avoid data leakage
    random.seed(42)
    paper_ids = sorted(set(s["paper_id"] for s in all_steps))
    random.shuffle(paper_ids)
    n_val_papers = max(1, int(len(paper_ids) * val_ratio))
    val_paper_set = set(paper_ids[:n_val_papers])

    val_steps = [s for s in all_steps if s["paper_id"] in val_paper_set]
    train_steps = [s for s in all_steps if s["paper_id"] not in val_paper_set]
    random.shuffle(train_steps)
    random.shuffle(val_steps)

    msg_counts = [len(s["messages"]) for s in train_steps]
    print(f"Train: {len(train_steps)}, Val: {len(val_steps)}")
    print(f"Messages per step: min={min(msg_counts)}, max={max(msg_counts)}, avg={sum(msg_counts)//len(msg_counts)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = str(out / "train.parquet")
    val_path = str(out / "val.parquet")

    pd.DataFrame(train_steps).to_parquet(train_path, index=False)
    pd.DataFrame(val_steps).to_parquet(val_path, index=False)

    # Verify no paper_id overlap between train and val
    val_paper_ids = sorted(set(s["paper_id"] for s in val_steps))
    train_paper_ids = sorted(set(s["paper_id"] for s in train_steps))
    overlap = set(val_paper_ids) & set(train_paper_ids)
    if overlap:
        raise ValueError(f"Train/val paper_id overlap detected ({len(overlap)} papers): {sorted(overlap)[:5]}")

    # Save eval mapping: unique paper_ids per split for joining with triplet data
    eval_map = {
        "val_paper_ids": val_paper_ids,
        "train_paper_ids": train_paper_ids,
    }
    eval_map_path = str(out / "eval_paper_ids.json")
    with open(eval_map_path, "w") as f:
        json.dump(eval_map, f, indent=2)

    print(f"Saved: {train_path}, {val_path}")
    print(f"Eval mapping: {eval_map_path} (val={len(val_paper_ids)}, train={len(train_paper_ids)} unique papers)")
    return train_path, val_path


def main():
    # Check if --convert_only is in argv (before hydra consumes args)
    if "--convert_only" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--convert_only", action="store_true")
        parser.add_argument("--traces_dir", default="outputs/sft_traces")
        parser.add_argument("--output_dir", default="outputs/sft_parquet")
        parser.add_argument("--val_ratio", type=float, default=0.05)
        parser.add_argument("--include_failed", action="store_true")
        args = parser.parse_args()
        convert_traces_to_parquet(
            args.traces_dir, args.output_dir, args.val_ratio,
            success_only=not args.include_failed,
        )
        return

    # Otherwise, run SFT training via hydra
    import hydra
    from omegaconf import DictConfig

    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer

    @hydra.main(
        config_path="pkg://rllm.trainer.config",
        config_name="agent_sft_trainer",
        version_base=None,
    )
    def train(config: DictConfig):
        trainer = AgentSFTTrainer(config=config)
        trainer.train()

    train()


if __name__ == "__main__":
    main()
