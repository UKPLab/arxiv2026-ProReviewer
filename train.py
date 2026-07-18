"""GRPO training entry point for ProReviewer.

This is the canonical training script for the Reviewer-R1 system.
It uses rLLM's AgentTrainer with the VERL backend to train ProReviewer
via Group Relative Policy Optimization (GRPO) with step-level advantages.

Prerequisites:
    # 1. Prepare dataset (register with rLLM DatasetRegistry)
    python -m reviewer.sft.prepare_review_dataset \
        --data_path data/paper_triplets/iclr2025_rl

    # 2. (Optional) Set up remote reward/embedding vLLM servers
    #    See train_scripts/setup_vllm_tunnel.sh

Usage:
    # Train with default config
    python train.py

    # Stage 1: format + score_diff (rule-based rewards, no LLM judge)
    python train.py \
        +reviewer.direct_mode=true \
        +reviewer.reward_mode='[syntactic, format, score_diff]'

    # Stage 2: adds evidence-based memory reasoning (requires LLM judge)
    python train.py \
        +reviewer.direct_mode=true \
        +reviewer.reward_mode='[syntactic, format, score_diff, memory_reasoning]'

    # Full training with all overrides — see train_scripts/curriculum/
"""

import logging

import hydra
from omegaconf import DictConfig

from reviewer.core.proreviewer import ProReviewer
from reviewer.core.review_env import ReviewEnv
from reviewer.core.review_workflow import ReviewWorkflow
from reviewer.prompts.reviewer_prompts_direct import REVIEWER_DIRECT_SYSTEM_PROMPT
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
)
def main(config: DictConfig):
    """Main GRPO training entry point.

    Configuration is managed via Hydra. Base config comes from rLLM's
    agent_ppo_trainer.yaml, with reviewer-specific overrides passed
    via the +reviewer.* namespace from the training shell scripts.
    """
    reviewer_cfg = config.get("reviewer", {})

    # --- Dataset ---
    dataset_name = reviewer_cfg.get("dataset_name", "paper_reviews")
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = DatasetRegistry.load_dataset(dataset_name, "val")

    if train_dataset is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Run prepare_review_dataset first:\n"
            "  python -m reviewer.sft.prepare_review_dataset "
            "--data_path data/paper_triplets/iclr2025_rl"
        )

    logger.info(f"Loaded datasets: train={len(train_dataset)}, val={len(val_dataset)}")

    # --- Mode / reward config ---
    use_direct_mode = reviewer_cfg.get("direct_mode", False)
    reward_mode = reviewer_cfg.get("reward_mode", "full")
    # Hydra may pass a ListConfig; convert to plain list
    if hasattr(reward_mode, "__iter__") and not isinstance(reward_mode, str):
        reward_mode = list(reward_mode)

    logger.info(f"Training mode: {'Direct' if use_direct_mode else 'Research'}")
    logger.info(f"Reward mode: {reward_mode}")

    # --- Agent args (ProReviewer.__init__) ---
    agent_args = {
        "accumulate_log_context": reviewer_cfg.get("accumulate_log_context", True),
        "max_claims_in_context": reviewer_cfg.get("max_claims_in_context", 10),
    }
    if use_direct_mode:
        agent_args["system_prompt"] = REVIEWER_DIRECT_SYSTEM_PROMPT

    # --- Environment args (ReviewEnv.__init__) ---
    env_args = {
        "reward_mode": reward_mode,
        "format_penalty": reviewer_cfg.get(
            "format_penalty",
            reviewer_cfg.get("incomplete_penalty", 0.0),
        ),
        "reward_weights": (
            dict(reviewer_cfg.get("reward_weights", {}))
            if reviewer_cfg.get("reward_weights")
            else None
        ),
        "judge_model": reviewer_cfg.get("judge_model", None),
        "min_finish_sections": reviewer_cfg.get("min_finish_sections", 5),
        "duplicate_detection": reviewer_cfg.get("duplicate_detection", False),
        "silent_duplicates": reviewer_cfg.get("silent_duplicates", False),
    }

    # --- Workflow args (ReviewWorkflow.__init__) ---
    workflow_args = {
        "agent_cls": ProReviewer,
        "env_cls": ReviewEnv,
        "agent_args": agent_args,
        "env_args": env_args,
        "broadcast_decay": reviewer_cfg.get("broadcast_decay", 0.0),
    }

    # --- Trainer ---
    trainer = AgentTrainer(
        config=config,
        workflow_class=ReviewWorkflow,
        workflow_args=workflow_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="verl",
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
