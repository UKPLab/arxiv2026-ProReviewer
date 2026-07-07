"""Training script for ProReviewer using rLLM AgentTrainer.

Example usage:
    # Prepare dataset first
    python -m reviewer.rllm_version.prepare_review_dataset \\
      --data_path data/paper_triplets/iclr2025

    # Train with default config (VERL backend, research mode)
    python -m reviewer.rllm_version.train

    # Train in direct mode (no research subagent)
    python -m reviewer.rllm_version.train reviewer.direct_mode=true

    # Train with custom settings
    python -m reviewer.rllm_version.train \\
      model.name="meta-llama/Llama-3.1-8B-Instruct" \\
      training.learning_rate=1e-5 \\
      agent.max_steps=30
"""

import hydra
from omegaconf import DictConfig
import logging

from reviewer.core.proreviewer import ProReviewer as ReviewAgent
from reviewer.core.review_env import ReviewEnv
from reviewer.core.review_workflow import ReviewWorkflow
from reviewer.prompts.reviewer_prompts_direct import REVIEWER_DIRECT_SYSTEM_PROMPT
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer"
)
def main(config: DictConfig):
    """Main training entry point."""

    # Load datasets
    dataset_name = config.reviewer.dataset_name
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = DatasetRegistry.load_dataset(dataset_name, "val")

    if train_dataset is None:
        raise ValueError(
            "Dataset not found! Please run prepare_review_dataset.py first:\n"
            "  python -m reviewer.rllm_version.prepare_review_dataset "
            "--data_path data/paper_triplets/iclr2025"
        )

    logger.info(f"Loaded datasets: train={len(train_dataset)}, val={len(val_dataset)}")

    # Check for direct mode flag
    use_direct_mode = config.get("reviewer", {}).get("direct_mode", False)
    reward_mode = config.get("reviewer", {}).get("reward_mode", "full")
    # Hydra may pass a ListConfig; convert to list
    if hasattr(reward_mode, '__iter__') and not isinstance(reward_mode, str):
        reward_mode = list(reward_mode)
    logger.info(f"Training mode: {'Direct (no research)' if use_direct_mode else 'Research'}")
    logger.info(f"Reward mode: {reward_mode}")

    # Configure agent
    agent_args = {
        "accumulate_log_context": config.get("reviewer", {}).get("accumulate_log_context", True),
        "max_claims_in_context": config.get("reviewer", {}).get("max_claims_in_context", 10),
        "memory_in_first_message": config.get("reviewer", {}).get("memory_in_first_message", False),
    }

    if use_direct_mode:
        agent_args["system_prompt"] = REVIEWER_DIRECT_SYSTEM_PROMPT

    # Configure environment (static args shared across all episodes)
    env_args = {
        "reward_mode": reward_mode,
        "format_penalty": config.get("reviewer", {}).get("format_penalty", config.get("reviewer", {}).get("incomplete_penalty", 0.0)),
        "reward_weights": dict(config.get("reviewer", {}).get("reward_weights", {})) if config.get("reviewer", {}).get("reward_weights") else None,
        "judge_model": config.get("reviewer", {}).get("judge_model", None),
        "min_finish_sections": config.get("reviewer", {}).get("min_finish_sections", 5),
        "duplicate_detection": config.get("reviewer", {}).get("duplicate_detection", False),
        "silent_duplicates": config.get("reviewer", {}).get("silent_duplicates", False),
    }

    # Create trainer using MultiTurnWorkflow with stepwise advantage broadcasting.
    # Each step becomes an independent training example. The terminal reward's
    # GRPO advantage is broadcast to all earlier steps.
    trainer = AgentTrainer(
        config=config,
        workflow_class=ReviewWorkflow,
        workflow_args={
            "agent_cls": ReviewAgent,
            "env_cls": ReviewEnv,
            "agent_args": agent_args,
            "env_args": env_args,
            "broadcast_decay": config.get("reviewer", {}).get("broadcast_decay", 0.0),
        },
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="verl",
    )

    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
