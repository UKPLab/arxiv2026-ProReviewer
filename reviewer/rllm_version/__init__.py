"""rLLM integration for ProReviewer.

This module adapts the ProReviewer paper review system to the rLLM framework
by implementing ReviewAgent (BaseAgent) and ReviewEnv (BaseEnv).

Example usage:
    from reviewer.rllm import ReviewAgent, ReviewEnv
    from rllm.trainer import AgentTrainer

    trainer = AgentTrainer(
        agent_class=ReviewAgent,
        env_class=ReviewEnv,
        agent_args={"accumulate_log_context": True},
        env_args={"research_model": model},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()
"""

from .review_agent import ReviewAgent
from .review_env import ReviewEnv

__all__ = [
    "ReviewAgent",
    "ReviewEnv",
]
