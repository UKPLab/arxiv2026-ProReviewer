
  1. rllm/experimental/common/advantage.py

  - Added a new "step_level" advantage mode implementation
  - Pools all steps from all trajectories and computes advantage across all steps (step-level GRPO)

  2. rllm/experimental/common/config.py

  - Extended stepwise_advantage_mode type to include "step_level" as a new option
  - Changed from: Literal["broadcast", "per_step"]
  - Changed to: Literal["broadcast", "per_step", "step_level"]

  3. rllm/experimental/verl/verl_advantage.py

  - Added logic to handle "step_level" mode by grouping steps by task_id instead of trajectory_id
  - Updated condition to check for both "per_step" and "step_level" modes when using step rewards

  4. rllm/trainer/config/agent_ppo_trainer.yaml

  - Updated comment to document the new step_level mode option
  - Changed from: mode: broadcast # [broadcast, per_step]
  - Changed to: mode: broadcast # [broadcast, per_step, step_level]

  5. rllm/trainer/verl/agent_workflow_trainer.py

  - Added step-level GRPO support by grouping steps by task_id when in "step_level" mode
  - Updated conditions to handle both "per_step" and "step_level" modes throughout (2 locations)

  6. rllm/utils/episode_logger.py

  - Added logging for reward_details in trajectory info
  - Added logging for log_snapshot in step info

  Key Feature: You've implemented a new step-level GRPO mode that groups all steps by task/prompt rather than by trajectory, allowing for step-level advantage computation across
  all trajectories of the same task.