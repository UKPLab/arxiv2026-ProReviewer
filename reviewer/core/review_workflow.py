"""Custom multi-turn workflow for ReviewAgent.

Extends rLLM's MultiTurnWorkflow to store ModelOutput on each step,
ensuring transform_results_for_verl uses original token IDs and logprobs
from inference rather than re-tokenizing from chat_completions text.

Without this, the decode/re-encode cycle causes token mismatches:
- skip_special_tokens=True strips <think>/<​/think> from Qwen3 output
- BPE boundaries may shift during decode → text → encode
- Rollout logprobs would not align with re-tokenized response IDs
"""

import logging
from collections import defaultdict

import numpy as np

from rllm.agents.agent import Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)


class ReviewWorkflow(MultiTurnWorkflow):

    def __init__(self, broadcast_decay: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        # broadcast_decay: 0.0 = uniform (no decay), 0.95 = later steps get more.
        # When > 0, broadcast weight at step t = γ^(T-1-t), so last step = 1x, first = γ^(T-1).
        self.broadcast_decay = broadcast_decay

        # Wire duplicate checker from env to agent
        duplicate_checker = getattr(self.env, 'duplicate_checker', None)
        if duplicate_checker is not None:
            self.agent.log.duplicate_checker_ = duplicate_checker
            self.agent.log.silent_duplicates_ = getattr(self.env, 'silent_duplicates', False)

    def reset(self, task: dict = None, uid: str = None):
        """Reset environment and rewire duplicate checker for new episode.

        Overrides parent to:
        1. Clear duplicate checker cache for new episode
        2. Wire checker to agent's new ReviewLog instance
        """
        # Reset duplicate checker cache
        duplicate_checker = getattr(self.env, 'duplicate_checker', None)
        if duplicate_checker is not None:
            duplicate_checker.reset()

        # Call parent reset (resets env and agent)
        observation, info = super().reset(task, uid)

        # Rewire checker to new log instance
        if duplicate_checker is not None:
            self.agent.log.duplicate_checker_ = duplicate_checker
            self.agent.log.silent_duplicates_ = getattr(self.env, 'silent_duplicates', False)

        return observation, info

    async def run(self, task: dict, uid: str, **kwargs):
        """Execute a multi-step review workflow.

        Identical to MultiTurnWorkflow.run except we attach the ModelOutput
        to the current trajectory step after each LLM call so that
        agent_workflow_engine uses original token IDs for training.
        """
        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)
        info["max_turns"] = self.max_steps
        info["current_turn"] = 1

        self.agent.update_from_env(observation, 0, False, info)

        for step_num in range(1, self.max_steps + 1):
            output: ModelOutput = await self.timed_llm_call(
                self.agent.chat_completions, application_id=uid, **kwargs
            )
            response = output.text

            action = self.agent.update_from_model(response)

            # Store ModelOutput on the step so transform_results_for_verl uses
            # original prompt_ids/completion_ids/logprobs instead of
            # re-tokenizing from chat_completions text.
            cur_step = self.agent.get_current_state()
            if cur_step is not None:
                cur_step.model_output = output

            next_obs, reward, done, info = await self.timed_env_call(self.env.step, action)
            info["max_turns"] = self.max_steps
            info["current_turn"] = step_num + 1

            self.agent.update_from_env(next_obs, reward, done, info)

            # Store per-step reward on the current step (the one that generated the action)
            # This is critical for step-level GRPO and ensures terminal rewards aren't lost
            # cur_step = self.agent.get_current_state()
            # if cur_step is not None:
            #     cur_step.reward = reward
            #     cur_step.done = done
            #     cur_step.info.update(info)

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def _broadcast_reward(self, steps, reward, failed_threshold):
        """Add trajectory reward to each step, optionally with decay.

        Decay weights are normalized as wi / sum(wi) so they sum to 1.
        Each step receives a fraction of the broadcast reward, with later
        steps receiving a larger share.  This keeps the total broadcast
        budget equal to `reward` regardless of T or γ, preserving the
        same reward scale as the no-decay case for critic/rewards/mean.
        """
        gamma = self.broadcast_decay
        T = len(steps)
        if gamma > 0 and T > 1:
            raw_weights = [gamma ** (T - 1 - t) for t in range(T)]
            mean_weight = sum(raw_weights) / T
            norm_weights = [w / mean_weight for w in raw_weights]  # wi/mean(wi), mean=1.0, range ~[0.8, 1.25]
        for t, step in enumerate(steps):
            if step.reward <= failed_threshold:
                continue
            if gamma > 0 and T > 1:
                step.reward += reward * norm_weights[t]
            else:
                step.reward += reward

    def adjust_step_rewards(self, trajectory: Trajectory) -> None:
        """Adjust step-level rewards and clip to [-1.0, 4.0].

        Calls parent class method for optional reward shaping/discounting,
        then clips each step reward to [-1.0, 4.0] to prevent extreme
        outliers from dominating the training signal. Max clipping at 4.0
        removes the top 2.7% of outlier steps (>6.0 max observed) while
        preserving 97.3% of the signal.
        """
        # Apply optional reward shaping and discounting from base class
        super().adjust_step_rewards(trajectory)

        # Clip step rewards to [-1.0, 4.0] to prevent extreme outliers
        for step in trajectory.steps:
            step.reward = max(-1.0, min(4.0, step.reward))

    def postprocess_episode(self, episode, termination_reason=None, error=None):
        """Override base class to handle judge-failed trajectories (reward=None).

        Follows the same sequential structure as the base class but skips
        adjust_step_rewards / correctness / metrics for trajectories where
        the LLM judge failed and trajectory.reward is None.  These
        trajectories are filtered out downstream in the batch builder.
        """
        # 1. assign a task id and task
        episode.id = self.uid
        episode.task = self.task

        for trajectory in episode.trajectories:
            if trajectory.steps and not trajectory.steps[-1].chat_completions:
                trajectory.steps.pop()

            # Skip empty trajectories (e.g. single step with no completions)
            if not trajectory.steps:
                trajectory.reward = 0.0
                trajectory.info["judge_failed"] = True
                continue

            # 2. compute trajectory-level rewards
            self.compute_trajectory_reward(trajectory)

            # 3. adjust the step level rewards
            if trajectory.info.get("judge_failed"):
                continue
            if len(trajectory.steps) > 1:
                self.adjust_step_rewards(trajectory)
            else:
                # Single-step trajectories: clip to [-1.0, 4.0]
                for step in trajectory.steps:
                    step.reward = max(-1.0, min(4.0, step.reward))

        # 4. assign an episode-level correctness flag (skip None rewards)
        total_reward = 0
        for trajectory in episode.trajectories:
            if trajectory.reward is not None:
                total_reward += trajectory.reward
        episode.is_correct = total_reward > 0

        # 5. collect additional metrics (skip None rewards)
        metrics = defaultdict(list)
        for traj in episode.trajectories:
            if traj.reward is not None:
                metrics[traj.name].append(traj.reward)
        episode.metrics = {f"{k}_acc": float(np.mean(v)) for k, v in metrics.items()}

        # 6. store error details if provided
        if error is not None:
            episode.info["error"] = error

        # 7. assign a termination reason
        episode.termination_reason = termination_reason or TerminationReason.UNKNOWN

        return episode

    def compute_trajectory_reward(self, trajectory: Trajectory) -> None:
        """Compute trajectory reward and broadcast to all steps.

        Overrides the default sum-of-step-rewards because the terminal step
        (which carries the reward in the default pattern) gets popped during
        postprocess_episode() since it has no chat_completions.

        Every step receives the same outcome reward (syntactic per-step
        reward + uniform broadcast of all review-level components).
        """
        # Compute trajectory-level reward (all non-syntactic components)
        raw_trajectory_reward = self.env.compute_final_reward()

        # If judge failed, skip this trajectory entirely
        if raw_trajectory_reward is None:
            trajectory.reward = None
            trajectory.info["judge_failed"] = True
            reward_result = getattr(self.env, '_reward_result', None)
            if reward_result is not None:
                trajectory.info['reward_details'] = reward_result
            return

        reward_result = getattr(self.env, '_reward_result')

        # Failed steps (syntactic reward <= -1.0) keep their penalty and
        # don't receive broadcast reward — they contributed nothing.
        FAILED_STEP_THRESHOLD = -1.0
        self._broadcast_reward(trajectory.steps, raw_trajectory_reward, FAILED_STEP_THRESHOLD)

        # Set trajectory.reward to the average of distributed rewards (for logging)
        trajectory.reward = sum(step.reward for step in trajectory.steps) / len(trajectory.steps)

        # Store detailed reward result in trajectory info for episode logging
        if reward_result is not None:
            trajectory.info['reward_details'] = reward_result
