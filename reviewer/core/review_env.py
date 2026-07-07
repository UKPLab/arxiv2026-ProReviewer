"""ReviewEnv - rLLM BaseEnv implementation for paper review.

This module implements the environment side of the agent-environment
interaction following the rLLM/MathAgent pattern with Gymnasium interface.
"""

from typing import Any, Dict, Optional, Tuple, Union
import logging
from .environment import PaperEnvironment
from reviewer.reward.score_review import async_score_review
from rllm.environments.base.base_env import BaseEnv
import asyncio
import concurrent.futures


class ReviewEnv(BaseEnv):
    """Review environment extending rLLM's BaseEnv pattern.

    The ReviewEnv:
    1. Wraps PaperEnvironment for section access
    2. Handles external actions (read_section, search_paper, finish)
    3. Computes rewards at terminal state

    Follows Gymnasium interface: reset() -> (obs, info), step(action) -> (obs, reward, done, info)
    """

    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        reward_mode: Union[str, list] = "full",
        judge_model: Optional[str] = None,
        format_penalty: float = 0.0,
        reward_weights: Optional[Dict[str, float]] = None,
        min_finish_sections: int = 4,
        duplicate_detection: bool = False,
        silent_duplicates: bool = False,
    ):
        """Initialize the review environment.

        Args:
            task: Task dict containing:
                - paper_content: Raw paper content (markdown or latex)
                - paper_id: Paper identifier
                - human_avg_score: Average human score (for reward)
            reward_mode: str or list of reward components. Can be "format", "syntactic", "utility", "score_diff", "full", or a list like ["syntactic", "utility"]. "full" includes all components.
            judge_model: LLM-as-a-Judge
            format_penalty: used for penalty when the finish is missing.
            reward_weights: Weights for ALL components (syntactic, format, utility, score_diff).
                           If not specified, all active components get equal weight.
            duplicate_detection: If True, use real-time embedding-based duplicate detection
                                 (duplicates rejected at insertion, counted as mem_error).
            silent_duplicates: If True (requires duplicate_detection=True), duplicates are
                               silently dropped without error or penalty. The model sees
                               "Successfully added" but the entry is not stored. No reward
                               signal for duplicates — they simply vanish.
        """
        self.task = task
        self.duplicate_detection = duplicate_detection
        self.silent_duplicates = silent_duplicates
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reward_modes = set(reward_mode) if isinstance(reward_mode, (list, tuple)) else {reward_mode}
        # Backward compat: accept old "review_quality" name as alias for "rubric"
        if "review_quality" in self.reward_modes:
            self.reward_modes.discard("review_quality")
            self.reward_modes.add("rubric")
        self.reward_modes.add("syntactic") # Always include syntactic for step-level feedback

        # Handle reward weights
        raw_weights = dict(reward_weights) if reward_weights else {}

        self._format_penalty = format_penalty

        # Store raw weights (no normalization); defaults match legacy equal-weight assumption
        self._raw_weights = {
            "syntactic": raw_weights.get("syntactic", 1.0),
            "format": raw_weights.get("format", 1.0),
            "score_diff": raw_weights.get("score_diff", 1.0),
            "rubric": raw_weights.get("rubric", raw_weights.get("review_quality", 1.0)),
        }

        # Duplicate checker: real-time embedding-based duplicate rejection
        # Replaces post-hoc info_gain when duplicate_detection=True
        self.duplicate_checker = None
        if self.duplicate_detection:
            from reviewer.reward.duplicate_checker import EmbeddingDuplicateChecker
            self.duplicate_checker = EmbeddingDuplicateChecker(
                embed_model="qwen3-embedding-8b",
                threshold=0.85
            )

        # Finish gate: require reading at least N unique sections before finish is accepted
        self.min_finish_sections = min_finish_sections

        self._finished_review = None
        self._reward_result = None  # Full reward result dict


    def reset(self, task: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for new episode.

        Args:
            task: Optional task dict. If provided, overrides the task set in __init__.
                  This is used by CumulativeWorkflow which passes task at reset time.

        Returns:
            Tuple of (observation, info) where:
            - observation: {"title": str, "sections": List[str]}
            - info: {"paper_id": str, ...}
        """
        if task is not None:
            self.task = task

        if self.task is None:
            raise ValueError("No task provided. Pass task to __init__ or reset().")

        self._finished_review = None
        self._reward_result = None  # Full reward result dict
        self._sections_read = set()  # Track unique sections read (for finish gate)

        paper_content = self.task.get("paper_content", "")
        if not paper_content:
            raise ValueError("Task must contain 'paper_content'")

        # Initialize paper environment
        self.paper_env = PaperEnvironment(paper_content)

        # Build initial observation
        title = self.paper_env.sections.get("title")
        if not title:
            raise ValueError(f"Paper has no title section. Paper ID: {self.task.get('paper_id', 'unknown')}. "
                             f"Ensure the paper content starts with 'Title: ...' or '# Title'.")
        sections = [s for s in self.paper_env.get_section_names() if s != "title"]

        observation = {
            "title": title.content,
            "sections": sections,
        }

        info = {
            "paper_id": self.task.get("paper_id", ""),
            "num_sections": len(sections),
        }

        return observation, info

    def step(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return result.

        Args:
            action: Action dict with "name" and "args"

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.paper_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Unwrap Action dataclass if needed
        if hasattr(action, "action"):
            action = action.action

        # Extract and consume meta info from agent
        meta = action.pop("_meta", {})
        # Update sections read from agent memory (single source of truth)
        self._sections_read = meta.get("sections_read", self._sections_read)

        action_name = action.get("name", "")
        action_args = action.get("args", {})

        # Compute per-step syntactic penalty (penalty-only: <= 0 for bad steps, 0 for correct steps)
        if "syntactic" in self.reward_modes:
            step_reward = self._compute_step_syntactic_reward(action_name, action_args, meta)
            # Apply syntactic weight (raw weight, no normalization)
            step_reward = step_reward * self._raw_weights["syntactic"]
        else:
            step_reward = 0.0

        # Format error — JSON didn't parse
        if action_name == "format_error":
            obs = {"action_result": f"Format error: {action_args.get('message', 'Invalid JSON')}. Please output valid JSON."}
            reward = step_reward
            return obs, reward, False, {"action_name": "format_error"}

        # Tool error from agent-side validation (e.g., research claim not found)
        if action_name == "tool_error":
            obs = {"action_result": f"Error: {action_args.get('message', 'Invalid tool call')}"}
            reward = step_reward
            return obs, reward, False, {"action_name": "tool_error"}

        # Track syntactic correctness for known vs unknown actions
        VALID_ACTIONS = {"read_section", "search_paper", "finish"}

        if action_name in VALID_ACTIONS:
            args_valid = self._validate_action_args(action_name, action_args)
        else:
            # Unknown action name
            args_valid = False

        # Dispatch to action handlers
        if action_name == "finish":
            obs, _, done, info = self._handle_finish(action_args)
        elif action_name == "read_section":
            obs, _, done, info = self._handle_read_section(action_args)
        elif action_name == "search_paper":
            obs, _, done, info = self._handle_search_paper(action_args)
        else:
            obs, _, done, info = self._handle_unknown_action(action)

        info.setdefault("args_valid", args_valid)
        info["syntactic_reward"] = step_reward
        reward = step_reward

        return obs, reward, done, info

    def _handle_finish(
        self,
        args: Dict,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Handle finish action - signal episode end.

        Reward is NOT computed here. It is computed in compute_final_reward(),
        which is called by compute_trajectory_reward() during postprocess_episode().
        This avoids the reward being silently discarded when the terminal step
        (which has no chat_completions) is popped during post-processing.

        Empty or partial reviews are allowed — compute_format_completeness
        provides the training signal (0.0 for empty, proportional for partial).

        Args:
            args: Action args containing "review_data"

        Returns:
            Terminal step tuple (observation, reward=0.0, done=True, info)
        """

        # Extract review from args (enriched by agent)
        review = args.get("review_data", {})

        # Store review for compute_final_reward() — reward computed there, not here
        # Empty/partial reviews are handled by compute_format_completeness (scores 0.0–1.0)
        self._finished_review = review

        observation = {"action_result": "Review complete."}
        info = {
            "action_name": "finish",
            "final_review": review,
        }

        return observation, 0.0, True, info

    def _handle_read_section(
        self,
        args: Dict,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Handle read_section action.

        Args:
            args: Action args with "section_name"

        Returns:
            Step tuple with section content
        """
        section_name = args.get("section_name", "")
        self.logger.info(f"Reading section: {section_name}")

        content = self.paper_env.read_section(section_name)

        if content is None:
            observation = {"action_result": f"Section not found: '{section_name}'. Please use a valid section name from the list provided."}
            info = {
                "action_name": "read_section",
                "section_name": section_name,
                "args_valid": False,
            }
            return observation, 0.0, False, info

        observation = {
            "action_result": f"Successfully read section '{section_name}'. Content:\n{content}",
        }
        info = {
            "action_name": "read_section",
            "section_name": section_name,
        }

        return observation, 0.0, False, info

    def _handle_search_paper(
        self,
        args: Dict,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Handle search_paper action.

        Args:
            args: Action args with "query"

        Returns:
            Step tuple with search results
        """
        query = args.get("query", "")
        if not query:
            observation = {"action_result": "Error: Missing query parameter"}
            return observation, 0.0, False, {"error": "missing_query"}

        self.logger.info(f"Searching paper for: {query}")
        results = self.paper_env.search_paper(query)

        if not results:
            action_result = f"No matches found for '{query}'"
        else:
            action_result = f"Search results for '{query}':\n\n"
            for result in results:
                action_result += f"**[{result['section']}]** ({result['match_count']} matches)\n"
                for snippet in result['snippets']:
                    action_result += f"  - {snippet}\n"
                action_result += "\n"

        observation = {"action_result": action_result}
        info = {
            "action_name": "search_paper",
            "query": query,
            "num_results": len(results),
        }

        return observation, 0.0, False, info

    def _validate_action_args(self, action_name: str, args: Dict) -> bool:
        """Validate action arguments are syntactically correct.

        Args:
            action_name: Name of the action
            args: Action arguments dict

        Returns:
            True if arguments are valid for the given action
        """
        if action_name == "read_section":
            return bool(args.get("section_name", "").strip())
        elif action_name == "search_paper":
            return bool(args.get("query", "").strip())
        elif action_name == "finish":
            return True  # finish has no required user-provided args
        elif action_name == "research":
            return bool(args.get("claim_id") or args.get("question_id"))
        return False

    def compute_final_reward(self) -> float:
        """Compute review-level rewards, store result for workflow credit assignment.

        Called by both ReviewWorkflow.compute_trajectory_reward() and
        AgentExecutionEngine. Must return a float for the engine contract.

        The workflow reads self._reward_result afterwards to distribute
        per-component rewards to individual steps via evidence credit.

        Returns:
            Scalar fallback reward (score_diff + format, weighted).
        """
        review = self._finished_review

        # Require at least min_finish_sections unique sections read before finishing
        has_coverage = len(self._sections_read) >= self.min_finish_sections

        # No review produced, empty review, or insufficient coverage — apply format penalty
        is_empty = (
            review is not None
            and not review.get("strengths")
            and not review.get("weaknesses")
        )
        if review is None or is_empty or not has_coverage:
            reward_components = {}
            if "format" in self.reward_modes:
                reward_components["format"] = -self._format_penalty
            penalty = sum(
                self._raw_weights[k] * reward_components[k]
                for k in reward_components
            )
            self._reward_result = {
                "reward_components": reward_components,
                "weights": {k: self._raw_weights[k] for k in self.reward_modes if k in self._raw_weights},
            }
            return penalty

        # Score the finished review (all non-syntactic components including rubric)
        coro = async_score_review(
            review=review,
            human_avg_score=self.task.get("human_avg_score"),
            reward_modes=self.reward_modes - {"syntactic"},
            paper_content=self.task.get("paper_content", ""),
            training=True,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                reward_components = pool.submit(asyncio.run, coro).result()
        else:
            reward_components = asyncio.run(coro)

        # Normalize: weighted sum divided by total weight → [0, 1]
        # This scalar is used as fallback when evidence credit is unavailable.
        ALL_SCALAR_COMPONENTS = {"score_diff", "format", "rubric"}
        active = self.reward_modes & ALL_SCALAR_COMPONENTS & set(reward_components.keys())
        weighted_sum = sum(
            self._raw_weights[k] * reward_components[k]
            for k in active
        )
        weight_sum = sum(self._raw_weights[k] for k in active) if active else 1.0
        final_reward = weighted_sum / weight_sum

        # Store for workflow consumption
        self._reward_result = {
            "reward_components": reward_components,
            "weights": {k: self._raw_weights[k] for k in self.reward_modes if k in self._raw_weights},
        }

        return final_reward

    def _compute_step_syntactic_reward(
        self,
        action_name: str,
        action_args: Dict,
        meta: Dict,
    ) -> float:
        """Compute per-step syntactic penalty for step-level GRPO.

        Penalty-only: correct steps get 0, failed steps get -1.
        Completely failed steps (no valid action executed) get -1.
        Partial failures (valid action but some memory ops failed) get
        a small per-error penalty.

        Args:
            action_name: Name of the action for this step
            action_args: Action arguments dict
            meta: Meta info dict from the agent (_meta field)

        Returns:
            Per-step syntactic reward (<= 0)
        """
        VALID_ACTIONS = {"read_section", "search_paper", "finish", "research"}

        # Completely failed steps: no valid action was executed
        if action_name == "format_error":
            return -1.0
        if action_name == "tool_error":
            return -1.0
        if action_name not in VALID_ACTIONS:
            return -1.0

        # Valid action but bad args
        args_valid = self._validate_action_args(action_name, action_args)
        if not args_valid:
            return -1.0

        # Hallucination (referencing non-existent IDs) and validation errors
        # (e.g. missing evidence tags on outline items). Soft proportional
        # penalty: -0.2 per error, consistent with mem_error below.
        # A flat -1.0 was too harsh — especially on the finish step which
        # has many outline ops and gets excluded from broadcast reward when
        # reward <= -1.0, causing training instability (step 7-10 drop).
        step_hallucinations = meta.get("mem_hallucinations", 0)
        validation_errors = meta.get("validation_errors", 0)
        mem_error = meta.get("mem_error", 0)
        n_hard_errors = step_hallucinations + validation_errors + mem_error
        if n_hard_errors > 0:
            return -0.2 * n_hard_errors

        return 0.0

    def _handle_unknown_action(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Handle unknown action.

        Args:
            action: The unknown action dict

        Returns:
            Step tuple with error message
        """
        action_name = action.get("name", "unknown")
        self.logger.error(f"Unknown action: {action_name}")
        observation = {"action_result": f"Unknown action: '{action_name}'. You need to carefully distinguish between memory operations and environment actions and their respective arguments."}
        return observation, 0.0, False, {"error": "unknown_action"}

    @staticmethod
    def from_dict(env_args: Dict[str, Any]) -> "ReviewEnv":
        """Factory method to create ReviewEnv from dict args.

        This is used by rLLM's AgentTrainer to instantiate environments.

        Args:
            env_args: Dict containing either:
              - task: Dict with paper data (legacy pattern)
              - OR paper_id, paper_content, etc. directly (training pattern)
              Plus: judge_model, reward_mode, etc.

        Returns:
            ReviewEnv instance
        """
        # Check if task is nested or flat
        if "task" in env_args:
            # Legacy pattern: task dict is nested
            task = env_args["task"]
        else:
            # Training pattern: dataset items come as flat dicts
            # Wrap them into task format
            task = {
                "paper_id": env_args.get("paper_id", ""),
                "paper_content": env_args.get("paper_content", ""),
                "human_avg_score": env_args.get("human_avg_score"),
            }

        return ReviewEnv(
            task=task,
            reward_mode=env_args.get("reward_mode", "full"),
            judge_model=env_args.get("judge_model"),
            format_penalty=env_args.get("format_penalty", env_args.get("incomplete_penalty", 0.0)),
            reward_weights=env_args.get("reward_weights"),
            duplicate_detection=env_args.get("duplicate_detection", False),
            silent_duplicates=env_args.get("silent_duplicates", False),
        )
