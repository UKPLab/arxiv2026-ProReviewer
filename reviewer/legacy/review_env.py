"""ReviewEnv - rLLM BaseEnv implementation for paper review.

This module implements the environment side of the agent-environment
interaction following the rLLM/MathAgent pattern with Gymnasium interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import json
import re
from reviewer.core.environment import PaperEnvironment
from reviewer.core.research_agent import ResearchSubagent, ResearchMemory
from reviewer.core.reviewer_memory import ReviewLog
from reviewer.reward.calculator import RewardCalculator
from reviewer.reward.score_review import score_review, async_score_review
from rllm.environments.base.base_env import BaseEnv
import asyncio
import concurrent.futures

class ReviewEnv(BaseEnv):
    """Review environment extending rLLM's BaseEnv pattern.

    The ReviewEnv:
    1. Wraps PaperEnvironment for section access
    2. Handles external actions (read_section, research, finish)
    3. Runs research subagent inline for research actions
    4. Computes rewards at terminal state

    Follows Gymnasium interface: reset() -> (obs, info), step(action) -> (obs, reward, done, info)
    """

    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        research_model: Optional[str] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        max_research_iterations: int = 20,
        enable_shaping_rewards: bool = False,
        reward_mode: Union[str, list] = "full",
        judge_model: Optional[str] = None,
        recall_model: str = "qwen35-122b",
        format_penalty: float = 0.0,
        reward_weights: Optional[Dict[str, float]] = None,
        min_finish_sections: int = 4,
        memory_reasoning_mode: str = "trajectory",
        memory_reasoning_format: str = "scirm",
        duplicate_detection: bool = False,
        silent_duplicates: bool = False,
    ):
        """Initialize the review environment.

        Args:
            task: Task dict containing:
                - paper_content: Raw paper content (markdown or latex)
                - paper_id: Paper identifier
                - human_avg_score: Average human score (for reward)
                - clustered_points: Clustered review points (for reward)
            research_model: Model identifier for research subagent
            reward_calculator: RewardCalculator instance (or creates default)
            max_research_iterations: Max iterations for research subagent
            enable_shaping_rewards: Whether to provide intermediate rewards
            reward_mode: str or list of reward components. Can be "format", "syntactic", "utility", "score_diff", "recall", "full", or a list like ["syntactic", "utility"]. "full" includes all components.
            judge_model: Judge model for utility reward calculator (e.g., "utility-score", "gpt-5mini")
            recall_model: Judge model for recall reward (defaults to judge_model)
            format_penalty: used for penalty when the finish is missing.
            reward_weights: Weights for ALL components (syntactic, format, utility, score_diff, recall).
                           If not specified, all active components get equal weight.
            memory_reasoning_mode: "trajectory" (LLM judge on step-by-step trajectory) or
                                   "snapshot" (LLM judge on final log snapshot only).
            memory_reasoning_format: "scirm" (default, uses <reasoning>/<score> tags) or
                                    "json" (legacy format).
            duplicate_detection: If True, use real-time embedding-based duplicate detection
                                 (duplicates rejected at insertion, counted as mem_error).
                                 If False, use post-hoc info_gain reward (current behavior).
            silent_duplicates: If True (requires duplicate_detection=True), duplicates are
                               silently dropped without error or penalty. The model sees
                               "Successfully added" but the entry is not stored. No reward
                               signal for duplicates — they simply vanish.
        """
        self.task = task
        self.research_model = research_model
        self.max_research_iterations = max_research_iterations
        self.enable_shaping_rewards = enable_shaping_rewards
        self.duplicate_detection = duplicate_detection
        self.silent_duplicates = silent_duplicates
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reward_modes = set(reward_mode) if isinstance(reward_mode, (list, tuple)) else {reward_mode}
        self.reward_modes.add("syntactic") # Always include syntactic for step-level feedback
        self._memory_reasoning_mode = memory_reasoning_mode
        self._memory_reasoning_format = memory_reasoning_format

        # Handle reward weights
        raw_weights = dict(reward_weights) if reward_weights else {}

        self._format_penalty = format_penalty  # Keep for backward compatibility in from_dict

        # Store raw weights (no normalization); defaults match legacy equal-weight assumption
        # Simplified: 6 core components only
        # - count_penalty folded into format (compute_format_completeness now checks weakness count)
        # - hallucination folded into syntactic (per-step penalty via _compute_step_syntactic_reward)
        # - finish_bonus dropped (broadcast of trajectory-quality rewards makes this unnecessary)
        self._raw_weights = {
            "syntactic": raw_weights.get("syntactic", 1.0),
            "format": raw_weights.get("format", 1.0),
            "utility": raw_weights.get("utility", 1.0),
            "score_diff": raw_weights.get("score_diff", 1.0),
            "recall": raw_weights.get("recall", 1.0),
            "memory_reasoning": raw_weights.get("memory_reasoning", 1.0),
            "info_gain": raw_weights.get("info_gain", 1.0),
            "duplicate_penalty": raw_weights.get("duplicate_penalty", 1.0),
            # "hallucination": raw_weights.get("hallucination", 1.0),  # folded into syntactic per-step
            # "count_penalty": raw_weights.get("count_penalty", 1.0),  # folded into format completeness
            # "finish_bonus": raw_weights.get("finish_bonus", 1.0),    # dropped
        }
                # Initialize research subagent if model specified
        self.research_subagent = None
        if self.research_model:
            print(f"Initializing research subagent with model: {self.research_model}")
            self.research_subagent = ResearchSubagent(self.research_model)

        # Reward calculator needed for utility, recall, memory_reasoning,
        # or info_gain (cosine-similarity-based per-step novelty reward).
        NEEDS_CALCULATOR = {"utility", "recall", "memory_reasoning", "info_gain"}
        if self.reward_modes & NEEDS_CALCULATOR:
            if reward_calculator:
                self.reward_calculator = reward_calculator
            else:
                # Create calculator with specified judge model
                self.reward_calculator = RewardCalculator(
                    judge_model=judge_model if judge_model else "utility-score",
                    recall_model=recall_model,
                    weights=reward_weights if reward_weights else None,
                    max_concurrent_judge_calls=32
                )
        else:
            self.reward_calculator = None

        # Duplicate checker: real-time embedding-based duplicate rejection
        # Replaces post-hoc info_gain when duplicate_detection=True
        self.duplicate_checker = None
        if self.duplicate_detection:
            from reviewer.reward.duplicate_checker import EmbeddingDuplicateChecker
            # Reuse embed_model from reward_calculator if available
            embed_model = "qwen3-embedding-8b"
            if self.reward_calculator is not None and hasattr(self.reward_calculator, 'embed_model'):
                embed_model = self.reward_calculator.embed_model
            self.duplicate_checker = EmbeddingDuplicateChecker(
                embed_model=embed_model,
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
        self._cumulative_syntactic_reward = 0.0  # Track syntactic rewards across steps
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

        # Precompute required sections for coverage shaping reward.
        # Only top-level numbered sections + abstract are required (not theorems,
        # proofs, etc.).  Reading a parent section credits its subsections too.
        all_names = self.paper_env.get_section_names()
        conclusion_idx = None
        for i, s in enumerate(all_names):
            if "conclusion" in s.lower():
                conclusion_idx = i
                break
        if conclusion_idx is not None:
            main_sections = set(all_names[1:conclusion_idx + 1])  # skip title
        else:
            # No conclusion — use all non-title, non-appendix sections
            main_sections = {s for s in all_names[1:]
                            if not s.startswith("appendix")}
        # Filter to numbered sections + abstract only
        self._required_sections = {s for s in main_sections
                                   if re.match(r'^(?:\d|abstract)', s)}
        # Build parent→children map for subsection expansion
        self._subsection_map = {}
        for sk in set(all_names):
            m = re.match(r'^(\d+)(?:\.|\s)', sk)
            if not m:
                continue
            sec_num = m.group(1)
            subs = [o for o in set(all_names) if o != sk
                    and re.match(rf'^{sec_num}\.', o)]
            if subs:
                self._subsection_map[sk] = subs
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
            # Accumulate for logging
            self._cumulative_syntactic_reward += step_reward
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
        VALID_ACTIONS = {"read_section", "search_paper", "finish", "research"}
        args_valid = True
        if action_name in VALID_ACTIONS:
            args_valid = self._validate_action_args(action_name, action_args)
        else:
            # Unknown action name
            args_valid = False

        # Dispatch to action handlers
        if action_name == "finish":
            obs, handler_reward, done, info = self._handle_finish(action_args)
        elif action_name == "read_section":
            obs, handler_reward, done, info = self._handle_read_section(action_args)
        elif action_name == "research":
            obs, handler_reward, done, info = self._handle_research(action_args)
        elif action_name == "search_paper":
            obs, handler_reward, done, info = self._handle_search_paper(action_args)
        else:
            obs, handler_reward, done, info = self._handle_unknown_action(action)

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

        # Optional shaping reward for exploration
        reward = 0.0
        if self.enable_shaping_rewards:
            reward = 0.01  # Small reward for reading sections

        observation = {
            "action_result": f"Successfully read section '{section_name}'. Content:\n{content}",
        }
        info = {
            "action_name": "read_section",
            "section_name": section_name,
        }

        return observation, reward, False, info

    def _handle_research(
        self,
        args: Dict,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Handle research action - run subagent synchronously.

        The research subagent runs as a synchronous sub-loop. From rLLM's
        perspective, this is a single action that returns findings.

        Args:
            args: Action args with "claim_data" or "question_data" (enriched by agent)

        Returns:
            Step tuple with research findings
        """
        if not self.research_subagent:
            observation = {"action_result": "Error: Research subagent not configured"}
            return observation, 0.0, False, {"error": "no_research_model"}

        target_id = args.get("claim_id") or args.get("question_id")
        target_type = "claim" if "claim_id" in args else "question"
        additional_context = args.get("additional_context", "")

        if not target_id:
            observation = {"action_result": "Error: Must provide either claim_id or question_id"}
            return observation, 0.0, False, {"error": "missing_target"}

        # Get target data from action args (enriched by agent) - as dict, not object
        if target_type == "claim":
            target_data = args.get("claim_data")
            if not target_data:
                observation = {"action_result": "Error: Missing claim_data in research action"}
                return observation, 0.0, False, {"error": "missing_claim_data"}

        else:  # question
            target_data = args.get("question_data")
            if not target_data:
                observation = {"action_result": "Error: Missing question_data in research action"}
                return observation, 0.0, False, {"error": "missing_question_data"}

        self.logger.info(f"Starting research for {target_type}: {target_id}")

        # Run research subagent synchronously - pass dict data instead of object
        try:
            findings = self.research_subagent.research(
                environment=self.paper_env,
                target_type=target_type,
                target=target_data,
                max_iterations=self.max_research_iterations,
            )
        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            observation = {"action_result": f"Error during research: {str(e)}"}
            return observation, 0.0, False, {"error": str(e)}

        # Format findings for main agent
        findings_text = self._format_research_findings(findings, target_type, target_data)

        # Optional shaping reward for research
        reward = 0.0
        if self.enable_shaping_rewards:
            reward = 0.02  # Slightly higher reward for research

        observation = {"action_result": findings_text}
        info = {
            "action_name": "research",
            "target_type": target_type,
            "target_id": target_id,
            "findings_summary": findings.get("summary", "")[:200],
            "sections_examined": findings.get("cross_references", []),
        }

        return observation, reward, False, info

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

        # Score the finished review
        raw_cp = self.task["clustered_points"]
        clustered_points = json.loads(raw_cp) if isinstance(raw_cp, str) else raw_cp
        
        coro = async_score_review(
            review=review,
            human_avg_score=self.task["human_avg_score"],
            clustered_points=clustered_points,
            reward_modes=self.reward_modes - {"syntactic"},
            reward_calculator=self.reward_calculator,
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

        # Scalar reward: all non-syntactic components, weighted.
        # Simplified: hallucination folded into syntactic, count_penalty folded into format,
        # finish_bonus dropped.

        if "memory_reasoning" in self.reward_modes and self.reward_calculator is not None:
            log_snapshot = getattr(self, "_log_snapshot", None)
            if log_snapshot is not None:
                import functools
                llm_judge_fn = functools.partial(
                    self.reward_calculator._call_llm_judge_async,
                    model_override=self.reward_calculator.recall_model,
                )
                if self._memory_reasoning_mode == "trajectory":
                    from reviewer.reward.trajectory_memory_reasoning import compute_trajectory_memory_reasoning_reward_async
                    step_snapshots = getattr(self, "_step_snapshots", None) or []
                    coro = compute_trajectory_memory_reasoning_reward_async(log_snapshot, llm_judge_fn, step_snapshots=step_snapshots)
                elif self._memory_reasoning_mode == "trajectory_v2":
                    from reviewer.reward.trajectory_memory_reasoning_v2 import compute_trajectory_memory_reasoning_reward_async
                    step_snapshots = getattr(self, "_step_snapshots", None) or []
                    coro = compute_trajectory_memory_reasoning_reward_async(log_snapshot, llm_judge_fn, step_snapshots=step_snapshots, format=self._memory_reasoning_format)
                elif self._memory_reasoning_mode == "trajectory_v2_evidence":
                    from reviewer.reward.trajectory_memory_reasoning_v2_evidence import compute_trajectory_memory_reasoning_reward_evidence_async
                    step_snapshots = getattr(self, "_step_snapshots", None) or []
                    paper_content = self.task.get("paper_content", "")
                    coro = compute_trajectory_memory_reasoning_reward_evidence_async(log_snapshot, llm_judge_fn, step_snapshots=step_snapshots, format=self._memory_reasoning_format, paper_content=paper_content)
                else:
                    from reviewer.reward.memory_reasoning import compute_memory_reasoning_reward_async
                    coro = compute_memory_reasoning_reward_async(log_snapshot, llm_judge_fn)
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    with concurrent.futures.ThreadPoolExecutor(1) as pool:
                        mem_score, mem_details = pool.submit(asyncio.run, coro).result()
                else:
                    mem_score, mem_details = asyncio.run(coro)

                # If judge failed, return None to skip this instance
                if mem_score is None:
                    self.logger.warning("memory_reasoning judge failed, skipping instance")
                    self._reward_result = {
                        "reward_components": reward_components,
                        "weights": {k: self._raw_weights[k] for k in self.reward_modes if k in self._raw_weights},
                        "memory_reasoning_details": mem_details,
                        "judge_failed": True,
                    }
                    return None

                reward_components["memory_reasoning"] = mem_score
                reward_components["memory_reasoning_details"] = mem_details

        # duplicate_penalty removed: subsumed by info-gain reward in
        # ReviewWorkflow._compute_info_gain_rewards(), which checks outline
        # items (strengths, weaknesses) against each other and against all
        # evidence entries using cosine similarity, with continuous penalty.

        # Normalize: weighted sum divided by total weight → [0, 1]
        # This scalar is used as fallback when evidence credit is unavailable.
        ALL_SCALAR_COMPONENTS = {"score_diff", "format", "utility", "recall", "memory_reasoning"}
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
        args_valid_override: Optional[bool] = None
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
            args_valid_override: Optional override for args validity (e.g., False for rejected finish)

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
        if args_valid_override is not None:
            args_valid = args_valid_override
        else:
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

        # mem_error > 0: update on nonexistent ID (not_found) or wrong entry type
        # (cannot_update on N*/W*/S*). Soft proportional penalty: -0.2 per error,
        # so a step with 5 bad ops gets -1.0 and a step with 1 gets -0.2. Softer
        # than hallucination (-1.0 flat) since the model may still do useful work
        # in the same step; uncapped so bulk-erroring steps are penalized strongly.
        # mem_error = meta.get("mem_error", 0)
        # if mem_error > 0:
        #     return -0.2 * mem_error

        # Duplicates are rejected at ReviewLog insertion time (when duplicate_detection=True)
        # and counted as mem_error above (line 723), receiving -0.2 per duplicate penalty.
        # When duplicate_detection=False, post-hoc info_gain reward handles duplicates instead.
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

    def _extract_review_from_log(self, agent_log: Optional[ReviewLog]) -> Dict:
        """Extract review dict from agent's log.

        Args:
            agent_log: Agent's review log

        Returns:
            Review dict with summary, strengths, weaknesses, questions, overall_score
        """
        if not agent_log:
            return {
                "summary": "",
                "strengths": [],
                "weaknesses": [],
                "questions": [],
                "overall_score": None,
            }

        outline = agent_log.review_outline
        # Convert OutlineItems to text for reward calculation
        return {
            "summary": outline.summary,
            "strengths": outline.get_strengths_text(),
            "weaknesses": outline.get_weaknesses_text(),
            "questions": outline.get_questions_text(),
            "overall_score": outline.overall_score,
        }

    def _format_research_findings(
        self,
        findings: Dict,
        target_type: str,
        target_data: Dict,
    ) -> str:
        """Format research findings for main agent.

        Args:
            findings: Research findings dict
            target_type: 'claim' or 'question'
            target_data: The target data dict (serialized from Claim/Question)

        Returns:
            Formatted findings string
        """
        summary = findings.get("summary", "")
        cross_refs = findings.get("cross_references", [])
        evidence = findings.get("evidence", [])

        if target_type == "claim":
            response = f"Research complete for {target_data.get('id')}.\n\n"
            response += f"**Claim being verified**: {target_data.get('text')}\n\n"
        else:
            response = f"Research complete for {target_data.get('id')}.\n\n"
            response += f"**Question being investigated**: {target_data.get('question')}\n\n"

        response += f"**Research findings**:\n"
        response += f"- Summary: {summary}\n"
        response += f"- Sections examined: {', '.join(cross_refs) if cross_refs else 'none'}\n\n"

        if evidence:
            response += f"**Evidence collected**:\n"
            for e in evidence[:3]:
                response += f"  - [{e.get('section', 'N/A')}] {e.get('finding', 'N/A')}\n"

        if target_type == "claim":
            response += f"\n**Your judgment needed**: Review these findings and decide how to update the claim's status using the update memory operation."
        else:
            response += f"\n**Your judgment needed**: Review these findings and decide how to resolve the question using the update memory operation."

        return response

    @staticmethod
    def from_dict(env_args: Dict[str, Any]) -> "ReviewEnv":
        """Factory method to create ReviewEnv from dict args.

        This is used by rLLM's AgentTrainer to instantiate environments.

        Args:
            env_args: Dict containing either:
              - task: Dict with paper data (legacy pattern)
              - OR paper_id, paper_content, etc. directly (training pattern)
              Plus: research_model, reward_calculator, judge_model, etc.

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
            raw_cp = env_args.get("clustered_points", [])
            clustered_points = json.loads(raw_cp) if isinstance(raw_cp, str) else raw_cp
            task = {
                "paper_id": env_args.get("paper_id", ""),
                "paper_content": env_args.get("paper_content", ""),
                "human_avg_score": env_args.get("human_avg_score"),
                "clustered_points": clustered_points,
            }

        return ReviewEnv(
            task=task,
            research_model=env_args.get("research_model"),
            reward_calculator=env_args.get("reward_calculator"),
            max_research_iterations=env_args.get("max_research_iterations", 20),
            enable_shaping_rewards=env_args.get("enable_shaping_rewards", False),
            reward_mode=env_args.get("reward_mode", "full"),
            judge_model=env_args.get("judge_model"),
            format_penalty=env_args.get("format_penalty", env_args.get("incomplete_penalty", 0.0)),
            reward_weights=env_args.get("reward_weights"),
            memory_reasoning_mode=env_args.get("memory_reasoning_mode", "trajectory"),
            memory_reasoning_format=env_args.get("memory_reasoning_format", "scirm"),
            duplicate_detection=env_args.get("duplicate_detection", False),
            silent_duplicates=env_args.get("silent_duplicates", False),
        )
