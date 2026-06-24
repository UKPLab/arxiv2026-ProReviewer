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
from typing import Dict, Optional

import numpy as np

from rllm.agents.agent import Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)


class ReviewWorkflow(MultiTurnWorkflow):

    def __init__(self, credit_assignment: str = "broadcast", broadcast_decay: float = 0.0,
                 credit_status_update_steps: bool = False,
                 credit_multiple_matched_ids: bool = True,
                 quality_bonus: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.credit_assignment = credit_assignment
        # broadcast_decay: 0.0 = uniform (no decay), 0.95 = later steps get more.
        # When > 0, broadcast weight at step t = γ^(T-1-t), so last step = 1x, first = γ^(T-1).
        self.broadcast_decay = broadcast_decay
        # credit_status_update_steps: if True, also credit the step that updated a claim's status.
        self.credit_status_update_steps = credit_status_update_steps
        # credit_multiple_matched_ids: if True, credit all IDs in a comma-separated matched_id
        # (e.g. "S2, S5, S6"); if False, only use the first one.
        self.credit_multiple_matched_ids = credit_multiple_matched_ids
        # quality_bonus: if True, give +0.1 per good weakness (tech>=3 & ground>=3)
        # to contributing steps, capped at 0.5 per step.
        self.quality_bonus = quality_bonus

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

    def _apply_quality_bonus(self, trajectory, reward_result, failed_threshold):
        """Give bonus reward to steps contributing to good weaknesses.

        A weakness qualifies if its raw technical_depth >= 3 AND grounding >= 3.
        Each qualifying weakness gives +0.1 to its contributing steps, capped at
        0.5 per step.
        """
        final_snapshot = trajectory.steps[-1].info.get("log_snapshot")
        if not final_snapshot or final_snapshot.get("review_outline") is None:
            return

        rc = reward_result.get("reward_components", {})
        mem_details = rc.get("memory_reasoning_details", {})
        raw_scores = mem_details.get("raw_scores", {})
        raw_tech = raw_scores.get("technical_depth", {})
        raw_ground = raw_scores.get("grounding", {})
        if not raw_tech:
            return

        outline = final_snapshot.get("review_outline", {})
        outline_weaknesses = outline.get("weaknesses", [])
        outline_strengths = outline.get("strengths", [])
        outline_questions = outline.get("questions", [])

        # Build evidence ID -> step mapping
        evidence_id_to_step = {}
        claim_id_to_status_updated_step = {}
        for evidence_type in ["claims", "questions", "notes"]:
            for item in final_snapshot.get(evidence_type, []):
                if item.get("step") is not None:
                    evidence_id_to_step[item["id"]] = item["step"]
                if evidence_type == "claims" and item.get("status_updated_step") is not None:
                    claim_id_to_status_updated_step[item["id"]] = item["status_updated_step"]

        step2cred = {}
        for item_id in raw_tech:
            tech_score = raw_tech.get(item_id, 0)
            ground_score = raw_ground.get(item_id, 0)
            if tech_score >= 3 and ground_score >= 3:
                outline_item = self._resolve_item_id(
                    item_id, outline_strengths, outline_weaknesses, outline_questions
                )
                if outline_item is None:
                    continue
                contributing_steps = self._get_contributing_steps(
                    outline_item, evidence_id_to_step, claim_id_to_status_updated_step
                )
                for step_idx in contributing_steps:
                    step2cred[step_idx] = step2cred.get(step_idx, 0) + 0.1

        for step_idx, cred in step2cred.items():
            if step_idx < len(trajectory.steps):
                step = trajectory.steps[step_idx]
                if step.reward > failed_threshold:
                    step.reward += min(cred, 0.5)

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

    @staticmethod
    def _resolve_item_id(item_id: str, strengths: list, weaknesses: list, questions: list) -> dict:
        """Resolve item_id (S1, W2, Q3) to outline item dict.

        Args:
            item_id: "S1", "W2", "Q1", etc.
            strengths, weaknesses, questions: Lists of outline items

        Returns:
            Outline item dict or None if not found
        """
        try:
            if item_id.startswith("S"):
                idx = int(item_id[1:]) - 1
                if 0 <= idx < len(strengths):
                    return strengths[idx]
            elif item_id.startswith("W"):
                idx = int(item_id[1:]) - 1
                if 0 <= idx < len(weaknesses):
                    return weaknesses[idx]
            elif item_id.startswith("Q"):
                idx = int(item_id[1:]) - 1
                if 0 <= idx < len(questions):
                    return questions[idx]
        except (ValueError, IndexError):
            pass
        return None

    def _get_contributing_steps(
        self,
        outline_item: dict,
        evidence_id_to_step: dict,
        claim_id_to_status_updated_step: dict
    ) -> set:
        """Get all steps that contributed to an outline item.

        Includes:
        - Step that created the outline item
        - Steps that created referenced evidence (claims, questions, notes)
        - Steps that updated claim status (if credit_status_update_steps is True)

        Args:
            outline_item: Outline item dict with 'step', 'related_claims', etc.
            evidence_id_to_step: Map from evidence ID (C1, Q2, N3) to creation step
            claim_id_to_status_updated_step: Map from claim ID to status update step

        Returns:
            Set of step indices
        """
        contributing_steps = set()

        # Credit outline creator step
        outline_step = outline_item.get("step")
        if outline_step is not None:
            contributing_steps.add(outline_step)

        # Credit evidence creator steps
        for tag in outline_item.get("related_claims", []):
            if tag in evidence_id_to_step:
                contributing_steps.add(evidence_id_to_step[tag])
            if self.credit_status_update_steps and tag in claim_id_to_status_updated_step:
                contributing_steps.add(claim_id_to_status_updated_step[tag])

        for tag in (outline_item.get("related_questions", []) +
                    outline_item.get("related_notes", [])):
            if tag in evidence_id_to_step:
                contributing_steps.add(evidence_id_to_step[tag])

        return contributing_steps

    def _compute_evidence_credit(self, trajectory: Trajectory, reward_result: dict) -> dict:
        """Compute per-step credit based on evidence contribution (stepwise training).

        Reward structure (5 components, 2 assignment types):

        Broadcast (trajectory-quality baseline, same for all steps):
          - format:     review completeness + weakness count     [0, 1]
          - score_diff: how close predicted score is to human    [0, 1]
          - recall:     fraction of human points covered         [0, 1]

        Evidence-based (only to contributing steps):
          - utility:    per-weakness quality, traced to evidence  [0, 1]

        Per-step (already applied in env.step(), not here):
          - syntactic:  tool format + hallucination penalty      [-1, 0]

        All components are normalised by the same total weight sum
        (w_format + w_score_diff + w_memory_reasoning [+ w_recall]) so that
        configured weights are faithfully respected.  Evidence-based per-item
        scores are multiplied by w_memory_reasoning / total_weight_sum before
        being assigned to contributing steps.

        Args:
            trajectory: The trajectory containing steps
            reward_result: Reward result dict from env with components and weights

        Returns:
            Dict mapping step_index -> total credit, or None to trigger uniform fallback
        """
        # Get the final log snapshot
        final_snapshot = trajectory.steps[-1].info.get("log_snapshot")
        if not final_snapshot or final_snapshot.get("review_outline") is None:
            return None

        # Initialize per-step credit
        num_steps = len(trajectory.steps)
        step_credits = {i: 0.0 for i in range(num_steps)}

        # Get reward components and weights
        reward_components = reward_result.get("reward_components", {})
        weights = reward_result.get("weights", {})

        # === BROADCAST: trajectory-quality baseline to all steps ===
        # Check if memory_reasoning is evidence-based
        mem_details = reward_components.get("memory_reasoning_details", {})
        has_evidence_mem = mem_details.get("evidence_based", False)

        # recall is excluded here — it is handled via evidence-based credit below.
        # Both broadcast and recall evidence are normalized by the same total weight
        # sum (broadcast components + recall) so their relative magnitudes faithfully
        # reflect the configured weights.
        if has_evidence_mem:
            # Evidence-based mode: memory_reasoning is distributed per-item below
            BROADCAST_COMPONENTS = {"format", "score_diff"}
        else:
            # Traditional mode: memory_reasoning is broadcast uniformly
            BROADCAST_COMPONENTS = {"format", "score_diff", "memory_reasoning"}

        active_broadcast = {k for k in BROADCAST_COMPONENTS if k in reward_components}
        has_recall = "recall" in reward_components

        # Include memory_reasoning weight in the denominator even in evidence
        # mode so all components share the same normalisation basis and the
        # configured weight ratio (e.g. memory_reasoning=2.0 vs format=1.0)
        # is preserved.
        total_weight_sum = (
            sum(weights.get(k, 1.0) for k in active_broadcast)
            + (weights.get("memory_reasoning", 1.0) if has_evidence_mem else 0.0)
            + (weights.get("recall", 1.0) if has_recall else 0.0)
        )
        if total_weight_sum == 0.0:
            total_weight_sum = 1.0

        if active_broadcast:
            broadcast_raw = sum(
                weights.get(k, 1.0) * reward_components[k]
                for k in active_broadcast
            )
            broadcast_normalized = broadcast_raw / total_weight_sum
        else:
            broadcast_normalized = 0.0

        for step_idx in step_credits:
            step_credits[step_idx] += broadcast_normalized

        # === EVIDENCE-BASED: utility credit to contributing steps ===
        # Utility is already in [0, 1] (avg over weaknesses), so no extra normalization needed.
        # Build evidence ID to step mapping from log snapshot
        evidence_id_to_step = {}
        claim_id_to_status_updated_step = {}
        for evidence_type in ["claims", "questions", "notes"]:
            for item in final_snapshot.get(evidence_type, []):
                if item.get("step") is not None:
                    evidence_id_to_step[item["id"]] = item["step"]
                if evidence_type == "claims" and item.get("status_updated_step") is not None:
                    claim_id_to_status_updated_step[item["id"]] = item["status_updated_step"]


        if "utility" in reward_components:
            utility_details = reward_components.get("utility_details", [])
            n_weakness_points = len(utility_details) if utility_details else 1

            # Get weaknesses from final outline
            outline_weaknesses = final_snapshot.get("review_outline", {}).get("weaknesses", [])

            # Distribute each weakness's utility score to its contributing steps.
            # Normalize by n_weakness_points so total utility credit ≈ avg_utility ∈ [0,1].
            for i, utility_per_weakness in enumerate(utility_details):
                utility_score = utility_per_weakness["utility_score"]
                outline_idx = utility_per_weakness["outline_idx"]

                if outline_idx < len(outline_weaknesses):
                    outline_item = outline_weaknesses[outline_idx]
                    contributing_steps = set()

                    # 1. Credit the step that created this weakness outline item
                    outline_step = outline_item.get("step")
                    assert outline_step is not None
                    contributing_steps.add(outline_step)

                    # 2. Credit steps that created the evidence referenced by this weakness,
                    #    and optionally steps that updated claim status (verification work).
                    for tag in outline_item.get("related_claims", []):
                        if tag in evidence_id_to_step:
                            contributing_steps.add(evidence_id_to_step[tag])
                        if self.credit_status_update_steps and tag in claim_id_to_status_updated_step:
                            contributing_steps.add(claim_id_to_status_updated_step[tag])
                    for tag in (outline_item.get("related_questions", []) +
                                outline_item.get("related_notes", [])):
                        if tag in evidence_id_to_step:
                            contributing_steps.add(evidence_id_to_step[tag])

                    if contributing_steps:
                        credit = utility_score / n_weakness_points  # [0, 1] total
                        for step_idx in contributing_steps:
                            step_credits[step_idx] += credit

        # === EVIDENCE-BASED: memory reasoning per-item credit ===
        if has_evidence_mem:
            # Apply the configured memory_reasoning weight so its relative
            # importance vs broadcast components is respected.  The weight
            # is already included in total_weight_sum (denominator), so we
            # multiply each per-item score by w_mr / total_weight_sum —
            # exactly the same normalisation the broadcast path uses.
            mem_weight_factor = (
                weights.get("memory_reasoning", 1.0) / total_weight_sum
            )

            outline = final_snapshot.get("review_outline", {})
            outline_strengths = outline.get("strengths", [])
            outline_weaknesses = outline.get("weaknesses", [])
            outline_questions = outline.get("questions", [])

            # 1. Factual correctness per-step penalties
            # factual_per_step = mem_details.get("factual_per_step", {})
            # for step_idx, penalty in factual_per_step.items():
            #     if 0 <= step_idx < num_steps:
            #         step_credits[step_idx] += penalty

            # 2. Technical depth per-item (W/Q only)
            tech_per_item = mem_details.get("technical_depth_per_item", {})
            n_tech_items = len(tech_per_item) if tech_per_item else 1
            for item_id, score_normalized in tech_per_item.items():
                outline_item = self._resolve_item_id(
                    item_id, outline_strengths, outline_weaknesses, outline_questions
                )
                if outline_item is None:
                    continue
                contributing_steps = self._get_contributing_steps(
                    outline_item, evidence_id_to_step, claim_id_to_status_updated_step
                )
                if contributing_steps:
                    credit = (score_normalized / n_tech_items) * mem_weight_factor
                    for step_idx in contributing_steps:
                        step_credits[step_idx] += credit

            # 3. Outline grounding per-item (W/Q only)
            ground_per_item = mem_details.get("grounding_per_item", mem_details.get("outline_grounding_per_item", {}))
            n_ground_items = len(ground_per_item) if ground_per_item else 1
            for item_id, score_normalized in ground_per_item.items():
                outline_item = self._resolve_item_id(
                    item_id, outline_strengths, outline_weaknesses, outline_questions
                )
                if outline_item is None:
                    continue
                contributing_steps = self._get_contributing_steps(
                    outline_item, evidence_id_to_step, claim_id_to_status_updated_step
                )
                if contributing_steps:
                    credit = (score_normalized / n_ground_items) * mem_weight_factor
                    for step_idx in contributing_steps:
                        step_credits[step_idx] += credit

            # 4. Factual correctness simple per-item (W/Q only)
            factual_simple_per_item = mem_details.get("factual_simple_per_item", {})
            n_factual_items = len(factual_simple_per_item) if factual_simple_per_item else 1
            for item_id, score_normalized in factual_simple_per_item.items():
                outline_item = self._resolve_item_id(
                    item_id, outline_strengths, outline_weaknesses, outline_questions
                )
                if outline_item is None:
                    continue
                contributing_steps = self._get_contributing_steps(
                    outline_item, evidence_id_to_step, claim_id_to_status_updated_step
                )
                if contributing_steps:
                    credit = (score_normalized / n_factual_items) * mem_weight_factor
                    for step_idx in contributing_steps:
                        step_credits[step_idx] += credit

        # === EVIDENCE-BASED: recall credit to steps that produced covered points ===
        # Capped-budget: total recall credit = recall_weight / total_weight_sum
        # (same scale as if recall were 1.0 in the broadcast formula).
        # Each covered point gets a proportional share of the budget based on
        # its coverage score (full=1.0, partial=0.5).  This gives each step
        # an undiluted signal for producing a covered point — credit does NOT
        # shrink as the model covers more points.
        if has_recall:
            recall_results = reward_components.get("recall_results", [])
            recall_weight = weights.get("recall", 1.5)
            recall_budget = recall_weight / total_weight_sum

            outline = final_snapshot.get("review_outline", {})
            outline_strengths = outline.get("strengths", [])
            outline_weaknesses = outline.get("weaknesses", [])

            # First pass: compute total coverage score for normalization
            coverage_entries = []  # (coverage_score, matched_id, recall_result)
            sum_coverage_scores = 0.0
            for r in recall_results:
                coverage = r.get("coverage")
                if coverage == "not_covered":
                    continue
                coverage_score = 1.0 if coverage == "full" else 0.5
                matched_id = r.get("matched_id")
                if not matched_id:
                    continue
                coverage_entries.append((coverage_score, matched_id, r))
                sum_coverage_scores += coverage_score

            if sum_coverage_scores == 0.0:
                sum_coverage_scores = 1.0  # Avoid division by zero

            # Second pass: distribute budget proportionally
            for coverage_score, matched_id, r in coverage_entries:
                # Resolve matched_id(s) to outline items.
                # The judge occasionally returns multiple IDs (e.g. "S2, S5, S6").
                # If credit_multiple_matched_ids=True, credit all; otherwise only the first.
                all_ids = [mid.strip() for mid in matched_id.split(",") if mid.strip()]
                matched_ids = all_ids if self.credit_multiple_matched_ids else all_ids[:1]
                contributing_steps = set()
                any_resolved = False
                for mid in matched_ids:
                    outline_item = None
                    try:
                        if mid.startswith("S"):
                            idx = int(mid[1:]) - 1
                            if 0 <= idx < len(outline_strengths):
                                outline_item = outline_strengths[idx]
                        elif mid.startswith("W"):
                            idx = int(mid[1:]) - 1
                            if 0 <= idx < len(outline_weaknesses):
                                outline_item = outline_weaknesses[idx]
                    except (ValueError, IndexError):
                        pass

                    if outline_item is None:
                        print(f"Warning: matched_id {mid} not found in outline, skipping.")
                        continue

                    any_resolved = True
                    if outline_item.get("step") is not None:
                        contributing_steps.add(outline_item["step"])
                    for tag in outline_item.get("related_claims", []):
                        if tag in evidence_id_to_step:
                            contributing_steps.add(evidence_id_to_step[tag])
                        if self.credit_status_update_steps and tag in claim_id_to_status_updated_step:
                            contributing_steps.add(claim_id_to_status_updated_step[tag])
                    for tag in (outline_item.get("related_questions", []) +
                                outline_item.get("related_notes", [])):
                        if tag in evidence_id_to_step:
                            contributing_steps.add(evidence_id_to_step[tag])

                if not any_resolved:
                    continue

                if contributing_steps:
                    # Proportional share of the fixed budget: credit per point
                    # is proportional to its coverage_score relative to the sum.
                    # Total credit across all points = recall_budget (capped).
                    credit = recall_budget * coverage_score 
                    # credit = recall_budget * (coverage_score / sum_coverage_scores)
                    for step_idx in contributing_steps:
                        step_credits[step_idx] += credit

        return step_credits

    @staticmethod
    def _log_content_fingerprint(log_snapshot: dict) -> str:
        """Return a hashable fingerprint of the log content (ignoring timestamps).

        Captures claims (id+status+issues), questions (id+status+answer),
        note count, and outline content counts + summary text.  Two snapshots
        with the same fingerprint contain identical review-relevant information.
        """
        if not log_snapshot:
            return ""
        outline = log_snapshot.get("review_outline") or {}
        claims = log_snapshot.get("claims") or []
        questions = log_snapshot.get("questions") or []
        notes = log_snapshot.get("notes") or []

        parts = []
        for c in claims:
            parts.append(f"C:{c.get('id','')}:{c.get('status','')}:{c.get('issues', [])}")
        for q in questions:
            parts.append(f"Q:{q.get('id','')}:{q.get('status','')}:{(q.get('answer','') or '')[:80]}")
        parts.append(f"N:{len(notes)}")
        for sec in ("strengths", "weaknesses", "questions"):
            items = outline.get(sec) or []
            parts.append(f"O_{sec}:{len(items)}")
        parts.append(f"SUM:{(outline.get('summary','') or '')[:80]}")
        parts.append(f"SC:{outline.get('overall_score')}")
        return "|".join(parts)

    def _compute_stagnation_penalties(
        self, trajectory: Trajectory, info_gain: Optional[Dict[int, float]]
    ) -> Dict[int, float]:
        """Penalise steps that fail to update memory after a read/search.

        Logic: at step N the model reads a section (or searches). At step N+1
        the model has seen that content and should update memory. If
        log_snapshot[N+1] == log_snapshot[N], the model added nothing
        meaningful → penalise step N+1.

        Returns:
            Dict mapping step_index → penalty (always <= 0).
        """
        STAGNATION_PENALTY = -0.5
        penalties: Dict[int, float] = {}
        steps = trajectory.steps

        # Pre-compute fingerprints for all steps
        fps = []
        for step in steps:
            snapshot = step.info.get("log_snapshot")
            fps.append(self._log_content_fingerprint(snapshot))

        READ_ACTIONS = {"read_section", "search_paper"}

        for step_idx in range(1, len(steps)):  # step_idx is the current step
            prev_action = steps[step_idx - 1].action or {}
            prev_action_name = prev_action.get("name", "")

            # Previous step was a read/search → check if current step added anything
            if prev_action_name in READ_ACTIONS:
                if fps[step_idx] == fps[step_idx - 1]:
                    penalties[step_idx] = STAGNATION_PENALTY

        return penalties

    def _compute_info_gain_rewards(self, trajectory: Trajectory) -> Optional[Dict[int, float]]:
        """Compute per-step info gain reward using cosine similarity.

        Collects all memory entries (notes, claims, questions) and outline items
        (strengths, weaknesses) from the final log snapshot, embeds them in one
        batched call, then scores each step by how novel its contributions are
        relative to earlier entries.

        Three zones per entry:
          - max_sim < 0.5  → novel  → +0.1 fixed bonus
          - 0.5 ≤ max_sim < 0.85 → gray zone → 0.0
          - max_sim ≥ 0.85 → duplicate → -(max_sim - 0.85) / 0.15 continuous

        Per-step reward is clamped to [-1.0, +0.3].

        Returns:
            Dict mapping step_index → info gain reward, or None on failure.
        """
        # Get final log snapshot
        final_snapshot = trajectory.steps[-1].info.get("log_snapshot")

        # Collect entries grouped by type — duplicates are only checked
        # within the same type.  A note and a question on the same topic are
        # both valuable; penalizing cross-type similarity would discourage
        # the model from building complementary evidence.
        entry_groups = {}  # type_key -> list of (step_idx, text)

        notes = []
        for item in final_snapshot.get("notes", []):
            notes.append((item.get("step", -1), item["text"]))
        if notes:
            entry_groups["notes"] = notes

        claims = []
        for item in final_snapshot.get("claims", []):
            claims.append((item.get("step", -1), item["text"]))
        if claims:
            entry_groups["claims"] = claims

        questions = []
        for item in final_snapshot.get("questions", []):
            text = item.get("question") or item.get("text", "")
            questions.append((item.get("step", -1), text))
        if questions:
            entry_groups["questions"] = questions

        # Outline items: strengths, weaknesses, outline questions
        outline = final_snapshot.get("review_outline") or {}
        for section_key in ("strengths", "weaknesses", "questions"):
            items = []
            for item in outline.get(section_key, []):
                if isinstance(item, dict) and item.get("text"):
                    items.append((item.get("step", -1), item["text"]))
            if items:
                entry_groups[f"outline_{section_key}"] = items

        # Need at least one group with 2+ entries
        if not any(len(g) >= 2 for g in entry_groups.values()):
            return None

        # Get embedding client from env's reward calculator
        reward_calculator = getattr(self.env, "reward_calculator", None)
        if reward_calculator is None:
            return None

        # Batch all texts in one embedding call, then split back by group
        all_texts = []
        group_slices = {}  # group_key -> (start, end) indices in all_texts
        for key, group in entry_groups.items():
            start = len(all_texts)
            all_texts.extend(text for _, text in group)
            group_slices[key] = (start, len(all_texts))

        if len(all_texts) < 2:
            return None

        try:
            from openai import OpenAI
            from utils.helpers.llm import MODEL_CONFIGS

            config = MODEL_CONFIGS.get(reward_calculator.embed_model, {})
            base_url = config.get("base_url", "http://localhost:8000/v1")
            api_key = config.get("api_key", "EMPTY")
            model_name = config.get("model", reward_calculator.embed_model).removeprefix("openai/")

            if reward_calculator._embed_client is None:
                reward_calculator._embed_client = OpenAI(base_url=base_url, api_key=api_key)

            response = reward_calculator._embed_client.embeddings.create(
                model=model_name,
                input=all_texts,
            )
            all_embeddings = np.array([
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ])
            # L2 normalize
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            all_embeddings = all_embeddings / norms
        except Exception as e:
            logger.warning(f"Info gain embedding call failed, skipping: {e}")
            return None

        # Per-step reward accumulation
        step_rewards: Dict[int, float] = defaultdict(float)

        # Process each group independently
        for key, group in entry_groups.items():
            if len(group) < 2: continue
                # Single entry — novel by definition
                # step_i = group[0][0]
                # if step_i >= 0:
                #     step_rewards[step_i] += 0.1

            start, end = group_slices[key]
            group_embeddings = all_embeddings[start:end]
            sim_matrix = group_embeddings @ group_embeddings.T

            for i, (step_i, _) in enumerate(group):
                if step_i < 0:
                    continue

                # Compare against earlier entries in this group:
                # entries from earlier steps, or earlier-indexed at same step
                compare_to = [
                    j for j, (step_j, _) in enumerate(group)
                    if step_j < step_i or (step_j == step_i and j < i)
                ]

                if not compare_to:
                    max_sim = 0.0
                else:
                    max_sim = max(float(sim_matrix[i, j]) for j in compare_to)

                # 3-zone reward
                # if max_sim < 0.5:
                #     step_rewards[step_i] += 0.2
                if max_sim >= 0.9:
                    step_rewards[step_i] -= 0.2
                # else: gray zone, 0.0

        # Clamp per-step totals
        clamped = {}
        for step_idx in step_rewards:
            clamped[step_idx] = max(min(step_rewards[step_idx], 0.0), -0.5)

        return clamped

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
        """Compute trajectory reward from env.compute_final_reward().

        Overrides the default sum-of-step-rewards because the terminal step
        (which carries the reward in the default pattern) gets popped during
        postprocess_episode() since it has no chat_completions.

        Supports two credit assignment modes controlled by
        ``self.workflow_args.get("credit_assignment", "broadcast")``:

        - ``"broadcast"`` (default): Every step receives the same outcome
          reward (syntactic per-step reward + uniform broadcast of all
          review-level components).  Simple, avoids double-counting from
          leaked near-duplicate outline items.

        - ``"evidence"``: Distributes utility/recall rewards to the steps
          that created evidence or outline items used in the final review.
          Provides finer-grained credit but can over-reward steps whose
          evidence is referenced by near-duplicate outline entries.
        """
        # Pass log_snapshot to env so compute_final_reward can include memory_reasoning
        if "memory_reasoning" in self.env.reward_modes:
            for step in reversed(trajectory.steps):
                if step.info.get("log_snapshot"):
                    self.env._log_snapshot = step.info["log_snapshot"]
                    break
            # Collect all per-step snapshots for trajectory analysis.
            # Include observation context (action, section, truncated content)
            # so the trajectory judge can assess whether memory is grounded.
            #
            # Step layout in the trajectory:
            #   step[i].observation = env result from step i-1's action
            #   step[i].info       = {action_name, log_snapshot, ...} for step i
            # So the observation that resulted from step i's action lives
            # in step[i+1].observation.
            import re as _re
            all_steps = trajectory.steps
            step_snapshots = []
            for i, step in enumerate(all_steps):
                snap = step.info.get("log_snapshot")
                if not snap:
                    continue
                # Attach lightweight observation context to the snapshot
                obs_ctx = {
                    "action_name": step.info.get("action_name", ""),
                    "section_name": step.info.get("section_name", ""),
                    "query": step.info.get("query", ""),
                }
                # The observation resulting from this step's action is on the
                # *next* step (step[i+1].observation).  It contains:
                #   <memory_operations_results>...</memory_operations_results>
                #   <action_result>...</action_result>
                # We extract both: memory ops as a prefix (shows what the
                # agent tried, including errors), action result as the
                # paper content snippet for grounding checks.
                raw_obs = ""
                if i + 1 < len(all_steps):
                    raw_obs = all_steps[i + 1].observation or ""
                if isinstance(raw_obs, dict):
                    raw_obs = raw_obs.get("action_result", "")
                if isinstance(raw_obs, str) and raw_obs:
                    # Extract action result (paper content)
                    ar_match = _re.search(
                        r"<action_result>\s*(.*?)\s*</action_result>",
                        raw_obs, _re.DOTALL,
                    )
                    snippet = ar_match.group(1) if ar_match else raw_obs
                    # Strip boilerplate prefix ("Successfully read section '...'. Content:\n[...]:")
                    # to maximise useful paper content in the snippet.
                    snippet = _re.sub(
                        r"^Successfully read section '[^']*'\.\s*Content:\s*(?:\[[^\]]*\]:\s*)?",
                        "", snippet,
                    )
                    obs_ctx["observation_snippet"] = snippet
                snap = {**snap, "_obs_ctx": obs_ctx}
                step_snapshots.append(snap)
            self.env._step_snapshots = step_snapshots

        # Compute trajectory-level reward (all non-syntactic components)
        raw_trajectory_reward = self.env.compute_final_reward()

        # If judge failed, skip this trajectory entirely
        if raw_trajectory_reward is None:
            trajectory.reward = None
            trajectory.info["judge_failed"] = True
            # Store reward result for logging even if failed
            reward_result = getattr(self.env, '_reward_result', None)
            if reward_result is not None:
                trajectory.info['reward_details'] = reward_result
            return

        # Get reward result with components and weights
        reward_result = getattr(self.env, '_reward_result')

        # --- Info gain: cosine-similarity-based per-step reward ---
        # Only used when duplicate_detection=False (post-hoc duplicate penalty).
        # When duplicate_detection=True, duplicates are rejected at insertion time
        # and counted as mem_error, so info_gain and stagnation are skipped.
        info_gain = None
        use_info_gain = "info_gain" in self.env.reward_modes and not getattr(self.env, 'duplicate_detection', False)
        if use_info_gain:
            info_gain = self._compute_info_gain_rewards(trajectory)
        if info_gain is not None:
            info_gain_weight = self.env._raw_weights.get("info_gain", 1.0)
            for step_idx, step in enumerate(trajectory.steps):
                if step_idx in info_gain:
                    step.reward += info_gain[step_idx] * info_gain_weight
            # Store in reward result for logging
            if reward_result is not None:
                reward_result.setdefault("reward_components", {})["info_gain"] = info_gain

        # --- Stagnation penalty: punish no-op steps that neither produce
        # info-gain nor change the review log. Prevents the model from
        # looping over already-read sections without contributing value. ---
        # Only applied when info_gain is enabled (duplicate_detection=False)
        if use_info_gain:
            stagnation = self._compute_stagnation_penalties(trajectory, info_gain)
            if stagnation:
                for step_idx, penalty in stagnation.items():
                    trajectory.steps[step_idx].reward += penalty
                if reward_result is not None:
                    reward_result.setdefault("reward_components", {})["stagnation"] = stagnation
                logger.debug(
                    "Stagnation penalty applied to %d / %d steps",
                    len(stagnation), len(trajectory.steps),
                )

        credit_mode = self.credit_assignment

        # Failed steps (syntactic reward <= -1.0) keep their penalty and
        # don't receive broadcast reward — they contributed nothing.
        FAILED_STEP_THRESHOLD = -1.0

        if credit_mode == "evidence" and reward_result is not None:
            # Evidence-based distribution
            # print(f"[CreditAssignment] Using EVIDENCE-BASED credit (credit_mode={credit_mode})")
            evidence_credit = self._compute_evidence_credit(trajectory, reward_result)
            if evidence_credit is not None:
                # n_unique = len(set(round(v, 6) for v in evidence_credit.values()))
                # print(f"[CreditAssignment] Evidence credit computed: {len(evidence_credit)} steps, {n_unique} unique values")
                for step_idx, step in enumerate(trajectory.steps):
                    step.info["syntactic_reward"] = step.reward
                    if step.reward <= FAILED_STEP_THRESHOLD:
                        continue  # skip broadcast for failed steps
                    step.reward += evidence_credit.get(step_idx, 0.0)
            else:
                # Fallback to broadcast if evidence credit fails
                logger.warning("Evidence credit assignment returned None, falling back to broadcast.")
                self._broadcast_reward(trajectory.steps, raw_trajectory_reward, FAILED_STEP_THRESHOLD)
        else:
            # print(f"[CreditAssignment] Using BROADCAST (credit_mode={credit_mode})")
            self._broadcast_reward(trajectory.steps, raw_trajectory_reward, FAILED_STEP_THRESHOLD)

        # === Quality bonus in broadcast mode ===
        # Reward steps that contributed to good weaknesses (tech_depth >= 3 AND
        # grounding >= 3).  +0.1 per qualifying weakness, capped at 0.5/step.
        if self.quality_bonus and reward_result is not None:
            self._apply_quality_bonus(trajectory, reward_result, FAILED_STEP_THRESHOLD)

        # finish_bonus dropped — broadcast of trajectory-quality rewards to all steps
        # already provides sufficient signal; concentrating extra reward on finish
        # creates magnitude imbalance in step_level GRPO.

        # Set trajectory.reward to the average of distributed rewards (for logging)
        trajectory.reward = sum(step.reward for step in trajectory.steps) / len(trajectory.steps)

        # Store detailed reward result in trajectory info for episode logging
        if reward_result is not None:
            if 'reward_details' not in trajectory.info:
                trajectory.info['reward_details'] = {}
            trajectory.info['reward_details'] = reward_result
