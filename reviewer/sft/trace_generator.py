"""Generate SFT training data by reconstructing review processes from gold human reviews.

The teacher LLM receives the gold review in its system prompt and reconstructs
the step-by-step reasoning process (read sections, log claims, build outline)
using the real ProReviewer + ReviewEnv loop. The saved traces preserve the
reconstruction system prompt, full LLM content, and reasoning_content.
"""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from reviewer.core.reviewer_memory import ReviewLog
from reviewer.prompts.reviewer_prompts_direct import (
    REVIEWER_RECONSTRUCTION_SYSTEM_PROMPT,
)
from reviewer.core.proreviewer import ProReviewer
from reviewer.core.review_env import ReviewEnv

logger = logging.getLogger(__name__)


class TraceGenerator:
    """Generates SFT training traces by reconstructing review processes from gold reviews."""

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        model_name: str,
        max_steps: int = 30,
        max_retries: int = 2,
        action_retries: int = 3,
    ):
        self.client = llm_client
        self.model = model_name
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.action_retries = action_retries

    async def generate_trace(
        self, paper_data: dict, human_review: dict
    ) -> Optional[tuple[list[dict], bool]]:
        """Generate one SFT example from a paper and its gold human review.

        Args:
            paper_data: Paper JSON dict with keys: id/paper_id, title, markdown.content
            human_review: Single review dict with keys: summary, strengths, weaknesses,
                          questions, rating

        Returns:
            Tuple of (messages, is_success) where messages is a list of
            {"role", "content"} dicts and is_success is True only when the agent
            finished within max_steps. Returns None if trace generation failed
            (e.g. LLM call error).
        """
        recon_prompt = self._build_recon_prompt(human_review)

        paper_id = paper_data.get("paper_id") or paper_data.get("id", "unknown")
        title = paper_data.get("title", "")
        content = paper_data.get("markdown", {}).get("content", "")
        paper_content = f"# {title}\n\n{content}" if title else content

        task = {
            "paper_id": paper_id,
            "paper_content": paper_content,
            "human_avg_score": human_review.get("rating"),
        }

        env = ReviewEnv(task=task, reward_mode="format")
        obs, info = env.reset()

        agent = ProReviewer(system_prompt=recon_prompt)
        agent.reset()
        agent.update_from_env(obs, 0, False, info)

        done = False
        reasoning_contents = []  # per-step reasoning_content from the LLM
        for step_i in range(self.max_steps):
            # Snapshot log state — memory ops in update_from_model mutate it
            log_snapshot = agent.log.model_dump()

            has_error = False
            next_obs, reward, step_done, step_info = None, 0.0, False, {}
            for attempt in range(1 + self.action_retries):
                if attempt > 0:
                    # Restore log and clear the failed step's model output
                    agent.log = ReviewLog.model_validate(log_snapshot)
                    agent.trajectory.steps[-1].model_response = None
                    agent.trajectory.steps[-1].chat_completions = None
                    logger.debug(f"[{paper_id}] Step {step_i} retry {attempt}/{self.action_retries}")

                llm_result = await self._call_llm(agent.chat_completions)
                if llm_result is None:
                    logger.warning(f"[{paper_id}] LLM call failed at step {step_i}")
                    return None

                action = agent.update_from_model(llm_result["content"])
                action_dict = action.action if hasattr(action, "action") else action
                meta = action_dict.get("_meta", {})

                has_error = (
                    meta.get("mem_error", 0) > 0
                    or meta.get("validation_errors", 0) > 0
                    or action_dict.get("name") in ("format_error", "tool_error")
                )
                if has_error:
                    continue

                next_obs, reward, step_done, step_info = env.step(action_dict)
                has_error = not step_info.get("args_valid", True)

                if not has_error:
                    break

            if has_error:
                logger.warning(f"[{paper_id}] Step {step_i}: all {self.action_retries} retries exhausted, skipping sample")
                return None
            reasoning_contents.append(llm_result.get("reasoning_content"))
            agent.update_from_env(next_obs, reward, step_done, step_info)

            if step_done:
                done = True
                break

        if not done:
            logger.warning(f"[{paper_id}] Trace did not finish within {self.max_steps} steps")

        # Validate outline has minimum required fields
        outline = agent.log.review_outline
        outline_complete = bool(
            outline.summary
            and outline.strengths
            and outline.weaknesses
            and outline.overall_score is not None
        )
        is_success = done and outline_complete

        if done and not outline_complete:
            missing = [
                field for field, val in [
                    ("summary", outline.summary),
                    ("strengths", outline.strengths),
                    ("weaknesses", outline.weaknesses),
                    ("overall_score", outline.overall_score),
                ]
                if not val and val != 0
            ]
            logger.warning(
                f"[{paper_id}] Trace finished but outline incomplete (missing: {missing})"
            )

        messages = self._extract_raw_messages(agent.trajectory, reasoning_contents)
        logger.info(
            f"[{paper_id}] Generated trace: {len(agent.trajectory.steps)} steps, "
            f"{len(outline.strengths)} strengths, {len(outline.weaknesses)} weaknesses, "
            f"is_success={is_success}"
        )
        return messages, is_success

    def _build_recon_prompt(self, review: dict) -> str:
        """Format the reconstruction system prompt with the gold review."""
        return REVIEWER_RECONSTRUCTION_SYSTEM_PROMPT.format(
            summary=review.get("summary", ""),
            strengths=review.get("strengths", ""),
            weaknesses=review.get("weaknesses", ""),
            questions=review.get("questions", ""),
            rating=review.get("rating", ""),
        )

    async def _call_llm(self, messages: list[dict]) -> Optional[dict]:
        """Call the teacher LLM with retries on failure.

        Returns:
            Dict with "content" and optional "reasoning_content", or None on failure.
        """
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.6,
                    top_p=0.95,
                    presence_penalty=0.0,
                    extra_body={
                        "top_k": 20,
                        "min_p": 0.0,
                        "repetition_penalty": 1.0,
                    },
                )
                msg = resp.choices[0].message
                return {
                    "content": msg.content,
                    "reasoning_content": msg.reasoning,
                }
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
        return None

    def _extract_raw_messages(self, trajectory, reasoning_contents: list) -> list[dict]:
        """Build multi-turn conversation preserving the reconstruction system prompt,
        full LLM content, and reasoning_content per assistant turn."""
        # First step's chat_completions[0] is the system message with reconstruction prompt
        system_content = None
        if trajectory.steps and trajectory.steps[0].chat_completions:
            system_content = trajectory.steps[0].chat_completions[0].get("content")
        messages = [{"role": "system", "content": system_content or ""}]

        rc_idx = 0  # index into reasoning_contents
        for step in trajectory.steps:
            if step.observation:
                messages.append({"role": "user", "content": step.observation})
            if step.model_response:
                assistant_msg = {"role": "assistant", "content": step.model_response}
                if rc_idx < len(reasoning_contents) and reasoning_contents[rc_idx]:
                    assistant_msg["reasoning_content"] = reasoning_contents[rc_idx]
                messages.append(assistant_msg)
                rc_idx += 1

        return messages

    @staticmethod
    def select_qualified_reviews(
        reviews: list[dict],
        min_confidence: int = 4,
        min_sw_length: int = 1500,
    ) -> list[dict]:
        """Return all reviews passing quality filters, sorted by detail (descending).

        Args:
            reviews: List of review dicts
            min_confidence: Minimum reviewer confidence score (1-5)
            min_sw_length: Minimum combined length of strengths + weaknesses

        Returns:
            List of qualified review dicts, sorted by detail score (most detailed first).
            Empty list if no review passes the filters.
        """
        def detail_score(r):
            return len(r.get("strengths", "")) + len(r.get("weaknesses", ""))

        candidates = [
            r for r in reviews
            if (r.get("confidence") or 0) >= min_confidence
            and detail_score(r) >= min_sw_length
        ]
        return sorted(candidates, key=detail_score, reverse=True)

    @staticmethod
    def select_best_review(
        reviews: list[dict],
        min_confidence: int = 4,
        min_sw_length: int = 1500,
    ) -> Optional[dict]:
        """Pick the most detailed review passing quality filters."""
        qualified = TraceGenerator.select_qualified_reviews(reviews, min_confidence, min_sw_length)
        return qualified[0] if qualified else None
