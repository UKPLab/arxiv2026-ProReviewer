"""ReviewAgent - rLLM BaseAgent implementation for paper review.

This module implements the agent side of the agent-environment interaction
following the rLLM/MathAgent pattern.
"""

from typing import Any, Dict, List, Optional, Tuple
import copy
import json
import re
import logging
from pydantic import ValidationError

from reviewer.core.reviewer_memory import ReviewLog
from reviewer.core.reviewer_prompts_direct import REVIEWER_DIRECT_SYSTEM_PROMPT
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

class ReviewAgent(BaseAgent):
    """Review agent extending rLLM's BaseAgent pattern.

    The ReviewAgent:
    1. Maintains the ReviewLog as internal state
    2. Builds chat completions for LLM inference
    3. Parses LLM responses into actions
    4. Executes memory operations locally
    5. Tracks trajectory for RL training

    Key design decisions:
    - accumulate_log_context: If True, includes log context in every message
    - Memory operations are executed by the agent (local state update)
    - External actions (read_section, research, finish) are passed to environment
    """

    def __init__(
        self,
        accumulate_log_context: bool = True,
        max_claims_in_context: int = 10,
        system_prompt: Optional[str] = None,
        memory_in_first_message: bool = False,
    ):
        """Initialize the review agent.

        Args:
            accumulate_log_context: Whether to include log context in messages
            max_claims_in_context: Maximum claims to show in log context
            system_prompt: Custom system prompt (defaults to REVIEWER_DIRECT_SYSTEM_PROMPT)
            memory_in_first_message: If True, place memory/log state in the first
                user message (after paper info) instead of in each observation.
                This separates "what you know" from "what just happened".
        """
        self.accumulate_log_context = accumulate_log_context
        self.max_claims_in_context = max_claims_in_context
        self._system_prompt = system_prompt or REVIEWER_DIRECT_SYSTEM_PROMPT
        self.memory_in_first_message = memory_in_first_message
        self.logger = logging.getLogger(self.__class__.__name__)

        # Internal state
        self.log = ReviewLog()
        self._trajectory = Trajectory()
        self._messages: List[Dict[str, str]] = []  # sliding window for LLM inference
        # Track last response for trajectory
        self._last_llm_response: Optional[str] = None
        self._last_action: Optional[Dict] = None
        self._last_memory_results: List[str] = []

    def reset(self, paper_id: str = "") -> None:
        """Reset agent state for a new episode.

        Args:
            paper_id: Identifier for the paper being reviewed
        """
        self.log = ReviewLog()
        self._trajectory = Trajectory()
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._last_llm_response = None
        self._last_action = None
        self._last_memory_results = []
        self._paper_intro = ""  # cached for memory_in_first_message mode

    def update_from_env(
        self,
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Update agent state from environment step result.

        Called after environment.step() returns. Updates:
        1. Messages with observation
        2. Trajectory with complete step
        3. Log state if first observation

        Args:
            observation: Dict with observation data
                - First call: {"title": str, "sections": List[str]}
                - Subsequent: {"action_result": str, ...}
            reward: Reward from environment (0 for intermediate, terminal at finish)
            done: Whether episode is complete
            info: Additional info from environment
        """
        # First observation - initialize messages with system prompt and paper info
        if not self._trajectory.steps and "title" in observation:
            title = observation["title"]
            sections = observation.get("sections", [])

            self._paper_intro = f"The paper you are reviewing is titled '{title}' and it has the following sections: {', '.join(sections)}."
            obs_content = self._paper_intro

            # Show turn budget from the very first turn
            max_turns = info.get("max_turns")
            current_turn = info.get("current_turn")
            if max_turns is not None and current_turn is not None:
                obs_content += f"\n\n[Turn {current_turn}/{max_turns}]"

            self._messages.append({"role": "user", "content": obs_content})

        else:
            # Build observation content for subsequent calls
            if self.memory_in_first_message:
                # In this mode, memory lives in msg[1]. The observation only
                # carries the result of the previous action. Frame the sliding
                # window so the model knows the assistant message above is its
                # own last response.
                obs_content = (
                    "The message above is your previous response. "
                    "Below is the environment's feedback from your last action.\n\n"
                )
            else:
                obs_content = "The observation for this step is:\n"

            # Add memory operation results if any
            if self._last_memory_results:
                obs_content += "<memory_operations_results>\n" + "\n".join(self._last_memory_results) + "\n</memory_operations_results>\n\n"

            # Add action result
            action_result = observation.get("action_result", "")
            if action_result:
                obs_content += f"<action_result>\n{action_result}\n</action_result>\n"

            # Include log context in observation (legacy mode)
            if self.accumulate_log_context and not self.memory_in_first_message:
                log_context = self.log.build_context(detailed=True)
                obs_content += f"\n<current_log_state>\n{log_context}\n</current_log_state>\n"

            # Show turn counter
            max_turns = info.get("max_turns")
            current_turn = info.get("current_turn")
            if max_turns is not None and current_turn is not None:
                obs_content += f"\n[Turn {current_turn}/{max_turns}]\n"

            # Sliding window for LLM inference (keeps context small)
            self._messages = self._messages[:2] + [{"role": "assistant", "content": self._last_llm_response}, {"role": "user", "content": obs_content}]

            # Place memory/log state in the first user message instead of observation
            if self.accumulate_log_context and self.memory_in_first_message:
                first_msg = self._paper_intro
                log_context = self.log.build_context(detailed=True)
                first_msg += f"\n\nYour current review log is:\n<current_log_state>\n{log_context}\n</current_log_state>"
                if max_turns is not None and current_turn is not None:
                    first_msg += f"\n\n[Turn {current_turn}/{max_turns}]"
                self._messages[1] = {"role": "user", "content": first_msg}
        
        # record the times of reading sections
        if "action_name" in info and "read_section" in info["action_name"]:
            section_name = info.get("section_name")
            self.log.record_section_visit(section_name)

        # record search queries
        if "action_name" in info and "search_paper" in info["action_name"]:
            query = info.get("query", "")
            if query:
                self.log.record_search_query(query)

        if self._trajectory.steps:
            # assgin the reward for the executed action of this step.
            current_state = self.get_current_state()
            current_state.reward = reward
            current_state.done = done
            current_state.action = self._last_action

            # Store log snapshot and env info
            # All credit assignment reads from log_snapshot (single source of truth)
            current_state.info = {
                **info,
                "log_snapshot": self.log.model_dump()
            }

        # Create a new step to trajectory
        step = Step(
            observation=obs_content
        )

        self._trajectory.steps.append(step)

        # Clear last action tracking
        self._last_memory_results = []

    def update_from_model(self, response: str) -> Action:
        """Parse and process LLM response.

        Called after LLM generates a response. This method:
        1. Parses the JSON response
        2. Executes memory operation s (local state update)
        3. Prepares action for environment

        Args:
            response: Raw LLM response text (expected JSON)

        Returns:
            Action wrapping the action dict for the environment

        Raises:
            ValueError: If response cannot be parsed
        """
        assert self.trajectory.steps, "Trajectory should not be empty when update_from_model is called."

        self._last_llm_response = response

        # Update the current step in the trajectory
        cur_step = self.get_current_state()
        cur_step.model_response = response
        # Snapshot the sliding window messages + this assistant response.
        # This matches exactly what the LLM saw during inference, so tokenize_and_mask
        # will produce the correct prompt/response split for training.
        cur_step.chat_completions = copy.deepcopy(self._messages) + [{"role": "assistant", "content": response}]

        # Parse decision from response
        decision = self._parse_decision(response)

        # if the decision is a string, it means parsing failed and we return a format_error action
        if isinstance(decision, str):
            self.logger.error(f"Failed to parse LLM response: {decision}")
            return Action(action={"name": "format_error", "args": {"message": decision}, "_meta": {"mem_success": 0, "mem_error": 0, "mem_duplicates": 0, "mem_hallucinations": 0}})

        # Execute memory operations locally
        self._last_memory_results = []
        validation_error_count = 0
        for mem_op in decision.get("memory_operations", []):
            result, is_validation_error = self._execute_memory_operation(mem_op)
            self._last_memory_results.append(result)
            if is_validation_error:
                validation_error_count += 1

        # Count memory operation successes vs errors
        mem_success = sum(1 for r in self._last_memory_results if r.startswith("Successfully"))
        mem_duplicates_skipped = sum(1 for r in self._last_memory_results if r.startswith("Skipped:"))
        mem_duplicates = sum(1 for r in self._last_memory_results if "too similar to" in r)
        mem_hallucinations = sum(1 for r in self._last_memory_results if "Hallucinated evidence references" in r)
        # mem_error: includes duplicates that raised errors (not silently skipped ones)
        # Silently skipped duplicates are neutral — no penalty, no reward.
        mem_error = len(self._last_memory_results) - mem_success - mem_duplicates_skipped - validation_error_count - mem_hallucinations
        meta = {
            "mem_success": mem_success,
            "mem_error": mem_error,
            "mem_duplicates": mem_duplicates,
            "mem_hallucinations": mem_hallucinations,
            "validation_errors": validation_error_count,
            "sections_read": set(self.log.section_visits.keys()),
        }

        # Store action for trajectory
        action = decision["action"]

        # Enrich action with data from memory so environment doesn't need agent_log
        action_name = action.get("name")

        # If it's a research action, enrich it with claim/question object
        if action_name == "research":
            args = action.get("args", {})

            # Determine target type and ID
            claim_id = args.get("claim_id")
            question_id = args.get("question_id")

            if claim_id:
                # Retrieve claim from memory
                claim = self.log.get_claim(claim_id)
                if not claim:
                    # Return tool_error action if claim not found
                    return Action(action={
                        "name": "tool_error",
                        "args": {"original_action": "research", "message": f"Claim {claim_id} not found in memory"},
                        "_meta": meta,
                    })

                # Serialize claim to dict instead of passing object
                args["claim_data"] = {
                    "id": claim.id,
                    "text": claim.text,
                    "section": claim.section,
                    "type": claim.type,
                    "status": claim.status,
                    "issues": claim.issues,
                    "cross_references": claim.cross_references,
                    "verifier_reason": claim.verifier_reason
                }
                action["args"] = args

            elif question_id:
                # Retrieve question from memory
                question = self.log.get_question(question_id)
                if not question:
                    # Return tool_error action if question not found
                    return Action(action={
                        "name": "tool_error",
                        "args": {"original_action": "research", "message": f"Question {question_id} not found in memory"},
                        "_meta": meta,
                    })

                # Serialize question to dict instead of passing object
                args["question_data"] = {
                    "id": question.id,
                    "question": question.question,
                    "source_section": question.source_section,
                    "status": question.status,
                    "type": question.type,
                    "answer": question.answer,
                    "answer_sections": question.answer_sections,
                    "related_claims": question.related_claims
                }
                action["args"] = args

            else:
                # Neither claim_id nor question_id provided
                return Action(action={
                    "name": "tool_error",
                    "args": {"original_action": "research", "message": "Research action must specify claim_id or question_id"},
                    "_meta": meta,
                })

        # If it's a finish action, enrich it with review data
        elif action_name == "finish":
            args = action.get("args", {})

            # Extract review outline from memory
            outline = self.log.review_outline
            # Convert OutlineItems to text for compatibility
            args["review_data"] = {
                "summary": outline.summary,
                "strengths": outline.get_strengths_text(),
                "weaknesses": outline.get_weaknesses_text(),
                "questions": outline.get_questions_text(),
                "overall_score": outline.overall_score
            }
            action["args"] = args

        # Attach meta info for syntactic reward tracking
        action["_meta"] = meta

        self._last_action = action

        return Action(action=action)

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Get messages for LLM chat completion.

        manage the current input messages for the LLM, which includes:
        1. System prompt (static)
        2. Initial observation with paper info (first step)
        3. updated_memory
        4. last action result
        """
        return self._messages

    @property
    def trajectory(self) -> Trajectory:
        """Get the current trajectory."""
        return self._trajectory

    def get_log(self) -> ReviewLog:
        """Get the current review log state."""
        return self.log

    def _parse_decision(self, response_text: str) -> Dict:
        """Parse LLM response JSON into decision dict.

        Args:
            response_text: Raw text from LLM

        Returns:
            Parsed decision dict with 'memory_operations' and 'action'

        Raises:
            ValueError: If parsing fails
        """
        before, sep, after = response_text.strip().partition("</think>")
        text = after if sep else before

        # Handle markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Fix invalid escape sequences (LaTeX notation like \hat, \alpha)
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        try:
            decision = json.loads(text)
            # Validate and normalize structure
            if "memory_operations" not in decision:
                decision["memory_operations"] = []

            if "action" not in decision:
                raise ValueError("Missing 'action' in response")

            if not isinstance(decision["memory_operations"], list):
                raise ValueError("'memory_operations' must be a list")

            # Validate each memory operation is a dict with 'op' field
            for i, mem_op in enumerate(decision["memory_operations"]):
                if not isinstance(mem_op, dict):
                    raise ValueError(f"memory_operations[{i}] must be a dict, got {type(mem_op).__name__}")
                if "op" not in mem_op:
                    raise ValueError(f"memory_operations[{i}] missing required 'op' field")

            if not isinstance(decision["action"], dict) or "name" not in decision["action"]:
                raise ValueError("'action' must be a dict with 'name' field")

            return decision

        except json.JSONDecodeError as e:
            return "The response could not be parsed as valid JSON. Please ensure your response follows the specified format. Error details: " + str(e)
        except ValueError as e:
            return f"JSON was parsed but has invalid structure: {e}. Please ensure your response includes a valid 'action' dict with a 'name' field."

    def _execute_memory_operation(self, mem_op: Dict) -> tuple[str, bool]:
        """Execute a memory operation on the review log.

        Handles 3 ops:
        - log: dispatches to add_claim/add_question/add_note based on args.type (claims use args.claim_type, questions use args.question_type)
        - update: dispatches to update_claim_status/resolve_question based on entry_id prefix
        - outline: maps to update_outline

        Args:
            mem_op: Memory operation dict with 'op' and 'args'

        Returns:
            Tuple of (result message string, is_validation_error boolean)
        """
        op_name = mem_op.get("op")
        args = mem_op.get("args", {})

        if op_name == "log":
            return self._handle_log(args)
        elif op_name == "update":
            return self._handle_update(args)
        elif op_name == "outline":
            return self._handle_outline(args)
        else:
            return (f"Error: Unknown memory operation '{op_name}'.", False)

    def _handle_log(self, args: Dict) -> tuple[str, bool]:
        """Handle 'log' operation -- dispatch based on args['type'].

        Returns:
            Tuple of (result message string, is_validation_error boolean)
        """
        entry_type = args.get("type")
        text = args.get("text")
        section = args.get("section")

        if not entry_type:
            return ("Error: 'log' operation requires 'type' field (claim|question|note).", False)

        # Gate: reject log operations that reference sections not yet visited.
        # Prevents reward hacking where the model pre-loads claims at step 0
        # before reading the paper to game evidence-based credit assignment.
        # Natural behavior: read section X first, then log about it in the next step.
        # Only empty section is exempt — to avoid misclassifying a missing 'section'
        # field (format error) as a hallucination; the type-specific checks below
        # will catch and report that as a proper format error.
        #
        # Exception: the SFT-trained model always emits a planning note at step 0
        # ("Starting Phase 1 - Orientation. Reading...") with section set to the
        # section it reads in the same action.  This is benign boilerplate, not
        # fabricated evidence, so skip the hallucination check for it.
        is_step0_planning_note = (
            not self.log.section_visits
            and entry_type == "note"
            and text
            and text.startswith("Starting Phase 1")
        )
        section_str = (section or "").strip().lower()
        if section_str and not is_step0_planning_note:
            visited_lower = {s.lower() for s in self.log.section_visits.keys()}
            if section_str not in visited_lower:
                return (
                    f"Hallucinated evidence references: section '{section}' has not been read yet. "
                    f"You must read the section first before logging claims, questions, or notes about it.",
                    False,
                )

        if entry_type == "claim":
            claim_type = args.get("claim_type")
            if not all([text, section, claim_type]):
                return ("Error: claim log requires 'text', 'section', and 'claim_type'.", False)
            issues = args.get("issues", None)
            try:
                claim_id = self.log.add_claim(text, section, claim_type, issues, step=len(self._trajectory.steps) - 1)
                if claim_id is None:
                    # Silently skipped duplicate — no penalty, but inform model
                    return (f"Skipped: this claim is too similar to an existing one. Try a different point.", False)
                self.logger.debug(f"Added claim {claim_id}: {text[:50]}...")
                return (f"Successfully added claim {claim_id} to log.", False)
            except ValueError as e:
                # Duplicate claim — route to mem_error (tool penalty), not validation_errors (format penalty)
                return (f"Error: {str(e)}", False)
            except ValidationError as e:
                error_msg = f"Validation error when adding claim: {str(e)}"
                self.logger.error(error_msg)
                return (f"Error: {error_msg}", True)

        elif entry_type == "question":
            if not all([text, section]):
                return ("Error: question log requires 'text' and 'section'.", False)
            question_type = args.get("question_type", "clarification")
            related_claims = args.get("related_claims", None)
            try:
                question_id = self.log.add_question(text, section, question_type, related_claims, step=len(self._trajectory.steps) - 1)
                if question_id is None:
                    return (f"Skipped: this question is too similar to an existing one. Try a different point.", False)
                self.logger.debug(f"Added question {question_id}: {text[:50]}...")
                return (f"Successfully added question {question_id} to log.", False)
            except ValueError as e:
                # Duplicate question — route to mem_error (tool penalty), not validation_errors (format penalty)
                return (f"Error: {str(e)}", False)
            except ValidationError as e:
                error_msg = f"Validation error when adding question: {str(e)}"
                self.logger.error(error_msg)
                return (f"Error: {error_msg}", True)

        elif entry_type == "note":
            if not all([text, section]):
                return ("Error: note log requires 'text' and 'section'.", False)
            tag = args.get("tag", [])
            # Ensure tag is always a list (handle cases where LLM returns a string)
            if isinstance(tag, str):
                tag = [tag]
            try:
                note_id = self.log.add_note(text, section, tag, step=len(self._trajectory.steps) - 1)
                if note_id is None:
                    return (f"Skipped: this note is too similar to an existing one. Try a different point.", False)
                self.logger.debug(f"Added note {note_id}: {text[:50]}...")
                return (f"Successfully added note {note_id} to log.", False)
            except ValueError as e:
                # Duplicate note — route to mem_error (tool penalty), not validation_errors (format penalty)
                return (f"Error: {str(e)}", False)
            except ValidationError as e:
                error_msg = f"Validation error when adding note: {str(e)}"
                self.logger.error(error_msg)
                return (f"Error: {error_msg}", True)

        else:
            return (f"Error: Unknown log type '{entry_type}'. Must be claim, question, or note.", False)

    def _handle_update(self, args: Dict) -> tuple[str, bool]:
        """Handle 'update' operation -- dispatch based on entry_id prefix.

        Returns:
            Tuple of (result message string, is_validation_error boolean)
        """
        entry_id = args.get("entry_id")
        if not entry_id:
            return ("Error: 'update' operation requires 'entry_id'.", False)

        if not isinstance(entry_id, str):
            return (f"Error: 'entry_id' must be a string, got {type(entry_id).__name__}: {entry_id}", True)

        if entry_id.startswith("C"):
            # Claim update
            status = args.get("status")
            reasoning = args.get("reasoning")
            if not all([status, reasoning]):
                return ("Error: claim update requires 'status' and 'reasoning'.", False)
            valid_statuses = ["to_be_verified", "supported", "weak", "invalid"]
            if status not in valid_statuses:
                return (f"Error: Invalid claim status '{status}'. Must be one of: {', '.join(valid_statuses)}", False)
            cross_references = args.get("cross_references", [])
            try:
                success = self.log.update_claim_status(entry_id, status, reasoning, cross_references, step=len(self._trajectory.steps) - 1)
                if success:
                    self.logger.debug(f"Updated claim {entry_id} status to {status}")
                    return (f"Successfully updated claim {entry_id} status to '{status}'.", False)
                else:
                    return (f"Error: Claim {entry_id} not found in log.", False)
            except ValidationError as e:
                error_msg = f"Validation error when updating claim {entry_id}: {str(e)}"
                self.logger.error(error_msg)
                return (f"Error: {error_msg}", True)

        elif entry_id.startswith("Q"):
            # Question update
            status = args.get("status", "resolved")
            answer = args.get("answer")
            if not answer:
                return ("Error: question update requires 'answer'.", False)
            valid_statuses = ["resolved", "partially_answered"]
            if status not in valid_statuses:
                return (f"Error: Invalid question status '{status}'. Must be one of: {', '.join(valid_statuses)}", False)
            answer_sections = args.get("answer_sections", [])
            try:
                success = self.log.resolve_question(entry_id, answer, answer_sections, status)
                if success:
                    self.logger.debug(f"Updated question {entry_id}")
                    return (f"Successfully updated question {entry_id}.", False)
                else:
                    return (f"Error: Question {entry_id} not found in log.", False)
            except ValidationError as e:
                error_msg = f"Validation error when updating question {entry_id}: {str(e)}"
                self.logger.error(error_msg)
                return (f"Error: {error_msg}", True)

        else:
            return (f"Error: Cannot update entry '{entry_id}'. Only claims (C*) and questions (Q*) can be updated.", False)

    def _handle_outline(self, args: Dict) -> tuple[str, bool]:
        """Handle 'outline' operation -- maps to update_outline.

        Returns:
            Tuple of (result message string, is_validation_error boolean)
        """
        section = args.get("section")
        content = args.get("content")
        tags = args.get("tags", [])

        if not all([section, content is not None]):
            return ("Error: 'outline' requires 'section' and 'content'.", False)

        # overall_score naturally comes as int/float — coerce to str
        if section == "overall_score" and isinstance(content, (int, float)):
            content = str(int(content))
            args["content"] = content

        if not isinstance(content, str):
            return ("Error: 'content' must be a string, not a list or object.", True)

        # Validate section name
        valid_sections = ["summary", "strengths", "weaknesses", "questions", "overall_score"]
        if section not in valid_sections:
            return (f"Error: Invalid outline section '{section}'. Must be one of: {', '.join(valid_sections)}", False)

        # Sections that require at least one evidence tag
        if section in ("strengths", "weaknesses", "questions") and not tags:
            return (
                "Error: 'tags' is required for outline strengths/weaknesses/questions. "
                "Provide at least one claim (C*), question (Q*), or note (N*) ID.",
                True,
            )

        # Parse tags into claim/question/note IDs
        related_claims = [t for t in tags if t.startswith('C')]
        related_questions = [t for t in tags if t.startswith('Q')]
        related_notes = [t for t in tags if t.startswith('N')]

        try:
            result = self.log.update_outline(
                section=section,
                content=content,
                append=True,
                related_claims=related_claims,
                related_questions=related_questions,
                related_notes=related_notes,
                step=len(self._trajectory.steps) - 1
            )
            if result == "duplicate_skipped":
                return (f"Skipped: this {section} point is too similar to an existing one. Try a different point.", False)
            self.logger.debug(f"Updated outline {section} with evidence tags: {tags}")
            return (f"Successfully updated outline {section}.", False)
        except ValueError as e:
            error_msg = str(e)
            if "Hallucinated evidence references" in error_msg:
                # Hallucination: model referenced IDs that don't exist in the log.
                # Counted in mem_hallucinations → hard -1.0 syntactic penalty.
                # is_validation_error=False so it doesn't also trigger format penalty.
                self.logger.warning(f"Hallucinated tags in outline {section}: {error_msg}")
                return (f"Error: {error_msg}", False)
            if "too similar to existing" in error_msg:
                # Duplicate outline entry: well-formed op, just redundant content.
                # Counted in mem_duplicates (no penalty — info_gain handles novelty).
                # is_validation_error=False so it doesn't trigger format penalty.
                self.logger.warning(f"Duplicate outline entry in {section}: {error_msg}")
                return (f"Error: {error_msg}", False)
            # Missing-evidence or other format errors → validation penalty (R_format)
            self.logger.error(f"Validation error in outline {section}: {error_msg}")
            return (f"Error: {error_msg}", True)

    def get_review_from_log(self) -> Dict:
        """Extract the final review from the log.

        Returns:
            Dict with summary, strengths, weaknesses, questions, overall_score
            (with strengths/weaknesses/questions as List[str] for compatibility)
        """
        outline = self.log.review_outline
        # Convert OutlineItems to text for compatibility with reward calculation
        return {
            "summary": outline.summary,
            "strengths": outline.get_strengths_text(),
            "weaknesses": outline.get_weaknesses_text(),
            "questions": outline.get_questions_text(),
            "overall_score": outline.overall_score,
        }
