"""ProReviewer - Simplified review agent with evidence-based review log for SFT+RL training."""

from typing import Optional, Dict, List, Union, Tuple
import json
from pydantic import ValidationError
from .base_agent import BaseReviewAgent
from .environment import PaperEnvironment
from .reviewer_memory import ReviewLog
from litellm.types.utils import Message
from utils.helpers.llm import call_llm
from utils.helpers.token_tracker import token_tracker


class ProReviewer(BaseReviewAgent):
    """ProReviewer agent optimized for SFT+RL training.

    This agent supports two modes controlled by `use_research_subagent`:

    **Mode 1: With Research Subagent (use_research_subagent=True, default)**
    - Hierarchical architecture: Main agent orchestrates, research subagent investigates
    - Actions: read_section, research, finish
    - Research subagent has full autonomy with its own agent loop
    - Separation of concerns: orchestration vs. deep research

    **Mode 2: Direct Investigation (use_research_subagent=False)**
    - Single-agent architecture: Agent performs all investigations directly
    - Actions: read_section, search_paper, finish
    - Better RL credit assignment (every action visible in trajectory)
    - Lower token cost (no subagent LLM calls)

    The agent maintains an evidence-based review log with:
    - Claims: Extracted statements (verified by research subagent or directly)
    - Questions: Unclear points and suspicions (answered by research subagent or directly)
    - Notes: Reviewer's thoughts triggered during reading
    - Review Outline: Final verdict (summary, strengths, weaknesses, questions, overall_score)

    Key characteristics:
    - Configurable architecture via use_research_subagent parameter
    - Minimal state management for main agent
    - Optimized for reinforcement learning with clear credit assignment
    """

    def __init__(
        self,
        model: str,
        research_model: Optional[str] = None,
        conference_format: str = "ICLR",
        use_research_subagent: bool = True
    ):
        """Initialize the ProReviewer agent.

        Args:
            model: Model identifier for the main policy
            research_model: Optional model for the research subagent (defaults to same as model)
            conference_format: Conference format for the review
            use_research_subagent: If True, uses hierarchical architecture with ResearchSubagent.
                                  If False, uses single-agent architecture with direct investigation.
        """
        # Initialize base class (writer_subagent created but not used)
        super().__init__(model, conference_format)

        self.use_research_subagent = use_research_subagent

        # Choose system prompt and initialize subagent based on mode
        if use_research_subagent:
            from .reviewer_prompts import REVIEWER_SYSTEM_PROMPT
            from .research_agent import ResearchSubagent
            self._system_prompt = REVIEWER_SYSTEM_PROMPT
            self.research_model = research_model or model
            self.research_subagent = ResearchSubagent(self.research_model)
        else:
            from .reviewer_prompts_direct import REVIEWER_DIRECT_SYSTEM_PROMPT
            self._system_prompt = REVIEWER_DIRECT_SYSTEM_PROMPT
            self.research_model = None
            self.research_subagent = None

        self.log = ReviewLog()

    def get_system_prompt(self) -> str:
        """Return the system prompt for reviewer agent."""
        return self._system_prompt

    def get_tools(self) -> List[dict]:
        """Return empty list - ProReviewer uses JSON output format, not function calling."""
        return []

    def _decide_next_action(self, trajectory: List[dict]) -> Tuple[Dict, Dict]:
        """Override base method to use JSON output format instead of function calling.

        Args:
            trajectory: Current conversation trajectory

        Returns:
            Tuple of (decision_dict, response_message_dict) where decision_dict contains:
            {
                "memory_operations": [...],
                "action": {"name": "...", "args": {...}}
            }
        """
        # Ensure all messages are dicts
        trajectory_dicts = [self._message_to_dict(msg) for msg in trajectory]

        # Call LLM WITHOUT tools parameter (so it returns text/JSON)
        llm_response = call_llm(
            model=self.model,
            messages=trajectory_dicts,
            temperature=0.7,
            response_format={"type": "json_object"}  # Request JSON output
        )
        response_message = llm_response.choices[0].message
        response_content = response_message.content

        self.logger.info(f"LLM response: {response_content[:200]}...")

        # Parse JSON
        try:
            decision = json.loads(response_content)

            # Validate structure
            if "memory_operations" not in decision or "action" not in decision:
                raise ValueError("Missing 'memory_operations' or 'action' in response")

            if not isinstance(decision["memory_operations"], list):
                raise ValueError("'memory_operations' must be a list")

            if not isinstance(decision["action"], dict) or "name" not in decision["action"]:
                raise ValueError("'action' must be a dict with 'name' field")

            self.logger.info(f"Parsed decision: {len(decision['memory_operations'])} memory ops, action={decision['action']['name']}")

            # Convert response_message to dict
            response_dict = {
                "role": "assistant",
                "content": response_content
            }

            return decision, response_dict

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response content: {response_content}")
            raise ValueError(f"LLM returned invalid JSON: {e}")
        except ValueError as e:
            self.logger.error(f"Invalid decision structure: {e}")
            raise

    def review_paper(self, environment: PaperEnvironment, max_iterations: int = 50) -> List[dict]:
        """Review paper using evidence-based methodology.

        Maintains minimal state:
        - Initial system and user prompts
        - Last assistant message
        - Last observations (tool responses)
        - Current log context

        Args:
            environment: The paper environment containing the paper to review
            max_iterations: Maximum number of iterations

        Returns:
            Full trajectory with all messages for analysis
        """
        self.logger.info("Starting evidence-based paper review...")

        # Build initial trajectory
        initial_trajectory, title, sections_list = self._build_initial_trajectory(environment)

        # Full trajectory for analysis
        full_trajectory = initial_trajectory.copy()

        # Reset state
        self.log = ReviewLog()
        last_assistant_message = None
        last_observations = []

        # Agent loop with token tracking
        for iteration in range(max_iterations):
            self.logger.info(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            self.log.current_iteration = iteration + 1

            # Build minimal trajectory for decision
            minimal_trajectory = self._build_minimal_trajectory(
                initial_trajectory, last_assistant_message, last_observations
            )

            # Decide next action (returns decision dict + response message)
            # Wrap with agent context for token tracking
            with token_tracker.agent_context("main_agent"):
                decision, response_message = self._decide_next_action(minimal_trajectory)

            # Add assistant message to trajectory
            last_assistant_message = response_message
            full_trajectory.append(last_assistant_message)

            # Execute memory operations
            memory_results = []
            for mem_op in decision["memory_operations"]:
                result = self._execute_memory_operation(mem_op)
                memory_results.append(result)

            # Execute external action
            action = decision["action"]
            action_name = action["name"]
            action_args = action.get("args", {})

            if action_name == "finish":
                # Terminal action - review complete
                self.logger.info("Agent finished review.")
                last_observation = {
                    "role": "user",
                    "content": "<observation>Review complete.</observation>"
                }
                full_trajectory.append(last_observation)
                return full_trajectory

            elif action_name == "read_section":
                action_response = self._execute_read_section_action(environment, action_args)

            elif action_name == "research":
                if self.use_research_subagent:
                    action_response = self._execute_research_action(environment, action_args)
                else:
                    action_response = "Error: 'research' action not available in direct mode (use 'search_paper' instead)"
                    self.logger.error(action_response)

            elif action_name == "search_paper":
                if not self.use_research_subagent:
                    action_response = self._execute_search_paper_action(environment, action_args)
                else:
                    action_response = "Error: 'search_paper' action not available with research subagent mode (use 'research' instead)"
                    self.logger.error(action_response)

            else:
                action_response = f"Error: Unknown action '{action_name}'"
                self.logger.error(action_response)

            # Create observation with memory operation results
            observation_content = "<observation>\n"
            if memory_results:
                observation_content += "<log_update>\n" + "\n".join(memory_results) + "\n</log_update>\n\n"
            observation_content += f"<action_result>\n{action_response}\n</action_result>\n"
            observation_content += "</observation>"

            last_observation = {
                "role": "user",
                "content": observation_content
            }
            full_trajectory.append(last_observation)
            last_observations = [last_observation]

        # Max iterations reached - force review synthesis
        self.logger.warning("Max iterations reached. Synthesizing review from current log.")
        synthesized_review = self._synthesize_review(environment)

        # Update the log's review outline with synthesized content
        if synthesized_review:
            self.log.review_outline.summary = synthesized_review.get('summary', '')
            self.log.review_outline.strengths = synthesized_review.get('strengths', [])
            self.log.review_outline.weaknesses = synthesized_review.get('weaknesses', [])
            self.log.review_outline.questions = synthesized_review.get('questions', [])
            self.log.review_outline.overall_score = synthesized_review.get('overall_score')

        full_trajectory.append({
            "role": "assistant",
            "content": f"[Max iterations reached - synthesized review]\n{json.dumps(synthesized_review, indent=2)}"
        })
        return full_trajectory

    def _build_minimal_trajectory(
        self,
        initial_trajectory: List[dict],
        last_assistant_message: Optional[Union[Dict, Message]],
        last_observations: List[Dict]
    ) -> List[dict]:
        """Build minimal trajectory with current state.

        Args:
            initial_trajectory: Initial system and user prompts
            last_assistant_message: Last LLM response (dict or Message)
            last_observations: List of tool responses for the last assistant message
            updated_memory: List of memory operation results

        Returns:
            Minimal trajectory for LLM decision
        """
        trajectory = initial_trajectory.copy()

        # Add last action (assistant message) if exists
        if last_assistant_message:
            assistant_dict = self._message_to_dict(last_assistant_message)

            # Ensure all tool calls have corresponding responses
            if assistant_dict.get('tool_calls'):
                tool_call_ids = {tc.get('id') for tc in assistant_dict.get('tool_calls', [])}
                observation_ids = {obs.get('tool_call_id') for obs in last_observations if obs.get('tool_call_id')}

                missing_ids = tool_call_ids - observation_ids
                if missing_ids:
                    raise ValueError(f"Missing tool responses for tool calls: {missing_ids}")

            trajectory.append(assistant_dict)

        # Add all tool observations
        for observation in last_observations:
            trajectory.append(observation)

        # Add current log context
        log_context = self.log.build_context(detailed=False, max_claims=10)
        trajectory.append({
            "role": "user",
            "content": f"<log_context>\n{log_context}\n</log_context>"
        })

        return trajectory

    def _execute_memory_operation(self, mem_op: Dict) -> str:
        """Execute a memory operation.

        Handles 3 ops:
        - log: dispatches to add_claim/add_question/add_note based on args.type (claims use args.claim_type, questions use args.question_type)
        - update: dispatches to update_claim_status/resolve_question based on entry_id prefix
        - draft: maps to update_outline

        Args:
            mem_op: Memory operation dict with 'op' and 'args'

        Returns:
            Result message string
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
            return f"Error: Unknown memory operation '{op_name}'."

    def _execute_read_section_action(self, environment: PaperEnvironment, args: Dict) -> str:
        """Execute read_section action."""
        section_name = args['section_name']
        self.logger.info(f"Agent decided to read section: {section_name}")

        content = environment.read_section(section_name)

        # Track section visit
        self.log.record_section_visit(section_name)

        return f"Successfully read section '{section_name}'. Content:\n{content}"

    def _execute_search_paper_action(self, environment: PaperEnvironment, args: Dict) -> str:
        """Execute search_paper action — search the paper for a query string.

        Only available in direct mode (use_research_subagent=False).
        """
        query = args.get('query', '')
        if not query:
            return "Error: Missing query parameter"

        self.logger.info(f"Agent searching paper for: {query}")
        self.log.record_search_query(query)
        results = environment.search_paper(query)

        if not results:
            return f"No matches found for '{query}'"

        # Format results
        output = f"Search results for '{query}':\n\n"
        for result in results:
            output += f"**[{result['section']}]** ({result['match_count']} matches)\n"
            for snippet in result['snippets']:
                output += f"  - {snippet}\n"
            output += "\n"

        return output

    def _handle_log(self, args: Dict) -> str:
        """Handle 'log' operation -- dispatch based on args['type']."""
        entry_type = args.get("type")
        text = args.get("text")
        section = args.get("section")

        if not entry_type:
            return "Error: 'log' operation requires 'type' field (claim|question|note)."

        if entry_type == "claim":
            claim_type = args.get("claim_type")
            if not all([text, section, claim_type]):
                return "Error: claim log requires 'text', 'section', and 'claim_type'."
            issues = args.get("issues", None)
            try:
                claim_id = self.log.add_claim(text, section, claim_type, issues)
                self.logger.info(f"Added claim {claim_id}: {text[:50]}...")
                return f"Successfully added claim {claim_id} to log."
            except ValidationError as e:
                error_msg = f"Validation error when adding claim: {str(e)}"
                self.logger.error(error_msg)
                return f"Error: {error_msg}"

        elif entry_type == "question":
            if not all([text, section]):
                return "Error: question log requires 'text' and 'section'."
            question_type = args.get("question_type", "clarification")
            related_claims = args.get("related_claims", None)
            try:
                question_id = self.log.add_question(text, section, question_type, related_claims)
                self.logger.info(f"Added question {question_id}: {text[:50]}...")
                return f"Successfully added question {question_id} to log."
            except ValidationError as e:
                error_msg = f"Validation error when adding question: {str(e)}"
                self.logger.error(error_msg)
                return f"Error: {error_msg}"

        elif entry_type == "note":
            if not all([text, section]):
                return "Error: note log requires 'text' and 'section'."
            tag = args.get("tag", [])
            try:
                note_id = self.log.add_note(text, section, tag)
                self.logger.info(f"Added note {note_id}: {text[:50]}...")
                return f"Successfully added note {note_id} to log."
            except ValidationError as e:
                error_msg = f"Validation error when adding note: {str(e)}"
                self.logger.error(error_msg)
                return f"Error: {error_msg}"

        else:
            return f"Error: Unknown log type '{entry_type}'. Must be claim, question, or note."

    def _handle_update(self, args: Dict) -> str:
        """Handle 'update' operation -- dispatch based on entry_id prefix."""
        entry_id = args.get("entry_id")
        if not entry_id:
            return "Error: 'update' operation requires 'entry_id'."

        if entry_id.startswith("C"):
            # Claim update
            status = args.get("status")
            reasoning = args.get("reasoning")
            if not all([status, reasoning]):
                return "Error: claim update requires 'status' and 'reasoning'."
            valid_statuses = ["to_be_verified", "supported", "weak", "invalid"]
            if status not in valid_statuses:
                return f"Error: Invalid claim status '{status}'. Must be one of: {', '.join(valid_statuses)}"
            cross_references = args.get("cross_references", [])
            try:
                success = self.log.update_claim_status(entry_id, status, reasoning, cross_references)
                if success:
                    self.logger.info(f"Updated claim {entry_id} status to {status}")
                    return f"Successfully updated claim {entry_id} status to '{status}'."
                else:
                    return f"Error: Claim {entry_id} not found in log."
            except ValidationError as e:
                error_msg = f"Validation error when updating claim {entry_id}: {str(e)}"
                self.logger.error(error_msg)
                return f"Error: {error_msg}"

        elif entry_id.startswith("Q"):
            # Question update
            status = args.get("status", "resolved")
            answer = args.get("answer")
            if not answer:
                return "Error: question update requires 'answer'."
            valid_statuses = ["resolved", "partially_answered"]
            if status not in valid_statuses:
                return f"Error: Invalid question status '{status}'. Must be one of: {', '.join(valid_statuses)}"
            answer_sections = args.get("answer_sections", [])
            try:
                success = self.log.resolve_question(entry_id, answer, answer_sections, status)
                if success:
                    self.logger.info(f"Updated question {entry_id}")
                    return f"Successfully updated question {entry_id}."
                else:
                    return f"Error: Question {entry_id} not found in log."
            except ValidationError as e:
                error_msg = f"Validation error when updating question {entry_id}: {str(e)}"
                self.logger.error(error_msg)
                return f"Error: {error_msg}"

        else:
            return f"Error: Cannot update entry '{entry_id}'. Only claims (C*) and questions (Q*) can be updated."

    def _handle_outline(self, args: Dict) -> str:
        """Handle 'outline' operation -- maps to update_outline."""
        section = args.get("section")
        content = args.get("content")
        tags = args.get("tags", [])

        if not all([section, content is not None]):
            return "Error: 'outline' requires 'section' and 'content'."

        # Validate section name
        valid_sections = ["summary", "strengths", "weaknesses", "questions", "overall_score"]
        if section not in valid_sections:
            return f"Error: Invalid outline section '{section}'. Must be one of: {', '.join(valid_sections)}"

        # Parse tags into claim/question/note IDs
        related_claims = [t for t in tags if t.startswith('C')]
        related_questions = [t for t in tags if t.startswith('Q')]
        related_notes = [t for t in tags if t.startswith('N')]

        try:
            self.log.update_outline(
                section=section,
                content=content,
                append=True,
                related_claims=related_claims,
                related_questions=related_questions,
                related_notes=related_notes
            )
            self.logger.info(f"Updated outline {section} with evidence tags: {tags}")
            return f"Successfully updated outline {section}."
        except ValueError as e:
            return f"Error: {str(e)}"

    def _execute_research_action(self, environment: PaperEnvironment, args: Dict) -> str:
        """Execute research action by delegating to research subagent.

        The research subagent will autonomously investigate to verify claims or answer questions.
        """
        target_id = args.get('claim_id') or args.get('question_id')
        target_type = 'claim' if 'claim_id' in args else 'question'
        additional_context = args.get('additional_context', None)

        if not target_id:
            return "Error: Must provide either claim_id or question_id"

        self.logger.info(f"Main agent delegating research for {target_type}: {target_id}")

        # Get the target (claim or question)
        if target_type == 'claim':
            target = self.log.get_claim(target_id)
            if not target:
                return f"Error: Claim {target_id} not found in log."
        else:
            target = self.log.get_question(target_id)
            if not target:
                return f"Error: Question {target_id} not found in log."

        # Delegate to research subagent with token tracking
        try:
            with token_tracker.agent_context("research_subagent"):
                findings = self.research_subagent.research(
                    environment=environment,
                    target_type=target_type,
                    target=target,
                    max_iterations=20
                )

            # Return findings to main agent for judgment
            # The main agent will decide how to update log based on these findings
            summary = findings['summary']
            cross_refs = findings['cross_references']
            evidence = findings.get('evidence', [])

            if target_type == 'claim':
                response = f"Research complete for {target_id}.\n\n"
                response += f"**Claim being verified**: {target.text}\n\n"
                response += f"**Research findings**:\n"
                response += f"- Summary: {summary}\n"
                response += f"- Sections examined: {', '.join(cross_refs) if cross_refs else 'none'}\n\n"
                # response += f"- Detailed reasoning: {reasoning}\n\n"
                if evidence:
                    response += f"**Evidence collected**:\n"
                    for e in evidence[:3]:  # Show top 3 pieces of evidence
                        response += f"  - [{e.get('section', 'N/A')}] {e.get('finding', 'N/A')}\n"
                response += f"\n**Your judgment needed**: Review these findings and decide how to update the claim's status using the update memory operation."

            else:  # question
                response = f"Research complete for {target_id}.\n\n"
                response += f"**Question being investigated**: {target.question}\n\n"
                response += f"**Research findings**:\n"
                response += f"- Summary: {summary}\n"
                response += f"- Sections examined: {', '.join(cross_refs) if cross_refs else 'none'}\n\n"
                # response += f"- Answer/Findings: {reasoning}\n\n"
                if evidence:
                    response += f"**Evidence collected**:\n"
                    for e in evidence[:3]:  # Show top 3 pieces of evidence
                        response += f"  - [{e.get('section', 'N/A')}] {e.get('finding', 'N/A')}\n"
                response += f"\n**Your judgment needed**: Review these findings and decide how to resolve the question using the update memory operation."

            self.logger.info(f"Research subagent completed with summary: {summary}. Main agent will judge findings.")
            return response

        except Exception as e:
            self.logger.error(f"Research failed: {e}")
            return f"Error during research: {str(e)}"

    def _synthesize_review(self, environment: PaperEnvironment) -> Dict:
        """Synthesize a review when max iterations reached.

        Uses the current ReviewLog state to generate a final review.

        Args:
            environment: The paper environment (for context if needed)

        Returns:
            Dictionary with review fields (summary, strengths, weaknesses, questions, overall_score)
        """
        # Build prompt with full log context
        log_context = self.log.build_context(detailed=True)

        synthesis_prompt = f"""You have reached the maximum iterations for reviewing this paper.
Based on all the evidence collected so far, you MUST now produce a final review.

{log_context}

Generate a complete review JSON with:
- summary: Brief summary of the paper
- strengths: List of key strengths
- weaknesses: List of key weaknesses
- questions: Questions for authors
- overall_score: Score from 1-10

Output ONLY valid JSON."""

        with token_tracker.agent_context("main_agent"):
            response = call_llm(
                model=self.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3
            )
        content = response.choices[0].message.content

        # Parse and return the review
        return self._parse_synthesized_review(content)

    def _parse_synthesized_review(self, content: str) -> Dict:
        """Parse synthesized review JSON from LLM response.

        Args:
            content: Raw LLM response containing JSON

        Returns:
            Parsed review dictionary
        """
        # Handle markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif content.startswith("```"):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1]).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse synthesized review: {e}")
            # Return minimal review from log state
            # Convert OutlineItems to text for compatibility
            return {
                "summary": self.log.review_outline.summary or "Review incomplete due to iteration limit.",
                "strengths": self.log.review_outline.get_strengths_text(),
                "weaknesses": self.log.review_outline.get_weaknesses_text(),
                "questions": self.log.review_outline.get_questions_text(),
                "overall_score": self.log.review_outline.overall_score
            }

    def get_log(self) -> ReviewLog:
        """Get the current review log state.

        Returns:
            The ReviewLog object
        """
        return self.log

    # Backward compatibility alias
    def get_memory(self) -> ReviewLog:
        """Get the current review log state (backward compatibility).

        Returns:
            The ReviewLog object
        """
        return self.log
