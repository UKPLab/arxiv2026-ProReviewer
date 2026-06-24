"""Research subagent for depth-first investigation of claims and questions."""

from typing import Optional, Dict, List, Union
from pydantic import BaseModel, Field
import json
import re
from .environment import PaperEnvironment
from .reviewer_memory import Claim, Question
from utils.helpers.llm import call_llm, get_content
import logging


class ResearchMemory(BaseModel):
    """Temporary memory for research subagent during investigation."""

    sections_visited: List[str] = Field(
        default_factory=list,
        description="Sections examined during research"
    )
    evidence: List[Dict] = Field(
        default_factory=list,
        description="Evidence found: [{section, finding, relevance}, ...]"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Working notes and observations"
    )
    current_hypothesis: Optional[str] = Field(
        default=None,
        description="Current working hypothesis about the claim/question"
    )

    def add_evidence(self, section: str, finding: str, relevance: str):
        """Add evidence from a section."""
        # Skip evidence with empty findings
        if finding and finding.strip():
            self.evidence.append({
                "section": section.strip() if section else "Unknown",
                "finding": finding.strip(),
                "relevance": relevance.strip() if relevance else ""
            })

    def add_note(self, note: str):
        """Add a working note."""
        # Skip empty or whitespace-only notes
        if note and note.strip():
            self.notes.append(note.strip())

    def visit_section(self, section: str):
        """Record section visit."""
        if section not in self.sections_visited:
            self.sections_visited.append(section)

    def build_context(self) -> str:
        """Build context string for the agent."""
        context = "# Research Progress\n\n"

        context += f"**Sections Visited**: {', '.join(self.sections_visited) if self.sections_visited else 'None yet'}\n\n"

        if self.current_hypothesis:
            context += f"**Current Hypothesis**: {self.current_hypothesis}\n\n"

        if self.evidence:
            context += "**Evidence Collected**:\n"
            for i, ev in enumerate(self.evidence, 1):
                context += f"{i}. [{ev['section']}] {ev['finding']}\n   Relevance: {ev['relevance']}\n"
            context += "\n"

        if self.notes:
            context += "**Working Notes**:\n"
            for i, note in enumerate(self.notes, 1):
                context += f"{i}. {note}\n"
            context += "\n"

        return context


class ResearchSubagent:
    """Subagent specialized for depth-first research and verification.

    The research subagent conducts thorough investigations to:
    - Verify claims by examining multiple sections
    - Answer questions by searching across the paper
    - Gather evidence and cross-references
    - Provide structured findings back to the main agent

    The subagent operates autonomously with its own agent loop.
    """

    def __init__(self, model: str):
        """Initialize the research subagent.

        Args:
            model: Model identifier for the research agent
        """
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

    def research(
        self,
        environment: PaperEnvironment,
        target_type: str,
        target: Union[Claim, Question, Dict],
        max_iterations: int = 20,
        trajectory_mode: str = "minimal"  # "minimal" or "full"
    ) -> Dict:
        """Conduct deep research to verify claim or answer question.

        Args:
            environment: Paper environment to read sections from
            target_type: Either "claim" or "question"
            target: The claim/question data (can be Pydantic object or dict)
            max_iterations: Maximum research iterations
            trajectory_mode: "minimal" (uses ResearchMemory) or "full" (complete history)

        Returns:
            Dictionary with structured findings:
            {
                "summary": str,  # Research agent's summary of what it found
                "cross_references": List[str],  # Sections examined
                "evidence": List[Dict]  # Evidence collected
            }
            NOTE: No "status" - main agent decides this
        """
        # Get target ID - works for both dict and Pydantic object
        target_id = target.get('id') if isinstance(target, dict) else getattr(target, 'id', 'target')
        self.logger.info(f"Starting research on {target_type}: {target_id}")
        self.logger.info(f"Trajectory mode: {trajectory_mode}")

        # Initialize memory only for minimal mode
        temp_memory = ResearchMemory() if trajectory_mode == "minimal" else None

        # Build initial trajectory
        initial_trajectory = self._build_initial_trajectory(
            target_type, target, environment
        )

        # Full trajectory for analysis
        full_trajectory = initial_trajectory.copy()

        last_assistant_message = None
        last_observations = []

        # Research loop
        for iteration in range(max_iterations):
            self.logger.info(f"Research iteration {iteration + 1}/{max_iterations}")

            # Build trajectory based on mode
            if trajectory_mode == "minimal":
                working_trajectory = self._build_minimal_trajectory(
                    initial_trajectory, last_assistant_message, last_observations, temp_memory
                )
            else:  # full - no memory, just all messages
                working_trajectory = self._build_full_trajectory(full_trajectory)

            # Decide next action (JSON output)
            try:
                decision, response_message = self._decide_next_action(working_trajectory)
                
            except ValueError as e:
                self.logger.error(f"Error parsing LLM response: {e}")
                # Add error message and continue
                error_observation = {
                    "role": "user",
                    "content": f"<error>Invalid response format: {e}. Please respond with valid JSON containing 'memory_update' and 'action' fields.</error>"
                }
                full_trajectory.append(error_observation)
                last_observations = [error_observation]
                continue

            # Add assistant message to full trajectory
            last_assistant_message = response_message
            full_trajectory.append(last_assistant_message)

            # Execute memory update if present (only tracked in minimal mode)
            memory_update = decision.get("memory_update")
            if memory_update and memory_update.get("type") and temp_memory:
                memory_result = self._execute_memory_update(memory_update, temp_memory)
                self.logger.info(f"Memory update: {memory_result}")

            # Execute action
            action = decision["action"]
            action_name = action["name"]
            action_args = action.get("args", {})

            if action_name == "finish_research":
                # Terminal action - build findings
                findings = self._build_findings(action_args, temp_memory, target_type)
                self.logger.info(f"Research completed. Summary: {findings['summary'][:100]}...")
                return findings

            elif action_name == "read_section":
                action_response = self._execute_read_section(environment, action_args, temp_memory)

            elif action_name == "search_paper":
                action_response = self._execute_search_paper(environment, action_args)

            else:
                action_response = f"Error: Unknown action '{action_name}'"

            # Create observation
            observation_content = f"<action_result>\n{action_response}\n</action_result>"
            last_observation = {"role": "user", "content": observation_content}
            full_trajectory.append(last_observation)
            last_observations = [last_observation]

        # Max iterations reached - synthesize what we have
        self.logger.warning("Max iterations reached. Synthesizing findings.")
        return self._synthesize_findings(temp_memory, target_type)

    def _build_initial_trajectory(
        self,
        target_type: str,
        target: Union[Claim, Question, Dict],
        environment: PaperEnvironment
    ) -> List[Dict]:
        """Build initial research trajectory."""
        from .research_prompts import RESEARCH_AGENT_SYSTEM_PROMPT

        # Helper to get attribute from dict or object
        def get_attr(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key)

        # Build research objective
        if target_type == "claim":
            objective = f"**Research Objective**: Verify the following claim\n\n"
            objective += f"**Claim Text**: {get_attr(target, 'text')}\n"
            objective += f"**Source Section**: {get_attr(target, 'section')}\n"
            objective += f"**Claim Type**: {get_attr(target, 'type')}\n"
            issues = get_attr(target, 'issues')
            if issues:
                objective += f"**Note from the main agent**: {', '.join(issues)}\n"
        else:  # question
            objective = f"**Research Objective**: Answer the following question\n\n"
            objective += f"**Question**: {get_attr(target, 'question')}\n"
            objective += f"**Source Section**: {get_attr(target, 'source_section')}\n"

        objective += "\n---\n\n"
        objective += "Your task is to conduct thorough research to verify this claim or answer this question. "
        title = environment.sections['title'].content
        sections_list = [s for s in environment.get_section_names() if s != 'title']
        objective += f"The paper is: {title} and it has the following sections: {', '.join(sections_list)} . You can read any sections you need, take notes, collect evidence, and then provide findings."

        return [
            {"role": "system", "content": RESEARCH_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": objective}
        ]

    def _build_minimal_trajectory(
        self,
        initial_trajectory: List[Dict],
        last_assistant_message: Optional[Dict],
        last_observations: List[Dict],
        temp_memory: ResearchMemory
    ) -> List[Dict]:
        """Build minimal trajectory with current state."""
        trajectory = initial_trajectory.copy()

        # Add last action if exists
        if last_assistant_message:
            trajectory.append(last_assistant_message)

        # Add observations
        for observation in last_observations:
            trajectory.append(observation)

        # Add memory context
        memory_context = temp_memory.build_context()
        trajectory.append({
            "role": "user",
            "content": f"<memory_context>\n{memory_context}\n</memory_context>"
        })
        print(f"---The current memroy status is {memory_context}-----")

        return trajectory

    def _build_full_trajectory(self, full_trajectory: List[Dict]) -> List[Dict]:
        """Build full trajectory with ALL messages - no memory summarization.

        The LLM sees the complete conversation history and learns from it directly.
        No intermediate ResearchMemory is injected.
        """
        return full_trajectory.copy()

    def _execute_memory_update(self, memory_update: Dict, temp_memory: ResearchMemory) -> str:
        """Execute a memory update operation."""
        update_type = memory_update.get("type")

        if update_type == "evidence":
            section = memory_update.get("section", "Unknown")
            finding = memory_update.get("finding", "")
            relevance = memory_update.get("relevance", "")
            temp_memory.add_evidence(section, finding, relevance)
            if finding and finding.strip():
                return f"Added evidence from {section}"
            else:
                return "Skipped empty evidence"

        elif update_type == "note":
            note = memory_update.get("note", "")
            temp_memory.add_note(note)
            if note and note.strip():
                return f"Added note: {note[:50]}..."
            else:
                return "Skipped empty note"

        elif update_type == "hypothesis":
            hypothesis = memory_update.get("hypothesis", "")
            # Only update if hypothesis is non-empty
            if hypothesis and hypothesis.strip():
                temp_memory.current_hypothesis = hypothesis.strip()
                return f"Updated hypothesis"
            else:
                return "Skipped empty hypothesis"

        else:
            return f"Unknown memory update type: {update_type}"

    def _decide_next_action(self, trajectory: List[Dict]) -> tuple:
        """Decide next research action using JSON output format.

        Returns:
            Tuple of (decision_dict, response_message_dict) where decision_dict contains:
            {
                "memory_update": {...} or None,
                "action": {"name": "...", "args": {...}}
            }

        Raises:
            ValueError: If LLM returns invalid JSON
        """
        response = call_llm(
            model=self.model,
            messages=trajectory,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        response_message = response.choices[0].message
        response_content = response_message.content

        # Parse JSON
        try:
            # clean up the prefix and suffix of the response_content
            response_content = response_content.replace("```json", "").replace("```", "").strip()
            # Fix invalid escape sequences (e.g., LaTeX notation like \hat, \alpha)
            # Escape backslashes that aren't part of valid JSON escape sequences
            response_content = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', response_content)
            decision = json.loads(response_content)

            # Validate structure
            if "action" not in decision:
                raise ValueError("Missing 'action' in response")

            if not isinstance(decision["action"], dict) or "name" not in decision["action"]:
                raise ValueError("'action' must be a dict with 'name' field")

            response_dict = {"role": "assistant", "content": response_content}
            return decision, response_dict

        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}")

    def _execute_read_section(
        self,
        environment: PaperEnvironment,
        args: Dict,
        temp_memory: Optional[ResearchMemory]
    ) -> str:
        """Read a section during research."""
        section_name = args.get('section_name', '')
        self.logger.info(f"Research agent reading section: {section_name}")

        content = environment.read_section(section_name)

        # Track section visit only in minimal mode
        if temp_memory:
            temp_memory.visit_section(section_name)

        return f"Successfully read section '{section_name}'. Content:\n{content}"

    def _execute_search_paper(self, environment: PaperEnvironment, args: Dict) -> str:
        """Search the paper for a query string."""
        query = args.get('query', '')
        if not query:
            return "Error: Missing query parameter"

        self.logger.info(f"Research agent searching for: {query}")
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

    def _build_findings(self, args: Dict, temp_memory: Optional[ResearchMemory], target_type: str) -> Dict:
        """Build findings dict from finish_research action.

        NOTE: Research agent does NOT decide status. It only presents evidence.
        The main agent will interpret these findings and decide the status.
        """
        summary = args.get('summary', '')  # What the evidence shows

        # In full trajectory mode, temp_memory is None
        if temp_memory:
            cross_refs = temp_memory.sections_visited
            evidence = temp_memory.evidence
        else:
            # Full trajectory mode - extract from args if provided
            cross_refs = args.get('cross_references', [])
            evidence = args.get('evidence', [])

        return {
            "summary": summary,  # Research agent's summary of what it found
            "cross_references": cross_refs,
            "evidence": evidence
            # NO "status" - main agent decides this
        }

    def _synthesize_findings(self, temp_memory: Optional[ResearchMemory], target_type: str) -> Dict:
        """Synthesize findings when max iterations reached using LLM with full memory context.

        NOTE: No status - main agent decides this.
        """
        if temp_memory:
            sections_visited = temp_memory.sections_visited
            evidence = temp_memory.evidence
            # Build full memory context for synthesis
            memory_context = temp_memory.build_context()
        else:
            sections_visited = []
            evidence = []
            memory_context = "No memory available (full trajectory mode)."

        # Build synthesis prompt with full memory
        synthesis_prompt = f"""Target type: {target_type}

{memory_context}

Based on all the information above (sections visited, evidence, notes, hypothesis), provide a concise summary of the key findings. Be honest and precisely highlight what was discovered."""

        # Call LLM to synthesize
        reasoning_content = ""
        try:
            response = call_llm(
                model=self.model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.3
            )
            summary = get_content(response)

        except Exception as e:
            self.logger.error(f"Error synthesizing: {e}")
            summary = f"Research incomplete. Examined {len(sections_visited)} sections, collected {len(evidence)} evidence pieces."

        return {
            "summary": summary,
            "cross_references": sections_visited,
            "evidence": evidence
        }

if __name__ == "__main__":
    """Simple test for the research subagent."""
    import sys
    sys.path.insert(0, "/Users/haishuo/Reviewer-R1")

    import json
    from reviewer.core.environment import PaperEnvironment
    from reviewer.core.reviewer_memory import Claim

    # Load real paper from JSON
    paper_path = "/Users/haishuo/Reviewer-R1/data/0Ag8FQ5Rr3.json"
    with open(paper_path, "r") as f:
        data = json.load(f)

    markdown_content = data["markdown"]["content"]
    # Check if content already starts with a title (# Title or Title:)
    # If not, prepend the title from metadata
    if not (markdown_content.startswith('# ') or markdown_content.startswith('Title:')):
        title = data.get('title', '')
        if title:
            paper_content = f"# {title}\n\n{markdown_content}"
        else:
            paper_content = markdown_content
    else:
        paper_content = markdown_content

    # Create environment
    env = PaperEnvironment(paper_content)
    print(f"Paper: {data['title']}")
    print(f"Sections: {env.get_section_names()}")

    # Create a claim to verify - this is a key claim from the paper
    claim = Claim(
        id="claim_1",
        text="Restoring super activations recovers approximately 42% of the quality loss when super weights are pruned",
        section="3.2 mechanisms of super weights",
        type="result",
        issues=["Need to verify this 42% recovery claim with the actual numbers in experiments"]
    )

    # Create subagent and run research
    # model = "deepseek-reasoner"  # Change to your model
    model = "glm-47"  # Change to your model
    subagent = ResearchSubagent(model)

    print(f"\nResearching claim: {claim.text}")
    print("-" * 50)

    # Test minimal trajectory mode
    print("\n=== Testing MINIMAL trajectory mode ===")
    findings = subagent.research(
        environment=env,
        target_type="claim",
        target=claim,
        max_iterations=20,
        trajectory_mode="minimal"
    )

    print(f"\nFindings (minimal mode):")
    print(f"  Summary: {findings['summary']}")
    print(f"  Sections visited: {findings['cross_references']}")
    print(f"  Evidence count: {len(findings['evidence'])}")
    print("  NOTE: No status - main agent decides this")

    # Uncomment to test full trajectory mode:
    # print("\n=== Testing FULL trajectory mode ===")
    # findings_full = subagent.research(
    #     environment=env,
    #     target_type="claim",
    #     target=claim,
    #     max_iterations=10,
    #     trajectory_mode="full"
    # )
    # print(f"\nFindings (full mode):")
    # print(f"  Summary: {findings_full['summary']}")
