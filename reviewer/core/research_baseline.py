"""Baseline research agent with full paper in context (no actions)."""

from typing import Dict, List, Union
import json
import re
import logging
from .environment import PaperEnvironment
from .reviewer_memory import Claim, Question
from utils.helpers.llm import call_llm, get_content


class ResearchBaseline:
    """Baseline research agent with full paper in context (no actions).

    Unlike the action-based ResearchSubagent, this baseline:
    - Receives the full paper content upfront
    - Completes verification in a single LLM call
    - Does not use ResearchMemory or iterative actions

    This enables controlled comparison with the action-based approach.
    """

    def __init__(self, model: str):
        """Initialize the baseline research agent.

        Args:
            model: Model identifier for the research agent
        """
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

    def research(
        self,
        environment: PaperEnvironment,
        target_type: str,
        target: Union[Claim, Question],
    ) -> Dict:
        """Verify claim or answer question with full paper in context.

        Args:
            environment: Paper environment containing the full paper
            target_type: Either "claim" or "question"
            target: The claim or question to investigate

        Returns:
            Dictionary with structured findings:
            {
                "summary": str,  # Research agent's summary of findings
                "cross_references": List[str],  # Sections cited in analysis
                "evidence": List[Dict]  # Evidence collected: [{section, finding, relevance}, ...]
            }
            NOTE: No "status" - main agent decides this
        """
        self.logger.info(f"Starting baseline research on {target_type}: {target.id if hasattr(target, 'id') else 'target'}")

        # Build the prompt with full paper content
        messages = self._build_messages(environment, target_type, target)

        # Single LLM call with JSON output
        try:
            response = call_llm(
                model=self.model,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            response_content = get_content(response)
            # Extract thinking/reasoning content if available (for models that support it)
            reasoning_content = getattr(response.choices[0].message, 'thinking', None) or ""

            # Parse JSON response
            response_content = response_content.replace("```json", "").replace("```", "").strip()
            # Fix invalid escape sequences (e.g., LaTeX notation like \hat, \alpha)
            # Escape backslashes that aren't part of valid JSON escape sequences
            response_content = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', response_content)
            findings = json.loads(response_content)

            # Validate required fields
            if "summary" not in findings:
                findings["summary"] = "No summary provided"
            if "cross_references" not in findings:
                findings["cross_references"] = []
            if "evidence" not in findings:
                findings["evidence"] = []
            findings["reasoning_content"] = reasoning_content

            self.logger.info(f"Baseline research completed. Summary: {findings['summary'][:100]}...")
            return findings

        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing LLM response as JSON: {e}")
            return {
                "summary": f"Error: Failed to parse research findings - {e}",
                "cross_references": [],
                "evidence": []
            }
        except Exception as e:
            self.logger.error(f"Error during baseline research: {e}")
            return {
                "summary": f"Error: Research failed - {e}",
                "cross_references": [],
                "evidence": []
            }

    def _build_messages(
        self,
        environment: PaperEnvironment,
        target_type: str,
        target: Union[Claim, Question]
    ) -> List[Dict]:
        """Build the message list for the LLM call.

        Args:
            environment: Paper environment containing the full paper
            target_type: Either "claim" or "question"
            target: The claim or question to investigate

        Returns:
            List of message dicts for the LLM
        """
        from .research_prompts import RESEARCH_BASELINE_SYSTEM_PROMPT

        # Build research objective
        if target_type == "claim":
            objective = "**Research Objective**: Verify the following claim\n\n"
            objective += f"**Claim Text**: {target.text}\n"
            objective += f"**Source Section**: {target.section}\n"
            objective += f"**Claim Type**: {target.type}\n"
            if target.issues:
                objective += f"**Note from the main agent**: {', '.join(target.issues)}\n"
        else:  # question
            objective = "**Research Objective**: Answer the following question\n\n"
            objective += f"**Question**: {target.question}\n"
            objective += f"**Source Section**: {target.source_section}\n"

        # Get full paper text
        full_paper = environment.get_full_text()
        title = environment.sections.get('title', None)
        title_text = title.content if title else "Unknown Title"
        sections_list = [s for s in environment.get_section_names() if s != 'title']

        # Build user message with objective and full paper
        user_content = f"{objective}\n"
        user_content += "---\n\n"
        user_content += f"**Paper Title**: {title_text}\n"
        user_content += f"**Available Sections**: {', '.join(sections_list)}\n\n"
        user_content += "---\n\n"
        user_content += "# Full Paper Content\n\n"
        user_content += full_paper

        return [
            {"role": "system", "content": RESEARCH_BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]


if __name__ == "__main__":
    """Simple test for the baseline research agent."""
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

    # Create a claim to verify
    claim = Claim(
        id="claim_1",
        text="Restoring super activations recovers approximately 42% of the quality loss when super weights are pruned",
        section="3.2 mechanisms of super weights",
        type="result",
        issues=["Need to verify this 42% recovery claim with the actual numbers in experiments"]
    )

    # Create baseline agent and run research
    # model = "deepseek-reasoner"  # Change to your model
    model = "glm-47"  # Change to your model
    baseline = ResearchBaseline(model)

    print(f"\nResearching claim (baseline): {claim.text}")
    print("-" * 50)

    findings = baseline.research(
        environment=env,
        target_type="claim",
        target=claim
    )

    print(f"\nFindings (baseline):")
    print(f"  Summary: {findings['summary']}")
    print(f"  Sections cited: {findings['cross_references']}")
    print(f"  Evidence count: {len(findings['evidence'])}")
    print(f"  Reasoning content: {findings.get('reasoning_content', 'N/A')}")
    print("  NOTE: No status - main agent decides this")
