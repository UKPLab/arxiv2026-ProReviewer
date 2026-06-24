"""Abstract base class for review agents."""

from abc import ABC, abstractmethod
from re import I
from typing import Optional, Dict, List, Tuple, Union
from .environment import PaperEnvironment
from utils.helpers.logger import logger
from litellm.types.utils import Message
from utils.helpers.llm import call_llm


class BaseReviewAgent(ABC):
    """Abstract base class for peer review agents.

    Provides shared functionality for:
    - LLM interaction (_decide_next_action, _parse_response)
    - Trajectory management
    - Writer subagent integration

    Subclasses implement specific review strategies by overriding abstract methods.
    """

    def __init__(
        self,
        model: str,
        conference_format: str = "ICLR"
    ):
        """Initialize the review agent.

        Args:
            model: Model identifier (config name or model string, e.g., "gpt-51", "openai/gpt-4")
            conference_format: The conference format to use for the review (e.g., "ICLR")
        """
        # Import here to avoid circular import
        self.logger = logger
        self.model = model
        self.conference_format = conference_format

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent type.

        Returns:
            System prompt string defining the agent's role and behavior
        """
        pass

    @abstractmethod
    def get_tools(self) -> List[dict]:
        """Return the tools available for this agent type.

        Returns:
            List of tool definitions in OpenAI function calling format
        """
        pass

    @abstractmethod
    def review_paper(self, environment: PaperEnvironment, max_iterations: int = 50) -> List[dict]:
        """Review a paper and return the trajectory.

        Args:
            environment: The paper environment containing the paper to review
            max_iterations: Maximum number of iterations for iterative agents

        Returns:
            List of message dictionaries representing the full review trajectory
        """
        pass

    def _decide_next_action(self, trajectory: List[dict], tools: Optional[List[dict]] = None) -> Tuple[List[Dict], Message]:
        """Use LLM to decide what action to take next.

        Args:
            trajectory: Current conversation trajectory (must be list of dicts)
            tools: Optional tools to use (defaults to self.get_tools())

        Returns:
            Memory operations:
            - List of memory operations to perform
            - List of action dictionaries with 'action', 'args', and 'tool_call_id' keys,
            or a string if no valid tool calls were found (invalid response)
        """
        if tools is None:
            tools = self.get_tools()

        # Ensure all messages in trajectory are dicts (not Message objects)
        # This is critical for API compatibility - Message objects with tool_calls
        # must be properly converted to dicts
        trajectory_dicts = [self._message_to_dict(msg) for msg in trajectory]

        # Call LLM using litellm helper
        llm_response = call_llm(model=self.model, messages=trajectory_dicts, tools=tools)
        response_message = llm_response.choices[0].message
        self.logger.info(f"len(trajectory): {len(trajectory)}")

        actions = self._parse_response(response_message)
        self.logger.info(f"Decision: actions: {actions}")
        return actions, response_message

    def _parse_response(self, response: Message) -> Union[List[Dict], str]:
        """Parse the LLM response and extract actions.

        Args:
            response: LLM response message

        Returns:
            List of action dictionaries if tool calls found,
            otherwise the response content string (indicates invalid response)
        """
        actions = []
        tool_calls = response.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                actions.append({
                    'action': tool_call["function"]["name"],
                    'args': tool_call["function"]["arguments"],
                    'tool_call_id': tool_call["id"],
                })
            return actions
        else:
            self.logger.warning(f"No tool calls found in the response: {response.content}")
            return response.content

    def _build_initial_trajectory(self, environment: PaperEnvironment) -> Tuple[List[dict], str, List[str]]:
        """Build the initial trajectory with system prompt and paper info.

        Args:
            environment: The paper environment

        Returns:
            Tuple of (initial_trajectory, title, sections_list)
        """
        sections = environment.get_section_names()
        self.logger.info(f"Paper sections: {', '.join(sections)}")

        trajectory = [{"role": "system", "content": self.get_system_prompt()}]
        title = environment.sections['title'].content
        sections_list = [s for s in sections if s != 'title']

        trajectory.append({
            "role": "user",
            "content": f"The paper you are reviewing is: {title} and it has the following sections: {', '.join(sections_list)}"
        })

        return trajectory, title, sections_list

    def _execute_read_section(self, environment: PaperEnvironment, section_name: str) -> str:
        """Execute a read_section action.

        Args:
            environment: The paper environment
            section_name: Name of the section to read

        Returns:
            Tool response string with section content
        """
        self.logger.info(f"Agent decided to read section: {section_name}")
        content = environment.read_section(section_name)
        return f"Successfully read section '{section_name}'. Content:\n{content}"

    def _message_to_dict(self, msg: Union[Message, dict]) -> dict:
        """Convert a Message object to a dictionary for API calls.

        Args:
            msg: Message object or dict

        Returns:
            Dictionary representation of the message
        """
        if isinstance(msg, dict):
            return msg
        
        # Try model_dump() first (Pydantic v2)
        if hasattr(msg, 'model_dump'):
            try:
                return msg.model_dump()
            except Exception:
                pass
        
        # Try dict() method (Pydantic v1)
        if hasattr(msg, 'dict'):
            try:
                return msg.dict()
            except Exception:
                pass
        
        # Fallback: manually extract attributes
        result = {"role": "assistant"}
        if hasattr(msg, 'role'):
            result['role'] = msg.role
        if hasattr(msg, 'content'):
            result['content'] = msg.content or ""
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            result['tool_calls'] = msg.tool_calls
        return result
