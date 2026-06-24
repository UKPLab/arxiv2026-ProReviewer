"""Token usage tracking for multi-agent paper review.

This module provides thread-safe token tracking with per-agent breakdown and aggregated totals.

Usage:
    from utils.helpers.token_tracker import token_tracker

    # Start tracking for a paper
    token_tracker.start_paper("paper_123")

    # Wrap agent code with context
    with token_tracker.agent_context("main_agent"):
        response = call_llm(...)  # Automatically recorded

    # Get usage summary
    summary = token_tracker.get_paper_summary("paper_123")

    # Cleanup
    token_tracker.end_paper()
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager


@dataclass
class TokenUsage:
    """Single LLM call token usage record."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentUsage:
    """Aggregated token usage for a specific agent."""
    agent_type: str
    calls: List[TokenUsage] = field(default_factory=list)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_type": self.agent_type,
            "call_count": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


class TokenTracker:
    """Thread-safe singleton for tracking token usage across agents.

    Provides:
    - Per-paper tracking with start_paper/end_paper lifecycle
    - Per-agent attribution with agent_context context manager
    - Automatic recording from LLM responses
    - Summary generation with per-agent breakdown and totals
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Thread-local storage for current context
        self._local = threading.local()

        # Paper-level tracking: {paper_id: {agent_type: AgentUsage}}
        self._paper_usage: Dict[str, Dict[str, AgentUsage]] = {}

        # Lock for thread-safe operations
        self._data_lock = threading.Lock()

    @property
    def _current_paper_id(self) -> Optional[str]:
        return getattr(self._local, 'paper_id', None)

    @_current_paper_id.setter
    def _current_paper_id(self, value: Optional[str]):
        self._local.paper_id = value

    @property
    def _current_agent(self) -> Optional[str]:
        return getattr(self._local, 'agent', None)

    @_current_agent.setter
    def _current_agent(self, value: Optional[str]):
        self._local.agent = value

    def start_paper(self, paper_id: str) -> None:
        """Start tracking token usage for a new paper.

        Args:
            paper_id: Unique identifier for the paper being reviewed
        """
        with self._data_lock:
            self._paper_usage[paper_id] = {}
        self._current_paper_id = paper_id

    def end_paper(self) -> None:
        """End tracking for the current paper."""
        self._current_paper_id = None
        self._current_agent = None

    @contextmanager
    def agent_context(self, agent_type: str):
        """Context manager to attribute token usage to a specific agent.

        Args:
            agent_type: Agent identifier (e.g., "main_agent", "research_subagent")

        Usage:
            with token_tracker.agent_context("main_agent"):
                response = call_llm(...)  # Automatically attributed to main_agent
        """
        previous_agent = self._current_agent
        self._current_agent = agent_type
        try:
            yield
        finally:
            self._current_agent = previous_agent

    def record(self, usage_obj: Any, model: str) -> None:
        """Record token usage from an LLM response.

        Args:
            usage_obj: Usage object from LLM response (has prompt_tokens, completion_tokens, total_tokens)
            model: Model identifier used for the call
        """
        paper_id = self._current_paper_id
        agent_type = self._current_agent

        if not paper_id:
            # Not tracking any paper, skip
            return

        if not agent_type:
            agent_type = "unattributed"

        # Extract token counts from usage object
        if usage_obj is None:
            return

        prompt_tokens = getattr(usage_obj, 'prompt_tokens', 0) or 0
        completion_tokens = getattr(usage_obj, 'completion_tokens', 0) or 0
        total_tokens = getattr(usage_obj, 'total_tokens', 0) or 0

        # If total_tokens is 0 but we have prompt and completion, calculate it
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens

        # Create usage record
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model
        )

        # Add to paper's agent usage
        with self._data_lock:
            if paper_id not in self._paper_usage:
                self._paper_usage[paper_id] = {}

            if agent_type not in self._paper_usage[paper_id]:
                self._paper_usage[paper_id][agent_type] = AgentUsage(agent_type=agent_type)

            self._paper_usage[paper_id][agent_type].calls.append(usage)

    def get_paper_summary(self, paper_id: str) -> Dict:
        """Get token usage summary for a paper.

        Args:
            paper_id: Paper identifier to get summary for

        Returns:
            Dictionary with per-agent breakdown and totals:
            {
                "paper_id": "paper_123",
                "by_agent": {
                    "main_agent": {...},
                    "research_subagent": {...}
                },
                "totals": {
                    "prompt_tokens": 195000,
                    "completion_tokens": 18000,
                    "total_tokens": 213000,
                    "total_calls": 40
                }
            }
        """
        with self._data_lock:
            agent_usage = self._paper_usage.get(paper_id, {})

        # Build per-agent breakdown
        by_agent = {
            agent_type: usage.to_dict()
            for agent_type, usage in agent_usage.items()
        }

        # Calculate totals
        total_prompt = sum(u.prompt_tokens for u in agent_usage.values())
        total_completion = sum(u.completion_tokens for u in agent_usage.values())
        total_tokens = sum(u.total_tokens for u in agent_usage.values())
        total_calls = sum(u.call_count for u in agent_usage.values())

        return {
            "paper_id": paper_id,
            "by_agent": by_agent,
            "totals": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "total_calls": total_calls
            }
        }

    def get_all_papers_summary(self) -> Dict:
        """Get aggregated token usage summary across all papers.

        Returns:
            Dictionary with aggregated totals across all tracked papers
        """
        with self._data_lock:
            paper_ids = list(self._paper_usage.keys())

        all_by_agent: Dict[str, Dict] = {}
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        total_calls = 0

        for paper_id in paper_ids:
            summary = self.get_paper_summary(paper_id)

            # Aggregate by agent across papers
            for agent_type, agent_data in summary["by_agent"].items():
                if agent_type not in all_by_agent:
                    all_by_agent[agent_type] = {
                        "agent_type": agent_type,
                        "call_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                all_by_agent[agent_type]["call_count"] += agent_data["call_count"]
                all_by_agent[agent_type]["prompt_tokens"] += agent_data["prompt_tokens"]
                all_by_agent[agent_type]["completion_tokens"] += agent_data["completion_tokens"]
                all_by_agent[agent_type]["total_tokens"] += agent_data["total_tokens"]

            # Aggregate totals
            total_prompt += summary["totals"]["prompt_tokens"]
            total_completion += summary["totals"]["completion_tokens"]
            total_tokens += summary["totals"]["total_tokens"]
            total_calls += summary["totals"]["total_calls"]

        return {
            "papers_count": len(paper_ids),
            "by_agent": all_by_agent,
            "totals": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "total_calls": total_calls
            }
        }

    def clear_paper(self, paper_id: str) -> None:
        """Clear usage data for a specific paper.

        Args:
            paper_id: Paper identifier to clear
        """
        with self._data_lock:
            if paper_id in self._paper_usage:
                del self._paper_usage[paper_id]

    def clear_all(self) -> None:
        """Clear all tracked usage data."""
        with self._data_lock:
            self._paper_usage.clear()
        self._current_paper_id = None
        self._current_agent = None


# Global singleton instance
token_tracker = TokenTracker()
