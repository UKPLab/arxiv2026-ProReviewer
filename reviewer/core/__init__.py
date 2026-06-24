"""Core ReviewerR1 components - active implementations."""

from .base_agent import BaseReviewAgent
from .reviewer_r1 import ReviewerR1
from .environment import PaperEnvironment, Section
from .reviewer_memory import ReviewLog, ReviewMemory, Claim, Question, Note, ReviewOutline
from .research_agent import ResearchSubagent


# Backward compatibility alias for ReviewerR1Direct (deprecated)
# Instead of importing ReviewerR1Direct from a separate file,
# we create an alias that instantiates ReviewerR1 with use_research_subagent=False
def ReviewerR1Direct(model: str, conference_format: str = "ICLR"):
    """Deprecated: Use ReviewerR1(model, use_research_subagent=False) instead.

    This is a backward compatibility wrapper that creates a ReviewerR1 instance
    in direct mode (without research subagent).
    """
    return ReviewerR1(model=model, conference_format=conference_format, use_research_subagent=False)


__all__ = [
    "BaseReviewAgent",
    "ReviewerR1",
    "ReviewerR1Direct",  # Backward compatibility alias
    "PaperEnvironment",
    "Section",
    "ReviewLog",
    "ReviewMemory",  # Backward compatibility alias
    "Claim",
    "Question",
    "Note",
    "ReviewOutline",
    "ResearchSubagent",
]
