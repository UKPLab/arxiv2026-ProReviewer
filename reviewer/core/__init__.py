"""Core ProReviewer components - active implementations."""

from .base_agent import BaseReviewAgent
from .proreviewer import ProReviewer
from .environment import PaperEnvironment, Section
from .reviewer_memory import ReviewLog, ReviewMemory, Claim, Question, Note, ReviewOutline
from .research_agent import ResearchSubagent


# Backward compatibility alias for ProReviewerDirect (deprecated)
# Instead of importing ProReviewerDirect from a separate file,
# we create an alias that instantiates ProReviewer with use_research_subagent=False
def ProReviewerDirect(model: str, conference_format: str = "ICLR"):
    """Deprecated: Use ProReviewer(model, use_research_subagent=False) instead.

    This is a backward compatibility wrapper that creates a ProReviewer instance
    in direct mode (without research subagent).
    """
    return ProReviewer(model=model, conference_format=conference_format, use_research_subagent=False)


__all__ = [
    "BaseReviewAgent",
    "ProReviewer",
    "ProReviewerDirect",  # Backward compatibility alias
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
