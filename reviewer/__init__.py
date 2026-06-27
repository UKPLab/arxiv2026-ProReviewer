"""Evidence-based peer review agent for scientific papers."""

__version__ = "0.1.0"

# Core components (active)
from .core.base_agent import BaseReviewAgent
from .core.proreviewer import ProReviewer
from .core import ProReviewerDirect  # Backward compatibility alias
from .core.environment import PaperEnvironment, Section
from .core.reviewer_memory import ReviewLog, ReviewMemory, Claim, Question as CoreQuestion, Note, ReviewOutline
from .core.research_agent import ResearchSubagent

# Reward system
from .reward import RewardCalculator, score_review

# rLLM integration (agent-environment pattern)
from .rllm_version import ReviewAgent as RLLMReviewAgent, ReviewEnv

# Evaluation (not imported at top level to avoid circular imports)
# Users should import directly: from reviewer.evaluation import run_evaluation

__all__ = [
    # Core (active)
    "BaseReviewAgent", "ProReviewer", "ProReviewerDirect", "PaperEnvironment", "Section",
    "ReviewLog", "ReviewMemory", "Claim", "CoreQuestion", "Note", "ReviewOutline",
    "ResearchSubagent",
    # Reward
    "RewardCalculator", "score_review",
    # rLLM integration
    "RLLMReviewAgent", "ReviewEnv",
]
