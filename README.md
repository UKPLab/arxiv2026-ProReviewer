# ProReviewer: From Passive Generation to Investigation: A Proactive Scientific Peer Review Agent

An RL-trained peer review agent featuring **ProReviewer**, an R1-style reasoning agent with structured memory, research delegation, and evidence-based judgment. The system includes a multi-dimensional reward function, SFT data generation pipeline, and RL training infrastructure built on [rLLM](https://github.com/agentica-project/rllm).

## Overview

**ProReviewer** trains a language model to produce high-quality scientific peer reviews through reinforcement learning. The core components are:

- **ProReviewer Agent**: R1-style reasoning agent with structured memory (claims, questions, assessments, review outline), research delegation via a verification subagent, and evidence-based judgment
- **Multi-Dimensional Reward**: Reward function evaluating recall coverage, format compliance, actionability, grounding, factual correctness, duplicate detection, and memory-based reasoning quality
- **RL Training Pipeline**: Multi-turn review environment integrated with the rLLM framework for GRPO-based training
- **SFT Data Generation**: Tools for generating supervised fine-tuning trajectories as warm-start for RL

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (for RL training and local model inference)

### Setup

```bash
git clone <repository-url>
cd ProReviewer

# Create and activate the virtual environment
micromamba create -f environment_rllm.yml
micromamba activate rllm

# Install the verl backend
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.6.1
uv pip install -e .
cd ..

# Install the modified rLLM framework (with step-level GRPO support)
cd rllm
uv pip install -e .
cd ..

# Configure API keys
cp config.toml.example config.toml
# Edit config.toml with your API keys and model paths
```


## Project Structure

```
ProReviewer/
├── config.toml.example                # Configuration template
├── pyproject.toml                     # Python dependencies
│
├── reviewer/                          # Core review system
│   ├── core/                          # ProReviewer agent
│   │   ├── proreviewer.py             # Main R1 agent
│   │   ├── reviewer_memory.py         # Memory structures (Claim, Assessment, etc.)
│   │   ├── reviewer_prompts.py        # System prompts
│   │   ├── verifier.py                # Claim verification
│   │   ├── base_agent.py              # Abstract base class
│   │   ├── environment.py             # Paper parsing and navigation
│   │   ├── review_agent.py            # RL review agent
│   │   ├── review_env.py              # Review environment for RL
│   │   └── review_workflow.py         # Multi-turn review workflow
│   ├── reward/                        # Multi-dimensional reward system
│   │   ├── calculator.py              # Reward computation
│   │   ├── components.py              # Individual reward components
│   │   ├── score_review.py            # Review scoring
│   │   ├── duplicate_checker.py       # Duplicate weakness detection
│   │   ├── factual_correctness_simple.py  # Factual correctness
│   │   ├── memory_reasoning.py        # Memory-based reasoning reward
│   │   └── trajectory_memory_reasoning*.py  # Trajectory-level rewards
│   ├── prompts/                       # Prompt templates
│   │   ├── reward_prompts.py          # Reward judge prompts
│   │   ├── rubric_dimensions.py       # Rubric evaluation dimensions
│   │   └── trajectory_v2_prompt*.py   # Trajectory generation prompts
│   └── sft/                           # SFT data generation
│       ├── train.py                   # SFT training entry point
│       ├── train_sft.py               # SFT training script
│       ├── generate_review_sft_data.py # SFT data generation
│       ├── prepare_review_dataset.py  # Dataset preparation
│       └── trace_generator.py         # Trace generation
│
├── rllm/                              # RL framework (modified rLLM)
│   └── ...                            # Training infrastructure
│
└── utils/                             # Shared utilities
    ├── helpers/
    │   ├── llm.py                     # LLM API wrapper (litellm-based)
    │   ├── logger.py                  # Logging utilities
    │   └── token_tracker.py           # Token usage tracking
    ├── sft/                           # SFT data tools
    │   ├── review_parser.py           # Parse review text
    │   ├── trajectory_generator.py    # Generate trajectories
    │   └── trajectory_validator.py    # Validate trajectories
    └── paper_fetcher/                 # Data pipeline
        ├── arxiv_client.py            # arXiv API
        ├── openreview_client.py       # OpenReview API
        ├── enricher.py                # Main orchestrator
        ├── pdf_processor.py           # PDF processing
        └── cache_manager.py           # Cache management
```

## Core Components

### ProReviewer Agent (`reviewer/core/`)

The main agent uses R1-style reasoning with structured memory:

- **Structured Memory**: Tracks claims, questions, assessments, and review outline as the agent reads through a paper
- **Research Delegation**: Delegates claim verification to a specialized research subagent
- **Evidence-Based Judgment**: Two-turn pattern — delegate investigation, then judge findings
- **Skeptical Reading**: Flags suspicious claims rather than passively accepting them

### Reward System (`reviewer/reward/`)

Multi-dimensional reward function for RL training:

| Component | Description |
|-----------|-------------|
| Recall Coverage | Does the review identify the key weaknesses? |
| Format Compliance | Is the review properly structured? |
| Actionability | Are suggestions concrete and actionable? |
| Grounding | Are claims grounded in paper evidence? |
| Factual Correctness | Are factual statements accurate? |
| Duplicate Detection | Penalizes repeated weaknesses |
| Memory Reasoning | Quality of reasoning based on memory state |

### RL Training (`reviewer/core/` + `rllm/`)

The training pipeline integrates with rLLM for GRPO-based reinforcement learning:

- `review_env.py`: Multi-turn environment where the agent reads and reviews a paper
- `review_workflow.py`: Orchestrates the review process across turns
- `review_agent.py`: RL review agent

We extend the rLLM framework with a **step-level GRPO** mode for step-wise advantage computation. Instead of computing advantages per trajectory, this mode pools all steps across trajectories of the same task and computes advantages at the step level. See [`rllm/change.md`](rllm/change.md) for details.

### SFT Data Generation (`reviewer/sft/` + `utils/sft/`)

Tools for generating supervised fine-tuning trajectories:

- Parse existing reviews into structured format
- Generate training trajectories using teacher models
- Validate trajectory quality

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fang2026passive,
  title={From Passive Generation to Investigation: A Proactive Scientific Peer Review Agent},
  author={Fang, Haishuo and Feng, Yue and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2606.13349},
  year={2026}
}
```

## License

[Add your license here]
