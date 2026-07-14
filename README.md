# ProReviewer: From Passive Generation to Investigation

[![arXiv](https://img.shields.io/badge/arXiv-2606.13349-b31b1b.svg)](https://arxiv.org/abs/2606.13349)
[![Skill](https://img.shields.io/badge/Skill-ProReviewer-blue)](ProReviewer-Skill/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-coming%20soon-yellow)](#)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-coming%20soon-yellow)](#)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)

**An AI agent that reviews papers the way human experts do — reading step by step, taking notes, and verifying claims before judging.**

**ProReviewer** formulates peer review as a Markov Decision Process (MDP): the paper is built as an environment, and the agent uses a structured working memory (review log) to track its reviewing artifacts across multiple steps.

---

## Quick Start

There are three ways to use ProReviewer. Pick the one that fits your setup:

| | **Skill** | **API Prompting** | **Trained Model** |
|---|---|---|---|
| **What** | Install as a skill in your CLI agent | Run the agent loop against any LLM API | Run inference with an RL-trained checkpoint |
| **GPU required?** | No | No | Yes |
| **Model** | Uses your CLI agent | Any model with API keys| Fine-tuned ProReviewer checkpoint |

---

## Option 1: As a Skill (No GPU)

Install the **ProReviewer as a Skill** with a CLI agent (Claude Code, Codex CLI, Gemini CLI).

```bash
# Clone the repo
git clone https://github.com/UKPLab/arxiv2026-ProReviewer.git

# Install for Claude Code (global)
cp -R arxiv2026-ProReviewer/ProReviewer-Skill/proreviewer ~/.claude/skills/

# Install for Codex CLI
mkdir -p ~/.agents/skills
cp -R arxiv2026-ProReviewer/ProReviewer-Skill/proreviewer ~/.agents/skills/
```

Then ask your agent to review a paper:

```
/proreviewer papers/submission.pdf
```

See the [Skill README](ProReviewer-Skill/README.md) for all installation options and supported formats (PDF, LaTeX, Markdown).

---

## Option 2: Via API Prompting (No GPU)

Run the ProReviewer agent loop with any LLM API (Claude, GPT, Gemini, DeepSeek, etc.) through litellm. Requires [full installation](#installation).

```bash
# 1. Configure API keys
cp config.toml.example config.toml
# Edit config.toml — add your API keys (see format below)

# 2. Run inference
python run_inference.py \
    --model "anthropic/claude-opus-4" \
    --test_data data/test_data \
    --output_dir outputs/reviews \
    --max_samples 1 --n_runs 1 --max_steps 25

# 3. Evaluate generated reviews
python evaluation.py \
    --mode score \
    --reviews_dir outputs/reviews \
    --triplets_dir data/test_data \
    --output_dir outputs/eval
```

**`config.toml` format** — define named model configs with API keys:

```toml
[deepseek-reasoner]
model = "deepseek/deepseek-reasoner"
api_key = "your-deepseek-api-key-here"

[gpt-52]
model = "openai/gpt-5.2"
api_key = "your-openai-api-key-here"
```

You can then reference configs by name: `--model "deepseek-reasoner"`.

---

## Option 3: With a Trained Model (GPU Required)

Run inference using a ProReviewer checkpoint trained via RL (GRPO). This runs the full multi-turn agent-environment loop with the fine-tuned model.

Requires [full installation](#installation) including CUDA and the verl/rLLM backends.

```bash
python run_inference.py \
    --model <path-to-checkpoint> \
    --test_data data/test_data \
    --output_dir outputs/reviews \
    --n_runs 4 --max_steps 25
```

---

## Installation

> **Note:** If you only want the Skill (Option 1), you don't need the steps below — see [Option 1](#option-1-as-a-skill-no-gpu).

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (for RL training and local model inference)

### Steps

```bash
git clone https://github.com/UKPLab/arxiv2026-ProReviewer.git
cd arxiv2026-ProReviewer

# Create and activate the virtual environment
micromamba create -f environment_rllm.yml
micromamba activate rllm

# Install the verl backend
git clone https://github.com/volcengine/verl.git
cd verl && git checkout v0.6.1
uv pip install -e .
cd ..

# Install the modified rLLM framework (with step-level GRPO support)
cd rllm && uv pip install -e .
cd ..

# Configure API keys (needed for API-based inference and LLM-judge rewards)
cp config.toml.example config.toml
# Edit config.toml with your API keys
```

---

## Training

Training follows a three-stage pipeline: **SFT warm-start → Stage 1 RL → Stage 2 RL**. Ready-to-run scripts are provided in [`reviewer/scripts/`](reviewer/scripts/).

| Script | Stage | Rewards | Extra requirements |
|---|---|---|---|
| `train_sft.sh` | SFT warm-start | — (supervised) | SFT traces from `reviewer/sft` |
| `train_stage1_format_score.sh` | Stage 1 RL | syntactic + format + score_diff (rule-based) | Embedding server for duplicate detection |
| `train_stage2_review_quality.sh` | Stage 2 RL | Stage 1 + rubric (LLM-as-judge) | LLM judge vLLM server on `localhost:8000` |

The RL stages use GRPO via rLLM. We extend rLLM with **step-level GRPO** — advantages are computed per step across all trajectories of the same task, rather than per trajectory. See [`rllm/change.md`](rllm/change.md) for details.

### Step 1: SFT Warm-Start

Converts JSON traces (from `reviewer.sft.generate_review_sft_data`) into MDP-decomposed parquet files, then runs multi-turn SFT with rLLM's `AgentSFTTrainer`:

```bash
MODEL_PATH="Qwen/Qwen3-8B" \
TRACES_DIR=outputs/sft_traces \
N_GPUS=8 \
bash reviewer/scripts/train_sft.sh
```

### Step 2: Stage 1 RL — Format and Score Calibration

Rule-based rewards only (no LLM judge). The agent learns tool usage, review structure, and score calibration:

```bash
MODEL_PATH=<sft_checkpoint_or_hf_id> \
DATA_PATH=data/ \
N_GPUS=8 \
bash reviewer/scripts/train_stage1_format_score.sh
```

The script prepares the dataset automatically (via `reviewer.sft.prepare_review_dataset`) if it doesn't exist yet.

### Step 3: Stage 2 RL — Review Quality

Resumes from a Stage 1 checkpoint and adds LLM-as-judge rubric rewards for technical depth and grounding specificity. Start an LLM judge vLLM server on `localhost:8000` first, then:

```bash
STAGE1_CKPT_DIR=outputs/checkpoints/stage1_format_score \
STAGE1_STEP=20 \
STAGE1_EPISODES_DIR=logs/proreviewer/<stage1_run> \
bash reviewer/scripts/train_stage2_review_quality.sh
```

The script automatically merges the FSDP Stage 1 checkpoint, verifies the judge server is reachable, and builds a curriculum dataset that excludes papers already seen in Stage 1 (set `STAGE1_EPISODES_DIR` to enable exclusion).

**Common knobs** (all scripts): `N_GPUS`, `RUN_NAME`, `lr`, `ROLLOUT_TP` — override as environment variables.


## Project Structure

```
ProReviewer/
├── run_inference.py              # Inference entry point (API or local model)
├── evaluation.py                 # Evaluation entry point
├── train.py                      # RL training entry point
├── config.toml.example           # API key configuration template
│
├── ProReviewer-Skill/            # Standalone skill for CLI agents
│
├── reviewer/                     # Core system
│   ├── core/                     # Agent, environment, memory, verification
│   ├── reward/                   # Multi-dimensional reward system
│   ├── prompts/                  # Prompt templates
│   ├── sft/                      # SFT data generation
│   └── scripts/                  # Training scripts (SFT, Stage 1, Stage 2)
│
├── rllm/                         # RL framework (modified rLLM)
│
└── utils/                        # LLM API wrapper, token tracking, data pipeline
```

---

## Citation

```bibtex
@article{fang2026passive,
  title={From Passive Generation to Investigation: A Proactive Scientific Peer Review Agent},
  author={Fang, Haishuo and Feng, Yue and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2606.13349},
  year={2026}
}
```
