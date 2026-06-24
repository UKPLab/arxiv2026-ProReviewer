---
license: apache-2.0
language:
- en
tags:
- reward-model
- scientific-writing
- evaluation
- reinforcement-learning
- text-generation
- grpo
pipeline_tag: text-generation
arxiv: 2601.11374
---

# SciRM / SciRM-Ref

[![Arxiv](https://img.shields.io/badge/Arxiv-2601.11374-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.11374)
[![GitHub](https://img.shields.io/badge/GitHub-UKPLab%2Farxiv2026--expert--rm-black?style=flat-square&logo=github)](https://github.com/UKPLab/arxiv2026-expert-rm)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

[**Paper**](https://arxiv.org/abs/2601.11374) | [**GitHub**](https://github.com/UKPLab/arxiv2026-expert-rm) | [**Dataset**](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/4980)

---

## Introduction

We present **SciRM** and **SciRM-Ref**, cost-efficient open-source reward models tailored for scientific writing evaluation. Our models are trained via a **two-stage reinforcement learning framework** (using GRPO) that first optimizes scientific evaluation preferences, then refines reasoning capabilities. The multi-aspect evaluation design and joint training across diverse tasks enable fine-grained assessment and robustness to dynamic criteria and scoring rubrics.

Key properties:
- ✅ **Multi-aspect evaluation** — assesses multiple evaluation dimensions per task
- ✅ **Dynamic scoring rubrics** — conditioned on an explicit evaluation constitution at both train and inference time
- ✅ **Cross-task generalization** — a single model handles diverse and previously unseen scientific writing tasks without task-specific retraining
- ✅ **Open-source and cost-efficient** — no reliance on proprietary LLMs

---

## Model Summary

| Property | Details |
|---|---|
| **Base LLM** | [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) |
| **Model Variants** | SciRM (Stage 1 only), SciRM-Ref (Stage 1 + Stage 2) |
| **Training Method** | Two-stage GRPO via [Unsloth](https://github.com/unslothai/unsloth) |
| **Evaluation Tasks (Seen)** | Related Work Section Generation, Scientific Review Writing |
| **Evaluation Tasks (Unseen)** | Novelty Evaluation Alignment, Paper Revision Evaluation |
| **License** | Apache 2.0 |
| **Paper** | [arXiv:2601.11374](https://arxiv.org/abs/2601.11374) |
| **Authors** | Furkan Şahinuç, Subhabrata Dutta, Iryna Gurevych (UKP Lab, TU Darmstadt) |

## Inference

Although SciRM and SciRM-Ref have increased reasoning capabitilies, they are not complete reasoning models. Therefore, adherence to the system prompt and providing a clear evaluation criteria is strongly recommended for best performance. SciRM is more stable than SciRM-Ref in terms of output formatting.

This is an example infrence for demonstration purposes. For more efficient implementation with full datasets please refer to our GitHub repo. 

```python
import torch
from vllm import LLM

SYSTEM_PROMPT = """\
You are an evaluator of expert-domain scientific writing. You will get a query-answer pair along with criteria explaining the specific evaluation aspect and the scoring rubric. You should evaluate whether the answer satisfy the query based on the given criteria. In addition, examples demonstrating how the evaluation should be performed will be provided. First output your reasoning enclosed between <reasoning> and </reasoning>. Then, output your score enclosed between <score> and </score>. Inside <score> provide only the numeric score and nothing else.
"""

QUERY = """\ 
[QUERY]: Your task is to write a review comment for a scientific paper. The comment should be actionable. Those actions should be clearly identifiable and concrete.\n\n\n
"""

# Scoring rubric shortened for brevity (Scores 2 and 4 removed).
CRITERIA = """\
[CRITERIA]: Explicit actions or suggestions are direct or apparent. Authors can directly identify modifications they should apply to their draft. Clarification questions should be treated as explicit statements if they give a direct action. However, implicit actions need to be inferred from the comment. This includes missing parts that need to be added. Authors can deduce what needs to be done after reading the comment. For concrete actions, the authors know exactly what needs to be done and how to apply the action. However, for vague actions the authors still don’t know how to carry out this action. Scoring rubric is as follows:\n1: The comment lacks meaningful information to help authors improve the paper. Authors do not know what they should do after reading the comment.\n3: The comment explicitly states an action but is vague on how to execute it.\n5: The comment contains an explicit action and concrete details on how to implement it. Authors know exactly how to apply it.\n\n\n
"""

# Examples shortened for brevity (Examples 2 and 4 removed).
EXAMPLES = """\n\n<START OF EXAMPLE 1>\n\nANSWER: The hGRU architecture seems pretty ad-hoc and not very well motivated.\n\nEVALUATION:\n\n<reasoning>The review comment, \"The hGRU architecture seems pretty ad-hoc and not very well motivated,\" lacks specificity and actionable guidance for the authors. While it expresses a concern about the hGRU architecture being \"ad-hoc\" and \"not very well motivated,\" it does not provide any detailed explanation or examples of why the reviewer perceives it this way. Without specific points or suggestions, the authors are left without a clear understanding of what aspects of the architecture need further clarification or improvement. hence, this comment is not actionable at all. Therefore the evaluation score should be 1.</reasoning>\n\n<score>1</score>\n\n<END OF EXAMPLE 1>\n\n\n<START OF EXAMPLE 3>\n\nANSWER: A number of claims from this paper would benefit from more in-depth analysis.\n\nEVALUATION:\n\n<reasoning>The comment points out that certain claims require more in-depth analysis but does not clarify which claims need further scrutiny. As a result, the authors may not know where to focus their efforts, leading to potential misinterpretation of the feedback. Since the suggested action is direct but still lacks the necessary details for precise implementation, this comment is somewhat actionable. Therefore the evaluation score should be 3.</reasoning>\n\n<score>3</score>\n\n<END OF EXAMPLE 3>\n\n\n<START OF EXAMPLE 5>\n\nANSWER: The abstract is written well and invokes intrigue early - could potentially be made even better if, for \"evaluating with gold answers is inconsistent with human evaluation\" - an example of the inconsistency, such as models get ranked differently is also given there.\n\nEVALUATION:\n\n<reasoning>The comment explicitly states that an example of inconsistency should be provided in the abstract, specifically where it mentions \"evaluating with gold answers is inconsistent with human evaluation.\" By directly instructing the authors to include an example, such as how models get ranked differently, it removes any uncertainty about how to proceed. Since the feedback is clear, specific, and directly actionable, the comment is fully actionable. Therefore the evaluation score should be 5.</reasoning>\n\n<score>5</score>\n\n<END OF EXAMPLE 5>\n\n\n
"""

EVALUATED_TEXT = """\
1. Table 2: the value \"9.2\" highlighted in the first column seems to be an error, as it is highlighted as the highest, which contradicts the data presented.
"""

model_path = "path_to_model"

model = LLM(model=model_path, dtype=torch.bfloat16, max_model_len=max_model_len, trust_remote_code=True)

sampling_params = model.get_default_sampling_params()
sampling_params.max_tokens = 2048
sampling_params.temperature = 1
sampling_params.top_p = 0.95

user_message = QUERY + CRITERIA + EXAMPLES + "[ANSWER]:" + EVALUATED_TEXT

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": user_message},
]

completions = model.chat(messages, sampling_params)

output = completions[0].outputs[0].text

print(output)
```

## Limitations

- The model is trained primarily on English scientific text from NLP/CS domains; performance may degrade on other scientific fields or languages.
- Evaluation quality depends on the quality and specificity of the provided evaluation constitution. Vague or incomplete criteria may yield less reliable scores.
- The model should not be the sole arbiter of quality in high-stakes scientific publishing decisions.

---

## Citation

If you use this model in your work, please cite:

```bibtex
@misc{sahinuc2026reward,
    title     = {Reward Modeling for Scientific Writing Evaluation},
    author    = {Furkan {\c{S}}ahinu{\c{c}} and Subhabrata Dutta and Iryna Gurevych},
    year      = {2026},
    eprint    = {2601.11374},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
    url       = {https://arxiv.org/abs/2601.11374}
}
```
---

## Contact

✉️ Contact person: [Furkan Şahinuç](mailto:furkan.sahinuc@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send an e-mail or open a GitHub issue if something is broken or if you have further questions.

---
