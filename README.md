# Robust Medical QA

Code for constructing MedDistractQA-Hard, an unreleased research version that strengthens MedDistractQA, evaluating models, and analyzing internal representations. Bystander is the primary analysis setting; Nonliteral is also supported.

```text
experiments/  Experiment arguments, execution, and analysis
common/       Shared computations, model calls, configuration, and tests
docs/         Experiment references
```

## Installation

Use Python 3.10 or later. Install the attention/probe analysis environment with:

```bash
pip install -e '.[inference]'
```

Dataset construction and model evaluation require CUDA and vLLM 0.24.0. GPT-OSS scoring requires Transformers, accelerate, kernels, Triton 3.4 or later, and a GPU that supports MXFP4. Supply the original dataset and model weights separately. Each script also lists its arguments with `--help`.

## Dataset construction

Generate eight candidates per incorrect answer option, validate them under two option orders, and select Hard examples using GPT-OSS-120B scores. For both Bystander and Nonliteral, only candidate slots rejected by the option-association check are regenerated, for up to two additional rounds.

```bash
python experiments/build_dataset.py --type bystander --input data/clean.jsonl --split internal --output outputs/bystander
python experiments/build_dataset.py --type nonliteral --input data/clean.jsonl --split internal --output outputs/nonliteral
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | Required | JSONL file containing the original MedQA questions |
| `--type` | Required | Distractor family to generate: `bystander` / `nonliteral` |
| `--split` | `internal` | Dataset label assigned to all input questions: `dev` / `internal` / `test` |
| `--retry-rounds` | `2` | Additional generation rounds for rejected candidate slots: `0` / `1` / `2` |
| `--output` | Required | Output directory |
| `--generation-python` | Current Python | Python executable in the vLLM environment used for candidate generation/validation |
| `--scoring-python` | Current Python | Python executable in the Transformers/MXFP4 environment used for GPT-OSS scoring |
| `--resume` | Off | Resume saved work with the same inputs and settings |

The input JSONL uses `idx`, `source_id`, `question`, `options` (A–D), and `answer_idx`. Generated data is saved as `meddistractqa-hard-bystander.jsonl` or `meddistractqa-hard-nonliteral.jsonl`, with the original MedQA question and its Hard version stored together. The pipeline also saves `views.jsonl`, `selection.jsonl`, and `summary.json`.

## Model evaluation

Compare three versions of the same original questions: MedQA (Clean, without a distractor), official MedDistractQA, and MedDistractQA-Hard. The official data is downloaded and cached automatically from [Hugging Face](https://huggingface.co/datasets/KrithikV/MedDistractQA), then matched to the Hard input by checking the original question, options, and correct answer.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/evaluate_models.py --input outputs/bystander --official bystander --model llama31_8b_instruct --output outputs/evaluation
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | Required | Dataset construction output directory, or a `meddistractqa-hard-bystander.jsonl` / `meddistractqa-hard-nonliteral.jsonl` file |
| `--official` | Omitted | Official comparison family: `bystander` / `nonliteral`. Use the same family as the Hard input. If omitted, evaluate only MedQA (Clean) and MedDistractQA-Hard |
| `--model` | `llama31_8b_instruct` | Model configuration key. See `primary_model_keys` in the [model configuration](common/model_eval/models.json) for supported models |
| `--output` | Required | Evaluation output directory |
| `--resume` | Off | Resume saved work with the same inputs and settings |

When given a construction output directory, the script locates the dataset file automatically. The Hard input determines which questions are evaluated. If an official counterpart cannot be found, execution stops before inference.

After evaluation, the terminal displays question counts, correct/wrong/unparseable answer counts, accuracy, and the accuracy difference from Clean for each condition. The same table is saved as `summary.txt` and `summary.csv`, detailed aggregates as `summary.json`, and individual responses as `predictions.jsonl`. Unparseable answers remain in the accuracy denominator.

## Attention blocking

Generate rationales and answers with Llama-3.1-8B-Instruct while blocking attention to the distractor or a token-count-matched source from the original clinical text. Use the Hard data generated above; the code computes source token positions automatically.

```bash
python experiments/attention_blocking.py --input outputs/bystander --model models/llama31-8b-instruct --device cuda --dtype bfloat16 --output outputs/attention
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | Required | Dataset construction output directory, or a `meddistractqa-hard-bystander.jsonl` / `meddistractqa-hard-nonliteral.jsonl` file |
| `--model` | Required | Local directory containing Llama model weights and configuration |
| `--tokenizer` | Model directory | Local tokenizer directory, if separate from the model |
| `--device` | `cpu` | Execution device, such as `cpu`, `cuda`, or `cuda:0` |
| `--dtype` | `float32` | Model computation precision: `float32` / `float16` / `bfloat16` |
| `--config` | [Default configuration](common/configs/attention_blocking.json) | Experiment JSON specifying blocking conditions, target layers, and related settings |
| `--output` | Required | Output directory |
| `--resume` | Off | Resume saved work with the same inputs and settings |

Outputs are `outcomes.csv` and `summary.json`. All questions in the input file are evaluated.

## Patient/third-party attribution probe

Extract hidden states at finding tokens and the answer position, then classify patient versus third-party conditions using standardization, PCA, and logistic regression. The repository includes the [118-item attribution dataset](common/attribution_data.jsonl) used in the original experiments. Patient (Ma)/third-party (H1) comparisons use the 53 questions with both variants. Omitting `--input` selects the included data.

```bash
python experiments/attribution_probe.py --model models/llama31-8b-instruct --device cuda --dtype bfloat16 --output outputs/attribution
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | [Included analysis data](common/attribution_data.jsonl) | Optional alternative JSONL file in the same format |
| `--model` | Required | Local Llama model directory used to extract representations |
| `--tokenizer` | Model directory | Local tokenizer directory, if separate from the model |
| `--device` | `cpu` | Model execution device, such as `cpu`, `cuda`, or `cuda:0` |
| `--dtype` | `float32` | Model computation precision: `float32` / `float16` / `bfloat16` |
| `--config` | [Default configuration](common/configs/attribution_probe.json) | JSON specifying extraction settings, hidden-state indices, and classifier/holdout settings |
| `--output` | Required | Directory for extracted representations and analysis results |
| `--resume` | Off | Resume saved work with the same inputs and settings |

The script runs extraction through classification/holdout evaluation and saves `results.json`.

## Attention transmission analysis

Extract H, V, source contributions per head, and contributions after the output projection from the included attribution data. Measure attribution decoding from each representation and diagnostic-score sensitivity.

```bash
python experiments/transmission.py --model models/llama31-8b-instruct --device cuda --dtype bfloat16 --output outputs/transmission
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | [Included analysis data](common/attribution_data.jsonl) | Optional alternative JSONL file in the same format |
| `--model` | Required | Local Llama model directory used to extract transmitted contributions |
| `--tokenizer` | Model directory | Local tokenizer directory, if separate from the model |
| `--device` | `cpu` | Model execution device, such as `cpu`, `cuda`, or `cuda:0` |
| `--dtype` | `float32` | Model computation precision: `float32` / `float16` / `bfloat16` |
| `--config` | [Default configuration](common/configs/transmission.json) | JSON specifying extraction, classification, sensitivity contrasts, and bootstrap settings |
| `--output` | Required | Directory for extracted representations and analysis results |
| `--resume` | Off | Resume saved work with the same inputs and settings |

Results are saved as `results.json`. Questions with additional Ua/Pg/Oa variants are included in the corresponding analyses.

## Head search

Run mask training, mask evaluation, and individual-head evaluation for janthonio03's Llama-3.1-8B base experiment. Exact reproduction requires the original `train.jsonl` / `dev.jsonl` / `test.jsonl` files (89/11/11 questions) and the complete `lsld-v1.0.0` code directory from janthonio03. The included snapshot contains seven files; the split data and some execution modules are not included.

```bash
python experiments/head_search.py --input data/head_search --code-dir /path/to/lsld-v1.0.0 --output outputs/heads
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | Required | Directory containing the three split files obtained from janthonio03 |
| `--code-dir` | Required | Complete code directory obtained from janthonio03, containing `src/lsld_repro/` |
| `--output` | Required | Directory for mask training and evaluation results |
| `--resume` | Off | Resume evaluation after completed training. If training was interrupted, use a new output directory |

Outputs are `head_search_report.json`, `best_mask.json`, `mask/`, and `heads/`. The original snapshot is in `common/vendor/`.

[Experiment references](docs/references.md)
