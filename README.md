# Robust Medical QA

Code for constructing MedDistractQA-Hard, an unreleased research version that strengthens MedDistractQA, evaluating models, and analyzing internal representations. Bystander is the primary analysis setting; Nonliteral is also supported.

```text
pipeline/     Dataset construction and benchmark evaluation
experiments/  Attention blocking, attribution probing, transmission, and head search
common/       Shared computations, model calls, configuration, and tests
docs/         Experiment references
```

## Installation

Use Python 3.12 on a CUDA-capable Linux system. Install the single environment used by dataset construction, model evaluation, and all analyses with:

```bash
pip install -r requirements.txt
```

The requirements fix the shared PyTorch, Transformers, vLLM, accelerate, kernels, and Triton versions. The host must provide a compatible NVIDIA driver. GPT-OSS-120B scoring additionally requires an MXFP4-capable GPU. MedQA and official MedDistractQA are downloaded automatically from Hugging Face. Supply model weights separately where a local model is required. Each script also lists its arguments with `--help`.

## Dataset construction

Load MedQA from [Hugging Face](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf), then generate eight candidates per incorrect answer option, validate them under two option orders, and select Hard examples using GPT-OSS-120B scores. For both Bystander and Nonliteral, only candidate slots rejected by the option-association check are regenerated, for up to two additional rounds.

```bash
python pipeline/build_dataset.py --type bystander --split test --output outputs/bystander
python pipeline/build_dataset.py --type nonliteral --split test --output outputs/nonliteral
python pipeline/build_dataset.py --type bystander --split dev --fallback-mode fixed-template --output outputs/bystander-fullcoverage-v2
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | Omitted | Optional local MedQA JSONL. If omitted, download MedQA from Hugging Face |
| `--type` | Required | Distractor family to generate: `bystander` / `nonliteral` |
| `--split` | Automatic: `test` without `--input` / `internal` with `--input` | Without `--input`, select the HF split: `train` / `dev` / `test`. With `--input`, assign one of those labels or `internal` to the local data |
| `--retry-rounds` | `2` | Additional generation rounds for rejected candidate slots: `0` / `1` / `2` |
| `--fallback-mode` | `none` | `fixed-template` adds three explicitly labeled, non-gate-validated candidates only when a source has no gate-valid candidate after the retry budget |
| `--output` | Required | Output directory |
| `--generation-python` | Current Python | Python executable in the vLLM environment used for candidate generation/validation |
| `--scoring-python` | Current Python | Python executable in the Transformers/MXFP4 environment used for GPT-OSS scoring |
| `--resume` | Off | Resume saved work with the same inputs and settings |

The HF test input preserves the source IDs and iteration order of the existing construction run. An optional local input JSONL uses `idx`, `source_id`, `question`, `options` (A–D), and `answer_idx`. Generated data is saved as `meddistractqa-hard-bystander.jsonl` or `meddistractqa-hard-nonliteral.jsonl`, with the original MedQA question and its Hard version stored together. The pipeline also saves `views.jsonl`, `selection.jsonl`, and `summary.json`.

`--fallback-mode fixed-template` is a separate full-coverage V2 policy. It does not replace individual rejected slots. After all generation and retries finish, it acts only on a source with zero gate-valid candidates and adds one deterministic candidate for each wrong option. Bystander uses a quoted clinical topic repeated by a neighbor's parrot; Nonliteral uses the quoted topic as a nickname for the patient's current mood. These candidates bypass the option-association gate by design and are recorded as `candidate_origin=fixed_template_fallback`, `gate_validated=false`, and `must_not_be_used_for_training=true`. The selected pair retains the fallback trigger and template version. Reports should keep generated-only and fallback strata separate because the fallback repeats answer-option text directly.

Full-coverage outputs are named `meddistractqa-hard-bystander-fullcoverage-v2.jsonl` or `meddistractqa-hard-nonliteral-fullcoverage-v2.jsonl`. The run fails unless every input source is included. `summary.json` reports generated coverage, fallback source/candidate counts, final coverage, and the full-coverage check. For a fallback source, the only same-target fallback candidate is also the Hard candidate, so `random_same_target_control_collapsed=true` explicitly marks that the control is not distinct.

## Model evaluation

Compare three versions of the same original questions: MedQA (Clean, without a distractor), official MedDistractQA, and MedDistractQA-Hard. [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf) and [official MedDistractQA](https://huggingface.co/datasets/KrithikV/MedDistractQA) are downloaded and cached automatically. Match by the original question, options, and correct answer while retaining the Hard input IDs and order.

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline/evaluate_models.py --official bystander --model llama31_8b_instruct --output outputs/evaluation
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | Auto-detected | Use the generated Hard dataset in repository `outputs/bystander/` or `outputs/nonliteral/`. Specify a file/directory only when stored elsewhere or when selecting between multiple datasets |
| `--medqa-split` | `test` | HF MedQA split containing the original Hard questions: `train` / `dev` / `test` |
| `--official` | Omitted | Official comparison family: `bystander` / `nonliteral`. Use the same family as the Hard input. If omitted, evaluate only MedQA (Clean) and MedDistractQA-Hard |
| `--model` | `llama31_8b_instruct` | Model configuration key. See `primary_model_keys` in the [model configuration](common/model_eval/models.json) for supported models |
| `--output` | Required | Evaluation output directory |
| `--resume` | Off | Resume saved work with the same inputs and settings |

Without `--input`, look in `outputs/bystander/` and `outputs/nonliteral/` relative to the repository. `--official` selects the matching family; otherwise use the only available dataset. If no dataset is found or both remain eligible, the script asks for `--input PATH`. An explicit path takes precedence. The Hard input determines which questions are evaluated. If a MedQA counterpart or a requested official counterpart cannot be found, execution stops before inference.

After evaluation, the terminal displays question counts, correct/wrong/unparseable answer counts, accuracy, and the accuracy difference from Clean for each condition. The same table is saved as `summary.txt` and `summary.csv`, detailed aggregates as `summary.json`, and individual responses as `predictions.jsonl`. Unparseable answers remain in the accuracy denominator.

## Attention blocking

Generate rationales and answers with Llama-3.1-8B-Instruct while blocking attention to the distractor or a token-count-matched source from the original clinical text. Use the Hard data generated above; the code computes source token positions automatically.

```bash
python experiments/attention_blocking.py --model models/llama31-8b-instruct --device cuda --dtype bfloat16 --output outputs/attention
```

| Argument | Required/default | Description |
| --- | --- | --- |
| `--input` | Auto-detected | Use the generated Hard dataset in repository `outputs/bystander/` or `outputs/nonliteral/`. Specify a file/directory only when stored elsewhere or when selecting between multiple datasets |
| `--model` | Required | Local directory containing Llama model weights and configuration |
| `--tokenizer` | Model directory | Local tokenizer directory, if separate from the model |
| `--device` | `cpu` | Execution device, such as `cpu`, `cuda`, or `cuda:0` |
| `--dtype` | `float32` | Model computation precision: `float32` / `float16` / `bfloat16` |
| `--config` | [Default configuration](common/configs/attention_blocking.json) | Experiment JSON specifying blocking conditions, target layers, and related settings |
| `--output` | Required | Output directory |
| `--resume` | Off | Resume saved work with the same inputs and settings |

Without `--input`, use the only Hard dataset found in repository `outputs/bystander/` or `outputs/nonliteral/`. If both exist or neither exists, provide `--input PATH`. The selected path is printed before execution.

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
