# Entrainment head search

## Source

janthonio03's experiment snapshot is included at [external/lsld-v1.0.0-medqa](../external/lsld-v1.0.0-medqa/README.md). [Source manifest](../provenance/entrainment_source.json) records each file's SHA-256.

## Experiment setup

The upstream README specifies `meta-llama/Llama-3.1-8B` (base model). The medical analysis set consists of 111 examples selected because the clean prompt predicts G and the hard-distractor prompt predicts T. Joint mask search uses 89 train / 11 dev / 11 test examples; the adapter checks these counts and rejects duplicate IDs across splits.

The prompt ends in `Answer:`. Medical evaluation compares the next-token logits for A/B/C/D; it does not generate a rationale. The all-111 mask evaluation and exhaustive scan combine the original three splits, so those outputs include training and development examples.

## Included files

| File | Purpose |
|---|---|
| [run_medqa_head_search.py](../external/lsld-v1.0.0-medqa/scripts/run_medqa_head_search.py) | Adapt joint head-mask search to the 89/11/11 split, with microbatch gradient accumulation. |
| [evaluate_medqa_all111_mask.py](../external/lsld-v1.0.0-medqa/scripts/evaluate_medqa_all111_mask.py) | Apply a saved mask and evaluate Clean/Hard predictions on all 111 examples. |
| [scan_medqa_entrainment_heads.py](../external/lsld-v1.0.0-medqa/scripts/scan_medqa_entrainment_heads.py) | Ablate candidate heads one at a time and save per-head/per-example effects. |
| [scan_medqa_all111_heads.py](../external/lsld-v1.0.0-medqa/scripts/scan_medqa_all111_heads.py) | Scan all 1024 heads on the combined 111 examples. |
| [paper_head.py](../external/lsld-v1.0.0-medqa/src/lsld_repro/paper_head.py) | Train masks, select a checkpoint, and produce reports. |
| [hooked_transformers.py](../external/lsld-v1.0.0-medqa/src/lsld_repro/official_code/head_search/circuit_lms/hooked_transformers.py) | Runtime adaptation, including optional last-position-only unembedding. |

## Joint masks and individual-head scores

Joint search learns head masks. The included driver optimizes a Gold-vs-Target cross-entropy term with a head-retention term, then selects an epoch using `dev_gap + kept_heads * HEAD_SELECTION_WEIGHT`. The coefficient is imported from the missing `protocol.py`; the driver also retains the upstream last-batch development aggregation for selection and separately records full-development diagnostics.

Single-head scanning uses a different score. Let `m = logit(G) - logit(T)` and let Δ denote the change after ablating one head:

```text
E = m_clean - m_hard
score = E_baseline - E_ablated
      = Δm_hard - Δm_clean
```

The scan also reports Clean/Hard accuracy and the two margin changes separately. A high score can reflect a Hard improvement, a Clean margin decrease, or both; the separate measurements explain the ranking.

## Execution requirements

This upstream folder is a compact result-producing code snapshot. It does not include a complete v1.0.0 package, dataset splits, trained masks, or numerical result artifacts. The imported scripts have not been executed against the full model in this repository.

The included files import missing local modules: `lsld_repro.data`, `io_utils`, `paper_head_data`, `protocol`, `tokenization`, `official_code.head_search.head_analysis`, and `official_code.head_search.circuit_lms.transformer_blocks`. Visible third-party imports include PyTorch, NumPy, SciPy, Hugging Face datasets, TransformerLens, and einops. A matching v1.0.0 environment and its dependency versions are needed from the upstream author; the earlier reproduction tree is not substituted for the missing modules.

Inputs are `train.jsonl`, `dev.jsonl`, and `test.jsonl`. Rows supply `clean_question`, `distracted_question`, A–D `options`, `gold_answer`, and `intended_target`; `question_id` or `example_id` identifies each example, and `added_distractor` is optional. Mask evaluation additionally requires `best_mask.json`.

Once the complete v1.0.0 package and data are available, the original entry points accept the following arguments. Run these inside that environment:

```bash
python scripts/run_medqa_head_search.py --data-dir /path/to/splits --output-dir /path/to/new_search --seed 0 --epochs 500 --batch-size 1 --grad-accum-steps 16
python scripts/evaluate_medqa_all111_mask.py --data-dir /path/to/splits --best-mask /path/to/best_mask.json --output-dir /path/to/mask_eval --seed 0
python scripts/scan_medqa_all111_heads.py --data-dir /path/to/splits --output-dir /path/to/head_scan --seed 0
```

The import was checked for exact source hashes and Python syntax. Existing package installation and tests continue to target `robust_medical_qa/` and `experiments/`; this snapshot has its own runtime requirements.
