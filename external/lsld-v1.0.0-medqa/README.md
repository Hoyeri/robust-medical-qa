# LSLD v1.0.0 — Medical hard-distractor experiments

This directory preserves the final medical-experiment code snapshot used after the earlier LSLD paper reproduction.

The original `lsld-reproduction/` directory in this repository is the earlier v0.4 paper-reproduction tree. The medical experiments were run later from the server environment:

```text
/nas2/data/janthonio03/robust_medqa/med_prm/lsld_v1.0.0
```

The files committed here are the result-producing v1.0.0 experiment files supplied from that environment. This is a compact experiment snapshot rather than a complete standalone export of every v1.0.0 file.

## Included code

- `scripts/run_medqa_head_search.py` — adapts the LSLD head-mask search to the 89/11/11 medical split and supports microbatch gradient accumulation.
- `scripts/evaluate_medqa_all111_mask.py` — applies the selected joint head mask to all 111 clean-to-hard target-flip examples and evaluates restricted A/B/C/D logits.
- `scripts/scan_medqa_entrainment_heads.py` — performs single-head ablation and computes per-head/per-example entrainment effects.
- `scripts/scan_medqa_all111_heads.py` — exhaustively evaluates all 32 x 32 = 1024 attention heads on all 111 examples.
- `src/lsld_repro/paper_head.py` — v1.0.0 experiment-driver snapshot used by the medical runs, including opt-in gradient accumulation.
- `src/lsld_repro/official_code/head_search/circuit_lms/hooked_transformers.py` — v1.0.0 runtime snapshot with the opt-in last-position-only unembedding path used for long medical prompts.

## Core protocol represented by these files

- Base model: `meta-llama/Llama-3.1-8B`
- Medical analysis set: 111 examples for which the clean prompt selects the gold answer and the hard-distractor prompt selects the intended hard target.
- Initial split for joint head search: 89 train / 11 dev / 11 test.
- Multiple-choice evaluation: next-token A/B/C/D logit comparison at the final prompt position after `Answer:`.
- Single-head analysis: ablate one attention head at a time while keeping the other 1023 heads active.
- Entrainment score: reduction in the clean-to-hard Gold-vs-Target preference shift after head ablation.

## Repository layout

```text
medqa_robustness/
├── lsld-reproduction/      # earlier v0.4 paper-reproduction code
└── lsld-v1.0.0-medqa/      # final medical-experiment snapshot
```

## Excluded intentionally

Datasets, checkpoints, learned masks, generated outputs, cache files, OOM/debug artifacts, backup files, and ad-hoc result-filtering snippets are not committed here.

If a fully standalone v1.0.0 reproduction environment is needed later, the remaining source/package files from the server-side `lsld_v1.0.0` directory should be added separately.
