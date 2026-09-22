# Reproducibility

## Verify results

```bash
python3 scripts/verify_release.py
python3 scripts/reproduce_tables.py
```

These commands verify file hashes, recompute correct-answer and transition counts from 480 attention outcome records, and check probe-count arithmetic. The outcome CSV contains question IDs and labels. Transmission decoding values are stored with their original rounded output.

## Recompute the probe

Provide `hidden_v5/Ma.npz` and `hidden_v5/H1.npz` outside the repository:

```bash
python3 -m pip install -r requirements-analysis.txt
python3 scripts/reproduce_probe.py --hidden-dir /absolute/path/to/hidden_v5 --output /tmp/probe.json
```

The script aligns M/H1 item IDs and reads the `phrase` and `answer` arrays. It recomputes the selected indices and writes a new output file. Use a new filename for each run.

On 2026-09-22, NumPy 2.0.2 reproduced the per-seed correct counts for all 12 selected position/index combinations. The local run emitted matrix-multiplication RuntimeWarnings; intermediate finite-value checks and count comparisons passed. The script exits with an error on non-finite intermediates or differing counts. Numerical behavior can depend on the OS and BLAS implementation.

The full probe and holdout scripts are in `reference/role/`. Holdout analysis also requires the original items JSONL. Transmission analysis requires variant NPZ files, `alignment.json`, `RUN_INFO.json`, and the items JSONL.

## GPU extraction and generation

The scripts in `reference/` retain the original server paths, execution guards, and project-module dependencies. GPU execution requires the following resources:

| Experiment | Resources |
|---|---|
| Attention E08/E11 | 0902 baseline/runtime/tokenizer, E02/E03/E05/E06 input and result bindings, MedQA-based paired inputs, model snapshot |
| Hidden-state extraction | cm_confirmatory_v5.jsonl, tokenizer, ariel harness/common dependencies, model snapshot |
| Transmission | Extraction resources, eager-attention implementation, automatic differentiation runtime |

Model: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`. BF16, eager attention, batch size 1, KV cache disabled. `requirements-analysis.txt` specifies the CPU analysis dependency.

## Artifacts

The repository contains selected experiment code and derived results. Source questions, generated rationales, activation NPZ files, and model weights are stored externally. Original source paths and hashes are recorded in `provenance/sources.json`.
