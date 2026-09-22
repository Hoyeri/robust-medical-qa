# Reproducibility

Run commands from the repository root. CPU probing requires Python 3.9+ and NumPy. Model execution uses PyTorch and Transformers 4.57.3.

## Required inputs

| Task | Inputs | Included here? |
|---|---|---|
| Verify counts and files | Outcome CSV, summaries, manifests | Yes |
| Recompute attribution probes | `Ma.npz`, `H1.npz`; item metadata for holdouts | External arrays; one input example included |
| Recompute transmission analysis | Transmission NPZs, `alignment.json`, item metadata | External |
| Extract model representations | Input JSONL, local model/tokenizer | One example JSONL; model external |
| Rerun attention blocking | Tokenized attention input manifest, local model/tokenizer | External |

The original inputs and activations are in the author's research workspace. Collaborators need a separate transfer from the repository owner; this repository does not have an artifact download endpoint. Source paths and hashes are in `provenance/sources.json`. See [Example inputs](../examples/README.md) for schema details.

## Install

```bash
python3 -m pip install -r requirements-analysis.txt
python3 -m pip install -e .
# For model extraction/generation and model integration tests:
python3 -m pip install -e '.[inference]'
```

The refactor was tested with NumPy 2.0.2, PyTorch 2.8.0, and Transformers 4.57.3. Model paths refer to local snapshots; device and dtype are explicit options.

## Verify stored results

```bash
python3 scripts/verify_release.py
python3 scripts/reproduce_tables.py
python3 scripts/reproduce_probe.py \
  --hidden-dir /path/to/hidden_v5 \
  --output /tmp/probe_recheck.json
```

The first two commands use the committed results. Probe recalculation requires external `Ma.npz` and `H1.npz` files with `phrase`, `answer`, and `item_ids` arrays. Frozen-result checks compare against `results/role_probe_exact.json`.

## Run probes on another sample

```bash
python3 -m experiments.attribution_probe \
  --hidden-dir /path/to/hidden_arrays \
  --config configs/attribution_probe.json \
  --output /tmp/probe.json

python3 -m experiments.attribution_probe \
  --hidden-dir /path/to/hidden_arrays \
  --items /path/to/items.jsonl \
  --config configs/attribution_holdouts.json \
  --output /tmp/holdouts.json
```

Set variants, indices, PCA components, folds, seeds, and preprocessing in the config. Counts are derived from paired input IDs. Holdouts use `meta.cue.hpo` and attribution-family metadata or the existing templates.

## Extract representations

```bash
python3 -m experiments.extract_representations \
  --kind hidden \
  --model /path/to/llama-snapshot --tokenizer /path/to/tokenizer \
  --device cuda:0 --dtype bfloat16 \
  --items /path/to/items.jsonl \
  --config configs/hidden_extraction.json \
  --output outputs/hidden

python3 -m experiments.extract_representations \
  --kind transmission \
  --model /path/to/llama-snapshot --tokenizer /path/to/tokenizer \
  --device cuda:0 --dtype bfloat16 \
  --items /path/to/items.jsonl \
  --config configs/transmission_extraction.json \
  --validate 5 --output outputs/transmission
```

The JSONL schema uses `item_id`, `options`, and `variants`. Transmission additionally uses `gold_G` and `target_T`. Each variant is a complete question string. The original dataset uses C/Ma/H1/Oa/Pg/Ua; select the variants in the config. Explicit source annotations are described in [Code structure](CODE_STRUCTURE.md).

Model dimensions are read from the model configuration, and answer token IDs are derived and checked against the configured answer prefix. The default prompt is in `configs/prompt.json`. Extraction writes NPZ arrays, per-variant counts, missing-variant records, input hashes, and runtime settings.

## Analyze transmission

```bash
python3 -m experiments.transmission_audit \
  --results-dir outputs/transmission \
  --items /path/to/items.jsonl \
  --config configs/transmission_analysis.json \
  --output /tmp/transmission_analysis.json
```

The analysis accepts original transmission artifacts as well as newly extracted arrays. It reads `Ma.npz`, `H1.npz`, and `alignment.json`, plus optional `validation.json`. Validation output reports the alpha stored in each record.

## Run attention blocking

Convert the original distractor and clinical-control prepared inputs once:

```bash
python3 -m experiments.prepare_attention \
  --distractor-input /path/to/distractor_generation.json \
  --clinical-control-input /path/to/clinical_control.json \
  --output /tmp/attention_inputs.json

python3 -m experiments.attention_blocking \
  --inputs /tmp/attention_inputs.json \
  --config configs/attention_blocking.json \
  --model /path/to/llama-snapshot --tokenizer /path/to/tokenizer \
  --device cuda:0 --dtype bfloat16 \
  --output outputs/attention
```

A prepared input contains `generation` settings and `rows`. Each row supplies `question_id`, `gold_answer`, `intended_target`, `prompt_ids`, input-option token positions in `options`, and named token-position lists in `sources`. Token inputs must come from the selected tokenizer. Each condition selects a source and receiver policy from the config. Generation uses full-prefix greedy decoding with exact-prefix replay checks.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_core.py' -v
python3 -m unittest discover -s tests -v
```

The full suite uses a small randomly initialized Llama with a locally created tokenizer. It runs entirely on CPU and downloads no model weights. Tests include source isolation, attention normalization, contribution reconstruction, gradient agreement, and cleanup after exceptions.

## Data and original implementations

Original questions, generated rationales, activation NPZ files, and model weights are stored externally. The historical experiments used `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`, BF16/eager attention, batch size 1, and no KV cache.

`reference/` retains the original implementations for comparison. New experiment entry points use only modules in this repository and the declared dependencies. Original hashes and source paths are recorded in `provenance/sources.json`.
