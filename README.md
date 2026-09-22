# Robust Medical QA

We use an unreleased, hardened version of MedDistractQA, internally referred to as MedDistractQA-Hard, with Bystander as the primary distractor type. Separate MedQA-based patient/third-party finding pairs are used for attribution probing and transmission analysis.

## Analyses

These analyses ask whether blocking a known distractor changes generated answers, whether the model represents patient/third-party attribution, and whether that attribution remains decodable in attention contributions. Attention blocking uses a separate sample from the paired attribution analyses.

| Analysis | Sample and evaluation | Main result | Details |
|---|---|---|---|
| Attention blocking | 96 questions, rationale and answer generated from scratch | Correct answers: 41 at baseline, 55 with input-option blocking, 73 with all-downstream blocking. | [Methods and results](docs/01_attention_blocking.md) |
| Attribution probe | 53 M/H1 pairs, finding representations and a fixed answer slot with an empty rationale | Finding-token hidden-state indices 8–16: 99.69–100% question-grouped classification accuracy. | [Methods and results](docs/02_role_probe.md) |
| Transmission audit | The same 53 M/H1 pairs, contributions to the fixed answer slot and diagnostic-score sensitivity | Attribution remained decodable after attention and output projection. Aggregate M/H1 sensitivity differences had confidence intervals including zero. | [Methods and results](docs/03_transmission.md) |

[Sample construction and example](docs/DATA.md) / [Notation](docs/NOTATION.md) / [Experiment references](docs/REFERENCES.md)

## Check stored results

From the repository root, using Python 3:

```bash
python3 scripts/verify_release.py
python3 scripts/reproduce_tables.py
```

These commands verify committed files and recompute counts from stored outcome labels.

## Rerun analyses or model experiments

Probe recalculation needs saved activation arrays. Model extraction and generation additionally need the input data and a local model snapshot. See [Reproducibility](docs/REPRODUCIBILITY.md) for installation, inputs, and commands. [Example inputs](examples/README.md) illustrate the data format.

## Structure

- `robust_medical_qa/`: shared model runtime, attention, extraction, probing, and evaluation
- `experiments/`: experiment entry points
- `configs/`: experiment settings and historical numerical profiles
- `docs/`: methods, sample construction, results, notation, and references
- `examples/`: a paired input example and input-format guide
- `results/`: aggregate results and per-question outcome labels
- `scripts/`: file verification, table reconstruction, and CPU probe analysis
- `tests/`: regression checks and CPU model integration tests
- `reference/`: historical implementations for comparison
- `provenance/`: original source paths, hashes, and validation records

[Code structure](docs/CODE_STRUCTURE.md)
