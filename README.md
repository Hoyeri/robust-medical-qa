# Robust Medical QA

## Analyses

| Analysis | Question | Result | Details |
|---|---|---|---|
| Attention blocking | Does blocking attention to a known distractor recover correct answers? | On the same 96 questions, correct answers increased from 41 at baseline to 55 with option-receiver blocking and 73 with all-downstream blocking. | [Methods and results](docs/01_attention_blocking.md) |
| Attribution probe | Do representations distinguish the patient's finding from the same finding attributed to a third party? | Finding-token representations at hidden-state indices 8–16 achieved 99.69–100% classification accuracy on 53 pairs. | [Methods and results](docs/02_role_probe.md) |
| Transmission audit | Is attribution information preserved through value projection and attention? | Attribution remained decodable in transmitted contributions. The confidence intervals for aggregate M/H1 diagnostic-score sensitivity differences included zero. | [Methods and results](docs/03_transmission.md) |

## Run the analysis

Verify the files and recompute outcome counts using Python 3:

```bash
python3 scripts/verify_release.py
python3 scripts/reproduce_tables.py
```

Recompute the probe from saved hidden states:

```bash
python3 -m pip install -r requirements-analysis.txt
python3 scripts/reproduce_probe.py --hidden-dir /path/to/hidden_v5 --output /tmp/role_probe_recheck.json
```

## Structure

- `docs/`: methods, sample selection, results, and notation
- `results/`: aggregate results and per-question outcome labels
- `scripts/`: file verification, table reconstruction, and CPU probe analysis
- `reference/`: selected experiment code, with original computational logic
- `provenance/`: source paths, SHA-256 hashes, and file manifest

See [Reproducibility](docs/REPRODUCIBILITY.md) for data and runtime requirements.

[Notation](docs/NOTATION.md) / [Related work](docs/REFERENCES.md)
