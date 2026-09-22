# Experiment code

Selected files from the original experiments:

- `attention/`: E08 free generation, E11 matched clinical-source control, and attention blocking
- `role/`: hidden-state extraction, attribution probes, and holdout analysis
- `transmission/`: H/V/contribution extraction, probing, and sensitivity analysis

The two transmission files have reformatted mathematical notation in comments and output strings. Computational logic is unchanged. Original hashes and reformatted-copy hashes are recorded in `../provenance/sources.json`.

The GPU scripts use the original server paths, execution guards, and external project modules. See [Reproducibility](../docs/REPRODUCIBILITY.md) for required resources and `../scripts/` for portable result verification and CPU analysis.
