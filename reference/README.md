# Historical experiment code

Historical snapshots used to compare the refactored implementations against the original experiments:

- `attention/`: distractor blocking during generation, matched clinical-source control, and attention blocking
- `role/`: hidden-state extraction, attribution probes, and holdout analysis
- `transmission/`: H/V/contribution extraction, probing, and sensitivity analysis

The two transmission files have reformatted mathematical notation in comments and output strings. The clinical-control snapshot also updates an import to the renamed distractor script. Computational logic is unchanged. Original hashes and reformatted-copy hashes are recorded in `../provenance/sources.json`.

The GPU scripts use the original server paths, execution guards, and external project modules. See [Reproducibility](../docs/REPRODUCIBILITY.md) for required resources and `../scripts/` for portable result verification and CPU analysis.

Active entry points are in `experiments/`; shared implementations are in `robust_medical_qa/`. See [Code structure](../docs/CODE_STRUCTURE.md).
