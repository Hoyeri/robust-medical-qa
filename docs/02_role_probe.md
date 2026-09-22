# Attribution probe

## Inputs

For each of 53 MedQA-based questions, the same finding selected to support competing answer T was attributed either to the patient (M, stored as Ma) or to a third party (H1). This produced 106 inputs across 53 independent questions and 35 finding concepts. The 53 pairs are the subset of 118 constructed questions eligible for third-party attribution. [Sample construction and an actual input pair](DATA.md) explain the eligibility rule and the distinction between G and T.

We extracted two 4096-dimensional representations from the frozen model:

| Position | Representation |
|---|---|
| `phrase` | Mean hidden state of tokens in the clinical phrase shared by M and H1 |
| `answer` | Hidden state of the final input token immediately before answer prediction |

The answer position used a fixed JSON answer prefix with an empty rationale (`rs`).

## Classifier

Standardization → PCA with 24 components → L2 logistic regression.

A separate classifier was fitted for each position, hidden-state index, and training fold. Five-fold cross-validation was grouped by original question, keeping each M/H1 pair in the same fold. Standardization and PCA were fitted on the training fold. Seeds were 0, 1, and 2.

For each seed, correct predictions across the five test folds were summed and divided by 106. The reported accuracy is the mean across three seeds, representing 318 repeated predictions on 53 independent questions.

## Question-grouped cross-validation results

| Position | Hidden-state index | Block interpretation, zero-based | Correct /106 by seed | Mean accuracy |
|---|---:|---|---|---:|
| Finding | 0 | Input embedding | 53, 53, 53 | 50% |
| Finding | 8 | Block 7 output | 106, 106, 106 | 100% |
| Finding | 9 | Block 8 output | 106, 106, 105 | 99.6855% |
| Finding | 10–16, each | Block 9–15 output | 106, 106, 106 | 100% |
| Answer | 12 | Block 11 output | 82, 85, 83 | 78.6164% |
| Answer | 32 | Final normalization after block 31 | 54, 59, 58 | 53.7736% |

These counts were recomputed on CPU from saved activations on 2026-09-22 using the original classifier operations. The original two-decimal output rounded the index-9 accuracy to `1.00`.

Patient/third-party attribution was linearly decodable from the finding representations. Answer-position decoding accuracy was higher at index 12 than at index 32.

## Additional holdouts

- Concept holdout: training and evaluation used disjoint HPO concepts.
- Template holdout: coworker questions (39) were evaluated against the remaining templates (14: classmate 5, another child 5, another baby 4), in both training directions.
- Templates followed the patient's age in the source question, so template transfer also involved adult/pediatric question differences.

These holdouts were evaluated separately from the question-level cross-validation table above. Related holdout analyses are described in the [transmission audit](03_transmission.md).

## Code and results

- [Current extraction](../experiments/extract_representations.py) / [Current probe](../experiments/attribution_probe.py)
- [Probe settings](../configs/attribution_probe.json) / [Holdout settings](../configs/attribution_holdouts.json)
- [Exact counts](../results/role_probe_exact.json) / [Run instructions](REPRODUCIBILITY.md)
- Historical implementations: [Extraction](../reference/role/extract_hidden_v5.py) / [Probe](../reference/role/probe_contextual_role.py) / [Holdouts](../reference/role/probe_robustness.py)
