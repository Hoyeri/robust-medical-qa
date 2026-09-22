# Transmission audit

## Question

Does patient/third-party attribution remain decodable after value projection, attention weighting, and output projection?

We applied M/H1 classifiers to four representations along this sequence. The extraction run covered 118 questions with available variants. Pairwise sample sizes were 53 for M/H1, 118 for M/U, and 70 for G+/U. The readout used the fixed `rs` answer slot with an empty rationale.

## Representations

For query head h, the finding's contribution to the answer position is:

```text
C^(h)_{finding→answer} = sum_{j in finding} A[answer,j,h] * V[j,kv(h)]
C_{finding→answer} = Concat(C^(1), ..., C^(32))
```

Attention is normalized over all allowed source tokens. The sum above selects contributions from the finding tokens.

| Classifier input | Representation | Dimensions |
|---|---|---:|
| H mean | Mean hidden state of finding tokens | 4096 |
| V mean | Mean value vectors of finding tokens, concatenated over 8 KV heads | 1024 |
| C_{finding→answer} | Concatenated attention-weighted contributions from 32 query heads | 4096 |
| C_{finding→answer} W_O | Finding contribution added to the answer-position residual | 4096 |

Grouped-query attention shares 8 KV heads across 32 query heads. A separate standardization → PCA24 → L2 logistic classifier was fitted for each representation and index.

## Attribution decoding

| Representation | Mean minimum holdout accuracy |
|---|---:|
| H | 0.94 |
| V | 0.95 |
| C_{finding→answer} | 0.89 |
| C_{finding→answer} W_O | 0.83 |

Values retain the original output's two-decimal precision. At each of indices **8, 10, 12, 14, 16**, the analysis selected the minimum of concept-holdout accuracy, coworker→other-template accuracy, and other-template→coworker accuracy. The table reports the mean of those five minima. Question-level cross-validation and shuffled-label controls were reported separately.

H[l] is the input to block l, while V/C/CW_O[l] describe computations inside block l. H[32] is the final normalized hidden state.

The transmission probe used clipping of standardized values and an unweighted mean of fold accuracies. The earlier hidden-state probe used its original preprocessing and pooled correct counts across folds.

## Diagnostic-score sensitivity

We defined `q = logit(T) - logit(G)` and computed `A_e = dq/dα`, the first-order sensitivity to scaling a source's contribution across downstream receivers.

| Source extent | Median M−H1 sensitivity | 95% CI |
|---|---:|---|
| Full sentence | +0.18 | [−0.19, +0.36] |
| Shared finding phrase | +0.01 | [−0.04, +0.12] |

The corrected answer-slot comparison on the same 53 pairs gave:

| q(M)−q(H1) | Estimate | 95% CI |
|---|---:|---|
| Median | 0.000 | [−0.125, +0.125] |
| Mean | +0.245 | [−0.075, +0.649] |

Attribution remained decodable in the transmitted vectors. Confidence intervals for aggregate sensitivity and answer-score differences included zero.

## Code and results

- [Extraction and differentiation](../reference/transmission/transmission_audit_v5.py)
- [Probe and aggregation](../reference/transmission/analyze_transmission.py)
- [Original decoding output](../results/transmission_decodability_original.txt)
- [Sensitivity and answer-score summary](../results/transmission_summary.json)

The same-pair answer-score values follow the correction in section 18.4 of the original report, recorded in the source manifest.
