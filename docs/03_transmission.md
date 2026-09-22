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

This table summarizes concept/template generalization. The earlier attribution table reports question-grouped cross-validation, so its near-100% values and the H value below use different evaluation summaries.

| Representation | Mean minimum holdout accuracy |
|---|---:|
| H | 0.94 |
| V | 0.95 |
| C_{finding→answer} | 0.89 |
| C_{finding→answer} W_O | 0.83 |

Values retain the original output's two-decimal precision. At each of indices **8, 10, 12, 14, 16**, the analysis selected the minimum of concept-holdout accuracy, coworker→other-template accuracy, and other-template→coworker accuracy. The table reports the mean of those five minima. For example, the original rounded H scores at index 8 are 0.99, 0.86, and 0.99; their minimum is 0.86. The summary averages five such minima using unrounded scores. Question-level cross-validation and shuffled-label controls are available in the [full decoding output](../results/transmission_decodability_original.txt).

H[l] is the input to block l, while V/C/CW_O[l] describe computations inside block l. H[32] is the final normalized hidden state.

The transmission probe used clipping of standardized values and an unweighted mean of fold accuracies. The earlier hidden-state probe used its original preprocessing and pooled correct counts across folds.

## Diagnostic-score sensitivity

We defined `q = logit(T) - logit(G)`. For a source e, the perturbation adds `α C_e W_O` to the attention output at each downstream non-source receiver. Thus α = 0 is baseline and α = 0.1 adds 10% of the baseline source contribution. This is an additive contribution intervention with the baseline contribution held fixed.

For each block, the gradient gives `A_e = dq/dα` at α = 0. The results below sum block sensitivities over zero-based blocks 8–16, then compute each question's M−H1 difference and the median across 53 pairs. This measures the local derivative for simultaneous perturbations across those blocks. The source extent is either the full added sentence or the shared finding phrase. Bootstrap resampling uses questions as the unit.

The decoding vectors above use only the answer receiver; this sensitivity table uses all downstream non-source receivers. Both are evaluated with the fixed empty-rationale answer prefix.

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

- [Current extraction](../experiments/extract_representations.py) / [Current analysis](../experiments/transmission_audit.py)
- [Extraction settings](../configs/transmission_extraction.json) / [Analysis settings](../configs/transmission_analysis.json)
- [Original decoding output](../results/transmission_decodability_original.txt) / [Sensitivity and answer-score summary](../results/transmission_summary.json)
- [Run instructions](REPRODUCIBILITY.md)
- Historical implementations: [Extraction](../reference/transmission/transmission_audit_v5.py) / [Analysis](../reference/transmission/analyze_transmission.py)

The tables retain the original reported values. The refactor's bootstrap recalculation gave mean q(M)−q(H1) = 0.245283 with CI [−0.077830, +0.639151]; the point estimate matches, while the resampled interval differs slightly. The bootstrap settings and recomputed values are recorded in [validation results](../provenance/refactor_validation.json).
