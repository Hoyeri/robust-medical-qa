# Attention blocking

## Method

We blocked attention from selected receiver positions to known distractor tokens across all 32 Transformer blocks and 32 query heads. Setting the corresponding pre-softmax scores to negative infinity assigns zero attention to those tokens and renormalizes attention over the remaining sources.

| Condition | Source | Blocked receivers |
|---|---|---|
| `baseline` | — | None |
| `distractor_option_cut` | Distractor | Input answer-option tokens |
| `distractor_all_source_cut` | Distractor | All subsequent non-source tokens, including generated rationale and answer tokens |
| `clinical_option_cut` | Matched clinical span | Input answer-option tokens |
| `clinical_all_source_cut` | Matched clinical span | All subsequent non-source tokens, including generated tokens |

The clinical control was selected by an ID hash from the original question's non-distractor token list. The blocked token count matched the distractor exactly in all 96 questions. Input text was retained in every condition. The control matched source length; source meaning, position, and baseline attention mass varied. The downstream receiver range followed the position of each source.

## Sample selection

The source pool contained 304 Bystander exploratory questions from our unreleased, hardened version of MedDistractQA (internal name: MedDistractQA-Hard); 302 supported the existing fixed-prefix evaluation. We ranked those IDs by SHA-256 with a fixed salt and selected the first 96, independently of model outcomes. The exact historical salt and selection function are preserved in [the selection code](../reference/attention/generation_routes.py).

The sample size was set within the experiment's time budget. Baseline and the two distractor-blocking conditions generated 288 responses; the two clinical-control conditions generated 192 responses on the same questions. Three clinical-control baseline smoke runs were recorded separately. The baseline/distractor generation loop took approximately 35.4 minutes.

## Free-generation results

| Condition | Correct /96 | Accuracy | Wrong→correct | Correct→wrong |
|---|---:|---:|---:|---:|
| Baseline | 41 | 42.71% | — | — |
| Distractor → input options blocked | 55 | 57.29% | 17 | 3 |
| Distractor → all downstream blocked | 73 | 76.04% | 36 | 4 |
| Clinical span → input options blocked | 44 | 45.83% | 7 | 4 |
| Clinical span → all downstream blocked | 33 | 34.38% | 9 | 17 |

Each condition generated the rationale and answer from scratch using greedy decoding. All 96 questions, including invalid and length-limited outputs, contributed to the denominator. The transition columns compare each condition with the baseline on these distracted inputs: 41 baseline-correct and 55 baseline-incorrect questions.

The paired accuracy difference between distractor and clinical-source blocking was +11.46 percentage points [2.08, 20.83] for option receivers and +41.67 [30.21, 53.13] for all downstream receivers. The confidence intervals for clinical-source blocking versus baseline included zero.

## Runtime

Llama-3.1-8B-Instruct, BF16, eager attention, batch size 1, greedy decoding, and a maximum of 1024 generated tokens. KV caching was disabled. Each generation step used a full-prefix forward pass with blocking verification.

With oracle source locations, distractor blocking recovered more correct answers than length-matched clinical-source blocking.

## Code and results

- [Current experiment](../experiments/attention_blocking.py) / [Input converter](../experiments/prepare_attention.py) / [Configuration](../configs/attention_blocking.json)
- [Per-question outcomes](../results/attention_outcomes.csv) / [Aggregated counts](../results/attention_summary.json)
- [Run instructions](REPRODUCIBILITY.md)
- Historical implementations: [Distractor blocking](../reference/attention/run_distractor_blocking.py) / [Clinical control](../reference/attention/run_clinical_control.py)
