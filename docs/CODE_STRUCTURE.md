# Code structure

Experiments call reusable components in `robust_medical_qa/`. JSON configurations define the analysis choices. The `reference/` directory preserves the historical implementations for comparison.

| Component | Responsibility |
|---|---|
| `runtime.py` | Local model/tokenizer loading, prompt construction, answer token IDs, model dimensions |
| `spans.py` | Source and finding spans, token offsets, attribution metadata |
| `attention.py` | Source/receiver masks and scoped attention interventions |
| `representations.py` | Hidden-state extraction, attention capture, contribution vectors, gradients |
| `extraction.py` | Shared item/variant extraction loop and artifact writing |
| `generation.py` | Greedy decoding, answer parsing, answer alignment, prefix replay |
| `probes.py` | Standardization, PCA, logistic regression, grouped cross-validation, template holdouts |
| `evaluation.py` | Outcome counts and bootstrap summaries |
| `io.py` | Input validation, JSON IO, hashes |

## Experiment entry points

- `experiments.attention_blocking`: source/receiver interventions using prepared token inputs
- `experiments.prepare_attention`: convert distractor and clinical-control input payloads into one portable manifest
- `experiments.extract_representations`: hidden-state or transmission extraction
- `experiments.attribution_probe`: question-level probes and optional concept/template holdouts
- `experiments.transmission_audit`: transmission probes, paired sensitivities, answer-score comparisons

Each entry point accepts configuration and input/output paths. Importing an experiment module defines its functions without starting an experiment.

## Numerical profiles

| Configuration | Standardized-value clipping | Logistic-score clipping | Non-finite replacement | CV aggregation |
|---|---|---|---|---|
| `attribution_probe.json` | None | None | None | Pooled predictions |
| `attribution_holdouts.json` | ±20 | ±30 | None | Pooled predictions |
| `transmission_analysis.json` | ±20 | ±30 | `nan_to_num` after clipping | Mean fold accuracy |

These profiles preserve the original analysis choices in one shared implementation. Sample counts and denominators are calculated from the input arrays. Frozen-result comparison is performed separately by `scripts/reproduce_probe.py`.

## Source annotations

Existing templates use the shared phrase extractor. New templates can provide explicit character offsets within the source sentence:

```json
{
  "source_annotations": {
    "H1": {
      "sentence": "A colleague reports signal red.",
      "phrase_char_span": [20, 30]
    }
  }
}
```

The character interval is half-open. Unknown templates require an explicit annotation. New attribution families can be supplied as `meta.attribution_family` for holdout analysis.

## Intervention validation

Transmission validation records the actual alpha in each output row. Baseline contribution vectors for every validation layer are saved before intervention forwards run. Attention wrappers and registered hooks are removed on normal exit and exceptions.

## Validation on 2026-09-22

- Frozen attention outcomes: all 480 records reproduce the original five-condition counts.
- Frozen attribution probe: all 12 selected position/index combinations reproduce per-seed correct counts on 53 pairs.
- Frozen transmission analysis: recomputed stage summaries round to 0.94, 0.95, 0.89, and 0.83.
- Unit/integration tests: historical probe profiles, nonstandard model dimensions, source isolation, contribution reconstruction, gradient checks, hook cleanup, local model loading, generation, extraction, and analysis.

Model integration tests use a two-block random Llama on CPU. Full Llama-3.1-8B generation parity remains a separate validation run.
