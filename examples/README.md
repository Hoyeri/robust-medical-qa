# Example inputs

[attribution_pair.jsonl](attribution_pair.jsonl) contains one actual M/H1 pair from the analyzed set. Original English question text, options, labels, and inserted sentences are preserved. Only C/Ma/H1 and the metadata needed for extraction/holdouts are included. This single pair illustrates extraction input; cross-validation requires multiple independent questions and concepts.

## Representation input

- `item_id`: unique question identifier shared across variants
- `options`: A–D option text
- `gold_G`, `target_T`: original correct label and competing target label
- `variants`: complete question text for C (original), Ma (patient finding), H1 (third-party finding)
- `meta.cue.hpo`: concept identifier used for concept holdouts
- `meta.attribution_family`: family identifier used for template holdouts

Additional variants can be selected in the extraction configuration. Explicit finding annotations for new templates are described in [Code structure](../docs/CODE_STRUCTURE.md).

## Attention input

The attention experiment reads one JSON object with `generation` and `rows`. Each row contains `question_id`, `gold_answer`, `intended_target`, `prompt_ids`, `options` (input-option token positions), and `sources` with `distractor` and `clinical` token-position lists. Positions refer to the exact tokenized prompt, not character offsets.

A schematic row is shown below. Its token IDs are placeholders; generate real IDs and source positions with the tokenizer used for the model before running inference.

```json
{
  "question_id": "example",
  "gold_answer": "A",
  "intended_target": "B",
  "prompt_ids": [101, 102, 103, 104, 105, 106],
  "options": [4, 5],
  "sources": {"distractor": [2, 3], "clinical": [0, 1]}
}
```

The converter combines original distractor and clinical-control payloads and verifies matching question IDs, prompts, labels, generation settings, and source lengths. See [Reproducibility](../docs/REPRODUCIBILITY.md) for commands.
