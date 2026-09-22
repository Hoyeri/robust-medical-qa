# Notation

| Symbol | Definition |
|---|---|
| G | Original correct answer |
| T | Competing target answer |
| M / Ma | T-supporting finding attributed to the patient |
| H1 | The same finding attributed to a third party |
| H | Hidden-state representation |
| Q, K, V | Query, key, and value vectors |
| W_Q, W_K, W_V, W_O | Learned projection matrices in the frozen model |
| A[t,j,h] | Attention weight from receiver t to source j in query head h |
| C^(h)_{finding→answer} | Sum of attention-weighted finding values for one head |
| C_{finding→answer} | Concatenated contributions from 32 query heads, 4096 dimensions |
| C_{finding→answer} W_O | Finding contribution after output projection |
| q | logit(T) − logit(G) |
| A_e | dq/dα, source-contribution sensitivity |

## Indices

The 32 Transformer blocks use indices 0–31. `output_hidden_states` contains 33 entries, including the input embedding at index 0. For l=0–31, H[l] is the input to block l. H[12] is the output of block 11; H[32] is the final normalized output after block 31. Transmission arrays V/AV/OAV[l] correspond to computations in block l.

## Evaluation units

- Attention: 96 independent questions, five conditions, 480 outcome records.
- Probe: 53 independent questions, two attribution conditions, 106 inputs, 318 predictions across three seeds.
- Template holdout: 39 coworker questions and 14 questions using other third-party templates.
- Probe accuracy: patient/third-party classification accuracy.
- Diagnostic accuracy: fraction of questions answered with G.
