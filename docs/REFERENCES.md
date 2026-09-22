# Experiment references

## Attention blocking

| Experiment | Reference used in the experiment design | Method used here |
|---|---|---|
| Source/receiver attention knockout, E05/E06, extended to free generation in E08/E11 | [Geva et al., Dissecting Recall of Factual Associations in Auto-Regressive Language Models (EMNLP 2023), Section 5, Eq. 7](https://aclanthology.org/2023.emnlp-main.751/) | Set selected pre-softmax attention scores to negative infinity. Apply the operator to input-option or downstream receiver sets, with distractor and matched clinical sources. |
| Initial distractor-source knockout, E03 | [Cheng et al., Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs (ACL 2025), Section 6.3 and Appendix H](https://aclanthology.org/2025.acl-long.1458/) | Block attention to a context source and measure the resulting change in the predicted answer. This was the starting reference for the source-flow experiments. |
| Fixed-rationale versus regenerated-rationale evaluation, E08 | [Lanham et al., Measuring Faithfulness in Chain-of-Thought Reasoning (2023)](https://arxiv.org/abs/2307.13702) | Evaluation-design reference for examining answer behavior after intervening on the reasoning process. E08 regenerates the rationale and answer while attention blocking is active. |

The original design records explicitly cite these papers:

- `E03_SOURCE_ATTENTION_KNOCKOUT_V1_KO.md`, "Prior method and adaptation"
- `E05_RECEIVER_COVERAGE_V1_KO.md` and `E06_FIRST_RECEIVER_ROUTES_V1_KO.md`
- `E08_NATURAL_GENERATION_V1_KO.md`, "Question and prior method"
- `E11_NATURAL_SOURCE_CONTROL_V1_KO.md`

These protocols are under `0909_mechanistic_discovery/experiments/` in the original workspace. Their paths and hashes are recorded in [the source manifest](../provenance/sources.json).

## Attribution probe and transmission audit

| Analysis | Experiment design record | Study-specific setup |
|---|---|---|
| Attribution probe | `0916/CM_PAPER_POSITION_20260920_KO.md`, Sections 12.6–12.8 | Compare the same finding attributed to the patient or a third party. Probe finding-token and answer-position representations, with question, concept, and template splits. |
| Transmission audit | `0916/CM_PAPER_POSITION_20260920_KO.md`, Section 12.13 | Follow that M/H1 contrast through H, V, attention-weighted source contributions, and output-projected contributions. Measure the diagnostic-score sensitivity of those contributions. |

These two analyses were designed within the project to investigate the preceding source-flow results. Their methods and implementations are documented in [Attribution probe](02_role_probe.md) and [Transmission audit](03_transmission.md).
