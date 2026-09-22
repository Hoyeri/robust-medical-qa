# Experiment references

## Attention blocking

| Experiment | Reference used in the experiment design | Method used here |
|---|---|---|
| Source/receiver attention knockout during generation | [Geva et al., Dissecting Recall of Factual Associations in Auto-Regressive Language Models (EMNLP 2023), Section 5, Eq. 7](https://aclanthology.org/2023.emnlp-main.751/) | Set selected pre-softmax attention scores to negative infinity. Apply this to distractor sources and length-matched clinical sources, with input-option positions or subsequent positions as receivers. |
| Distractor-source knockout | [Cheng et al., Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs (ACL 2025), Section 6.3 and Appendix H](https://aclanthology.org/2025.acl-long.1458/) | Block attention to a context source and measure changes in the predicted answer. This paper was a starting reference for the source-flow experiments. |
| Comparing fixed-rationale and regenerated-rationale evaluation | [Lanham et al., Measuring Faithfulness in Chain-of-Thought Reasoning (2023)](https://arxiv.org/abs/2307.13702) | Reference for evaluating answer behavior after intervening in reasoning. In the generation experiments, regenerate the rationale and answer while maintaining attention blocking. |

## Attribution probing and transmission analysis

| Analysis | Study design |
|---|---|
| Attribution probe | Compare conditions attributing the same finding to the patient or a third party. Classify representations at finding tokens and the answer position using item, concept, and template splits. |
| Transmission analysis | Apply the M/H1 contrast to H, V, attention-weighted source contributions, and contributions after the output projection. Measure diagnostic-score sensitivity to those contributions. |

These two analyses were designed within the project to investigate the preceding source-flow findings. See the implementations for the [attribution probe](../experiments/attribution_probe.py) and [transmission analysis](../experiments/transmission.py).
