# Related work

| Paper | Relevant method | Connection to these analyses |
|---|---|---|
| [Vaswani et al., Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | QKV projections and multi-head attention | Basis for the attention notation; the analyzed Llama model uses RMSNorm, RoPE, and GQA |
| [Geva et al., Dissecting Recall of Factual Associations… (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.751/) | Pre-softmax attention knockout | Source/receiver blocking, applied here to medical inputs and free generation |
| [Cheng et al., Stochastic Chameleons (ACL 2025)](https://aclanthology.org/2025.acl-long.1458/) | Attention-path interventions on irrelevant context and queries | Context influence in factual recall; our experiments evaluate medical rationale and answer generation |
| [Belinkov, Probing Classifiers… (CL 2022)](https://aclanthology.org/2022.cl-1.7/) | Probing internal representations | Classifier-based decoding, applied here to paired patient/third-party attribution |
| [Kobayashi et al., Attention is Not Only a Weight (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.574/) | Contribution vectors incorporating values and output projections | Analysis of transmitted representations, extended here with attribution probes and score sensitivity |

These references place the analyses in the broader literature on context influence, representation probing, and attention contributions. Llama See, Llama Do provides a related head-search perspective.
