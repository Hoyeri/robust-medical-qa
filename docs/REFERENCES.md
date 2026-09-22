# 관련 문헌과 직접 재현 여부

아래는 분석 기법의 선행연구 또는 관련 연구입니다. 모든 연구를 직접 재현했다고 주장하지 않습니다. 특히 probe/transmission 관련 문헌 연결은 실험 후 정리 과정에서 추가한 것이며 실제 실험 설계 당시의 인용으로 소급하지 않습니다.

| 논문 | 연결 | 이번 분석과 차이 |
|---|---|---|
| [Vaswani et al., Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | QKV, multi-head attention의 기본식 | Llama는 RMSNorm/RoPE/GQA 등 구현 차이가 있음 |
| [Geva et al., Dissecting Recall of Factual Associations… (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.751/) | pre-softmax attention knockout | 우리의 의료 source, receiver 범위, 자연생성 확장은 동일 설정 재현 아님 |
| [Cheng et al., Stochastic Chameleons (ACL 2025)](https://aclanthology.org/2025.acl-long.1458/) | 무관 문맥과 query의 attention 경로 차단 | 사실 회상 다음 답 예측과 긴 의료 rationale 자연생성은 다름 |
| [Belinkov, Probing Classifiers… (CL 2022)](https://aclanthology.org/2022.cl-1.7/) | 내부 표현을 분류기로 읽는 probing 및 한계 | 우리 환자/제3자 대응쌍은 자체 구성 |
| [Kobayashi et al., Attention is Not Only a Weight (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.574/) | value, 출력 변환까지 포함한 기여 벡터 분석 | norm 분석과 우리의 단계별 역할 probe, 점수 민감도는 다름 |

Llama See, Llama Do와의 관련성은 연구 질문의 연결이며, 여기의 세 분석이 그 논문의 head-mask 탐색 재현은 아닙니다. 후배의 head-search 결과와 별도로 보고합니다.
