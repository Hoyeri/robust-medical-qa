# 3. 귀속 정보는 attention 전달 뒤에도 읽히는가?

## 목적

소견 표현에 귀속 정보가 있어도 value 사영이나 attention 전달 과정에서 잃을 수 있습니다. H → V → 소견의 attention 기여 → 출력 변환 후 기여에서 같은 M/H1 분류를 수행해 이 가능성을 검사했습니다. 판독 가능성과 진단 점수에 대한 영향은 구분했습니다.

전체 실행에는 변형이 존재하는 118문항을 사용했지만 M/H1 비교는 **53쌍**입니다. M/U는 118, G+/U는 70입니다. 동일한 빈 rationale의 rs 답 슬롯을 사용하며 자연생성 정확도 실험이 아닙니다.

## 분류기 입력

한 query head의 소견→답 기여는 `C^(h) = sum_{j in finding} A[answer,j,h] * V[j,kv(h)]`입니다. QK softmax는 허용된 전체 토큰을 대상으로 계산하고, 그중 소견 source의 기여만 합산합니다.

| 입력 | 실제 저장값 | 차원 |
|---|---|---:|
| H 평균 | 소견 토큰의 hidden state 평균 | 4096 |
| V 평균 | 소견 토큰의 value 평균, 8 KV head concat | 1024 |
| Concat(C^(1),…,C^(32)) | head별 attention 가중 소견 기여 concat | 4096 |
| Concat(C^(1),…,C^(32)) W_O | 답 위치 residual에 더해지는 소견 기여 | 4096 |

각 입력, 층에 별도의 표준화→PCA24→L2 로지스틱 분류기를 적합했습니다. GQA에서 query head 32개는 KV head 8개를 공유합니다.

## 역할 분류 결과와 집계 규칙

| 표현 | 원 보고서의 최저 holdout 정확도 평균 |
|---|---:|
| H | 0.94 |
| V | 0.95 |
| Concat(C) | 0.89 |
| Concat(C) W_O | 0.83 |

이 값들은 원 출력의 소수 둘째 자리 반올림값입니다. 정확한 새 자릿수를 추정하지 않았습니다.

**실제 코드의 집계:** index **8,10,12,14,16** 다섯 지점 각각에서 `min(개념 holdout, 동료→나머지, 나머지→동료)`를 취하고 다섯 값을 평균했습니다. 8–16의 아홉 층 전부 평균이 아닙니다. seed 최솟값, CI 하한도 아닙니다. 모든 지점에서 0.83 이상이라는 뜻도 아닙니다. by-item 정확도와 shuffled-label 대조는 이 최솟값에 포함되지 않습니다.

**층 표기:** H 배열의 index l은 block l의 입력(즉 l>0이면 앞 block 출력)입니다. V/C/CW_O의 index l은 실제 block l 연산입니다. H index 8을 block 8의 출력이라고 표시하지 않습니다. 마지막 H index32만 최종 norm 이후입니다.

기존 hidden-state probe와 transmission의 전처리 구현 및 CV 집계도 완전히 동일하지 않습니다. transmission은 표준화값 clipping과 fold 정확도 단순 평균 등을 사용합니다. H 100%→94%를 정보 손실량으로 해석하지 않습니다.

## 진단 점수 민감도

`q = logit(T) - logit(G)`로 두고 source 기여에 작은 배율 변화를 주었을 때 `dq/dα`를 계산했습니다. 여기의 민감도 A_e는 attention weight A와 다른 변수입니다. 전체 downstream receiver에 대한 기여의 1차 민감도 차이:

- 문장 전체 source: M−H1 중앙값 +0.18, 95% CI [−0.19,+0.36]
- 공통 소견 source: +0.01 [−0.04,+0.12]

역할 정보가 전달된 벡터에서도 읽히지만, 이 aggregate 민감도에서 일관된 역할 차이는 확인되지 않았습니다. 동등성 검정이 아니므로 역할을 전혀 사용하지 않는다거나 모든 head가 같은 역할을 한다고 결론짓지 않습니다. 원 보고서 §18.4의 동일53문항 정정 역시 유지합니다: q(M)−q(H1) 중앙값 0 [−0.125,+0.125], 평균 +0.245 [−0.075,+0.649].

## 코드, 증거

- [실제 추출, 미분 코드](../reference/transmission/transmission_audit_v5.py)
- [실제 분석, 집계 코드](../reference/transmission/analyze_transmission.py)
- [원 출력의 단계별 분류 표](../results/transmission_decodability_original.txt)
- [민감도와 판독 차이 요약](../results/transmission_summary.json)

원 코드와 출력의 해석을 위 범위로 제한합니다. 이 레포 정리에서 새로운 value intervention을 실행하지 않았습니다.
