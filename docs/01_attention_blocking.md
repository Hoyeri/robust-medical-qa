# 1. 알려진 distractor에 대한 attention 차단

## 질문과 방법

특정 head 전체를 없애는 대신 **지정한 receiver가 distractor 토큰을 읽는 attention만** 차단했습니다. Softmax 이전 해당 score에 −∞를 주므로 해당 weight는 0이 되고 나머지 attention은 재정규화됩니다. 전체 32개 block, 32개 query head에 적용했습니다.

- `baseline`: 무개입
- `distractor_option_cut`: 입력 선택지 토큰이 D를 읽지 못함. 생성 토큰은 차단 receiver에 포함되지 않음.
- `distractor_all_source_cut`: D 이후 모든 non-source 토큰이 D를 읽지 못함. 생성되는 설명, 답 위치에도 차단 유지.
- `clinical_*`: 동일한 연산을 D 대신 토큰 수가 같은 원문 임상 구간에 적용.

입력 텍스트를 삭제하거나 새 대조 문장을 넣지 않았습니다. 원문 질문에서 D를 제외한 토큰 목록의 구간을 ID hash로 선택했습니다. 96/96문항에서 차단 source 토큰 수가 정확히 같았습니다. 의미, 거리, 원래 attention mass까지 일치한 대조는 아닙니다. all-source 조건의 후속 receiver 범위는 각 source 위치에 따라 달라질 수 있습니다.

## 표본

동결 탐색 304문항 중 기존 fixed-prefix 평가가 가능한 302문항에서 `sha256('e08-natural-v1:'+question_id)` 오름차순 첫 96개를 선택했습니다. 성능, 정오답, 개입 효과를 선정에 사용하지 않았습니다. E08은 96×3=288회, E11은 같은 96×2=192회 새 생성이며 E11 baseline smoke 3회는 별도입니다.

96은 시간 예산 안에서 고정한 탐색 부분표본입니다. 정확히 96이어야 한다는 검정력 계산은 원 명세에 없습니다. E08 실행 루프 기록은 약 35.4분이며 일반 inference 대비 속도 배수 비교는 아닙니다.

## 자연생성 결과

| 조건 | 정답 /96 | 정답률 | 무개입 오답→정답 | 무개입 정답→오답 |
|---|---:|---:|---:|---:|
| 무개입 | 41 | 42.71% | — | — |
| D → 입력 선택지 차단 | 55 | 57.29% | 17 | 3 |
| D → 모든 후속 위치 차단 | 73 | 76.04% | 36 | 4 |
| 임상 구간 → 입력 선택지 차단 | 44 | 45.83% | 7 | 4 |
| 임상 구간 → 모든 후속 위치 차단 | 33 | 34.38% | 9 | 17 |

모든 조건에서 설명과 답을 처음부터 greedy 생성했습니다. Invalid와 길이 초과를 제외하지 않았습니다. 무개입에서 맞힌 **Hard 41개**에 대한 손상이며 별도 Clean 실험이 아닙니다. 73/96은 전체 정답률의 분자이지 회복 문항 수가 아닙니다.

같은 receiver 정책에서 D 차단과 임상 구간 차단의 paired 정답률 차이는 선택지 +11.46%p [2.08,20.83], 전체후속 +41.67%p [30.21,53.13]였습니다. 임상 구간 차단 자체의 baseline 대비 변화 CI는 0을 포함하므로 손상을 통계적으로 확정하지 않습니다.

## 실행, 해석

Llama-3.1-8B-Instruct, BF16/eager/batch1, greedy, 최대 1024 생성 토큰, KV cache 없음. 매 생성 단계 full-prefix forward 및 차단 검증을 실행했습니다. 이 설정은 원 실험의 재입력, hook 일치 계약이며, 최적화된 배포 구현이 아닙니다.

**결론:** source 선택이 중요하다는 oracle 근거입니다. 일반적인 자동 source 판별이나 head-output scaling 연산 대비 단독 우월성을 입증하지 않습니다.

## 코드와 결과

- [원 실행 코드](../reference/attention/run_e08.py), [임상 대조](../reference/attention/run_e11.py)
- [차단 구현](../reference/attention/receiver_knockout.py), [source 선정](../reference/attention/source_spans.py)
- [문항별 판정](../results/attention_outcomes.csv), [표 집계](../results/attention_summary.json)
- 원 출처: `0909_mechanistic_discovery/experiments/E08_NATURAL_GENERATION_V1_KO.md`, `E11_NATURAL_SOURCE_CONTROL_V1_KO.md` 및 해당 results 디렉터리. SHA-256은 provenance에 기록했습니다.
