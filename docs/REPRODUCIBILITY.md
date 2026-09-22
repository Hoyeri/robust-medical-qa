# 실행 가능 범위

## 1. 지금 레포만으로 확인 가능한 것

```bash
python3 scripts/verify_release.py
python3 scripts/reproduce_tables.py
```

원문/생성 rationale 없는 480행 attention 판정 CSV에서 96문항 분모와 정답, 회복, 손상을 재집계합니다. Probe 정확한 정답 개수 및 transmission의 원 반올림 출력도 확인합니다. 이는 GPU 실행의 재현과 구분됩니다.

## 2. 원 activation을 가진 경우

`hidden_v5/Ma.npz`, `hidden_v5/H1.npz`를 레포 밖에 두고 실행합니다. NumPy 2.0.2에서 검증했습니다.

```bash
python3 scripts/reproduce_probe.py --hidden-dir /absolute/path/to/hidden_v5 --output /tmp/probe.json
```

동결 npz의 phrase/answer, item_ids를 읽어 M/H1 공통 ID를 맞추고 선택된 index의 원 probe를 재계산합니다. 원 inference는 실행하지 않습니다. OS/BLAS 차이에 따른 수치 차이가 가능하며 자동으로 기존 결과를 덮어쓰지 않습니다.

2026-09-22 이 공유본의 CPU 스크립트로 12개 위치/index의 seed별 정답 개수 일치를 확인했습니다. 로컬 NumPy 실행에서는 행렬곱의 `RuntimeWarning`이 출력됐지만, PCA 결과, 회귀 계산, 예측 점수의 유한값 검사와 기존 정답 개수 비교는 모두 통과했습니다. 경고를 숨기지 않았으며 다른 환경에서 non-finite 값이나 정답 개수 차이가 나오면 스크립트가 실패하도록 했습니다.

전체 probe 및 robustness의 원 실행 코드는 `reference/role/`에 있습니다. Robustness 원 스크립트는 별도의 원문 items JSONL이 필요합니다. Transmission 재계산은 `reference/transmission/analyze_transmission.py`와 variant NPZ, alignment.json, RUN_INFO.json, items JSONL이 필요합니다.

## 3. GPU 원 실험 재실행

**현재 레포는 GPU 실험을 독립 실행할 수 있는 완전한 패키지가 아닙니다.** `reference/`는 실제 사용 코드의 선별 사본입니다. 과거 서버 절대 경로, GPU 제한, 실행 마감, 다른 실험의 input binding과 모듈 의존성을 유지합니다. 이 파일을 바로 실행하거나 guard만 제거해 기존 결과 재현이라고 부르면 안 됩니다.

필요한 외부 자료:

| 실험 | 외부 자료 |
|---|---|
| Attention E08/E11 | 0902 baseline/runtime/tokenizer, E02/E03/E05/E06 input 및 결과 binding, MedQA 기반 paired inputs, 모델 snapshot |
| Hidden-state 추출 | cm_confirmatory_v5.jsonl, tokenizer, ariel harness/common dependencies, 모델 snapshot |
| Transmission | 위 자료 및 정확한 eager attention 구현/자동미분 runtime |

모델: `meta-llama/Llama-3.1-8B-Instruct`, revision `0e9e39f249a16976918f6564b8830bc894c89659`. BF16, eager, batch1, cache 미사용. 실제 runtime은 각 실험의 원 명세가 기준입니다. `requirements-analysis.txt`는 CPU 분석용이며 GPU 환경을 정의하지 않습니다.

다음 포팅 단계는 경로의 설정 분리, 필요한 의존성 closure 확정, 고정 입력의 no-op/차단/trace parity 검증입니다. 이 작업을 완료한 것으로 표시하지 않습니다.

## 데이터와 코드 취급

이 첫 공유본은 코드와 소규모 파생 결과 중심입니다. 원 MedQA 문항, 생성 설명, 원시 activation, 모델 가중치, 인증 정보, 채팅 기록은 포함하지 않았습니다. 논문 공개를 위한 데이터 재배포와 라이선스 선택은 별도입니다. 오픈소스 라이선스는 아직 지정하지 않았습니다.
