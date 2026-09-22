# Robust Medical QA

## 읽는 순서

| 분석 | 질문 | 확인한 결과 | 상세 |
|---|---|---|---|
| Attention 차단 | 알려진 distractor를 읽지 못하게 하면 답이 회복되는가? | 같은 96문항에서 정답 41 → 선택지 차단 55 → 모든 후속 위치 차단 73. 각 조건은 별도 실행. | [방법, 결과](docs/01_attention_blocking.md) |
| 귀속성 probe | 같은 소견의 환자/제3자 귀속이 표현되는가? | 53쌍의 소견 hidden-state index 8–16에서 분류 정확도 99.69–100%. | [방법, 결과](docs/02_role_probe.md) |
| Transmission audit | 귀속 정보가 value 변환과 attention 전달 뒤에도 읽히는가? | 전달 기여에서도 분류 가능. 전체 head의 진단 점수 민감도에는 일관된 M/H1 차이를 확인하지 못함. | [방법, 결과](docs/03_transmission.md) |

이 세 결과는 서로 다른 평가입니다. **진단 정답률, 귀속 분류 정확도, 점수 민감도**를 같은 지표로 해석하지 않습니다.

## 바로 확인하기

Python 3 표준 라이브러리만으로 공유된 결과의 행 수, 분모, 집계를 검증할 수 있습니다.

```bash
python3 scripts/verify_release.py
python3 scripts/reproduce_tables.py
```

저장된 hidden state를 별도로 보유한 경우, GPU 없이 앞의 정확한 probe 수치를 재계산할 수 있습니다.

```bash
python3 -m pip install -r requirements-analysis.txt
python3 scripts/reproduce_probe.py --hidden-dir /path/to/hidden_v5 --output /tmp/role_probe_recheck.json
```

## 구성

- `docs/`: 실험별 질문, 방법, 분모, 해석 한계와 표기 규칙
- `results/`: 원문 문항, 생성 설명을 제외한 결과표와 문항별 판정
- `scripts/`: 저장 결과 검증, 재집계 및 CPU probe 재계산
- `reference/`: 실제 사용한 코드의 선별 사본. 일부 주석과 출력 표기만 정리했습니다. 서버 의존 GPU 코드는 읽기/감사용이며 독립 실행 패키지가 아닙니다.
- `provenance/`: 원본 상대 경로, SHA-256, 배포 파일 무결성 목록

원문 데이터, tokenizer/model, activation NPZ, 전체 생성 로그는 업로드하지 않았습니다. [실행 가능 범위와 필요한 자료](docs/REPRODUCIBILITY.md)를 확인하세요.

## 연구상의 경계

- Attention 차단은 **distractor 위치를 아는 oracle 실험**입니다. Clean 입력의 안전성을 이 표에서 주장하지 않습니다.
- M은 T를 지지하는 소견을 현재 환자에게 귀속한 조건, H1은 같은 소견을 제3자에게 귀속한 조건입니다. T 유도 여부와 진단 근거의 정당성은 별개입니다.
- Probe가 읽을 수 있는 정보와 모델이 실제 진단에서 사용하는 정보는 다릅니다.
- 동일 문항의 여러 버전, seed 반복을 독립 문항으로 세지 않습니다.
- 이 저장소에는 DIAT 학습, AGSA 정책 전체, 후배의 head search를 포함하지 않았습니다.

[표기와 용어](docs/NOTATION.md) / [관련 문헌과 직접 재현 여부](docs/REFERENCES.md)
