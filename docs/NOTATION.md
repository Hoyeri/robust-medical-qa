# 표기와 분모

- G: 원문 정답, T: 경쟁/표적 오답. M은 T를 지지하는 소견을 환자에게 귀속한 조건. H1은 동일 소견의 의도적 제3자 대조. 실제로 모든 M이 진단을 바꿔야 하거나 모든 제3자 정보가 부적합하다는 뜻은 아닙니다.
- H(수식): hidden state. H1(조건 이름)과 구분합니다.
- Q,K,V: query/key/value. W_Q,W_K,W_V,W_O: 고정된 모델의 학습 가중치.
- A[t,j,h]: query head h에서 receiver t가 source j를 읽는 attention 가중치.
- C^(h)_{소견→답}: 소견 토큰의 attention-weighted value 합. 평균이 아닙니다.
- C_{소견→답}: 32개 query head의 C^(h)를 concat한 4096차원 벡터.
- C W_O: 소견에서 오는 attention 기여. 답 위치의 전체 residual 표현이 아닙니다.
- A_e=dq/dα: 원 민감도 분석의 변수. attention 가중치 A와 혼동하지 않습니다.

## 층

32개 Transformer block의 0-based 번호는 0–31입니다. `output_hidden_states`는 입력 임베딩을 포함한 33개(index 0–32)입니다. H[l]은 block l의 입력, H[12]는 block11 출력, H[32]는 block31 이후 최종 norm입니다. Transmission의 V/AV/OAV[l]은 실제 block l에 대응합니다. 모든 표는 어느 index를 썼는지 표시합니다.

## 통계 단위

- Attention: 서로 다른 96문항. 5조건의 결과 행 480개를 독립 문항으로 세지 않습니다.
- Probe: 53문항×2귀속조건=106개 입력. 3개 seed의 평가 318개는 반복 예측입니다.
- 관계 holdout: 동료39/나머지14. 가족력, 접촉력 일반화와 다릅니다.
- Probe 정확도는 환자/제3자 조건 분류 정확도. 임상 정답률 또는 AUROC가 아닙니다.
