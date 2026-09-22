# 원본 코드 사본

이 폴더는 기존 실험에서 사용한 파일을 선별 복사한 것입니다. Transmission 코드 두 파일은 주석과 출력 문자열의 수식 표기만 정리했고 계산은 바꾸지 않았습니다. 원본 경로와 SHA-256, 표기를 바꾼 사본의 SHA-256은 `../provenance/sources.json`에 기록했습니다.

- attention: E08 자연생성, E11 matched clinical-source 대조와 attention 차단 구현
- role: hidden-state 추출 및 원 probe/robustness 분석
- transmission: H/V/attention 기여 추출과 분류, 민감도 분석

GPU 실행 파일은 과거 서버 경로, 마감 guard, 포함되지 않은 프로젝트 모듈을 참조합니다. 독립 실행용 명령으로 사용하지 마세요. 현재 실행 가능한 검증, 재집계는 `../scripts/`이며, 필요한 외부 자료는 `../docs/REPRODUCIBILITY.md`에 정리했습니다.
