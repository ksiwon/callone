# AIHub 데이터 다운로드 안내 (선택 — §13 보강)

AIHub 방언/저음질 통화 데이터는 **선택**이다. 없어도 본인 통화만으로
ASR 적응(§13)이 동작한다. 라이선스 동의(무료)만 필요, 결제 아님(§2).

## 필요 계정
- `.env` 의 `AIHUB_ID`, `AIHUB_PW` (AIHub 회원가입 무료)

## 권장 데이터셋
- **한국어 방언 발화** — 경상(#119), 전라(#120), 충청, 강원
  - S2.5 에서 추정된 지역에 맞춰 선택 (asr_adapt.yaml `aihub.region_match`)
- **저음질 전화망 음성인식 데이터** — 전화 음질 적응

## 다운로드 (aihubshell)
```bash
pip install aihubshell    # 또는 AIHub 제공 CLI
aihubshell -mode l                       # 데이터셋 목록
aihubshell -mode d -datasetkey 119       # 경상 방언 (키는 사이트에서 확인)
```

## 배치
- 압축 해제 후 `data/aihub/{region}/` 에 둔다.
- `configs/asr_adapt.yaml` 에서 `aihub.enabled: true` 로 켠다.

## 주의
- AIHub 데이터는 외부 전송 금지 원칙(§20)과 별개로, 라이선스 약관 준수.
- 제주 방언은 본 프로젝트 데이터에 없음 → 후보 제외(폴백만, §11.1).
