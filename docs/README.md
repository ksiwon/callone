# callone 사용 가이드

## 🚀 처음 세팅(권장) → **[FRESH_SETUP.md](FRESH_SETUP.md)**
새 GPU 인스턴스에서 클론 → 스크립트 3개 → 실행까지 한 번에. 제로샷(5~10초 음성+사진) 영상통화는 이거 하나면 끝.

상위 개요·두 가지 사용방식(제로샷 vs 풀튜닝)·두 GPU(A100/4090)는 [최상위 README](../README.md).

---

## 문서 맵

| 하고 싶은 것 | 문서 |
|---|---|
| **새 인스턴스 세팅 + 제로샷 영상통화** | [FRESH_SETUP.md](FRESH_SETUP.md) |
| 토킹헤드(Ditto) 상세·문제해결 | [AVATAR_RUN.md](AVATAR_RUN.md), [avatar_talking_head_design.md](avatar_talking_head_design.md) |

### 풀 파인튜닝(고급, mode B) — 긴 녹음으로 화자 학습
긴 통화 녹음(1시간+)에서 화자를 분리하고 목소리·말투를 **학습**하는 경로. 제로샷보다 무겁지만 충실도↑.

| 단계 | 문서 |
|---|---|
| 노트북에서 녹음 처리(데이터셋·방언프로필) | [1. 로컬에서 학습](1_로컬에서_학습.md) |
| GPU에서 화자분리·전사 + 말투 LoRA 학습 | [2. GPU에서 학습](2_GPU에서_학습.md) |
| 노트북에서 통화 | [3. 노트북에서 통화](3_노트북에서_통화.md) |
| 화자 목소리 학습(Piper TTS) | [5. 화자 A 목소리 학습](5_화자A_목소리_학습.md) |

> 초기 설계 기록(spec·스택 결정)은 [`legacy/design/`](../legacy/design/) 에 보관(코드 주석이 §인용).
