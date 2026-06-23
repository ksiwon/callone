# callone — call + clone

사진 1장과 짧은 목소리만으로 **그 사람과 영상통화하듯** 대화하는 로컬 시스템.
음성 복제(TTS) + 한국어 대화(LLM 페르소나) + 말하는 얼굴(토킹헤드)을 **전부 로컬 오픈소스**로 돌린다(유료 API 0원).

> ⚠️ **윤리:** 결과물은 그 사람의 **근사(近似)** 다. 사칭·기만 금지, 사적/추모/연구 목적 한정.
> 통화 녹음·개인정보 관련 관할 법규를 준수하라. 본인이 권리를 가진 음성·사진만 사용할 것.

---

## 처음이라면 → **[docs/FRESH_SETUP.md](docs/FRESH_SETUP.md)**

현재 확정 스택과 남은 GPU 검증은 **[docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)**.
새 GPU 인스턴스에서 클론 → 스크립트 3개 → 실행까지 **한 번에** 세팅하는 절차. 이 문서 하나면 된다.

---

## 두 가지 사용 방식 (목적에 맞게 선택)

| | **A. 제로샷 (빠른 복제)** ← 기본·권장 | **B. 풀 파인튜닝 (고충실도)** ← 고급 |
|---|---|---|
| 입력 | **5~10초** 깨끗한 음성 1개 + 사진 1장 | **1시간 이상** 통화/녹음 + 화자 분리 |
| 학습 | **없음**(업로드 즉시 사용) | 화자별 TTS·페르소나 **파인튜닝**(수 시간) |
| 음성 | CosyVoice3 제로샷 클론 | Piper 화자학습(onnx, 최고 충실도) |
| 대화 | EXAONE + 페르소나 프롬프트 | + 화자 실제 발화 학습/RAG |
| 쓰는 곳 | UI에서 파일 업로드 → 바로 통화 | `docs/` 학습 파이프라인(아래) |
| 문서 | **[FRESH_SETUP.md](docs/FRESH_SETUP.md)** | [학습 파이프라인](#풀-파인튜닝-파이프라인-고급) |

대부분은 **A(제로샷)** 이면 충분하다. 음색을 더 끌어올리고 싶고 긴 녹음이 있으면 B.

---

## 두 가지 GPU 타깃 (둘 다 지원)

| | **A100 / H100** (예: Elice) | **RTX 4090 / 3090** (예: RunPod) |
|---|---|---|
| 아키텍처 | Ampere/Hopper | Ada/Ampere |
| LLM | **EXAONE-4.0-32B-abliterated Q6_K** | EXAONE-3.5-7.8B-abliterated Q6_K(VRAM 한계) |
| TTS·ASR | CosyVoice3·whisper | 동일 |
| 토킹헤드(Ditto) | **프리빌트 TensorRT 엔진**(`ditto_trt_Ampere_Plus`) 그대로 사용 | **Ada는 엔진 재빌드 필요**(onnx→trt) 또는 PyTorch 폴백 |
| VRAM | 여유(80GB) | 24GB — 7.8B LLM + 0.5B TTS + Ditto 들어감 |

세팅 스크립트는 GPU를 감지해 자동 처리한다. 4090 관련 주의는 [FRESH_SETUP.md](docs/FRESH_SETUP.md) "알아둘 것" 참고.

---

## 구성 (4개 독립 서비스 = 각자 venv/프로세스)

| 서비스 | 포트 | 역할 | 모델 |
|---|---|---|---|
| `llama-server` | 8090 | 한국어 대화 LLM | EXAONE-4.0-32B(A100/H100) / 3.5-7.8B(24GB GPU) |
| `cosyvoice-server` | 8092 | 제로샷 음색 복제 TTS | CosyVoice3-0.5B (conda env) |
| `avatar-server` | 8091 | 사진→말하는 얼굴 | Ditto (TensorRT, `.venv-avatar`) |
| `callone-serve` | 8000 | 오케스트레이터(ASR+LLM+TTS+아바타 조립, WS) | faster-whisper large-v3-turbo (`.venv-serve`) |

무거운 스택끼리 의존성 충돌을 피하려 **별 프로세스 + HTTP/WS**로만 연결(llama-server 패턴). UI는 `ui/`(React).

우선순위: **① 목소리 유사도 → ② 한국어 자연스러움 → ③ 얼굴 매칭 → ④ 속도.**

---

## 빠른 실행 (이미 세팅된 인스턴스)
```bash
cd ~/callone && source ~/.bashrc
bash scripts/run_all.sh          # llama·cosy·avatar·serve 한 방(+health)
cd ui && npm run dev             # :5173 (별 터미널)
# 노트북: ssh -L 5173:localhost:5173 ... 후 http://localhost:5173/call/me
```
처음 세팅은 **[docs/FRESH_SETUP.md](docs/FRESH_SETUP.md)**.

---

## 프라이버시
음성·사진·대화는 **프론트(브라우저) 소유**. 서버는 인메모리(`/dev/shm`)로만 받고 **통화 종료 시 즉시 폐기** — 디스크/로그에 본문 0. 대화 이력은 브라우저에서 내보내기/불러오기/리셋.

## 하드 제약
1. **외부 유료 API 0원** (`tests/test_no_paid_api.py` 가 정적 스캔으로 강제).
2. 개인데이터(`data/`/`models/`/`db/`)는 gitignore + 암호화. 외부 전송 금지.
3. 한국어 우선.

---

## 풀 파인튜닝 파이프라인 (고급)
긴 통화 녹음에서 화자를 분리하고 화자별 TTS·페르소나를 **학습**하는 경로(방식 B). 제로샷보다 무겁지만 충실도가 높다.
스테이지: 적재(S0) → 음질복원(S1) → 화자분리(S2) → 라벨링(S2.5) → 전사/데이터셋(S3) → TTS학습(S4) → 페르소나(S5).
- 학습 절차: [docs/1_로컬에서_학습.md](docs/1_로컬에서_학습.md), [docs/2_GPU에서_학습.md](docs/2_GPU에서_학습.md), [docs/5_화자A_목소리_학습.md](docs/5_화자A_목소리_학습.md)
- 각 스테이지: 독립 CLI + `configs/*.yaml` + `tests/test_sX.py`. 무거운 모델 없으면 안전 폴백으로 배관만 검증.
- `pip install -e .` (코어) / `pip install -e ".[heavy]"` (학습용).

## 디렉토리
```
callone/        오케스트레이터·파이프라인 (serve, ingest, diarize, tts, llm, asr ...)
avatar_server/  토킹헤드(Ditto/static) 별 프로세스
cosyvoice_server/ CosyVoice3 TTS 별 프로세스
configs/        스테이지별 yaml 설정
scripts/        세팅·실행 스크립트 (bootstrap_gpu, setup_cosyvoice_gpu, setup_avatar_gpu, run_all ...)
ui/             React 통화 화면
docs/           세팅·사용 문서 (FRESH_SETUP 우선)
tests/          pytest (폴백 경로 검증)
legacy/         초기 설계 기록(spec·스택 결정) — 코드 주석이 §인용, 보관용
```

## 라이선스
Apache-2.0. 사용 모델은 각자 라이선스 준수.
