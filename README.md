# callone — call + clone

두 사람만의 통화 녹음(m4a, 단일 채널 두 목소리 혼합)에서 **두 화자를 분리·학습**해
각 사람의 **목소리(TTS)** 와 **말투·성격(LLM 페르소나)** 을 복제하고,
**그 사람과 전화하는 듯한 UI**로 대화한다.
전 과정 **로컬 오픈소스(유료 API 0원)**, 최종 사용은 **서버(고품질) + 폰(온디바이스 경량) 2단**.

> ⚠️ **윤리 고지:** 결과물은 그 사람 *그대로*가 아니라 **근사(近似)** 다.
> 사칭·기만 금지. 사적/추모/연구 목적에 한정. 통화 녹음·개인정보 관련 관할 법규를 준수하라.

## 📖 사용 가이드 → [`docs/`](docs/README.md)
1. [로컬에서 학습](docs/1_로컬에서_학습.md) · 2. [GPU에서 학습](docs/2_GPU에서_학습.md) · 3. [노트북에서 통화](docs/3_노트북에서_통화.md) · 4. [휴대폰에서 통화](docs/4_휴대폰에서_통화.md)

(아래는 기술 명세·Model Decisions. 사용법만 필요하면 위 docs 만 보면 됨.)

---

## 빠른 시작

```bash
# 1) 설치 (코어만 — 파이프라인 배관 + 폴백)
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .

# 2) 무거운 모델(GPU/학습/실모델) — H100 노드 등
pip install -e ".[heavy]"     # torch 는 https://pytorch.org 에서 플랫폼 맞춰 먼저

# 3) 환경
cp .env.example .env          # HF_TOKEN(무료) 등 채우기
python -m callone.common.crypto genkey   # ENCRYPTION_KEY 생성 → .env

# 4) 더미 데이터로 배관 검증 (신규 설치 시. 실데이터가 이미 data/raw 에 있으면 건너뛰기)
python scripts/make_dummy_data.py --n 6        # 실데이터 있으면 자동 중단(섞임 방지)
python -m callone.pilot --n 6 --speaker A --stages s0 s1 s2 s3 s2b s25

# 5) 실데이터 파일럿 (50통 + A) — §18
bash scripts/run_pilot.sh 50 A

# 6) 서버 + UI
callone-serve                 # FastAPI :8000
cd ui && npm install && npm run dev   # :5173
```

테스트: `pytest -q` (무거운 모델 없이 31개 통과 — 폴백 경로 검증).

---

## 파이프라인 (S0~S7)

| 스테이지 | 모듈 | CLI | 산출물 |
|---|---|---|---|
| S0 적재/정규화 | `callone/ingest` | `callone-ingest` | wav16k + manifest + DB |
| S1 음질 복원 | `callone/restore` | `callone-restore` | restored/*.wav (음색 보존 가드) |
| S2 분리/전역연결/제3자제거 | `callone/diarize` | `callone-diarize`, `callone-link` | global_assignment.parquet |
| S2.5 방언/라벨링 | `callone/profile` | `callone-profile`, `callone-dialect` | profile.json |
| ASR 방언 적응 | `callone/asr_adapt` | `callone-correct`, `callone-asr-train` | models/asr_dialect |
| S3 전사/데이터셋/PII | `callone/asr`, `callone/dataset` | `callone-transcribe`, `callone-build-tts`, `callone-build-dlg` | TTS셋/대화셋/페르소나카드 |
| S4 음성 클론 | `callone/tts` | `callone-tts-train`, `callone-tts-phone`, `callone-tts-infer` | models/tts_{server,phone} |
| S5 페르소나 두뇌 | `callone/llm` | `callone-llm-sft`, `callone-llm-train` | models/llm_{12b,e4b} |
| S6 실시간 대화 | `callone/serve` | `callone-serve` | WebSocket 통화 |
| S7 UI + 배포 | `ui/` | `npm run dev` | 전화 화면 + 라벨링 편집기 |

각 스테이지는 **독립 CLI + `configs/*.yaml` + `tests/test_sX.py`**. 중간 산출물은 디스크 저장(재현성).
무거운 모델이 없으면 **안전 폴백**(더미 분리/리샘플/placeholder 합성)으로 배관이 끊기지 않는다.

---

## Model Decisions (§3 — 최신성 검증 의무 기록)

> 이 분야는 주 단위로 바뀐다. 아래는 **2026-06-08 기준 baseline**(스펙 §3 표)을 채택한 것.
> **각 모델을 코드에 고정하기 전, 해당 스테이지 착수 시 웹으로 최신본을 재확인하고 이 표를 갱신할 것.**
> 설정값은 모두 `configs/*.yaml` 로 파라미터화되어 있어 교체가 쉽다.

**검증일: 2026-06-08 (웹 확인 완료).** 출처는 절 하단 "Sources".

| 구성요소 | 채택 | config 키 | 검증 | 근거(2026-06) |
|---|---|---|---|---|
| 두뇌 LLM (GPU 서버) | **Gemma 4 12B** (네이티브 오디오) | `llm_server.base_model` | ✅ 2026-06-08 | 16GB 통합메모리, 12B Unified 2026-06-03 출시 |
| 두뇌 LLM (노트북 온디바이스) | **Qwen 3.5-4B + LoRA** (llama.cpp GGUF Q4, Arc iGPU) | `serve.llm.backend=llama` | ✅ 2026-06-11 **실측** | 갤럭시북5 Pro(Arc 140V) Vulkan **~20 tok/s 실시간**. ⚠️ **OpenVINO는 qwen3_5 변환 불가**(아키텍처 GDN+MoE+MTP, optimum-intel #1628) → **llama.cpp 로 전환**. llama-server 별프로세스+HTTP라 torch/OV segfault 회피 |
| 두뇌 LLM (노트북) | Gemma 4 E4B (+QAT) | (티어 자동) | ✅ 2026-06-08 | CPU 2~5 tok/s. 노트북엔 E4B |
| ASR (오프라인+방언적응) | **Whisper large-v3 + LoRA** | `asr.model` | ✅ 2026-06-08 | 사투리 적응(§13)이 Whisper-LoRA 기반. 정확도 우선 |
| ASR (실시간, GPU) | **large-v3-turbo** (greedy) | `serve.asr.model=auto` | ✅ 2026-06-08 | 속도. 대안 **Voxtral**(한국어·스트리밍·WER 5.9<7.4) 도입 가능 |
| ASR (실시간, CPU) | **Whisper small/int8** | (티어 자동) | ✅ 2026-06-08 | 노트북 CPU 실시간엔 small. turbo는 GPU에서 |
| 화자 분리 | **pyannote community-1** (4.0) | `s2_diarize.diarizer` | ✅ 2026-06-08 **변경** | 2026-02 출시, **3.1보다 우수**(잡음 통화 강함), 무료 CC-BY-4.0. `precision-2`(유료) **금지** |
| 화자 임베딩 | ECAPA-TDNN (SpeechBrain) | `s2_diarize.embedding` | ✅ | WeSpeaker 대안 |
| 음질 복원 | DeepFilterNet / Resemble Enhance | `s1_restore.*` | ✅ | 오픈 |
| TTS 서버 | **Qwen3-TTS**(한국어·스트리밍, 2026-01) / **VoxCPM2**(48k·LoRA 5~10분, 2026-04) | `tts_server.backend` | ✅ 2026-06-08 | ⚠️ **로컬 가중치만**, 클라우드 보이스 API 금지 |
| TTS 노트북/폰 온디바이스 | **Piper (화자별 학습, onnx)** | `serve.tts.backend` | ✅ 2026-06-11 **채택** | 화자 TTS셋(~73분)으로 파인튜닝=**최고 충실도+CPU 실시간**(onnx, torch 불필요). `scripts/train_piper.md`. 학습 전엔 Kokoro(제로샷)/placeholder 폴백 |
| RAG 임베딩 | EmbeddingGemma | `serve.rag.embedder` | ✅ | 오픈 |
| 장기기억 | Mem0 (JSON 폴백) | `serve.memory.backend` | ✅ | 오픈 |
| 서빙 | vLLM(GPU) / Ollama·LiteRT(노트북·폰) | `serve.llm.backend` | ✅ | 오픈 |
| 실시간 오케스트레이션 | 경량 자체 루프(→Pipecat/LiveKit 확장) | `serve.orchestration` | ✅ | 오픈 |

### 기기별 배치 (자동 선택 — `common/hardware.detect_tier`)
`serve.yaml tier: auto` 면 하드웨어를 감지해 **티어별로 올바른 모델을 자동 선택**한다.
(`CALLONE_TIER` 환경변수로 강제 가능. 검증: `python -c "from callone.common.hardware import describe; print(describe())"`)

| 티어 | 감지 조건 | LLM | 실시간 ASR | TTS | 비고 |
|---|---|---|---|---|---|
| **server_gpu** | NVIDIA GPU(H100 등) | Gemma 4 **12B** (vLLM) | large-v3-turbo fp16 | Qwen3-TTS/VoxCPM2 48k | 첫 음성 <0.8~1.2s 달성 |
| **laptop (Arc iGPU)** | Intel Arc(예: 갤북5 Pro) | **Qwen3.5-4B+LoRA** (llama.cpp GGUF/Vulkan) | large-v3-turbo / small int8 | **Piper(화자학습)** | ~20 tok/s 실시간. docs/3 |
| **phone** | 온디바이스(명시) | Gemma 4 **E4B/E2B**+QAT | small / Gemma4 오디오 | per-speaker 소형 | LiteRT/MediaPipe/MLX |

> **통화 UI(화면)** 는 폰이든 노트북이든 **동일한 반응형 웹 클라이언트**(`ui/`)다.
> 폰/노트북 구분은 "UI를 어디서 보느냐"가 아니라 **"모델을 어느 기기가 돌리느냐(=callone-serve가 뜬 곳)"** 로 갈린다.
> 폰으로 통화 화면만 열고 모델은 H100 서버가 도는 게 기본(하이브리드). 완전 오프라인이면 폰 온디바이스 티어.

### 디바이스 자동 적응 (코드 레벨)
- 모든 모델 로더는 `common/io.resolve_device()` 경유 → `.env DEVICE=cuda` 라도 GPU 없으면 **자동 cpu+int8 강등**(경고 1회).
- 모델 선택은 `common/hardware.tier_defaults()` 경유 → 티어별 LLM/ASR/TTS 자동.
- faster-whisper/SpeechBrain/WhisperX/Resemble Enhance/PersonaLLM/StreamASR 전부 이 두 헬퍼 사용.

### 실시간성 / 지연 예산 (§16, 진짜 전화처럼)
- 오케스트레이터는 **문장 단위 스트리밍**: LLM 첫 문장 즉시 TTS→스피커, 나머지는 뒤에서 생성(`Turn.first_audio_latency_ms` 측정).
- 목표(GPU): 첫 음성 <0.8~1.2s = VAD~200ms + turbo ASR + LLM 첫 토큰 200~400ms + TTS 첫 청크 150~300ms.
- ⚠️ 이 속도는 **server_gpu 티어에서만**. laptop_cpu/phone-CPU 는 느리다(데모·오프라인용).

**Sources (2026-06-08 검증):**
[Gemma 4 12B(MarkTechPost)](https://www.marktechpost.com/2026/06/03/google-deepmind-releases-gemma-4-12b-an-encoder-free-multimodal-model-with-native-audio-that-runs-on-a-16-gb-laptop/) ·
[Gemma 4 하드웨어/E2B·E4B](https://ai.google.dev/gemma/docs/core) ·
[Gemma 4 CPU 속도(Ollama)](https://codersera.com/blog/how-to-run-gemma-4-with-ollama-setup-guide/) ·
[Voxtral vs Whisper 2026](https://weesperneonflow.ai/en/blog/2026-03-31-voxtral-whisper-open-source-speech-models-comparison-2026/) ·
[오픈 STT 2026(Northflank)](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks) ·
[pyannote community-1](https://www.pyannote.ai/blog/community-1) ·
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) ·
[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) ·
[온디바이스 TTS 2026](https://getstream.io/blog/best-on-device-tts-models/)

### 구현 가정 (Appendix B-7)
- 실데이터·HF 토큰·AIHub 계정은 **사용자가 제공**(경로/토큰은 `.env`). 없으면 더미 데이터로 배관 검증.
- 무거운 학습(ASR/TTS/LLM 파인튜닝)은 **H100 노드**에서 실행. 코어 설치만으로는 레시피/명령을 출력하는 폴백.
- 제주 방언은 데이터에 없음 → 방언 후보에서 기본 제외(폴백만 유지, §11.1).

---

## 하드 제약 (준수, §2)
1. **외부 유료 API 0.** OpenAI/ElevenLabs/AssemblyAI/Gemini API/pyannote `precision-2` 등 금지.
   → `tests/test_no_paid_api.py` 가 금지 도메인/패키지를 정적 스캔하여 발견 시 **빌드 실패**.
2. **데이터 로컬 보관 + 외부 전송 금지.** `data/`/`models/`/`db/` 는 gitignore + 암호화(`crypto.py`).
3. **한국어 + 방언 필수.** 방언 지역·세기는 사전 정의 없이 **데이터에서 자동 측정**(S2.5).
4. **모노믹스** → diarization 풀 파이프라인.
5. **2단 배포**(서버 고품질 + 폰 경량). 일상 대화라 경량 우선.
6. **모듈성:** 각 스테이지 독립 실행/재실행 + CLI + 디스크 산출물.

## 보안·윤리 (§20)
- 통화 원본/전사/학습셋: `data/`(gitignore) + 디스크 암호화(`ENCRYPTION_KEY`).
- PII 마스킹은 학습셋 생성 파이프라인에 **강제**(`callone/asr/pii.py`): 이름/전화/주소/주민번호/계좌.
- 폰 온디바이스 모드 = 데이터가 기기를 벗어나지 않는 프라이버시 기본값 권장.

## 평가 (§19)
- DER < 12%, SECS > 0.70, 자가 WER < 10%, 첫 음성 < 1.2s(목표 0.8s), 블라인드 A/B ≈ 50%, PII 누출 0.
- `tests/test_sX.py` + `reports/` 리포트.

## 미스보이스 UI
`ui/` 는 missvoice(https://github.com/ksiwon/missvoice) 화면 골격을 차용하되,
**외부 유료 API 레이어(`audioProcessingAPI.ts`)는 제거**하고 `ui/src/api/calloneClient.ts`
(로컬 백엔드 REST + WebSocket)로 교체했다. missvoice 로직은 쓰지 않고 UI 톤만 차용.

## 라이선스
Apache-2.0. 사용 모델은 각자 라이선스(대부분 Apache-2.0/MIT, 게이트 모델은 무료 동의) 준수.
