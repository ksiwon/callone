# callone — 종합 기술 기획서 & 구현 명세서 (Build Spec)

> **이 문서 하나만으로 callone 전체를 처음부터 설계·구현할 수 있도록 작성된 단일 사양서입니다.**
> 대상 구현자: **Claude Code**(자율 코딩 에이전트). 이 문서를 처음부터 끝까지 읽고, §17의 빌드 순서대로 레포를 스캐폴딩하고 단계별로 구현·테스트하세요.
> 버전: v1.0 (handoff) · 작성 기준일: 2026-06-08

---

## 목차
0. 프로젝트 한 줄 요약 / 1. 목표·범위·성공기준 / 2. 하드 제약(반드시 준수) / 3. ⚠️ 모델 최신성 검증 의무 / 4. 시스템 아키텍처 / 5. 레포 구조 / 6. 환경·의존성 / 7. 데이터 스키마(전부) / 8. S0 적재 / 9. S1 음질 복원 / 10. S2 화자 분리·전역연결·제3자 제거 / 11. S2.5 방언 자동 프로파일링 + 화자 라벨링 편집기 / 12. S3 전사·데이터셋·PII / 13. ASR 방언 적응 / 14. S4 음성 클론(서버+폰 2단) / 15. S5 페르소나 두뇌(Gemma 4) / 16. S6 실시간 대화 / 17. S7 UI + 2단 배포 / 18. 빌드 순서·마일스톤(파일럿 우선) / 19. 평가·수용 테스트 / 20. 보안·윤리·법률 / 21. 참고 / 부록

---

## 0. 프로젝트 한 줄 요약
**callone** = call + clone. 두 사람만의 통화 녹음(m4a, 1,000+개, 평균 5분, **단일 채널에 두 목소리 혼합**)에서 두 화자를 분리·학습하여, 각 사람의 **목소리(TTS)** 와 **말투·성격(LLM 페르소나)** 을 복제하고, **그 사람과 전화하는 듯한 UI**로 대화하게 만든다. 전 과정 **로컬 오픈소스(유료 API 0원)**, 최종 사용은 **서버(고품질) + 폰(온디바이스 경량) 2단**.

---

## 1. 목표·범위·성공기준

### 1.1 산출물
1. **데이터 파이프라인**: m4a 1,000+개 → 복원 → 화자 분리 → 전역 화자 연결(A/B 확정) → 전사 → 사람별 (a)음성 학습셋 / (b)대화 학습셋 + 페르소나 카드.
2. **화자 라벨링 편집기(UI)**: 분리된 A/B에 사람이 직접 이름·나이·성별·관계·특징 입력. 방언은 **자동 프로파일링 결과를 초안으로 제시**하고 사람이 확인/수정.
3. **음성 클론(2단)**: 서버 고품질 TTS 파인튜닝 + 폰용 사람별 소형 TTS.
4. **페르소나 두뇌(2단)**: Gemma 4 12B(서버) / E4B·E2B(폰) LoRA + 페르소나 카드 + RAG/장기메모리.
5. **실시간 대화 시스템**: (스트리밍 ASR 또는 Gemma 오디오) → LLM → 스트리밍 TTS, VAD/턴테이킹.
6. **전화 UI**: "A에게 전화 / B에게 전화" 통화 화면.

### 1.2 범위 밖(Non-goals)
- 영어 코드스위칭 처리(데이터에 없음). 제주 방언(데이터에 없음 — 일반 폴백만 유지). 다수(3인+) 동시 대화 모델(1:1 통화 기준).

### 1.3 성공 기준(정량/정성)
- 화자 분리: 전역 화자 귀속 정확도 높음, DER < 12%(전화 환경 현실 목표). 제3자 세그먼트 제거율 높음.
- 음성: 화자 유사도(SECS, 화자검증 임베딩 코사인) > 0.70, 자가 ASR 재인식 WER < 10%, MOS 청취 양호.
- 페르소나: 블라인드 A/B(실제 vs 클론) 식별률이 우연(50%)에 근접, 입버릇·사투리 재현, 연속 질문 일관성.
- 시스템: 첫 음성 응답 지연 < 1.2s(스트리밍), 목표 < 0.8s.

---

## 2. 하드 제약 (반드시 준수)

1. **외부 유료 API 금지.** OpenAI/GPT, ElevenLabs, AssemblyAI, Gemini API, pyannote `precision-2`(유료) 등 **결제형 API를 코드/문서/의존성 어디에도 넣지 말 것.** 전부 **로컬 실행 오픈소스**로 구현. (이전 시도가 GPT‑4·ElevenLabs 비용으로 실패했음.)
   - 단, **무료 계정/토큰**은 허용: Hugging Face 무료 토큰(게이트 모델 다운로드), Gemma·pyannote·AIHub **라이선스 동의**(무료). 이는 결제가 아님. 모두 `.env`로 파라미터화.
   - ⚠️ 주의: **Qwen3‑TTS는 로컬 가중치로만 사용**(Apache‑2.0, 무료). Alibaba 클라우드 보이스 API는 유료이므로 호출 금지.
2. **데이터는 전부 로컬/서버에 보관, 외부 전송 금지.** 통화엔 민감정보 다수 → 저장 시 암호화, PII 마스킹(§20).
3. **언어 = 한국어. 방언 처리 필수. 방언의 지역·세기는 사전 정의하지 말고 데이터에서 자동 측정**(§11).
4. **녹음은 모노믹스**(한 채널에 두 목소리) → diarization 풀 파이프라인 필수.
5. **2단 배포**: 서버(고품질) + 폰(온디바이스 경량). 일상 대화 수준이라 경량 우선.
6. **모듈성**: 각 스테이지(S0~S7)는 독립 실행/재실행 가능한 모듈 + CLI. 중간 산출물은 디스크에 저장(재현성).
7. **모델 버전은 §3의 검증 의무를 따른다.**

---

## 3. ⚠️ 모델 최신성 검증 의무 (가장 중요한 운영 규칙)

이 분야는 **주 단위로 바뀐다.** 아래 표는 **2026‑06‑08 기준 확인된 baseline**이다. **Claude Code는 각 모델을 코드에 고정(pin)하기 전에, 웹에서 해당 모델의 최신 버전/대체 모델을 반드시 재확인하고, 더 적합한 최신본이 있으면 채택한 뒤 그 결정을 README에 기록할 것.**

검증 절차(스테이지 착수 직전마다 1회):
1. 해당 구성요소(예: "한국어 오픈소스 TTS 보이스 클론")를 웹 검색.
2. baseline 대비 (a) 더 최신 버전, (b) 한국어/방언 지원, (c) 라이선스(상업·무료), (d) 온디바이스 가능 여부를 비교.
3. 채택 모델·버전·날짜·이유를 `README.md`의 "Model Decisions" 섹션과 해당 `configs/*.yaml`에 기록.

### Baseline 모델표 (2026‑06‑08 확인)
| 구성요소 | 서버(고품질) | 폰(온디바이스) | 라이선스 | 검증 포인트 |
|---|---|---|---|---|
| 두뇌 LLM | **Gemma 4 12B (Unified, 네이티브 오디오)** | **Gemma 4 E4B/E2B + QAT** | Apache‑2.0 | 신규 12B(2026‑06‑03). 더 최신 Gemma/대체 확인 |
| ASR | **Whisper large‑v3** | Whisper turbo/small, Gemma4 오디오 | MIT | Voxtral/Canary 등 한국어 성능 재확인 |
| 화자 분리 | **pyannote community‑1 / 3.1** + WhisperX | — | 오픈(무료) | `precision-2`(유료) 금지 |
| 화자 임베딩 | ECAPA‑TDNN(SpeechBrain) / WeSpeaker | — | 오픈 | |
| 음질 복원 | ClearerVoice‑Studio(FRCRN/MossFormer2/SR) + Resemble Enhance | DeepFilterNet(실시간) | 오픈 | |
| TTS | **Qwen3‑TTS**(한국어·스트리밍·FT) / **VoxCPM2**(48k·LoRA) | **Piper/VITS · MeloTTS · Kokoro+KokoClone · GPT‑SoVITS**(사람별 학습) | Apache‑2.0/MIT | 한국어·온디바이스·FT 가능 재확인 |
| RAG 임베딩 | EmbeddingGemma | 경량 | 오픈 | |
| 장기기억 | Mem0 / Zep | — | 오픈 | |
| 서빙/런타임 | vLLM | LiteRT‑LM / MediaPipe LLM / Ollama(QAT) / MLX / llama.cpp | 오픈 | |
| 실시간 오케스트레이션 | Pipecat + LiveKit(WebRTC) | — | 오픈 | |
| 벡터DB | FAISS / Chroma / Qdrant | SQLite‑VSS | 오픈 | |

---

## 4. 시스템 아키텍처

```
[S0] 적재/정규화      m4a(모노믹스) → wav 표준화(16k/원본), 메타 DB
   ↓
[S1] 음질 복원        denoise + 대역확장(8k→48k), 근단/원단 비대칭 처리
   ↓
[S2] 화자 분리/전역연결 통화별 2화자 분리 → 임베딩 → 전 통화 A/B 통합 → 제3자 이상치 제거
   ↓
[S2.5] 방언 자동 프로파일링 + 화자 라벨링 편집기
        (사투리 지역·세기 자동 측정 → 초안 → 사람이 이름/나이/성별/관계/특징 확정)
   ↓
[ASR-adapt] 자체 통화 일부 수동교정 → Whisper 방언/저음질 적응 (+선택 AIHub)
   ↓
[S3] 전사/데이터셋    적응 ASR로 전사(사투리 보존) → (a)음성셋 (b)대화셋 + 페르소나 카드, PII 마스킹
   ↓
[S4] 음성 클론        서버: Qwen3-TTS/VoxCPM2 FT │ 폰: 사람별 소형(Piper/MeloTTS) 학습
   ↓
[S5] 페르소나 두뇌    Gemma 4 (12B 서버/E4B·E2B 폰) LoRA + 페르소나 카드 + RAG/메모리
   ↓
[S6] 실시간 대화      (스트리밍 ASR 또는 Gemma 오디오) → LLM → 스트리밍 TTS, VAD/턴테이킹
   ↓
[S7] UI + 배포        LiveKit/WebRTC 통화 화면 + 화자 라벨링 편집기 / 서버·폰 2단
```

**설계 원칙:** 모듈식(cascade) 우선(부품 독립 교체·평가 용이, Gemma 두뇌 결합 용이). 풀듀플렉스(말끊기/맞장구)는 완성 후 선택적 고도화. Gemma 4가 오디오 입력을 직접 받으므로 ASR을 흡수하는 준‑E2E 단순화도 허용.

---

## 5. 레포 구조 (Claude Code: 이대로 스캐폴딩)

```
callone/
├─ README.md                      # 개요 + "Model Decisions"(§3 기록) + 실행법
├─ pyproject.toml | requirements.txt
├─ .env.example                   # HF_TOKEN, AIHUB creds(optional), CALLONE_DATA_DIR, DEVICE, ENCRYPTION_KEY
├─ configs/
│   ├─ default.yaml
│   ├─ s0_ingest.yaml  s1_restore.yaml  s2_diarize.yaml  s25_profile.yaml
│   ├─ asr.yaml  asr_adapt.yaml  s3_dataset.yaml
│   ├─ tts_server.yaml  tts_phone.yaml  llm_server.yaml  llm_phone.yaml  serve.yaml
├─ callone/                       # 파이썬 패키지
│   ├─ common/        schemas.py(pydantic) audio.py io.py db.py crypto.py logging.py
│   ├─ ingest/        s0_convert.py manifest.py            # CLI: callone-ingest
│   ├─ restore/       s1_restore.py                        # CLI: callone-restore
│   ├─ diarize/       s2_diarize.py embeddings.py s2b_link.py filter_thirdparty.py
│   ├─ profile/       s25_dialect.py s25_profile.py        # 방언 자동측정 + 화자카드
│   ├─ asr/           s3_transcribe.py pii.py
│   ├─ asr_adapt/     make_correction_set.py whisper_finetune.py
│   ├─ dataset/       build_tts.py build_dialogue.py persona_card.py
│   ├─ tts/           train_server.py train_phone.py infer.py eval.py
│   ├─ llm/           prepare_sft.py train_lora.py persona_prompt.py rag.py memory.py
│   ├─ serve/         vad.py asr_stream.py llm_server.py tts_stream.py orchestrator.py app.py
│   └─ pilot.py                                            # 50통 파일럿 엔드투엔드
├─ scripts/          run_pilot.sh  run_full.sh  download_aihub.md
├─ ui/               # React(미스보이스 골격 재사용) — CallScreen, SpeakerCardEditor, ContactList
├─ data/             raw/ wav16k/ restored/ diarized/ speakers/ datasets/   # gitignore, 암호화
├─ models/           asr_dialect/ tts_server/ tts_phone/ llm_12b/ llm_e4b/  # gitignore
├─ db/               callone.sqlite  vectors/
└─ tests/            test_s0.py ... test_s6.py  conftest.py
```

**미스보이스 UI 재사용:** `git clone https://github.com/ksiwon/missvoice.git` 후 `src/`(Upload/Processing/Chat 페이지 + styled-components 테마)를 `ui/`로 이식하고, `src/api/audioProcessingAPI.ts`(외부 유료 API 호출/목 데이터)는 **삭제하고 callone 로컬 백엔드(WebSocket/LiveKit) 클라이언트로 교체**. (missvoice 자체는 백엔드 없는 데모이므로 로직은 쓰지 말고 UI 골격만 차용.)

---

## 6. 환경·의존성

- **OS/HW:** Linux(Ubuntu 22.04+), CUDA 12.x, GPU = Elice **H100 80GB 1대**(학습). 서빙은 더 작은 GPU 가능. 폰: Android(LiteRT/MediaPipe)·iOS(MLX/llama.cpp).
- **Python:** 3.10–3.12. 가상환경 권장. `pip install --break-system-packages` 류 회피(venv 사용).
- **핵심 패키지(무료 오픈):** `torch torchaudio`, `transformers peft accelerate bitsandbytes datasets`, `vllm`, `faster-whisper whisperx`, `pyannote.audio`, `speechbrain` 또는 `wespeaker`, `ClearerVoice-Studio`, `resemble-enhance`, `deepfilternet`, `qwen-tts`(또는 VoxCPM repo), `piper-tts`/`MeloTTS`/`GPT-SoVITS`(폰), `faiss-cpu`/`chromadb`/`qdrant-client`, `mem0`/`zep`, `pipecat-ai`, `livekit-server-sdk`, `fastapi uvicorn websockets`, `pydantic`, `ffmpeg`(시스템), `librosa soundfile`.
- **토큰/계정(.env):** `HF_TOKEN`(필수, 무료 — pyannote·Gemma 게이트), AIHub 계정(선택 — 방언/저음질 데이터), `CALLONE_DATA_DIR`, `DEVICE=cuda`, `ENCRYPTION_KEY`.
- **모델 다운로드:** HF/Kaggle에서 가중치 로컬 캐시. 네트워크 제한 환경이면 사전 다운로드 안내.

---

## 7. 데이터 스키마 (전부 — `callone/common/schemas.py` pydantic으로 구현)

### 7.1 통화 메타 (db `calls` 테이블 / `manifest.parquet`)
```json
{"call_id":"call_00001","src_path":"data/raw/call_00001.m4a",
 "wav16k_path":"data/wav16k/call_00001.wav","restored_path":"data/restored/call_00001.wav",
 "duration_sec":312.4,"orig_sr":8000,"orig_channels":1,"codec":"aac","created_at":"..."}
```

### 7.2 통화별 분리 결과 (`data/diarized/{call_id}.json`)
```json
{"call_id":"call_00001",
 "segments":[{"start":0.0,"end":3.2,"local_speaker":"SPK_00","text":"...","words":[...],
              "asr_conf":0.91,"snr_db":18.2,"overlap":false,"embedding_ref":"emb_000123"}]}
```

### 7.3 전역 화자 연결 결과 (`data/speakers/global_assignment.parquet`)
```json
{"segment_uid":"call_00001#0","call_id":"call_00001","start":0.0,"end":3.2,
 "global_speaker":"A","sim_A":0.83,"sim_B":0.21,"is_thirdparty":false,"is_overlap":false,"clean":true}
```

### 7.4 화자 프로필 (`data/speakers/{A|B}/profile.json`) — S2.5
```json
{"speaker_id":"A",
 "auto":{                                   // 자동 추출(초안, 수정 가능)
   "gender_est":"F","age_band_est":"60s",
   "dialect":{"region_est":"gyeongsang","confidence":0.78,
              "intensity_0to1":0.62,                       // 데이터에서 측정한 사투리 '세기'
              "markers":[{"form":"~카이","count":214,"std":"~니까"},
                         {"form":"머라카노","count":33,"std":"뭐라고"}],
              "examples":["밥은 묵었나","이래가 안 된다카이"]},
   "speech":{"avg_sentence_len":8.4,"banmal_ratio":0.95,"top_fillers":["아이고","마"],
             "question_rate":0.18}},
 "user":{                                    // 사람이 입력/확정(라벨링 편집기)
   "name":"화자 A","age":63,"gender":"F","relation":"어머니",
   "register":"반말","traits":["걱정 많고 따뜻함"],"catchphrases":["밥은 묵었나"],
   "taboo":[],"dialect_confirmed":true},
 "tts":{"server_model":"qwen3-tts-ft-A","phone_model":"piper-A"},
 "llm":{"lora_12b":"models/llm_12b/A","lora_e4b":"models/llm_e4b/A"}}
```

### 7.5 TTS 학습셋 (`data/datasets/{A|B}/tts/metadata.csv`, LJSpeech 류)
```
wav_path|text|duration|snr
data/datasets/A/tts/wavs/A_000001.wav|밥은 묵었나|2.1|19.3
```

### 7.6 대화 학습셋 (`data/datasets/{A|B}/dialogue/train.jsonl`)
```json
{"messages":[
  {"role":"system","content":"<페르소나 카드 from profile.json>"},
  {"role":"user","name":"화자 B","content":"화자 A 나 왔어"},
  {"role":"assistant","content":"<thinking>반갑게 맞고 밥 챙기는 게 평소</thinking>아이고 왔나~ 밥은 묵었나?"}
]}
```
- `thinking` 태그는 TAU(Think-Aloud) 증강(선택). 상대 발화엔 관계 기반 `name`(화자 B/친구 등) 부여.

### 7.7 페르소나 카드(시스템 프롬프트, `persona_card.py`가 profile.json에서 생성)
- 포함: 정체성(이름/관계), 말투(반말/존댓말, 평균 문장 길이), **사투리(지역·세기·대표 어미 — 자동 측정값)**, 입버릇/감탄사, 자주 묻고/답하는 주제, 유머 스타일, 금기, "질문 유형별 전형 답변" 요약, "모르면 모른다"는 원칙.

---

## 8. S0 — 적재 & 정규화 (`callone/ingest`)
- **목적:** m4a → 표준 wav 2종 + 메타 DB.
- **입력:** `data/raw/*.m4a`. **출력:** `data/wav16k/*.wav`(16kHz mono, ASR/분리용), `data/restored/`는 S1에서 채움, `manifest.parquet` + `db.calls`.
- **로직:** `ffprobe`로 sr/channels/codec/duration 수집 → `ffmpeg`로 16k mono 변환(라우드니스 정규화 `loudnorm I=-23`, DC offset 제거). 원본 보존본 경로 기록.
```bash
ffmpeg -i {src} -ac 1 -ar 16000 -af "loudnorm=I=-23:LRA=7,highpass=f=40" {wav16k}
```
- **수용기준:** 모든 파일이 manifest에 1행, 16k mono wav 생성, 손상 파일은 `status=error`로 격리.

## 9. S1 — 음질 복원 (`callone/restore`)
- **목적:** 협대역·압축·잡음 복원 + 대역확장(8k→48k). 근단/원단 비대칭 고려.
- **도구:** ① denoise: ClearerVoice `FRCRN`/`MossFormer2_SE_48K` 또는 DeepFilterNet. ② 대역확장/초해상: Resemble Enhance(denoise+enhance 44.1k) 또는 ClearerVoice SR. ③ 겹말 정리(옵션): ClearerVoice 타깃화자추출.
- **출력:** `data/restored/{call_id}.wav`(48k, 복원). TTS 학습셋은 **복원본** 기준.
- **⚠️ 과복원 가드:** 강한 enhance는 음색(=화자 정체성)을 바꿈. **원본 대비 화자 임베딩 코사인 유사도 모니터링**(임계값 미만으로 떨어지면 enhance 강도 자동 완화). 샘플 청취 리포트 생성.
- **수용기준:** 복원 전후 SNR/PESQ 개선, 임베딩 유사도 > 임계값(음색 보존), 청취 샘플 10개 리포트.

## 10. S2 — 화자 분리 + 전역 연결 + 제3자 제거 (`callone/diarize`)
**전제:** 95%+ 고정 A↔B, 제3자 극소량. 모노믹스 → diarization 필수.
- **(a) 통화별 분리:** WhisperX(Whisper large‑v3/적응본 + pyannote) 또는 pyannote community‑1. `min_speakers=2, max_speakers=3`(제3자 대비). 단어 타임스탬프 정렬. SNR·overlap 플래그.
- **(b) 임베딩:** 화자별 대표 발화 → ECAPA‑TDNN(SpeechBrain) 또는 WeSpeaker 임베딩.
- **(c) 전역 연결:** 깨끗 통화로 A·B 센트로이드 2개 → 전 세그먼트 코사인 유사도로 A/B 귀속.
- **(d) 제3자/잡음 제거:** 전체 임베딩 HDBSCAN(또는 임계값). **2개 주군집=A·B, 나머지 소군집/이상치=제3자 → 제거.** 두 센트로이드 모두에 유사도 낮은 세그먼트 드롭. overlap 세그먼트는 TTS 제외(대화 transcript엔 보존).
- **출력:** `global_assignment.parquet`(7.3). 화자 통계(시간/세그먼트 수) 리포트.
- **수용기준:** A·B 정확히 2개 주군집 형성, 제3자 분리, (가능하면 소량 수동 라벨로) DER<12% 확인.

## 11. S2.5 — 방언 자동 프로파일링 + 화자 라벨링 편집기 (`callone/profile`)
> **방언의 지역·세기를 미리 정의하지 않고, 데이터에서 화자별로 측정한다.**

### 11.1 방언 자동 프로파일링 (`s25_dialect.py`)
- **입력:** 화자별 전사 텍스트(S3 1차 전사) + (선택)음성.
- **방법:**
  1. **사투리 마커 사전 구축:** AIHub 방언 데이터의 **방언↔표준어 대응쌍**에서 지역별(경상/전라/충청/강원) 특징 어미·어휘 목록을 추출해 `resources/dialect_markers/{region}.json` 생성(없으면 규칙 시드 + 코퍼스 확장).
  2. **지역 추정:** 화자 전사에서 각 지역 마커 출현 빈도 → 최빈 지역 + confidence(소프트맥스). **제주는 데이터에 없으므로 후보에서 기본 제외(폴백만 유지).**
  3. **세기(intensity) 측정:** `사투리 마커 토큰 수 / 전체 토큰 수` 비율, 그리고 (선택) 표준어 역정규화 모델로 문장별 변형 정도(편집거리/perplexity gap)를 평균 → `intensity_0to1`. 같은 지역이라도 사람마다 값이 다르게 나옴(요구사항 반영).
  4. (선택) **음향 dialect‑ID 분류기**로 보조 신호. 텍스트 마커가 1차, 음향은 보조.
- **출력:** `profile.json`의 `auto.dialect`(지역·confidence·intensity·markers·examples). 이 값이 ASR 적응 데이터 선택, 페르소나 카드, 라벨링 편집기 초안에 사용.

### 11.2 화자 라벨링 편집기 (UI + `s25_profile.py` 백엔드)
- **흐름:** 각 정체성 A/B에 대해 (i) 대표 발화 5~10개 **재생**, (ii) `auto`(추정 성별/나이대/사투리 지역·세기/말투) **초안 표시**, (iii) 사람이 `user` 필드 확정.
- **필드:** 이름·나이·성별·관계·호칭(반말/존댓말)·특징·입버릇·금기·**사투리 확인(dialect_confirmed)**. 사람이 지역/세기를 덮어쓸 수 있음.
- **저장:** `data/speakers/{id}/profile.json`. 백엔드 REST `GET/PUT /api/speakers/{id}/profile`.

## 12. S3 — 전사·데이터셋·PII (`callone/asr`, `callone/dataset`)
- **전사:** **방언 적응 Whisper**(§13) 또는 Gemma 4 오디오. **사투리 어형 보존**(표준어로 교정 금지; 표준어 대응은 별도 필드 보관 가능). 단어 타임스탬프 정렬.
- **PII(`pii.py`):** 정규식 + 한국어 NER로 이름·전화·주소·주민번호·계좌 등 마스킹/치환(`[NAME]`,`[PHONE]`). 학습셋 저장 시 적용. 원본은 암호화 보관.
- **(a) 음성 학습셋(`build_tts.py`):** `clean=true & overlap=false & snr≥임계` 세그먼트만, 3–15초 컷 → `metadata.csv`(7.5). 한국어 텍스트 정규화(숫자/기호→한글). 사람당 정제 후 5–20h 목표(후보 30h+ → 품질 선별).
- **(b) 대화 학습셋(`build_dialogue.py`):** 시간순 멀티턴 복원, **assistant=클론 대상**, 상대=`user`(관계 name). 선택 TAU 증강. → `train.jsonl`(7.6). 페르소나 카드(`persona_card.py`)는 profile.json(자동+사용자)에서 생성.
- **수용기준:** TTS 세그먼트 품질 필터 통과, 대화 jsonl 역할 정합, PII 누출 0(샘플 검수).

## 13. ASR 방언 적응 (`callone/asr_adapt`) — 정확도의 핵심
- **최우선:** `make_correction_set.py`로 본인 통화에서 **2~5시간 분량 세그먼트를 샘플링**(다양한 화자/주제) → 사람이 전사 **수동 교정**(간단 교정 UI 또는 CSV). → `whisper_finetune.py`로 Whisper large‑v3 **LoRA/파인튜닝**(faster-whisper/transformers). 그 **특정 사투리 + 두 화자 + 전화 음질**에 동시 적응.
- **보강(선택):** S2.5에서 추정된 지역에 맞춰 AIHub **방언 발화 데이터** + **저음질 전화망 음성인식 데이터**로 부트스트랩. HF의 한국어 저음질 통화 Whisper 파인튜닝 체크포인트를 출발점으로 사용 가능.
- **출력:** `models/asr_dialect/`. **수용기준:** 보류(held-out) 자체 통화에서 적응 전후 WER 유의미 개선.

## 14. S4 — 음성 클론 (서버 + 폰 2단) (`callone/tts`)
> 사람당 30~40h로 데이터가 풍부 → **per‑speaker 학습**이 제로샷보다 유리, 특히 폰.
- **서버(고품질):** **Qwen3‑TTS 로컬 파인튜닝**(한국어·스트리밍). 더 고음질은 **VoxCPM2(48k, LoRA)** 또는 Llasa‑Korean. 감정엔 IndexTTS2. → `models/tts_server/{A,B}`.
- **폰(온디바이스):** 사람별 **소형 단일화자 모델 학습** — **Piper/VITS**(초경량) · **MeloTTS**(한국어, CPU 실시간) · Kokoro+KokoClone · GPT‑SoVITS. 동일 정제 코퍼스 재사용. → `models/tts_phone/{A,B}`.
- **사투리:** 음성 학습으로 **자동 반영**(전사가 사투리 보존이면 됨). **비대칭:** 원단(B) 복원 강도 별도 튜닝.
- **추론(`infer.py`):** 텍스트→음성, 스트리밍 청크. **평가(`eval.py`):** SECS(임베딩 코사인)>0.70, 자가 WER<10%, MOS.

## 15. S5 — 페르소나 두뇌 Gemma 4 (서버 + 폰 2단) (`callone/llm`)
### 15.1 모델
- **서버:** **Gemma 4 12B**(통합·인코더리스·네이티브 오디오, 16GB 노트북 구동). 한국어·사투리 생성 품질↑.
- **폰:** **Gemma 4 E4B(또는 E2B) + QAT(Q4)**. Android=LiteRT/MediaPipe, iOS=MLX/llama.cpp, 간편=Ollama.
- **둘 다 학습:** 같은 SFT 데이터로 12B·E4B LoRA 각각 → `models/llm_12b/{A,B}`, `models/llm_e4b/{A,B}`.
### 15.2 세 층
1. **페르소나 카드 → 네이티브 system 프롬프트**(profile.json 기반: 이름·관계·사투리 지역/세기·대표 어미·말투·입버릇·금기).
2. **LoRA/QLoRA 파인튜닝**(`prepare_sft.py`→`train_lora.py`, HF+QLoRA 또는 Unsloth): `train.jsonl`(assistant=본인, 상대 role name, TAU 선택). Gemma 4 채팅 템플릿 사용.
3. **기억(`rag.py`,`memory.py`):** 실제 발화를 EmbeddingGemma로 임베딩→벡터DB 검색(RAG), 세션 간 Mem0/Zep. "예전에 ~했잖아" 일관성·사실성.
### 15.3 환각 억제 & 평가
- RAG 그라운딩, "모르면 모른다/기억 안 난다" 학습 예시, 사실질문 temp↓(≤0.7).
- 블라인드 A/B 식별률, 입버릇·사투리 재현율, 일관성. 참고: TwinVoice, REALTALK, Character‑LLM, CloneMem.

## 16. S6 — 실시간 대화 시스템 (`callone/serve`)
```
마이크 →(DeepFilterNet denoise)→ VAD(발화끝) → [스트리밍 ASR 또는 Gemma4 오디오]
      → Gemma 4(스트리밍 토큰) → 문장단위 스트리밍 TTS → 스피커
```
- **서빙:** 서버=vLLM(Gemma 4 12B). 폰=LiteRT/MediaPipe/llama.cpp. TTS 스트리밍. 오케스트레이션 **Pipecat**, 전송 **LiveKit(WebRTC)**.
- **API(서버 모드, `app.py` FastAPI + WebSocket/LiveKit):** 클라이언트가 통화 시작 시 `speaker_id`(A/B) 지정 → 오디오 청크 업스트림, 음성 청크 다운스트림. 제어 메시지(시작/종료/음소거).
- **지연 예산(첫 음성 < 0.8–1.2s):** VAD~200ms + ASR(겹침 처리)~0 + LLM 첫 토큰 ~200–400ms + TTS 첫 청크 ~150–300ms.
- **풀듀플렉스(선택 고도화):** SoulX‑Duplug류 LISTEN/SPEAK 상태예측 추가. 1:1 통화라 다화자 끼어들기 난점 낮음.
- **수용기준:** 엔드투엔드 통화 동작, 첫 음성<1.2s, 화자별(A/B) 올바른 음성·페르소나.

## 17. S7 — UI + 2단 배포 (`ui/`, `callone/serve`)
### 17.1 UI(React, 미스보이스 골격 재사용)
- **화면:** `ContactList`(A/B를 라벨한 이름·관계로 표시) → 통화 시작("전화 거는 중…"→수신음) → `CallScreen`(타이머·음성 파형·음소거·종료) → 종료 후 자막 로그(옵션). + `SpeakerCardEditor`(S2.5 라벨링 편집기) + 데이터 업로드/처리 진행 화면.
- **연결:** WebSocket/LiveKit로 `callone/serve`와 통신. (missvoice의 외부 API 레이어 제거.)
### 17.2 2단 배포
- **서버 모드(고품질):** Elice/자체 서버에 Gemma 4 12B(vLLM)+Qwen3‑TTS/VoxCPM2+방언 적응 ASR. 폰은 스트리밍 클라이언트.
- **폰 온디바이스 모드(오프라인·경량):** Gemma 4 E4B(QAT)+per‑speaker 소형 TTS(Piper/MeloTTS)+경량 ASR을 폰 탑재. ⚠️ **온디바이스 TTS 클론이 가장 까다로움** → 데이터 풍부함을 살린 per‑speaker 전용 모델이 정답. 초기엔 LLM·ASR만 온디바이스 + TTS는 로컬 서버 보조 하이브리드도 허용.

---

## 18. 빌드 순서 · 마일스톤 (Claude Code: 이 순서로 — 파일럿 우선)

> **원칙: 1,000통 전체를 돌리기 전에 50통 + 한 사람(A)으로 전 과정을 수직 관통한다.**

- **M0 스캐폴딩:** 레포 구조(§5), 환경(§6), 스키마(§7), `.env.example`, 공통 모듈, missvoice UI 골격 이식. README "Model Decisions" 섹션 생성.
- **M1 파일럿 데이터(50통):** S0→S1(복원·음색 보존 가드)→S2(2군집+제3자 제거 검증). 게이트: DER<12%, 복원 음색 보존.
- **M2 방언/라벨링:** S2.5 자동 프로파일링(지역·세기 측정) + 라벨링 편집기. 게이트: A/B 초안 생성 + 편집 저장.
- **M3 ASR 적응:** A 통화 30~120분 수동 교정 → Whisper 적응. 게이트: 보류셋 WER 개선.
- **M4 데이터셋:** S3로 A의 (a)음성셋/(b)대화셋 + 페르소나 카드, PII 마스킹.
- **M5 A 음성 클론:** 서버 Qwen3‑TTS FT → SECS>0.70 → 폰 Piper/MeloTTS 소형. 게이트: 유사도·명료도.
- **M6 A 페르소나:** Gemma 4 12B + E4B LoRA + 페르소나 카드 + RAG. 게이트: 텍스트 대화에서 사투리·말투 재현, 환각 통제.
- **M7 실시간 + UI:** S6 스트리밍 통합 + 통화 화면. 게이트: 첫 음성<1.2s, A와 음성 통화.
- **M8 B 확장 + 전량:** B에 M3~M6 반복, 1,000통 전량 처리. 2단 배포 마무리.
- **M9 고도화(선택):** 풀듀플렉스, 감정 제어, 메모리 강화.

각 마일스톤 끝에 해당 `tests/test_sX.py` 통과 + 짧은 리포트(README/`reports/`).

---

## 19. 평가 · 수용 테스트 (`tests/`)
- **test_s0:** 모든 manifest 행, 16k mono, 손상 격리.
- **test_s1:** 복원 후 SNR/PESQ 개선 + 임베딩 유사도(음색 보존) 임계 통과.
- **test_s2:** 정확히 2개 주군집(A/B) + 제3자 이상치 분리; (수동 라벨 일부로) DER<12%.
- **test_s25:** dialect intensity가 화자별로 산출(연속값), 라벨링 편집기 저장 라운드트립.
- **test_asr_adapt:** 보류 자체통화 WER(적응)<WER(기본).
- **test_s4:** SECS>0.70, 자가 WER<10%.
- **test_s5:** 블라인드 A/B 식별률 측정 스크립트, "모름" 캘리브레이션 체크.
- **test_s6:** 첫 음성 지연<1.2s, 화자별 올바른 음성/페르소나.
- **공통:** PII 누출 0(정규식 스캔), 외부 유료 API 호출 0(코드 정적 검사로 금지 도메인/패키지 차단).

---

## 20. 보안 · 윤리 · 법률 (구현 요건)
- **로컬·암호화:** 통화 원본·전사·학습셋은 `data/`(gitignore) + 디스크 암호화(`crypto.py`, `ENCRYPTION_KEY`). 외부 전송 금지.
- **PII 마스킹:** 학습셋 생성 파이프라인에 강제 적용(§12). 원본 보관 시 접근 통제.
- **유료 API 차단:** CI/정적 검사로 `openai/elevenlabs/assemblyai/api.openai.com` 등 금지 패턴 발견 시 빌드 실패.
- **동의/목적 한정:** README에 동의·목적 한정(사칭·기만 금지, 사적/추모/연구용) 명시. 통화 녹음·개인정보 관련 관할 법규 준수 안내.
- **폰 온디바이스 우선:** 데이터가 기기를 벗어나지 않는 모드를 프라이버시 기본값으로 권장.
- **정서적 고지:** 결과물은 그 사람 그대로가 아닌 **근사(近似)** 임을 UI/README에 부드럽게 고지.

---

## 21. 참고 문헌 / 도구 (2026, 무료 오픈 우선 — 착수 시 최신본 재확인)
- **Gemma 4:** 12B 발표 https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/ · 12B 가이드 https://developers.googleblog.com/gemma-4-12b-the-developer-guide/ · 코어 https://ai.google.dev/gemma/docs/core · 오디오 https://ai.google.dev/gemma/docs/capabilities/audio · QLoRA https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
- **ASR/방언:** Whisper large‑v3(MIT)·`faster-whisper`·`whisperX`(https://github.com/m-bain/whisperX) · AIHub 한국어 방언 발화(경상 #119/전라 #120 등) https://www.aihub.or.kr · 한국어 저음질 통화 Whisper FT 예 https://huggingface.co/INo0121/whisper-base-ko-callvoice · 2026 오픈 STT 비교 https://www.gladia.io/blog/best-open-source-speech-to-text-models
- **화자분리/복원:** pyannote.audio https://github.com/pyannote/pyannote-audio · SpeechBrain/WeSpeaker · ClearerVoice‑Studio https://github.com/modelscope/ClearerVoice-Studio · Resemble Enhance https://github.com/resemble-ai/resemble-enhance · DeepFilterNet · DPDFNet(arXiv:2512.16420)
- **TTS 서버:** Qwen3‑TTS https://github.com/QwenLM/Qwen3-TTS · VoxCPM2 https://github.com/OpenBMB/VoxCPM · Llasa‑Korean(arXiv:2509.18531) · IndexTTS2
- **TTS 폰:** Piper(MIT) · MeloTTS(한국어·CPU) · Kokoro(82M, Apache‑2.0)+KokoClone · GPT‑SoVITS · 비교 https://offlinetts.com/blog/voice-cloning-offline-tts-kokoro-kitten-piper/
- **실시간 음성대화/페르소나(연구):** PersonaPlex(arXiv:2602.06053) · FlashLabs Chroma 1.0(arXiv:2601.11141) · SoulX‑Duplug(arXiv:2603.14877) · X‑Talk(arXiv:2512.18706) · ICASSP 2026 HumDial(arXiv:2604.21406) · TwinVoice(arXiv:2510.25536) · TAU(arXiv:2510.09158) · CloneMem(arXiv:2601.07023) · REALTALK(arXiv:2502.13270)
- **런타임/서빙:** vLLM · LiteRT‑LM · MediaPipe LLM Inference · Ollama(Gemma 4 QAT) · MLX · llama.cpp · Pipecat · LiveKit · FAISS/Chroma/Qdrant · Mem0/Zep
- **UI 골격(로직 미사용, 화면만):** missvoice https://github.com/ksiwon/missvoice

---

## 부록 A. 설정 파일 예시 (`configs/`)
```yaml
# s2_diarize.yaml
asr_model: "models/asr_dialect"        # 적응본, 없으면 "large-v3"
language: "ko"
min_speakers: 2
max_speakers: 3                         # 제3자 대비
embedding: "speechbrain/spkrec-ecapa-voxceleb"
link:
  n_ref_calls: 20
  sim_threshold: 0.55                   # 두 센트로이드 모두 미만이면 drop
  cluster: "hdbscan"                    # 주군집 2 + 이상치
filters: {min_snr_db: 10, drop_overlap_for_tts: true}
```
```yaml
# llm_server.yaml
base_model: "google/gemma-4-12b-it"     # ※ §3에 따라 최신본 재확인 후 확정
method: "qlora"
lora: {r: 16, alpha: 32, dropout: 0.05, target: "all-linear"}
train: {epochs: 3, lr: 1e-4, bsz: 8, grad_accum: 4, max_seq_len: 4096}
use_tau: true
chat_template: "gemma4"
```

## 부록 B. Claude Code를 위한 시작 지침
1. 이 문서를 처음부터 끝까지 읽는다. §2 하드 제약과 §3 모델 검증 의무를 최우선 규칙으로 삼는다.
2. §5 레포를 스캐폴딩하고 §6 환경/`.env.example`/§7 스키마(pydantic)를 만든다. missvoice를 clone해 UI 골격만 `ui/`로 이식한다.
3. **M0→M9 순서(§18)** 로 구현하되, **반드시 50통 + 한 사람(A)으로 수직 관통(M1~M7)** 후 전량/두 번째 사람으로 확장한다.
4. 각 스테이지는 독립 CLI + `configs/*.yaml` + `tests/test_sX.py`로 구현하고, 중간 산출물을 디스크에 저장한다.
5. 각 모델을 코드에 고정하기 전 **웹으로 최신 버전을 확인**하고 결정을 README "Model Decisions"에 적는다.
6. 실제 통화 녹음·HF 토큰·AIHub 계정은 사용자가 제공한다(경로/토큰은 `.env`로 주입). 데이터가 없으면 더미/소량 샘플로 파이프라인을 검증할 수 있게 설계한다.
7. 막히면, 해당 스테이지의 §본문 + 부록 설정 + 참고 링크를 근거로 결정하고, 가정을 README에 기록한다.
