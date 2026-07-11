# callone v2 전면 재구축 기획서 (2026-07)

목표 하나: **비상업·개인용 최고 성능** — 한국어 대화 품질과 말→말(voice-to-voice) 지연의 극한.
법적 제약으로 상업화·nsfw 는 폐기됨(재도입 금지). 프라이버시 ephemeral 원칙(디스크/로그 영속 0)은 유지.

---

## 0. 벤치마킹 — 남들은 어떻게 하나 (2026-07 조사)

| 시스템 | 방식 | 배울 것 |
|---|---|---|
| **Kyutai Unmute/Moshi** | Moshi=오디오네이티브 풀듀플렉스(이론 160ms, L4 실측 200ms) / Unmute=모듈러 캐스케이드(아무 LLM + 스트리밍 STT/TTS) | 캐스케이드로도 사람급(230ms) 가능 검증. 단 Moshi 는 영어+고정목소리 → 한국어+음성클론엔 캐스케이드가 유일해 |
| **Pipecat / LiveKit Agents** | 프레임 프로세서 파이프라인, 자동 인터럽션 | ① **STT 엔드포인팅이 기본값**(VAD-only 는 지연↑) ② 세만틱 턴감지(+20ms 로 오인터럽트 87%↓) ③ 재생 중에도 턴감지 상시 활성(barge-in) ④ 800ms = 로봇같음의 경계 |
| **Character.AI Calls** | 원탭 통화 시작, 탭-투-인터럽트 버튼, 목소리 프리셋 갤러리 | UX: 설정 최소화, 통화 중 개입수단 2개(음성+버튼) |
| **제타(Zeta)** | 캐릭터 갤러리 중심(400만 캐릭터), 채팅→통화 확장 | UX: 캐릭터 카드가 입구, 대화 이력이 자산 |

**아키텍처 결론: 캐스케이드(ASR→LLM→TTS) 유지.** 풀듀플렉스 오디오네이티브는 한국어·제로샷 클론 불가.
지연은 파이프라인 재설계(스트리밍화)로 잡는다.

---

## 1. 모델 선정 (2026-07 조사 + 기존 실측)

### TTS — 이중 경로 + 게이트
- **후보 1순위: Qwen3-TTS-12Hz** (0.6B/1.7B, Apache2.0, 2026-01). 한국어 WER 1.755·SIM 0.799,
  완전 causal 스트리밍 첫패킷 97ms, 레퍼런스 3초, 공식 파인튜닝 코드.
- **⚠️ 기존 실측 경고**: 이 레포는 과거 Qwen3 계열 TTS 를 시험했고 **"턴마다 음색 튐"** 으로 기각,
  CosyVoice3 채택함(serve.yaml 주석). 신형(12Hz, 2026-01)은 별개 모델이지만 **같은 함정 검증 필수**.
- **게이트(통과 못 하면 교체 안 함)**: ① 턴 간 음색 편차 A/B(고정 seed + ICL 프롬프트 캐시로 10턴 합성
  → 블라인드 청취) ② 한국어 자연성 vs CosyVoice3 블라인드 ③ 3090 실측 첫패킷 < 300ms.
- **차점(품질 도전자)**: Fish S2 Pro(4B, 아레나 오픈 1위, 단 레퍼런스 10~30초 + 무거움).
  게이트 실패 시의 백업 후보로만.
- **폴백 체인**: qwen3tts → cosyvoice3(현행 유지) → piper/kokoro.

### ASR — 스트리밍화 (지연 개선 1순위)
- **Qwen3-ASR-0.6B/1.7B** (Apache2.0, 한국어 포함 52언어, 스트리밍+오프라인 단일 모델).
- 현행 구조는 "말 끝난 뒤 전사 시작" → **말하는 도중 부분 전사(partial)** 로 바꾸면 턴 종료 시점에
  전사가 이미 끝나 있음 = ASR 지연 ≈ 0. 여기가 최대 개선 포인트.
- 폴백: faster-whisper large-v3-turbo(현행).

### LLM — 티어별
- ⚠️ **SEED Think 는 llama.cpp 미지원**(GGUF 없음, llama.cpp feature request 계류) — vLLM
  `--trust_remote_code` 경로만 가능. 현 스택은 llama-server 라 **기본값은 EXAONE GGUF 유지**.
- 24GB: 기본 EXAONE-3.5-7.8B Q6_K(현행, 검증됨). 옵션 `LLM_PRESET=qwen3-14b`(Qwen3-14B GGUF,
  llama.cpp 지원 확인). SEED Think 14B 는 vLLM 도입 시 A/B(bench_llm_korean.py).
- 80GB: 기본 EXAONE-4.0-32B Q6_K(현행). 옵션 Qwen3-32B GGUF / SEED Think 32B(vLLM).
- 비상업이므로 라이선스 제약 없음. DRY 샘플러·RAG·prefix 캐시 등 기존 자산 유지.

### 아바타 — 현행 유지(Ditto), 이번 라운드 범위 밖.

---

## 2. 시스템 플로우 v2

### 프로세스 토폴로지 (run_all.sh 확장)
```
:8090 llama-server / vLLM ── LLM (티어별 모델)
:8091 avatar-server        ── Ditto (선택)
:8092 cosyvoice-server     ── TTS 폴백(현행)
:8093 qwen-tts-server      ── TTS 신규(별 venv, cosyvoice_server 와 동일 API 계약)
:8000 callone-serve        ── 오케스트레이터 + WS + ASR(인프로세스)
:5173 ui (vite)
```

### 한 턴의 파이프라인 (핵심 변경 = ①②)
```
mic ─→ VAD(speech start 감지)
        │  ① 말하는 동안: Qwen3-ASR 스트리밍 partial 전사 (기존: 침묵 후 일괄 전사)
        │     partial 은 UI 자막으로도 송출(체감 반응성↑)
        ▼
      엔드포인팅 ② = VAD 무음(end_silence_ms) + 세만틱 보조(문장 미완이면 +유예)
        │  턴 확정 시점에 최종 전사 이미 손에 있음 → ASR 단계 지연 ≈ 0
        ▼
      LLM 스트리밍(첫 문장 완성 즉시) ─→ 태그/이모지 정제(_strip_unspoken 유지)
        ▼
      TTS 스트리밍(qwen3tts 12Hz 첫패킷 ~100ms 급) ─→ WS audio 청크
        │                                              └→ avatar 프레임(병렬, 현행)
        ▼
      barge-in: 재생 중 VAD 상시 활성(현행 interrupt 유지) + UI 탭-투-인터럽트 버튼 추가
```

### 지연 예산 (목표, 말끝→첫음성)
| 단계 | 현행(실측) | v2 목표 |
|---|---|---|
| ASR | ~300-500ms (턴 후 일괄) | **~0ms** (스트리밍 선행) |
| LLM 첫 문장 | ~300-500ms | 200-400ms (모델 크기 trade) |
| TTS 첫 패킷 | ~300ms-1.3s | **100-300ms** (12Hz 스트리밍) |
| **합계** | **~1-2s** | **H100 ~250-350ms · 4090 ~350-550ms · 3090/Ti ~400-700ms** |

---

## 3. GPU 티어 (자동 감지: VRAM 기준, CALLONE_TIER 로 강제)

| 티어 | 감지 | LLM | TTS | ASR | 아바타 |
|---|---|---|---|---|---|
| **ultra** | VRAM ≥ 70GB (H100/A100-80) | SEED Think 32B AWQ(vLLM) 또는 EXAONE-4.0-32B | Qwen3-TTS-12Hz-**1.7B** | Qwen3-ASR-**1.7B** | ditto 512px |
| **high** | 20 ≤ VRAM < 70 (3090/3090Ti/4090) | SEED Think **14B** Q4_K_M | Qwen3-TTS-12Hz-**0.6B** | Qwen3-ASR-**0.6B** | ditto 256px |
| **mid** | 10 ≤ VRAM < 20 | EXAONE-3.5-7.8B Q4 (현행) | Qwen3-TTS-0.6B | faster-whisper turbo int8 | static |
| **cpu** | GPU 없음 (노트북) | EXAONE-3.5-2.4B OV (현행) | piper/kokoro (현행) | faster-whisper small int8 | 없음 |

**24GB(high) VRAM 예산**: LLM 14B Q4 ~8.5GB + KV/컨텍스트 ~2GB + TTS 0.6B ~2GB + ASR 0.6B ~1.5GB
+ Ditto ~4GB + CUDA 오버헤드 ~2GB ≈ **20GB** (headroom 4GB). 1.7B TTS 승격은 아바타 끌 때만.

기존 `hardware.py` 의 2단(server_gpu/laptop_cpu) → 4단으로 확장. 기존 이름은 별칭으로 유지(하위호환).

---

## 4. UI 재구성 (ui/src)

### 화면 구조 (3씬)
```
[홈]  캐릭터 갤러리(카드: 이름·사진·한줄상황, 제타 스타일) + [새 캐릭터] + [이어하기 배지]
  └→ [셋업 위저드]  ① 목소리: 내 목소리(파일/마이크 녹음 3~10s, 미리듣기) ↔ 프리셋 갤러리(청취 버튼)
                     ② 사진(선택, 미리보기)
                     ③ 캐릭터 카드(예시 프리셋 원탭 채움 — 현행 유지)
                     ④ [통화 시작] — 준비되면 원탭
  └→ [통화]  풀스크린 아바타/파형
             상태 배지: 연결중 → 듣는중 → 생각중 → 말하는중  (서버 이벤트 매핑)
             partial 자막(내 말, 실시간) + 응답 자막 (토글)
             [탭-투-인터럽트] 큰 버튼 (음성 barge-in 과 병행)
             [종료]  /  개발자 모드: latency HUD (asr/llm/tts ms — timing 이벤트 활용)
```
- 대화 이력: 현행 유지(localStorage, 클라 소유, export/import).
- WS 프로토콜 추가 이벤트: `partial`(스트리밍 전사), `state`(listening/thinking/speaking).

### 4-1. 목소리 입수 — 두 유저 플로우 (①단계 "어떤 자료를 갖고 있나요?")

| 유저 | 플로우 |
|---|---|
| **A. 10초 목소리 파일** | 짧은 파일 탭 → 업로드 → 🔊 미리듣기(전사 자동) → 다음 |
| **B. 몇 시간짜리 2인 통화 녹음** | 긴 통화 녹음 탭 → 업로드(raw body → tmpfs) → 서버 job: 화자분리(pyannote 체인)+겹침 제외+구간 점수화(ref_clip_score) → **화자 카드(들어보기 샘플)** "누가 그 사람?" → 선택+이름 → best 클립을 프리셋 저장(전사 포함) → '준비된 목소리'로 합류 |

- API: `POST /api/voice/analyze`(job 시작) / `GET .../analyze/{job}`(폴링) / `POST .../analyze/{job}/save`.
- 프라이버시: 원본은 tmpfs, 분석 직후 삭제. 저장 버튼을 눌러야만 프리셋(디스크) 생성. job TTL 1h.
- CLI 동일 기능: `scripts/pick_ref_clip.py` (같은 점수 함수 `common.audio.ref_clip_score` 공유).
- 확장(후속): B 플로우에서 그 화자의 전사→기억(memories) 구축 버튼, 말투 통계→캐릭터 카드 자동 채움.

---

## 5. 마이그레이션 순서 (각 단계 독립 커밋, 폴백 유지로 무중단)

1. **티어 시스템** — hardware.py 4단 + serve.yaml 티어 프리셋 (기존 동작 불변, auto 만 확장)
2. **qwen-tts-server** (:8093, cosyvoice_server 와 동일 API: /health /synth /synth_stream)
   + serve 어댑터 `tts_qwen3.py` + `backend: auto` 체인(qwen3tts 살아있으면 우선, 죽으면 cosy)
3. **Qwen3-ASR 스트리밍** — `asr_qwen3.py`(partial 콜백) + 오케스트레이터 연결, whisper 폴백
4. **오케스트레이터 v2** — 발화 중 스트리밍 전사, 엔드포인트 시 즉시 LLM, `partial`/`state` 이벤트
5. **LLM 티어** — SEED Think 14B GGUF 다운로드 스크립트 + llm 설정, EXAONE 경로 유지
6. **UI** — 홈 갤러리 / 위저드 정리 / 통화 화면(상태·자막·인터럽트 버튼·HUD)
7. **벤치 게이트** — scripts/bench_v2.py: 음색 안정성 A/B(10턴) + 첫패킷/E2E 지연 실측 → 결과로 기본값 결정

## 6. 리스크

| 리스크 | 대응 |
|---|---|
| Qwen3-TTS 신형도 음색 튐(과거 실측 재발) | 게이트 통과 전 cosyvoice3 기본값 유지, seed 고정 + ICL 캐시 시도 |
| Qwen3-TTS/ASR 소비자 GPU 실측 데이터 부재 | bench_v2 로 3090Ti 실측 후 티어표 보정 |
| SEED Think 14B 대화체(존댓말/구어) 미검증 | EXAONE 폴백 유지, 페르소나 A/B 후 기본값 교체 |
| 24GB 에 4모델 공존 OOM | VRAM 예산표 기준 로드 순서 고정 + 아바타 자동 강등(static) |
| vLLM은 Qwen3-TTS 오프라인만 지원 | 자체 서버는 transformers 스트리밍(12Hz causal)으로 구현 |
