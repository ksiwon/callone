# callone 현재 상태

> 정본 날짜: 2026-07-02. 과거 모델 검토와 구현 전 설계는 `legacy/design/`에 보관한다.

## 설치 환경 실측 (2026-07-02, RTX 3090 Ti · RunPod)

RunPod RTX 3090 Ti(Ampere cc 8.6, 24GB)에서 `install.sh` 원샷으로 음성+영상 통화까지 동작 확인.
GPU 선택 기준: 세 후보(3090/3090Ti/4090) 모두 24GB라 LLM은 EXAONE-3.5-7.8B로 동일(32B 불가).
**아바타 Ditto 프리빌트 TRT 엔진이 Ampere 전용**이라 3090/3090Ti가 무빌드로 바로 동작(4090=Ada는
custom TRT 재빌드 필요). 3090Ti가 3090 대비 RAM 72GB/vCPU 16 여유(멀티 서비스 4개)로 스윗스팟.

설치 중 발견·수정한 두 함정(코드 반영 완료):

1. **Node.js** — NodeSource `curl|bash`가 컨테이너서 파이프 깨짐(`curl:23`)으로 실패하고,
   수동 `apt-get install nodejs`는 **Node 12(구버전)+npm 미포함**을 깔아 Vite(Node18+) 불가.
   → `install.sh`/`FRESH_SETUP.md`를 **공식 바이너리 tarball을 /usr/local에 직접 푸는 방식**으로 교체.
2. **cuDNN 8/9 충돌** — `setup_avatar_gpu.sh`가 torch(cuDNN9) 설치 뒤 TRT용 cuDNN8을 **같은 pip
   패키지**로 덮어써 `libcudnn.so.9 없음`으로 torch import 실패(아바타 static 폴백).
   → cuDNN9 재설치 + so.8 파일 보존으로 **so.8(TRT)·so.9(torch) 실제 공존** 복구(자동).
   ※ tensorrt 메타패키지 `bdist_wheel` 에러는 `|| true`로 무시(libs/bindings는 정상)—기존 의도, 무해.

## 현행 스택

| 영역 | 현행 |
|---|---|
| 서버 LLM | A100/H100: EXAONE-4.0-32B-abliterated Q6_K / 24GB GPU: EXAONE-3.5-7.8B-abliterated Q6_K |
| LLM 런타임 | llama.cpp `llama-server` (:8090) |
| TTS | CosyVoice3-0.5B (:8092), 실패 시 Piper → Kokoro placeholder |
| ASR | faster-whisper large-v3-turbo |
| 아바타 | Ditto TensorRT (:8091), 실패 시 정지사진 |
| 오케스트레이터 | FastAPI + WebSocket `callone-serve` (:8000) |
| UI | React/Vite (:5173) |

지원하지 않는 현행 경로: Qwen3-TTS 서빙, MuseTalk, 폰 온디바이스, Pipecat/LiveKit, Mem0/Zep.
이들은 실험 기록에만 남기고 실행 코드·설정에서는 제거했다.

## 확정된 운영 원칙

- 우선순위: 목소리 → 한국어 → 얼굴 → 속도.
- 개인 음성·사진은 인메모리로만 보관하고 세션 종료 시 폐기한다.
- 외부 유료 API는 사용하지 않는다.
- CosyVoice 스트리밍은 저장 WAV와 라이브 A/B 검증 전까지 `tts.stream: false`가 기본이다.
- Ditto 프레임 drain gap 기본값은 0.4초이며 `DITTO_DRAIN_GAP_S`로 조정할 수 있다.

## 남은 GPU 검증

1. CosyVoice `/synth`와 `/synth_stream` 음색·경계 노이즈 A/B.
2. EXAONE 4.0 32B 실통화의 한국어 자연스러움, 첫 토큰 지연, 동시구동 VRAM.
3. Ditto 멀티턴 프레임 누락·OOM 여부와 0.4초 drain gap 검증.
4. `callone-bench`로 첫 음성·첫 프레임 지연 기록.
5. 현 llama-server의 `dry_*`와 `min_p` 지원 확인.

## 문서 우선순위

1. 새 GPU 설치: `FRESH_SETUP.md`
2. 현재 결정과 남은 일: 이 문서
3. 아바타 실행·문제해결: `AVATAR_RUN.md`
4. 풀 파인튜닝: `1_로컬에서_학습.md` → `2_GPU에서_학습.md` → `3_노트북에서_통화.md`
5. 과거 설계·시행착오: `../legacy/design/`