# callone 토킹헤드(사진→움직이는 얼굴) 통합 설계서

> **작성 2026-06-17.** 상태 = **설계만**(구현 전). 음성 통화 4090 안정화가 선행.
> 목적: 사진 1장 + 통화 TTS 음성 → 입·표정·고개가 살짝 움직이는 실시간 토킹헤드.
> 모델/아키텍처를 [[server-serve-stack-verify]] 음성 스택 위에 충돌 없이 얹는 방법을 확정한다.

---

## 0. 목표·범위

- **입력:** 화자 사진 1장(정면, 얼굴 또렷). **구동:** 통화 중 합성되는 TTS 오디오.
- **출력:** 오디오에 동기된 얼굴 영상(입모양 + "조금씩" 고개/표정). 전화 영상통화 느낌.
- **비범위:** 전신·배경 생성, 감정 과장 연기, 오프라인 고해상 렌더. (필요 시 후속)
- **callone 원칙 유지:** 무거운 모델 미설치/저사양/VRAM부족 → **정지 사진 또는 음성전용**으로
  안전 폴백(다른 스테이지와 동일 철학). 영상은 음성 통화의 *부가 레이어*지 의존 대상이 아니다.

---

## 1. 모델 결정 (웹검증 2026-06-17)

| 순위 | 모델 | 움직임 | 실시간 | 라이선스 | 입력 | 비고 |
|---|---|---|---|---|---|---|
| **1순위** | **Ditto** (antgroup/ditto-talkinghead, ACM MM 2025) | 입+표정+고개(motion 세밀제어) | **RTF<1, 첫프레임<400ms** | **Apache-2.0** | 단일 사진 | "조금씩 움직"에 부합. DiT가 identity-독립 motion 생성→face renderer |
| 2순위 | **MuseTalk** (TMElyralab) | 입/하관 위주(고개 거의 고정) | 30fps@256² | MIT | 단일 이미지 가능 | 가장 가벼움. latent inpainting 1-step(확산 아님). mmcv/mmdet/mmpose+diffusers 필요 |
| 폴백 | **StaticImage** | 없음(사진 정지) | — | — | 사진 | 모델 없음/VRAM부족/저사양. 음성만 정상 |

**채택: Ditto 1순위**(사용자 요구 "조금씩 움직" = 고개/표정 포함, Apache-2.0, 첫프레임<400ms로 전화 허용범위).
MuseTalk은 VRAM/지연 빠듯할 때의 경량 대안. 최종 확정 전 **착수 시점에 4090 실측**(FFD·fps·VRAM) 의무.

---

## 2. 아키텍처 결정 — **별도 프로세스(avatar-server)**

### 2-1. 왜 분리하나 (핵심 근거)
- Ditto/MuseTalk은 **자체 diffusers·mmcv/mmdet/mmpose·torch 스택**을 요구한다.
- 우리 서빙 venv(`.venv-serve`)는 **faster-qwen3-tts → qwen-tts 0.1.1 → `transformers==4.57.3` 하드핀**.
- 토킹헤드 스택과 TTS 스택은 transformers/torch에서 **충돌 불가피** → 한 venv 불가.
- 결론: **llama-server와 동일 패턴.** 토킹헤드를 별도 venv·별도 프로세스(`avatar-server`)로 띄우고
  callone 서빙은 **HTTP/WS로만 호출**. 서빙 파이썬엔 토킹헤드 의존성 0 → 충돌·segfault 원천차단.

```
┌─ .venv-serve (callone-serve) ──────────────┐     ┌─ .venv-llm 없음 ─┐
│ ASR(faster-whisper) · TTS(faster-qwen3-tts)│ HTTP│ llama-server     │ (GGUF, 별 바이너리)
│ orchestrator · app(WS)                     │────▶│ :8080            │
│                                            │     └──────────────────┘
│                         avatar 호출(HTTP/WS)│     ┌─ .venv-avatar ───┐
│                                            │────▶│ avatar-server    │ (Ditto/MuseTalk)
└────────────────────────────────────────────┘ :8091│ torch+diffusers │
                                                    └──────────────────┘
```

### 2-2. avatar-server 인터페이스 (신규, 별 repo/venv)
- `POST /session/start`  body: `{image: <path|base64>, fps: 25}` → `{session_id, w, h}`
  (사진→얼굴 검출·정렬·identity 사전추출 = **통화당 1회**. precompute_voice_emb 패턴과 동형.)
- `WS /session/{id}/stream`  ← 오디오청크(PCM f32, 24kHz) 업, → 영상프레임(JPEG/H264) 다운.
  - Ditto/MuseTalk 스트리밍 API로 오디오→프레임, 25~30fps로 emit.
- `POST /session/{id}/stop` → 리소스 해제.
- 헬스: `GET /health` → callone 측이 probe(없으면 StaticImage 폴백).

---

## 3. 파이프라인 통합 (음성 스택 최소 변경)

현재 [orchestrator.stream_turn](../callone/serve/orchestrator.py)는 문장 단위로
`("text",...)`/`("audio", np)`/`("latency",...)` 이벤트를 yield. 여기에 영상을 **병렬 레이어**로 추가:

- **오디오가 진실원본(master clock).** TTS가 낸 같은 오디오를 (a)클라 스피커로, (b)avatar-server로
  동시에 보낸다. avatar-server는 그 오디오 타임라인에 맞춰 프레임을 emit.
- orchestrator(또는 app WS 송출부)가 avatar-server WS에 audio chunk를 forward하고,
  돌아온 프레임을 `("frame", bytes)` 이벤트로 클라이언트에 중계.
- **barge-in:** 사용자 발화 감지 시 기존 `interrupt()`가 TTS 중단 → avatar-server에도
  `interrupt`(현 세션 프레임 중단) 신호. (입이 계속 움직이는 것 방지)
- **동기 허용오차:** 입모양은 ±80ms 내 동기면 자연스러움. 프레임이 약간 늦으면 오디오 우선
  재생(영상 살짝 지연 허용). 영상이 막히면 **정지 프레임 유지**(음성은 끊기지 않게).

### 3-1. 추상화 (TTS 백엔드 패턴 그대로)
```
# callone/serve/avatar.py (신규)
class AvatarBackend:                     # 인터페이스
    def start_call(self, image_path: str) -> None: ...
    def frames_for(self, audio: np.ndarray, sr: int) -> Iterator[bytes]: ...  # JPEG/H264
    def interrupt(self) -> None: ...
    def stop(self) -> None: ...

def _pick_avatar(cfg) -> AvatarBackend:  # _pick_tts 와 동형 폴백 체인
    # 1) DittoAvatar(HTTP avatar-server probe) → 2) MuseTalkAvatar → 3) StaticImageAvatar
```
- `DittoAvatar`/`MuseTalkAvatar` = avatar-server HTTP/WS 클라이언트(LlamaPersonaLLM의 probe 패턴 재사용).
- `StaticImageAvatar` = 사진 JPEG 1장 반복(움직임 없음). 모델/서버 없을 때 graceful.

---

## 4. 클라이언트·전송 프로토콜

- **1차(간단):** 기존 WS 그대로. 오디오=바이너리 f32(현행), 영상=바이너리 프레임 앞에 1바이트
  타입태그 또는 별도 메시지. UI(CallScreen)에 `<img>`/`<canvas>` 프레임 갱신 추가.
- **스케일 경로(저지연):** **WebRTC(LiveKit)** — 오디오+비디오 트랙 네이티브 동기(7880).
  실사용 품질 필요 시 이쪽. (1차 검증 후 결정)
- 프레임 포맷: 1차는 **JPEG**(구현 단순). 대역/품질 필요 시 H264/VP8(WebRTC와 함께).

---

## 5. 지연·VRAM 예산 (4090 24GB)

| 점유 | 모델 | VRAM(대략) |
|---|---|---|
| LLM | EXAONE-3.5-7.8B Q6_K(llama-server) | ~6.5 GB |
| TTS | CosyVoice3-0.5B(별 conda env) | ~1.5 GB |
| ASR | faster-whisper turbo | ~1.5 GB |
| **Avatar** | **Ditto / MuseTalk** | **~3~6 GB(실측 필요)** |
| 합계 | | **~12.5~15.5 GB → 24GB 안에 여유** |

- 별 GPU 프로세스 = **GPU 시분할** → LLM/TTS가 약간 느려질 수 있음. 첫음성 지연에 avatar FFD(<400ms) 가산.
- VRAM 부족 시: avatar를 256² 저해상/낮은 fps로, 또는 MuseTalk 경량, 최후 StaticImage.
- **실측 의무(착수 시):** ① avatar 단독 FFD·fps·VRAM ② LLM/TTS와 동시구동 시 음성 첫음성 지연 변화.

---

## 6. 통화 시작 준비 (사진 1회 전처리)

1. 사진 업로드 → 얼굴 검출·정렬·크롭(avatar-server `/session/start` 내부).
2. identity/외형 특징 사전추출(통화당 1회) → 세션 핸들. 이후 프레임 생성은 오디오만 입력.
3. 사진 품질 가이드: 정면, 얼굴이 프레임의 충분한 비율, 또렷, 정상 조명(음성 ref_wav 가이드와 짝).
4. (선택) 음성 ref_wav와 같은 사람의 사진을 쓰면 음색·외형 일치.

---

## 7. configs / 코드 변경 지점 (구현 시)

```yaml
# configs/serve.yaml (신규 블록 — 설계안)
avatar:
  enabled: false              # 기본 off(음성전용). 켜면 _pick_avatar 폴백체인 동작
  backend: auto               # auto | ditto | musetalk | static
  base_url: "http://127.0.0.1:8091"   # avatar-server
  image: ""                   # data/speakers/A/portrait.jpg
  fps: 25
  resolution: 256             # 256 | 512 (VRAM/지연 trade)
  frame_format: jpeg          # jpeg | h264
```
- 신규: `callone/serve/avatar.py`(추상화+백엔드), avatar-server(별 repo/venv), `scripts/setup_avatar_gpu.sh`,
  `requirements-avatar-gpu.txt`(Ditto/MuseTalk 전용 — **절대 .venv-serve 와 분리**).
- 변경(작음): `orchestrator`/`app.py` 송출부에 frame 이벤트 중계 + interrupt 전파. UI CallScreen 프레임 렌더.
- pytest: avatar 미설치 시 StaticImage 폴백 경로 + frame 이벤트 스키마 테스트(모델 없이).

---

## 8. 마일스톤 순서 (디버깅 면 최소화 = 비용 최소)

1. **(선행) 음성 통화 4090 안정화·실측** — 영상 없이 첫음성 지연 확정.
2. avatar-server **단독** 기동(별 venv) → 사진+wav 파일로 오프라인 영상 생성 검증(FFD·fps·VRAM 실측).
3. callone `_pick_avatar` + HTTP/WS 연동 → 턴제(녹음→영상응답)로 동기·품질 확인.
4. 실시간 WS 프레임 중계 + UI 렌더 → barge-in 시 프레임 중단까지.
5. (선택) WebRTC(LiveKit)로 오디오·비디오 트랙 동기 업그레이드.

---

## 9. 열린 이슈 / 착수 시 검증

- [ ] Ditto vs MuseTalk **4090 실측**(FFD/fps/VRAM, LLM·TTS 동시구동 영향).
- [ ] Ditto 스트리밍(청크 입력) API 형태 — 문장단위 vs 연속 오디오. (저장소 inference 코드 확인)
- [ ] 한국어 발음↔입모양 정합(모델이 보통 음소 비의존이라 무난하나 확인).
- [ ] 프레임 전송 포맷(JPEG vs WebRTC) 대역·지연 trade.
- [ ] 사진 1회 전처리 시간이 통화 시작 지연에 주는 영향(미리 예열).
- [ ] 윤리: 실존 인물 사진 사용 = 로컬 전용·동의 전제(callone_spec §20 준수, 외부배포 금지).
