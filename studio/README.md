# callone studio — 통합 진입점

`callone`(풀클론·통화)과 `voice_clone`(제로샷 TTS)을 **하나의 Gradio 앱**으로 합쳤다.
진입하면 헤더에서 3가지를 고르고, 나머지는 알아서 맞는 파이프라인으로 라우팅된다.

> 위치: `coding/callone/studio/`. 같은 루트의 `callone`(패키지)·`voice_clone` 을
> 자동 연결한다(env.REPO_ROOT = coding/callone).

## 실행

```bash
cd callone                              # 반드시 repo 루트에서(configs/·models/·voice_clone/ 상대참조)
pip install -r studio/requirements.txt
python -m studio                        # http://localhost:50000
# pip install -e . 했으면 진입점도 동일:
callone-studio                          # = studio.app:main
```

> 메뉴와 **전사**는 위 설치만으로 바로 동작. 나머지 칸은 해당 백엔드 설치 시 켜진다.

## 헤더 3축

| 축 | 선택지 | 효과 |
|---|---|---|
| ① 환경 | 자동 / GPU 강제 / CPU 강제 | `CALLONE_TIER` 강제 → 모델·compute 스왑 |
| ② 목적 | 전사 / TTS 출력 / 실시간 통화 | 패널 전환 |
| ③ 데이터 모드 | 제로샷(5~10초) / 풀클론(대량 녹음) | 백엔드 전환 |

## 라우팅 매트릭스 (목적 × 모드)

| 목적 ↓ \ 모드 → | 제로샷 | 풀클론 |
|---|---|---|
| **전사** | faster-whisper (GPU:large-v3/fp16, CPU:small/int8) | 방언적응 Whisper(있으면) |
| **TTS** | CosyVoice3 제로샷 (참조 WAV) | callone 화자학습 (GPU:Qwen3-TTS LoRA / CPU:Piper) |
| **통화** | 참조음색 + 페르소나-프롬프트 LLM | 화자 LoRA TTS + LLM 페르소나 SFT |

각 칸은 **🟢 바로 실행** 또는 **🟡 미설치(레시피 폴백)** 로 헤더 배지에 표시된다.
미설치 칸은 "어떻게 설치/학습하는지" 명령을 그대로 보여준다 — callone 의 폴백 철학과 동일.

## 구조

```
coding/callone/              # 단일 루트
  callone/                   핵심 파이프라인 패키지(S0~S7)
  configs/ scripts/ tests/ docs/ ui/   callone 본체(상대경로 결합 — 그대로)
  data/ db/ models/ resources/         런타임 데이터·산출물
  voice_clone/               제로샷 CosyVoice 앱 + voices/(저장 목소리)
  studio/                    ◀ 통합 진입점(이 폴더)
    env.py        환경 해석(GPU/CPU→티어) + 루트 경로 배선
    router.py     (목적×모드×환경) → Plan(백엔드/모델/가용성/레시피)
    backends.py   lazy 백엔드 래퍼(전사/제로샷TTS/풀클론TTS/통화) + 폴백
    app.py        Gradio UI(헤더 3드롭다운 + 목적별 패널)
    __main__.py   런처(python -m studio) · requirements.txt · README.md
```

## 통화 참고

앱 내장 통화는 **녹음→응답 턴제 간이판**(`Orchestrator.handle_utterance` 재사용).
풀 실시간(WebRTC, barge-in)은 기존 `callone-serve`(FastAPI+React) 그대로:

```bash
cd callone && callone-serve                      # :8000
cd callone/ui && npm install && npm run dev      # :5173
```

## 음성 프로필

제로샷 저장 목소리는 `voice_clone/voices/<이름>/`(ref_16k.wav + meta.json)에 공유 저장.
기존 voice_clone 앱과 동일 포맷 → 호환된다.

## 미설치 함정(코드에 폴백 반영됨)
`studio/requirements.txt` 하단 참고: ffmpeg(브라우저 마이크 opus), ctranslate2↔cuDNN(GPU 전사
→ CPU 자동 폴백), Qwen3.5 MoE는 bf16 LoRA 권장.
