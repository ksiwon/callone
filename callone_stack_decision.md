# callone — 최종 스택 결정서 (Claude Code 실행용)

> **작성 기준일:** 2026-06-14  
> **대상:** Claude Code 자율 코딩 에이전트  
> **목적:** 기존 callone_spec.md(v1.0)의 §3 Model Decisions를 웹 검색 검증 결과로 전면 업데이트  
> **우선순위:** 이 문서가 callone_spec.md의 모델 관련 섹션보다 우선 적용된다.

---

## 0. 변경 배경 요약

callone_spec.md(v1.0)은 Gemma 4 12B(LLM) + CosyVoice3(TTS) + vLLM(서빙)을 기본 스택으로 설정했다.
2026-06-14 웹 검색 및 실측 검증 결과 세 가지 문제가 확인되어 전면 교체한다.

| 항목 | 기존(spec v1.0) | 교체 이유 | 신규 채택 |
|---|---|---|---|
| LLM | Gemma 4 12B | 한국어 약함, vLLM GDN 버그, 속도 미검증 | **Qwen3.5-9B (abliterated)** |
| TTS | CosyVoice3 | zero-shot + instruct 동시 제어 불완전 | **Qwen3-TTS 1.7B** |
| 서빙 백엔드 | vLLM | Gemma4 TRITON fallback → ~9 tok/s | **llama.cpp (llama-server)** |

---

## 1. LLM 결정: Qwen3.5-9B abliterated

### 1-1. 선택 근거

- **한국어 성능:** Qwen3.5는 201개 언어/방언 지원. 한국어 포함 CJK 언어에서 Gemma 4 12B 대비 확연히 우위.
- **속도:** RTX 4090 기준 llama.cpp Q4_K_M 실측 **~126 tok/s**. Gemma 4 12B + vLLM 조합은 TRITON fallback 버그로 ~9 tok/s.
- **벤치마크:** Intelligence Index 32점으로 10B 미만 모델 전체 1위. 2위(16점) 대비 2배 수준.
- **아키텍처:** Gated DeltaNet 하이브리드 + sparse MoE. 262K 컨텍스트 네이티브 지원.
- **SFT 생태계:** Unsloth에서 Qwen3.5-9B LoRA/QLoRA 완전 지원. A100 80GB에서 bf16 LoRA VRAM ~22GB.
- **라이선스:** Apache 2.0. 상업적 사용 제한 없음.

### 1-2. 검열 해제(Abliteration) 전략

callone의 페르소나 복제 목적상 자연스러운 사투리, 거친 표현, 감정적 욕설 등이 기본 모델에서 거부(refusal)될 수 있다.
아래 두 가지 abliterated 베이스를 검증하고 **lukey03/Qwen3.5-9B-abliterated**를 1순위로 사용한다.

#### 사용 가능한 abliterated 모델 (HuggingFace)

| 우선순위 | HF ID | 형식 | 특징 |
|---|---|---|---|
| **1순위** | `lukey03/Qwen3.5-9B-abliterated-GGUF` | GGUF (text+vision) | 2단계 abliteration: orthogonal projection 3회 + QLoRA. 465개 적대적 프롬프트 0% 거부율 |
| 2순위 | `huihui-ai/Huihui-Qwen3.5-9B-abliterated` | Safetensors | huihui-ai의 빠른 abliteration 방식. Ollama에서 `huihui_ai/qwen3.5-abliterated:9b`로 직접 pull 가능 |
| 참고 | `HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive` | GGUF | Aggressive 버전. 더 공격적으로 검열 해제 |

> **⚠️ 주의:** abliterated 모델은 안전 필터가 제거되어 있다. 본 프로젝트는 개인/추모/연구 목적의 로컬 전용이며, 외부 배포 금지(callone_spec.md §20 준수).

#### Abliteration 적용 순서 (A100에서 SFT 시)

```
[권장 순서]
1. lukey03/Qwen3.5-9B-abliterated (safetensors) 다운로드
   → Unsloth로 QLoRA SFT (callone train.jsonl 적용)
   → GGUF Q4_K_M으로 변환
   → llama-server로 서빙

[대안 순서 — abliteration을 직접 적용할 경우]
1. Qwen/Qwen3.5-9B (공식 원본) 다운로드
2. Unsloth QLoRA SFT 먼저 수행 (페르소나 주입)
3. SFT 완료 후 abliterate-with-transformers 라이브러리로 abliteration
   pip install ablate-transformers
   python -m ablate --model ./output/sft_merged --output ./output/abliterated
4. GGUF Q4_K_M 변환
```

> **권장은 순서 1번(기존 abliterated 베이스 위에 SFT).** abliteration은 weight projection이므로 SFT로 덮어씌워도 refusal이 부분 복원되지 않는다.

### 1-3. SFT 설정 (A100 기준)

```python
# A100 80GB 기준, Unsloth QLoRA
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    # abliterated 베이스 사용
    model_name="lukey03/Qwen3.5-9B-abliterated",
    max_seq_length=4096,
    dtype=None,          # bf16 자동
    load_in_4bit=True,   # QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules="all-linear",   # 2026 권장: q,k,v,o,gate,up,down 전부
    lora_alpha=16,                 # alpha=r로 시작
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 학습 설정
trainer_args = dict(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    max_seq_length=4096,
    fp16=False,
    bf16=True,
    optim="adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# GGUF 변환 (학습 후)
model.save_pretrained_gguf(
    "output/callone_llm_A",
    tokenizer,
    quantization_method="q4_k_m",   # 속도/품질 최적 균형
)
```

#### 핵심 설정 포인트
- `target_modules="all-linear"`: 2026 기준 q_proj+v_proj만 타겟하던 구관습 버리고 전체 레이어 타겟. 품질 향상 검증됨.
- `use_gradient_checkpointing="unsloth"`: VRAM 절약 + 컨텍스트 길이 확장 동시 달성.
- Qwen3.5의 `<|im_start|>/<|im_end|>` 컨트롤 토큰은 LoRA PEFT에 최적화되어 있어 embedding 파인튜닝 불필요.
- **no-thinking 모드 강제**: 대화용이므로 system 프롬프트에 `/no_think` 또는 `enable_thinking=False`. thinking 토큰이 응답 지연을 크게 늘린다.

### 1-4. 학습 데이터 (callone_spec.md §12 유지)

`callone/dataset/build_dialogue.py`가 생성하는 `train.jsonl` 그대로 사용.
Qwen3.5 채팅 템플릿:

```python
# Qwen3.5 chat template (ChatML 기반)
messages = [
    {"role": "system", "content": "<persona_card_내용>"},
    {"role": "user",   "content": "발화_A"},
    {"role": "assistant", "content": "발화_B"},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,   # 대화용: thinking 비활성화
)
```

---

## 2. TTS 결정: Qwen3-TTS 1.7B (faster-qwen3-tts)

### 2-1. 선택 근거

CosyVoice3 대비 Qwen3-TTS를 선택한 핵심 이유:

| 항목 | CosyVoice3 | Qwen3-TTS 1.7B |
|---|---|---|
| zero-shot + 감정 동시 제어 | 구조적으로 분리 (zero-shot / instruct 별도 모델) | **단일 모델에서 통합 지원** |
| TTFA (첫 오디오) | 150ms | **97~159ms** |
| 한국어 개선율 | 68.7% WER 개선 (v3) | 네이티브 지원 (10개 언어) |
| 스트리밍 레이턴시 | 150ms | **97ms** |
| 파인튜닝 | 별도 recipe | LoRA 지원 (공식 sft_12hz.py) |
| 참조 오디오 | 3~10초 | **3초** |

### 2-2. 모델 선택

```
# 권장: 1.7B Base (파인튜닝 목적)
Qwen/Qwen3-TTS-12Hz-1.7B-Base

# 제로샷만 쓸 경우 (파인튜닝 불필요):
Qwen/Qwen3-TTS-12Hz-1.7B-Instruct
```

### 2-3. 감정 제어 연동 (callone orchestrator)

Qwen3-TTS는 참조 오디오(화자 정체성) + 자연어 instruct_text(감정/속도)를 **동시에** 받는다.
Gemma 4 대신 Qwen3.5가 JSON으로 감정 상태를 출력하도록 SFT하고, orchestrator에서 동적으로 주입:

```python
# callone/serve/tts_qwen.py — 실 API(검증 2026-06-17, andimarafioti/faster-qwen3-tts)
from faster_qwen3_tts import FasterQwen3TTS

tts = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device="cuda")

EMOTION_MAP = {
    "happy":   "Speak with a bright, cheerful, and warm tone.",
    "sad":     "Speak with a low, slow, comforting, and sad tone.",
    "angry":   "Speak with an annoyed, sharp, and frustrated voice.",
    "neutral": "Speak in a natural, relaxed, conversational tone.",
    "excited": "Speak with high energy and an excited, upbeat tone.",
}

def synthesize_streaming(text: str, emotion: str, ref_wav_path: str, ref_text: str):
    """
    ref_wav_path: 화자의 참조 WAV (7~10초, 24kHz, 잡음 없음)
    ref_text:     참조 WAV에서 실제 발화된 텍스트 (전사와 정확히 일치해야 함)
    emotion:      Qwen3.5가 판단한 감정 키
    """
    instruct = EMOTION_MAP.get(emotion, EMOTION_MAP["neutral"])

    # 실 API: ref_audio(경로), language 필수, instruct 로 감정 동시제어, yield=(chunk, sr, timing)
    for audio_chunk, sr, _timing in tts.generate_voice_clone_streaming(
        text=text,
        language="Korean",
        ref_audio=ref_wav_path,
        ref_text=ref_text,
        instruct=instruct,
        chunk_size=8,            # 97~159ms TTFA 달성
    ):
        yield audio_chunk        # FastAPI StreamingResponse로 전달
    # ⚠️ faster_qwen3_tts 는 LoRA 미지원 — adapter_dir/lora_scale 인자 없음(zero-shot 참조만)
```

### 2-4. Qwen3-TTS LoRA 파인튜닝 (A100, 화자당)

callone 스펙에서 화자당 30~40h 데이터가 있으므로 파인튜닝으로 유사도 대폭 향상 가능.

```bash
# 사전 준비: 반드시 24kHz로 리샘플링 (이걸 안 하면 학습 중 크래시)
find data/speakers/A/tts_segments/ -name "*.wav" | \
  xargs -I{} ffmpeg -i {} -ar 24000 {}.24k.wav

# 공식 repo 클론 (버그 픽스 버전 사용)
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
git checkout 680d4e9   # text_projection 버그 픽스 커밋

# ⚠️ 반드시 적용할 버그 픽스 2가지:
# 1) text_projection 누락 호출 → commit 680d4e9 이후 버전에서 수정됨
# 2) double label-shift → PR #178 확인 후 미병합 시 수동 패치

# LoRA 학습 (A100 80GB)
python sft_12hz.py \
  --base_model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --data_dir data/speakers/A/tts_24k/ \
  --output_dir models/tts_server/A \
  --lora_r 16 \
  --lora_alpha 32 \
  --learning_rate 2e-6 \     # ⚠️ 기본값 2e-5 금지 — 2e-6 고정
  --num_epochs 10 \
  --batch_size 4

# 추론 시 LoRA scale 스윕 테스트 (0.2, 0.3, 0.35, 0.5 중 최적값 선택)
python infer_lora.py \
  --base_model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --adapter_dir models/tts_server/A \
  --lora_scale 0.3 \
  --text "테스트 문장입니다." \
  --out test_A.wav
```

> **필수 체크리스트 (파인튜닝 전)**
> - [ ] 모든 오디오 24kHz 리샘플링 완료
> - [ ] 전사 텍스트에서 annotation 태그 제거
> - [ ] 각 클립 끝에 1초 묵음 추가
> - [ ] LR=2e-6 설정
> - [ ] commit 680d4e9 이후 버전 확인
> - [ ] PR #178 double label-shift 픽스 적용 여부 확인

### 2-5. 참조 오디오 품질 기준 (zero-shot)

```python
# callone/tts/ref_selector.py
"""
참조 WAV 선택 기준:
1. 잡음 없는 또렷한 발화 (SNR >= 20dB 권장)
   → S1 복원 단계(ClearerVoice)를 거친 cleaned WAV 사용
2. 평온하고 중립적인 음성 (감정 중립 구간 선택)
   → 너무 화가 났거나 웃고 있는 구간은 기준점으로 부적합
3. 실제 발화 텍스트와 전사 100% 일치
   → ref_text 파라미터에 정확한 글자 그대로 입력
4. 길이: 7~10초 (3초 최소이나 7초+ 권장)
5. 샘플레이트: 16kHz (S1 복원 출력) → 24kHz 변환
"""
```

---

## 3. 서빙 백엔드 결정: llama.cpp (llama-server)

### 3-1. 선택 근거

| 백엔드 | 단일 사용자 tok/s (RTX 4090) | Qwen3.5 GDN 최적화 | 설정 난이도 | callone 적합성 |
|---|---|---|---|---|
| **llama.cpp** | **~126 tok/s** | 최적 | 보통 | ✅ 최적 |
| Ollama | ~90 tok/s (GDN 미최적화, 15~20 tok/s 보고 사례 있음) | 미흡 | 쉬움 | △ |
| vLLM | ~9 tok/s (Gemma4 TRITON bug), Qwen3.5는 정상 | 최적 | 어려움 | △ 멀티유저 시 |
| SGLang | 멀티유저 최강 (vLLM+29%) | 최적 | 어려움 | △ 멀티유저 시 |

> **결론:** callone M7은 1:1 통화로 단일 사용자. Ollama는 Qwen3.5의 Gated DeltaNet 레이어를 제대로 처리하지 못해 5~6배 속도 저하 사례가 보고됨. llama.cpp가 현 시점 가장 빠르고 안정적.

### 3-2. llama-server 실행 (RunPod RTX 4090)

```bash
# 1. llama.cpp 빌드 (CUDA)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# 2. 모델 다운로드 (파인튜닝 완료 GGUF)
# $CALLONE_HOME에 저장 (RunPod Stop/Start 후에도 유지)
huggingface-cli download \
  --local-dir $CALLONE_HOME/models/llm_A \
  lukey03/Qwen3.5-9B-abliterated-GGUF \
  "qwen3.5-9b-abliterated-Q4_K_M.gguf"

# 3. llama-server 실행 (OpenAI 호환 API)
./build/bin/llama-server \
  -m $CALLONE_HOME/models/llm_A/callone_A_q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 8192 \              # 실시간 대화용 컨텍스트 (262K 불필요)
  -n 512 \               # 최대 생성 토큰 (대화 응답 충분)
  --n-gpu-layers 99 \    # 전체 GPU 오프로드
  --flash-attn \         # Flash Attention 활성화
  -t 8                   # CPU 스레드 (fallback용)
```

### 3-3. no-thinking 모드 강제

Qwen3.5는 기본적으로 thinking 모드(내부 추론 체인)가 활성화됨. 대화 응답에서 수백 토큰의 `<think>...</think>` 블록이 먼저 나와 지연이 생긴다. 반드시 비활성화:

```python
# callone/serve/llm_server.py
# llama-server OpenAI 호환 엔드포인트 사용

import httpx

async def generate_response(messages: list, speaker_id: str) -> str:
    payload = {
        "model": "callone_A",
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "max_tokens": 256,
        # Qwen3.5 no-thinking 모드
        "chat_template_kwargs": {"enable_thinking": False},
        # 또는 system 프롬프트에 /no_think 추가
        "stream": True,          # 스트리밍으로 TTFT 최소화
    }
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", "http://localhost:8080/v1/chat/completions",
                                  json=payload) as r:
            async for chunk in r.aiter_text():
                yield chunk
```

---

## 4. 실시간 파이프라인 아키텍처 (M7)

```
[폰/웹 클라이언트]
    ↕ WebRTC (LiveKit) / WebSocket
[callone-serve FastAPI :8000]
    │
    ├─ VAD (DeepFilterNet, 발화 종료 감지 ~200ms)
    │
    ├─ ASR (Whisper large-v3-turbo, 방언 적응 완료)
    │     → 텍스트 출력
    │
    ├─ LLM (llama-server :8080, Qwen3.5-9B abliterated + SFT)
    │     → 스트리밍 JSON {"emotion": "sad", "reply": "어이구..."}
    │     → no-thinking 모드, 1~2문장 단위 청크 즉시 TTS로
    │
    └─ TTS (faster-qwen3-tts :8002, Qwen3-TTS 1.7B LoRA)
          → 감정 instruct 동적 주입
          → 스트리밍 오디오 청크 → 클라이언트

[지연 예산]
VAD        ~200ms
ASR         ~0ms (스트리밍 겹침 처리)
LLM TTFT   ~100~150ms (llama.cpp, ~126 tok/s)
TTS TTFA   ~97~159ms  (faster-qwen3-tts, chunk_size=8)
─────────────────────
합계        ~400~500ms → 목표 1.2초 충분히 달성
```

### 4-1. orchestrator 핵심 코드 패턴

```python
# callone/serve/orchestrator.py

async def handle_turn(audio_bytes: bytes, speaker_id: str, ref_wav: str, ref_text: str):
    """한 번의 발화 턴을 처리하는 전체 파이프라인"""
    
    # 1. ASR
    transcript = await asr_stream(audio_bytes)
    
    # 2. LLM (스트리밍, 문장 단위 청크)
    sentence_buffer = ""
    async for token in llm_stream(transcript, speaker_id):
        sentence_buffer += token
        
        # 문장 완성 감지 (마침표, 쉼표, 느낌표 등)
        if any(sentence_buffer.endswith(p) for p in [".", "!", "?", "~", "ㅋ", "요.", "다.", "네."]):
            # JSON 파싱 (emotion + reply)
            try:
                parsed = json.loads(sentence_buffer)
                emotion = parsed.get("emotion", "neutral")
                reply_text = parsed.get("reply", sentence_buffer)
            except json.JSONDecodeError:
                emotion = "neutral"
                reply_text = sentence_buffer
            
            # 3. TTS (문장 단위로 즉시 합성 → 전체 응답 완성 기다리지 않음)
            async for audio_chunk in synthesize_streaming(
                text=reply_text,
                emotion=emotion,
                ref_wav_path=ref_wav,
                ref_text=ref_text,
            ):
                yield audio_chunk   # 클라이언트로 즉시 전송
            
            sentence_buffer = ""
```

---

## 5. 인프라: RunPod Community Cloud

### 5-1. 인스턴스 설정

```
GPU:      RTX 4090 24GB (Community Cloud)
가격:     $0.34/hr (Gemini가 말한 $0.69는 Secure Cloud 가격)
스토리지: Volume 150GB (모델 캐시 포함)
```

### 5-2. 환경 변수 설정 (Stop/Start 후에도 모델 재다운로드 방지)

```bash
# 인스턴스 시작 시 반드시 설정. $CALLONE_HOME = 영속 폴더(RunPod=/workspace, Elice 등=$HOME)
if [ -d /workspace ] && [ -w /workspace ]; then export CALLONE_HOME=/workspace; else export CALLONE_HOME=$HOME; fi
export HF_HOME=$CALLONE_HOME/hf_cache
export CALLONE_DATA_DIR=$CALLONE_HOME/data
# HF_TOKEN 은 게이트 모델 쓸 때만(현 스택 9B GGUF·Qwen3-TTS 는 비게이트 → 불필요)
```

### 5-3. 포트 설정

| 서비스 | 내부 포트 | 타입 | 용도 |
|---|---|---|---|
| FastAPI 오케스트레이터 | 8000 | **TCP** | 메인 API + WebSocket |
| llama-server (LLM) | 8080 | HTTP (내부) | LLM 추론 |
| faster-qwen3-tts (TTS) | 8002 | HTTP (내부) | TTS 합성 |
| LiveKit (WebRTC) | 7880 | **TCP** | 실시간 오디오 전송 |

> ⚠️ **RunPod TCP 포트 설정 필수:** Pod 생성 시 8000, 7880은 반드시 HTTP → TCP로 변경. 프록시 우회 직통 연결로 레이턴시 최소화.

### 5-4. Stop/Start 절약 운용

```
코딩할 때만 START → 작업 종료 시 STOP
  - GPU 비용: $0.34/hr (STOP 시 $0.00)
  - Storage 비용: 150GB × $0.10~$0.20/GB/월 = 약 $15~$30/월 유지
  - $CALLONE_HOME에 저장한 모델은 STOP/START 후에도 유지됨
  - TCP 포트 번호는 START할 때마다 새로 배정됨 → 클라이언트 코드의 ws:// 주소 업데이트 필요

며칠 이상 쉴 때: 코드를 GitHub에 push → Pod를 Terminate(삭제)
  → 재시작 시 HF에서 모델 재다운로드 (Gemma 4 12B ~24GB 대신 Qwen3.5-9B ~5.7GB라 빠름)
```

---

## 6. 전체 스택 요약 (callone_spec.md §3 업데이트 버전)

```yaml
# configs/llm_server.yaml (업데이트)
base_model: "lukey03/Qwen3.5-9B-abliterated"   # abliterated 베이스
method: "qlora"
lora:
  r: 16
  alpha: 16
  dropout: 0.05
  target: "all-linear"
train:
  epochs: 3
  lr: 1e-4
  bsz: 4
  grad_accum: 4
  max_seq_len: 4096
  bf16: true
chat_template: "qwen3.5"
enable_thinking: false          # 대화용: thinking 비활성화
export_format: "q4_k_m"        # GGUF 변환

# configs/serve.yaml (업데이트)
llm:
  backend: "llama_server"       # llama.cpp llama-server
  model_path: "$CALLONE_HOME/models/llm_A/callone_A_q4_k_m.gguf"
  port: 8080
  n_gpu_layers: 99
  flash_attn: true
  context: 8192
  enable_thinking: false

tts:
  backend: "qwen3_tts"          # faster-qwen3-tts (FasterQwen3TTS.from_pretrained)
  model: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  language: "Korean"            # 실 API 필수 인자
  ref_wav: "data/speakers/A/ref_24k.wav"   # zero-shot 클론 참조(필수). 7~10초·24kHz·깨끗
  ref_text: ""                  # 참조 WAV 의 실제 발화 텍스트
  # ⚠️ faster_qwen3_tts 는 LoRA 미지원(adapter_dir/lora_scale 인자 없음) — 충실도 필요 시 Piper.
  chunk_size: 8                 # TTFA ~159ms
  port: 8002

asr:
  model: "models/asr_dialect"   # Whisper large-v3 방언 적응 완료
  backend: "faster_whisper"

llm_tier: "server_gpu"
tts_tier: "server_gpu"
```

---

## 7. Claude Code 실행 순서 (M7 파이프라인 관통)

```
[M0 이미 완료] 스캐폴딩, 환경, 스키마

[M1~M6 A100에서 완료 가정]
  - S0~S3: 데이터 전처리, ASR 적응, 데이터셋 빌드
  - M4: Qwen3-TTS 1.7B LoRA 파인튜닝 (화자 A)
  - M5: Qwen3.5-9B abliterated QLoRA SFT (화자 A 페르소나)
  - 산출물: callone_A_q4_k_m.gguf, models/tts_server/A/

[M7 클라우드 GPU — RunPod RTX 3090/4090 또는 Elice A100/H100]
Step 1. 환경 설정
  # $CALLONE_HOME = 영속 폴더(RunPod=/workspace, Elice 등=$HOME) 자동선택
  if [ -d /workspace ] && [ -w /workspace ]; then export CALLONE_HOME=/workspace; else export CALLONE_HOME=$HOME; fi
  export HF_HOME=$CALLONE_HOME/hf_cache       # 비게이트라 HF_TOKEN 불필요

Step 2. 모델 다운로드 ($CALLONE_HOME에 저장)
  huggingface-cli download [모델들]

Step 3. llama-server 실행 (백그라운드)
  ./llama-server -m callone_A_q4_k_m.gguf --port 8080 ...

Step 4. faster-qwen3-tts 서버 실행 (백그라운드)
  python tts_stream.py --port 8002

Step 5. callone-serve FastAPI 실행
  callone-serve --port 8000

Step 6. TCP 포트 확인 및 React UI 클라이언트 주소 업데이트
  RunPod Console → Connect → TCP Port Mapping 확인

Step 7. 통화 테스트
  목표: 첫 음성 < 0.8s (지연 예산 400~500ms)
```

---

## 8. callone_spec.md에서 변경되지 않는 부분

다음 섹션은 이 문서에서 언급하지 않았어도 **그대로 유지**한다:

- §2 하드 제약 (외부 유료 API 0, 데이터 로컬 보관) — **유지**
- §4 아키텍처 S0~S3 파이프라인 — **유지**
- §10 S2 화자 분리 (pyannote community-1) — **유지**
- §11 S2.5 방언 자동 프로파일링 — **유지**
- §13 ASR 방언 적응 (Whisper large-v3 LoRA) — **유지**
- §17 UI (missvoice 골격, React) — **유지**
- §19 평가 기준 (DER<12%, SECS>0.70 등) — **유지**
- §20 보안·윤리 (로컬 전용, PII 마스킹, 암호화) — **유지**
- `tests/test_no_paid_api.py` — **유지 (abliterated 모델은 로컬이므로 해당 없음)**

---

## 9. 알려진 이슈 및 대응

| 이슈 | 상태 | 대응 |
|---|---|---|
| Qwen3.5 + Ollama 속도 저하 (15~20 tok/s) | 확인됨 | llama.cpp 사용으로 우회 |
| Qwen3.5 + SGLang GDN thinking 루프 | 구버전 이슈, nightly로 수정 | llama.cpp 사용으로 회피 |
| Qwen3-TTS sft_12hz.py 이중 label-shift | PR #178 추적 중 | 학습 전 commit 680d4e9 확인 |
| Qwen3-TTS 24kHz 강제 요구 | 주의 필요 | 모든 오디오 사전 리샘플링 |
| Gemma 4 12B + vLLM TRITON fallback (~9 tok/s) | issue #38887 추적 중 | 본 문서에서 Qwen3.5로 교체 완료 |

---

*이 문서는 callone_spec.md의 §3 Model Decisions를 대체하는 2026-06-14 버전이다.*
*다음 검토 시점: M7 파이프라인 관통 후, 또는 Qwen3.6/3.7 open-weight 출시 시.*
