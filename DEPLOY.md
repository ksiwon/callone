# callone 배포·실행 매뉴얼 (서버 + 로컬 통합)

> 이 문서 하나로 **서버 올리기 → 모델 받기 → 띄우기 → 확인**까지. 신스택 기준
> (Qwen3.5-9B uncensored/aggressive + Qwen3-TTS 1.7B + llama.cpp). 모델 결정은
> [`callone_stack_decision.md`](callone_stack_decision.md), 통합 앱은 [`studio/README.md`](studio/README.md).
>
> 🚀 **RunPod 4090 켜서 바로 따라가는 순서(확정·복붙) = [`docs/RUNPOD_RUN.md`](docs/RUNPOD_RUN.md).**
> 이 DEPLOY 는 근거·대안·트러블슈팅 레퍼런스. 실행은 RUNPOD_RUN 을 위→아래로.

전제: 한 폴더 `callone/` 안에 `callone`(패키지)·`studio`(통합앱)·`voice_clone`·configs·scripts 공존.

---

## 0. 두 가지 실행 형태 — 뭘 띄울지부터

| 형태 | 언제 | 진입점 | 필요 |
|---|---|---|---|
| **A. studio 통합 앱** | 전사/제로샷TTS/턴제통화를 한 화면에서. 데모·운영 | `python -m studio` (:50000) | gradio + 선택 백엔드 |
| **B. callone-serve 풀 실시간** | WebRTC·barge-in 진짜 전화 | `callone-serve` (:8000) + `ui` (:5173) | llama-server + TTS |
| **(공통) llama-server** | LLM 추론(둘 다 이걸 HTTP로 호출) | `llama-server ... :8080` | GGUF 모델 |

> studio 의 **통화 칸**도 내부적으로 llama-server 가 떠 있으면 그걸 쓴다(없으면 폴백).

---

## 1. 서버 준비 (GPU: RunPod RTX 4090 / Elice A100 등 Linux+CUDA)

### 1-1. 코드 올리기
```bash
# (택1) git
git clone <your-repo> callone && cd callone
# (택2) 로컬에서 업로드 — Windows: scripts/upload_to_elice.ps1 참고(.venv/.git/node_modules 제외)
```

### 1-2. 환경 설치 — ⚠️ **통화 서빙은 setup_serve_gpu.sh** (heavy 와 섞지 말 것)
실시간 통화(4090)는 **서빙 전용 환경**이다. 데이터 파이프라인용 `.[heavy]`(transformers<4.50 +
pyannote 3.x)와 Qwen3-TTS(transformers==4.57.3)는 **충돌**하므로 별 venv(`.venv-serve`)에 깐다.
```bash
cd callone
bash scripts/setup_serve_gpu.sh              # ffmpeg→.venv-serve→torch(cu124)→faster-whisper+faster-qwen3-tts
# CUDA 버전 다르면:
CUDA_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/setup_serve_gpu.sh
source .venv-serve/bin/activate
```
> LLM 은 이 venv 에 없다 — `llama-server`(컴파일 바이너리)로 HTTP 서빙(§3-1)이라 파이썬 충돌 0.
> 화자분리·학습까지 같은 박스에서 하려면 그건 **별도** `bash scripts/setup_server.sh`(.[heavy], 다른 venv).

### 1-3. 환경변수 (Stop/Start 후에도 모델 유지 — /workspace 에 캐시)
```bash
export HF_HOME=/workspace/hf_cache
export HF_TOKEN=hf_xxx                        # 게이트 모델용(무료 동의)
export CALLONE_TIER=server_gpu                # GPU 강제(자동감지 대신 명시)
# 위 3줄을 ~/.bashrc 에 넣어두면 재접속 편함
```

---

## 2. 모델 받기 (/workspace 에 저장 → Stop/Start 유지)

### 2-1. LLM — Qwen3.5-9B aggressive uncensored GGUF (서빙 직행, 학습 불필요)
```bash
huggingface-cli download \
  HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive \
  --local-dir /workspace/models/llm_A \
  --include "*Q4_K_M*.gguf"
# configs/llm_server.yaml 의 serve_gguf_repo/serve_gguf_file 와 일치 확인.
```
> 페르소나 SFT 까지 하려면: §4 학습 경로(safetensors uncensored 베이스 + LoRA → GGUF 변환).

### 2-2. TTS — Qwen3-TTS 1.7B (faster-qwen3-tts)
```bash
pip install faster-qwen3-tts                  # 없으면 studio TTS 칸은 piper/placeholder 폴백
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir /workspace/models/qwen3_tts
```

### 2-3. (제로샷 TTS만) CosyVoice3 — studio 제로샷 칸용
```bash
bash voice_clone/GPU_A100/setup.sh            # CosyVoice repo + Fun-CosyVoice3-0.5B
export COSYVOICE_MODEL_DIR=/path/to/pretrained_models/Fun-CosyVoice3-0.5B
```

---

## 3. 띄우기

### 3-1. llama-server (LLM, 별도 터미널 — 계속 켜둠)
```bash
# llama.cpp 빌드(최초 1회): scripts/run_llama_server.md 참고
./llama-server \
  -m /workspace/models/llm_A/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 8192 -n 512 --n-gpu-layers 99 --flash-attn
# health: curl http://127.0.0.1:8080/health  → {"status":"ok"}
```
> 포트는 **8080 으로 통일**(코드 기본값 = `serve.yaml llm.base_url` = 결정서). 다른 포트를 쓰려면
> llama-server `--port` 와 `configs/serve.yaml` 의 `llm.base_url` 을 같은 값으로 맞춘다.

### 3-2-A. studio 통합 앱
```bash
cd callone
python -m studio                              # http://0.0.0.0:50000
# 포트 바꾸기: PORT=7000 python -m studio
# 원격(RunPod 등) 브라우저+마이크 — 포트 노출/재시작 없이 공개 HTTPS 링크:
GRADIO_SHARE=1 python -m studio               # 출력의 https://....gradio.live 열기
```
헤더에서 [환경=GPU] [목적] [데이터모드] 고르면 끝. 배지에 🟢/🟡 표시.
통화칸: 상대 화자 ID(제로샷은 학습본과 다른 ID), 페르소나·상황 입력 → 마이크.

### 3-2-B. callone-serve 풀 실시간 (대안)
```bash
cd callone && callone-serve                   # FastAPI :8000 (+WebSocket)
cd callone/ui && npm install && npm run dev   # :5173
```

### 3-3. RunPod 포트 매핑 (외부 접속)
- Pod 생성 시 **8000·50000·7880(WebRTC) 은 HTTP→TCP 로 변경**(프록시 우회 = 저지연).
- TCP 포트번호는 **START 할 때마다 새로 배정** → 클라이언트 `ws://`/주소 갱신.

---

## 4. (선택) 학습 경로 — 풀클론 화자 모델

```bash
# 데이터 파이프라인(S0~S3) + SFT 데이터
bash scripts/run_full.sh                      # 또는 scripts/run_pilot.sh 50 A

# LLM 페르소나 LoRA (A100, Qwen3.5 = MoE → bf16 LoRA 기본)
callone-llm-train --config llm_server --speakers A
python scripts/make_gguf.py --speaker A --config llm_server --llama-cpp /path/llama.cpp

# TTS 화자 LoRA (Qwen3-TTS, ⚠️ 24kHz 리샘플 + lr 2e-6)
callone-tts-train --speakers A
```
> ⚠️ Windows 파인튜닝은 unstable(unsloth #2742) → **학습은 Linux/A100**.

---

## 5. 동작 확인 (스모크)

```bash
cd callone
# 1) 앱 기동
python -m studio &  sleep 3
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:50000/   # 200 기대
# 2) 파이프라인 테스트(모델 없이 폴백 경로)
python -m pytest -q                                                # 31 passed 기대
# 3) llama-server 단독
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"안녕"}],"max_tokens":32}'
```

---

## 6. 트러블슈팅 (실측 + 웹검증 반영, 폴백은 코드에 박힘)

| 증상 | 원인 | 해결 |
|---|---|---|
| `numpy 2.4` numba ImportError | numpy≥2.4 ↔ numba/librosa | `pip install "numpy<2.4"` |
| 마이크 녹음 전사 실패 | 브라우저는 opus/webm | ffmpeg 설치(`apt install ffmpeg`). 없으면 wav 업로드만 |
| GPU 전사 크래시 | ctranslate2≥4.5 ↔ cuDNN9/CUDA12.3 불일치 | 코드가 **CPU int8 자동 폴백**. GPU 쓰려면 torch(cu12x)·ct2 버전 맞춤 |
| TTS 칸 🟡 (Qwen3-TTS) | faster-qwen3-tts 미설치 | `pip install faster-qwen3-tts` (또는 piper 폴백) |
| 제로샷 TTS 🟡 (CosyVoice) | repo/가중치 없음 | `voice_clone/*/setup.sh` + `COSYVOICE_MODEL_DIR` |
| CosyVoice 중국어로 새어나옴 | zero-shot 언어 디폴트 버그 | 참조 **한국어 prompt_text** 정확히 입력 |
| LLM content 빈데 reasoning_content만 참 | Qwen3.5 thinking 기본 ON | `chat_template_kwargs.enable_thinking:false`(코드 처리) 또는 서버 `--reasoning-budget 0` |
| `hf_transfer` ModuleNotFound (모델 다운) | 베이스 이미지 `HF_HUB_ENABLE_HF_TRANSFER=1` | `pip install hf_transfer` (또는 `export HF_HUB_ENABLE_HF_TRANSFER=0`) |
| TTS `ref_text is required ... ICL mode` | Qwen3-TTS 클론은 참조 전사 필수 | `data/speakers/{spk}/ref_text.txt`(large-v3 전사) 두기 — 없으면 자동전사 폴백(품질↓) |
| TTS 톤 끊김/흔들림 | chunk_size 작음·temperature 높음 | `chunk_size:25` + `temperature:0.5`(코드 기본, 실측 최적) |
| 통화 톤이 항상 neutral | base LLM이 감정 라벨 안 냄 | `tts.emotion:true` 면 orchestrator 가 LLM 감정라벨링 자동 on |
| 통화 응답이 단순/엉뚱 | 모델 미준비 → PersonaLLM 폴백 | llama-server 띄우고 `serve.yaml llm.backend=llama` |
| MoE 4bit 학습 에러 | bitsandbytes MoE 4bit 버그 | `load_in_4bit:false`(bf16 LoRA, 기본값) |

---

## 7. 포트 요약

| 서비스 | 포트 | 비고 |
|---|---|---|
| studio 앱 | 50000 | `PORT` 로 변경 |
| callone-serve | 8000 | RunPod TCP |
| llama-server (LLM) | 8080 | `serve.yaml llm.base_url` |
| faster-qwen3-tts | 8002 | (별도 서버 구성 시) |
| LiveKit WebRTC | 7880 | RunPod TCP |
| React UI(dev) | 5173 | |
