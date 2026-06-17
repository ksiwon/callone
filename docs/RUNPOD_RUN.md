# RunPod RTX 4090 — 실시간 음성 통화 실행 순서 (확정판)

> 켜자마자 **위에서 아래로 그대로** 복붙. 각 단계 끝 **✅ 체크**가 통과해야 다음으로.
> 싼 단계(설치·헬스)부터 검증 → 비싼 GPU 시간 낭비 방지. 음성 통화만(토킹헤드·학습 제외).
> 근거/대안은 [DEPLOY.md](../DEPLOY.md), 모델결정 [callone_stack_decision.md](../callone_stack_decision.md).

전제: 음성 LoRA/페르소나는 A100에서 이미 끝났거나, **base 9B+페르소나로 바로 통화**(학습 없이 동작).

---

## 0. Pod 생성 (RunPod 콘솔, UI)
- GPU: **RTX 4090 24GB** / Template: PyTorch(CUDA 12.x) / **Volume 100GB+ 를 `/workspace` 에 마운트**.
- **Expose TCP Ports: `8000`**(callone-serve) **, `50000`**(studio). HTTP 아닌 **TCP** 로(저지연).
- ⚠️ TCP 포트번호는 START마다 **새로 배정** → 클라이언트 주소는 그때 콘솔 Connect 에서 확인.

---

## 1. 코드 + 환경변수
```bash
cd /workspace
git clone <your-repo> callone && cd callone     # 또는 로컬 업로드(.venv/.git 제외)

export HF_HOME=/workspace/hf_cache              # 모델 캐시 → Stop/Start 유지
export CALLONE_TIER=server_gpu                  # GPU 강제(자동감지 대신 명시)
echo 'export HF_HOME=/workspace/hf_cache'   >> ~/.bashrc
echo 'export CALLONE_TIER=server_gpu'       >> ~/.bashrc
```
> HF_TOKEN 불필요(9B GGUF·Qwen3-TTS·whisper 전부 비게이트). 게이트 모델 쓸 때만 `export HF_TOKEN=hf_...`.

---

## 2. 서빙 환경 설치 (서빙 전용 venv — heavy 와 분리)
```bash
bash scripts/setup_serve_gpu.sh                 # ffmpeg→.venv-serve→torch(cu124)→faster-whisper+faster-qwen3-tts→numpy<2.4
# CUDA 다르면:  CUDA_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/setup_serve_gpu.sh
source .venv-serve/bin/activate
```
✅ **체크:** 스크립트 끝의 `pip check` 통과 + `transformers 4.57.3` 표시. (충돌 나면 여기서 멈추고 해결 — GPU 안 씀)

---

## 3. llama.cpp 빌드 (LLM 서버 바이너리)
```bash
bash scripts/build_llama_cuda.sh                # → /workspace/llama.cpp/build/bin/llama-server
```
✅ **체크:** 마지막에 `✅ 빌드 완료: .../llama-server` 출력.

---

## 4. 모델 다운로드 (`/workspace` → Stop/Start 유지)
```bash
# 4-1. LLM: Qwen3.5-9B uncensored aggressive GGUF (Q4_K_M, 5.3GB)
huggingface-cli download HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive \
  --include "*Q4_K_M*.gguf" --local-dir /workspace/models/llm_A

# 4-2. TTS: Qwen3-TTS 1.7B Base (4.54GB, Apache2.0 비게이트)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir /workspace/models/qwen3_tts
```
✅ **체크:**
```bash
ls -lh /workspace/models/llm_A/*Q4_K_M*.gguf      # 5.3GB 파일 1개
ls /workspace/models/qwen3_tts                    # config/모델 파일 존재
```

---

## 5. 화자 참조 음성 (Qwen3-TTS 클론 필수입력)
Qwen3-TTS는 **zero-shot 참조음성**으로 음색을 만든다. 화자당 깨끗한 **7~10초·24kHz·모노** 클립 1개 필요.
```bash
mkdir -p data/speakers/A
# 깨끗한 발화 wav 하나를 24kHz 모노로 변환(8초 예시). <clean.wav> = 잡음없는 화자 A 단독 발화.
ffmpeg -y -i <clean.wav> -ac 1 -ar 24000 -t 8 data/speakers/A/ref_24k.wav
```
- 코드가 `data/speakers/A/ref_24k.wav` 를 **자동 인식**(serve.yaml 수정 불필요).
- (선택) 참조 발화의 실제 텍스트를 알면 품질↑: `configs/serve.yaml` 의 `tts.ref_text` 에 정확히 입력.

✅ **체크:** `ls -lh data/speakers/A/ref_24k.wav` (수백 KB, ~8초).

---

## 6. llama-server 기동 (LLM, 백그라운드 — 계속 켜둠)
```bash
nohup /workspace/llama.cpp/build/bin/llama-server \
  -m /workspace/models/llm_A/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 8192 -n 512 --n-gpu-layers 99 \
  > /workspace/llama.log 2>&1 &
sleep 8
# (속도 옵션, 선택) 위 명령이 잘 뜨면 다음 기동부터 `--flash-attn on` 추가.
#   ⚠️ 최신 빌드는 값 인자(on|off|auto) — bare `--flash-attn` 는 에러날 수 있어 기본에선 뺐다.
```
✅ **체크(둘 다):**
```bash
curl -s http://127.0.0.1:8080/health                                  # {"status":"ok"}
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"안녕"}],"max_tokens":32}'   # 한국어 응답
```
> 실파일명은 `ls /workspace/models/llm_A/` 로 확인해 맞춰라(repo가 파일명 바꿨을 수 있음).

---

## 7. TTS 단독 검증 (5번 ref + 모델 + 패키지 한 번에 확인)
```bash
python - <<'PY'
import numpy as np, soundfile as sf
from callone.serve.tts_qwen import QwenTTS
tts = QwenTTS("A")                                  # ref_24k.wav 자동 인식
y = np.concatenate(list(tts.synth_stream("안녕하세요. 잘 지냈어요?", emotion="happy")))
sf.write("/workspace/tts_test.wav", y, tts.sr); print("OK", y.shape, tts.sr)
PY
```
✅ **체크:** `OK (N,) 24000` 출력 + `/workspace/tts_test.wav` 생성(들어보면 화자 음색).
> 실패해 Piper 폴백 로그가 뜨면: faster-qwen3-tts 설치/ref_wav/CUDA 중 하나 — 메시지대로 처리(아직 통화 안 켰으니 비용 적음).

---

## 8. 앱 기동 (택1)
```bash
# A) studio 통합 앱(데모·턴제 통화, 가장 간단)
PORT=50000 python -m studio        # http://0.0.0.0:50000

# B) callone-serve 풀 실시간(WebSocket, barge-in) + UI
callone-serve                      # FastAPI :8000
#   (UI 쓰려면 별 터미널)  cd ui && npm install && npm run dev   # :5173
```
✅ **체크:** `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:50000/` → `200`
(B면 `http://127.0.0.1:8000/api/health` → `{"status":"ok"}`)

---

## 9. 외부 접속 + 통화 테스트
- RunPod 콘솔 **Connect → TCP 포트 매핑** 확인(8000/50000 의 외부 host:port).
- 브라우저로 그 주소 접속 → 마이크 한마디 → 화자 음색 응답.
✅ **목표:** 첫 음성 < 1.2초(예산 ~0.8초). studio 통화칸 상태에 `첫음성 …ms` 표시.

---

## 통화 품질 올리기 (선택, 통화 된 다음)
- **페르소나/RAG:** `data/speakers/A/` 에 profile.json + utterances(말투·기억) 두면 system 주입으로 진해짐.
- **LoRA 진한 말투:** A100에서 만든 화자 GGUF(병합본)를 6번 `-m` 으로 교체.
- **감정:** LLM이 emotion 키 내면 TTS instruct 자동 주입(이미 코드 처리). 평소 통화는 그대로 OK.

## 막힐 때 (전부 코드에 폴백 박힘)
| 증상 | 위치 | 처리 |
|---|---|---|
| 설치 transformers 충돌 | 2번 | heavy venv 와 섞였는지 확인. `.venv-serve` 단독 재설치 |
| llama-server health 무응답 | 6번 | `cat /workspace/llama.log` — VRAM/파일명/포트 |
| TTS가 Piper로 폴백 | 7번 | faster-qwen3-tts 미설치 or ref_24k.wav 없음 or CUDA |
| ASR 빈 전사 | 통화 | ct2/cuDNN 불일치 → 코드가 CPU int8 자동 폴백(로그 확인) |
| 통화 응답 단순/엉뚱 | 통화 | llama-server 안 떠서 PersonaLLM 폴백 → 6번 재확인 |

## Stop 전
- 모델은 `/workspace` 라 유지. 코드 변경은 `git push`. 다음 START 때 **TCP 포트 재배정** → 9번 주소만 갱신.
