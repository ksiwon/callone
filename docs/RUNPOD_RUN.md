# RunPod GPU — 실시간 음성 통화 실행 순서 (검증 확정판, 2026-06-17 실측)

> 켜자마자 **위에서 아래로 그대로** 복붙. 각 단계 끝 **✅ 체크**가 통과해야 다음으로.
> 싼 단계(설치·헬스)부터 → 비싼 GPU 시간 낭비 방지. 음성 통화만(토킹헤드·학습 제외).
> 이 문서는 **실제로 4090/3090 24GB에서 관통 검증한 순서**다. 근거/대안은 [DEPLOY.md](../DEPLOY.md).

전제: 학습 없이 **base 9B + 페르소나(+제로샷 음색)로 바로 통화** 동작 확인됨. (LoRA/페르소나 학습본 있으면 더 진해짐.)

---

## 0. Pod 생성 (RunPod 콘솔)
- GPU: **RTX 4090** 권장. 매진 시 **RTX 3090 / 3090 Ti(24GB)** 도 OK(메모리대역폭 비슷 → 토큰생성 속도 거의 동급, prefill만 약간 느림). **24GB VRAM 필수**(16GB 비추).
  - Community Cloud = 절반 가격(4090 ~$0.34/hr). 매진 잦음 → 24GB 아무거나 available 한 것.
- 템플릿: **순정 PyTorch (CUDA 12.x~13.x)**. ComfyUI/AI-Rebels 류 금지. (CUDA 13 베이스라도 우리 torch는 cu124로 동작 확인.)
- 디스크: **Container 30GB**(이미지용, 줄이면 안 뜸) + **Volume 40~50GB**(`/workspace`, 실사용 ~25GB).
- 포트 노출은 **선택** — 아래 9번에서 **gradio 공개 링크(GRADIO_SHARE)** 로 접속하면 RunPod 포트 설정·재시작 불필요.
- 접속: Pod Running → **Connect → JupyterLab(8888)** 또는 **Web terminal**(SSH키 불필요).

---

## 1. 코드 + 환경변수
```bash
cd /workspace
git clone https://github.com/ksiwon/callone.git && cd callone

export HF_HOME=/workspace/hf_cache              # 모델 캐시 → Stop/Start 유지
export CALLONE_TIER=server_gpu                  # GPU 강제
echo 'export HF_HOME=/workspace/hf_cache'   >> ~/.bashrc
echo 'export CALLONE_TIER=server_gpu'       >> ~/.bashrc
```
> HF_TOKEN 불필요(9B GGUF·Qwen3-TTS·whisper 전부 비게이트).

---

## 2. 서빙 환경 설치 (서빙 전용 venv — heavy 와 절대 분리)
```bash
bash scripts/setup_serve_gpu.sh                 # ffmpeg→.venv-serve→torch(cu124)→faster-whisper+faster-qwen3-tts→numpy<2.4
source .venv-serve/bin/activate
```
✅ **체크:** 끝에 `No broken requirements found` + 버전표에 **`transformers 4.57.3`**, `torch 2.x+cu124`, `faster-qwen3-tts 0.2.6`.
> CUDA 13 베이스에서 cu124 torch 정상. 혹시 torch import 에러면: `rm -rf .venv-serve && CUDA_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_serve_gpu.sh`

---

## 3. llama.cpp 빌드 (LLM 서버 바이너리, CUDA)
```bash
nvcc --version                                  # CUDA 컴파일러 확인(없으면 runtime 이미지 → 프리빌트 필요)
bash scripts/build_llama_cuda.sh                # → /workspace/llama.cpp/build/bin/llama-server
```
✅ **체크:** `✅ 빌드 완료: /workspace/llama.cpp/build/bin/llama-server`

---

## 4. 모델 다운로드 (`/workspace` → Stop/Start 유지)
```bash
pip install hf_transfer                          # ⚠️ 베이스 이미지가 HF_HUB_ENABLE_HF_TRANSFER=1 → 이거 없으면 다운로드 에러

# 4-1. LLM: 9B uncensored aggressive GGUF (Q4_K_M, 5.3GB)
huggingface-cli download HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive \
  --include "*Q4_K_M*.gguf" --local-dir /workspace/models/llm_A

# 4-2. TTS: Qwen3-TTS 1.7B Base (4.5GB, 비게이트)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir /workspace/models/qwen3_tts

# TTS 모델을 로컬 다운본으로 지정(재다운로드 방지). repo config 는 HF id 유지 → git pull 충돌 없음.
export CALLONE_TTS_MODEL=/workspace/models/qwen3_tts
echo 'export CALLONE_TTS_MODEL=/workspace/models/qwen3_tts' >> ~/.bashrc
```
✅ **체크:** `ls -lh /workspace/models/llm_A/*Q4_K_M*.gguf` (5.3GB 1개) + `ls /workspace/models/qwen3_tts` (config/model.safetensors 등)

---

## 5. 화자 참조 음성 + ref_text (제로샷 클론 필수)
Qwen3-TTS는 **참조음성(7~10초·깨끗) + 그 전사(ref_text)** 로 음색을 만든다(ICL 모드 → ref_text 필수).
> ⚠️ **화자 ID 규칙:** 학습본(예: 엄마=`A`)과 제로샷 테스트는 **다른 ID**로. 여기선 예시로 `sis`.
> 참조 클립은 git 에 없으니(개인 음성) **JupyterLab 드래그 업로드** 또는 runpodctl 로 올린다.

```bash
apt-get install -y sox >/dev/null 2>&1          # qwen-tts 오디오 처리(경고 제거)
mkdir -p data/speakers/sis
# 업로드한 깨끗한 클립 → 24kHz 모노 변환(앞 10초)
ffmpeg -y -i data/speakers/sis/Record.wav -ac 1 -ar 24000 -t 10 data/speakers/sis/ref_24k.wav

# ref_text 저장(large-v3 = 정확. 이게 품질 큰 영향). 있으면 자동전사 안 타고 이걸 씀.
python - <<'PY'
from faster_whisper import WhisperModel
m=WhisperModel("large-v3",device="cuda",compute_type="float16")
segs,_=m.transcribe("data/speakers/sis/ref_24k.wav",language="ko")
open("data/speakers/sis/ref_text.txt","w",encoding="utf-8").write("".join(s.text for s in segs).strip())
print("ref_text.txt saved")
PY
```
✅ **체크:** `ls data/speakers/sis/` → `ref_24k.wav` + `ref_text.txt`
> ref_text.txt 없으면 코드가 자동전사(small)로 폴백하나 품질↓ → **large-v3 ref_text.txt 권장.**
> 품질 튜닝은 코드에 박힘: `chunk_size=25` + `temperature=0.5`(실측 최적: 8/0.9는 끊김·흔들림).

---

## 6. llama-server 기동 (LLM, 백그라운드 — 계속 켜둠)
```bash
nohup /workspace/llama.cpp/build/bin/llama-server \
  -m /workspace/models/llm_A/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 8192 -n 512 --n-gpu-layers 99 \
  > /workspace/llama.log 2>&1 &
sleep 15
curl -s http://127.0.0.1:8080/health ; echo
```
✅ **체크:** `{"status":"ok"}`. 그다음 thinking 끈 생성(우리 코드가 보내는 방식과 동일):
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"엄마 나 왔어"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}' ; echo
```
→ `content` 에 **한국어 답변**이 차야 정상. (이 모델 thinking 기본 ON — `enable_thinking:false` 없으면 `reasoning_content` 로만 가고 content 빈다. 우리 LLM 코드는 이미 false 로 보냄. 서버 자체를 끄려면 `--reasoning-budget 0` 추가 기동.)
> 속도 옵션(선택): 잘 뜨면 다음 기동부터 `--flash-attn on`(bare `--flash-attn`는 최신 빌드서 에러).

---

## 7. TTS 단독 검증 (warm 지연 = 통화 TTS 몫)
```bash
python - <<'PY'
import time, numpy as np, soundfile as sf
from callone.serve.tts_qwen import QwenTTS
tts = QwenTTS("sis")                              # cs=25·temp=0.5·ref_text.txt·env모델 자동
print(f"chunk_size={tts.chunk_size} temperature={tts.temperature} ref_text={tts.ref_text[:25]!r}")
def run(txt,tag):
    t=time.time(); first=None; ch=[]
    for i,c in enumerate(tts.synth_stream(txt, emotion="neutral")):
        if i==0: first=time.time()-t
        ch.append(c)
    print(f"[{tag}] 첫청크 {first:.2f}s"); return np.concatenate(ch)
run("워밍업.","warmup")                            # CUDA 그래프 캡처(1회성 ~5s, 버림)
y=run("안녕하세요. 잘 지냈어요? 오랜만이에요.","warm")
sf.write("/workspace/tts_test.wav", y, tts.sr); print("OK", y.shape, tts.sr)
PY
```
✅ **체크:** `chunk_size=25 temperature=0.5` + `[warm] 첫청크 ~0.5~0.7s` + `tts_test.wav`(화자 음색).
> 워밍업 첫 실행만 ~5s(그래프 캡처). 서버는 한 번 뜨면 계속 warm.

---

## 8. 전체 파이프라인 관통 (ASR→LLM→TTS, 마이크 없이)
참조 클립을 "들어온 말"로 넣어 한 턴 돌린다.
```bash
# ① 감정이 문맥 따라 바뀌나 (LLM 단독)
python - <<'PY'
from callone.serve.llama_llm import LlamaPersonaLLM
llm = LlamaPersonaLLM("sis", use_rag=False); llm.set_emotion_labeling(True)
llm.set_context(persona="너는 사용자의 친한 여동생이야. 항상 반말로 짧게.", situation="")
for msg in ["나 오늘 승진했어!","할머니가 편찮으셔서 걱정돼.","왜 전화 안 받았어 진짜."]:
    print(msg,"→",llm.chat(msg)[:90])
PY
# → 각 응답 앞 [emotion:happy/sad/...] 가 문맥 맞게 붙으면 OK

# ② 전체 한 턴 (ASR→LLM→TTS→음성)
python - <<'PY'
import numpy as np, soundfile as sf
from callone.serve.orchestrator import Orchestrator
orch = Orchestrator("sis")
orch.set_context(persona="너는 사용자의 친한 여동생이야. 항상 반말로 짧고 자연스럽게.",
                 situation="오랜만에 오빠한테 전화가 왔다.")
audio, sr = sf.read("data/speakers/sis/ref_24k.wav")
if getattr(audio,"ndim",1)>1: audio=audio.mean(1)
orch.handle_utterance(audio.astype("float32"), sr)            # 워밍업
turn = orch.handle_utterance(audio.astype("float32"), sr)     # 측정
print("내 말(ASR):", turn.user_text)
print("클론 응답 :", turn.reply_text)
print("첫음성 지연: %.0f ms" % turn.first_audio_latency_ms)
sf.write("/workspace/call_reply.wav", np.concatenate(turn.audio_chunks), orch.tts.sr)
PY
```
✅ **체크:** ①감정 태그 문맥별 변화 + ②`call_reply.wav`(화자 음색·문맥 톤).
> 지연 주의: 위 측정은 **11초 클립**을 ASR 하느라 부풀려짐(~2.3s). 실제 통화는 짧게 말하니 **첫음성 ~1.2~1.5s**.
> 감정→톤: TTS 감정 켜지면(serve.yaml `tts.emotion:true`) orchestrator 가 LLM 감정라벨링 자동 on → 톤 동적 변화.

### 8-1. 멀티턴 맥락 누적 확인 (통화가 앞 대화를 기억하나)
orchestrator(또는 studio 의 화자별 캐시)는 한 세션 내내 `history` 를 쌓고, LLM 에 **최근
`llm.max_history` 개 메시지(기본 24=12턴)** 를 넘긴다 → 앞 맥락 기억.
```bash
python - <<'PY'
from callone.serve.llama_llm import LlamaPersonaLLM
llm = LlamaPersonaLLM("sis", use_rag=False)
llm.set_context(persona="너는 사용자의 친한 여동생이야. 항상 반말로 짧게.", situation="")
hist=[]
def turn(u):
    r="".join(llm.chat_stream(u, hist)); hist.append({"role":"user","content":u}); hist.append({"role":"assistant","content":r}); print(u,"→",r[:70])
turn("나 내일 부산 내려가서 3일 있을 거야.")
turn("내가 어디 간다고 했지?")          # ← '부산' 을 기억하면 맥락 누적 OK
PY
```
✅ 둘째 답에 **부산/3일** 류가 나오면 누적 맥락 작동. (창 더 넓히려면 `serve.yaml llm.max_history`↑ — RAG off 면 prefix 캐시로 길어도 저렴.)

---

## 9. 브라우저 마이크 실시간 통화 (studio)
RunPod 포트 설정/재시작 없이 **gradio 공개 링크**로 바로:
```bash
GRADIO_SHARE=1 python -m studio                   # 출력의 https://....gradio.live 링크 열기
```
→ 브라우저에서 그 링크 → **통화 패널**: `상대 화자 ID = sis`, 페르소나·상황 입력 → 마이크로 한마디 → 멈춤 → 📞 보내기.
✅ 화자 음색 + 문맥 톤 응답. 상태에 `첫음성 …ms` 표시.
> 포트로 직접 쓰려면(대안): Pod에 50000 노출 후 `PORT=50000 python -m studio` → RunPod Connect 의 50000 주소.
> 풀 실시간(WebSocket·barge-in)은 `callone-serve`(:8000) + `ui`(:5173).

> ⚠️ **체감 지연:** studio 턴제는 **응답 합성이 다 끝나야** 음성이 나온다(첫음성 ms 와 별개).
> + **gradio.live 터널**이 왕복 지연을 더한다(~2초). 빠르게 하려면 ①**직접 포트**(50000 노출, 터널 회피)
> ②`llm.max_new_tokens`↓(짧은 응답) ③진짜 저지연은 **WS 스트리밍**(`callone-serve`, 음성이 도착하는 대로 재생).

---

## 통화 품질 올리기 (선택)
- **문장 톤 튐 / 급박함:** `serve.yaml tts.synth_mode: full`(기본) = 응답을 통째 1회 합성 →
  운율 일관(톤 안 튐) + 구두점 자연 텀. `sentence_pause_ms`(기본 180) 로 문장 사이 무음 조절.
  (저지연이 더 급하면 `synth_mode: sentence` = 문장별 스트리밍, 단 톤 약간 튐.)
- **사람처럼 말하기:** LLM system 에 강한 스타일 지시 내장(구어체·추임새·비서말투 금지·감정 반응).
  더 생생하게는 `serve.yaml llm.temperature`↑(기본 0.7, 횡설수설하면 ↓). 페르소나로 캐릭터 고정.
- **다채로운 감정:** 13종 팔레트(happy/sad/angry/excited/surprised/tender/playful/worried/shy/tired/
  disappointed/proud/neutral). LLM 이 맥락 따라 `[emotion:..]` 선택 → TTS instruct 로 톤 변화.
  추가하려면 `tts_qwen.EMOTION_MAP`+`tts_server.yaml emotion_instruct`+`llama_llm` 프롬프트에 같이.
- **페르소나:** studio 통화칸 페르소나/상황을 구체적으로(예 "항상 반말, 오빠라고 불러"). base 모델 호칭/말투 흔들림 잡힘.
- **맥락 길이:** `serve.yaml llm.max_history`(기본 24=12턴) ↑ 로 긴 통화도 앞 맥락 유지.
- **학습본 음색(최고):** A100에서 만든 화자 LoRA→GGUF 병합본을 6번 `-m` 으로 교체(말투 내재화). 음색은 Piper 학습본이 제로샷보다 충실.
- **데이터/RAG:** `data/speakers/{spk}/` 에 profile.json + utterances 두면 기억·말투 진해짐.

## 막힐 때 (실측 함정 + 해결)
| 증상 | 위치 | 처리 |
|---|---|---|
| `hf_transfer` ModuleNotFound | 4번 | `pip install hf_transfer` (또는 `export HF_HUB_ENABLE_HF_TRANSFER=0`) |
| transformers 충돌 | 2번 | heavy venv 와 섞였는지. `.venv-serve` 단독(transformers 4.57.3) |
| llama 응답 content 빈데 reasoning_content만 참 | 6번 | thinking ON. `chat_template_kwargs.enable_thinking=false`(코드 처리) 또는 서버 `--reasoning-budget 0` |
| TTS `ref_text is required ... ICL mode` | 5/7번 | ref_text.txt 두기(large-v3 전사) — ICL 필수 |
| TTS 톤 끊김/흔들림 | 7번 | chunk_size=25·temperature=0.5(코드 기본). 더 매끄럽게는 cs=40(지연↑) |
| `sox not found` 경고 | 5번 | `apt-get install -y sox` (경고일 뿐, 무시 가능) |
| TTS가 Piper로 폴백 | 7번 | faster-qwen3-tts 미설치 / ref 없음 / CUDA |
| ASR 빈 전사 | 8/9번 | ct2/cuDNN 불일치 → 코드가 CPU int8 자동 폴백(로그 확인) |
| 통화 응답 단순/엉뚱 | 8/9번 | llama-server 안 떠서 PersonaLLM 폴백 → 6번 health 재확인 |
| gradio 링크 마이크 안 됨 | 9번 | gradio.live 는 HTTPS라 OK. 브라우저 마이크 권한 허용 확인 |

## Stop / 재시작
- 모델·venv·코드 전부 `/workspace` → **Stop 후에도 유지**. 재접속 시 `source .venv-serve/bin/activate` 만.
- llama-server 는 Stop 시 죽음 → 재시작 후 **6번 재기동**. `~/.bashrc` 의 env(HF_HOME/CALLONE_TIER/CALLONE_TTS_MODEL)는 새 셸에서 자동.
- 코드 변경은 `git pull`(로컬 수정 있으면 `git checkout <file>` 후). 며칠 안 쓰면 Terminate(디스크 0, 다음에 이 문서 1번부터 ~30분 복구).
```
