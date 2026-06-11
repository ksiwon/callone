# Qwen3.5-4B + LoRA 를 노트북(Arc 140V)에서 실시간 — llama.cpp 런북

> 왜 llama.cpp? **OpenVINO 는 qwen3_5 아키텍처(Gated Delta Net + MoE + MTP)를 아직 변환 못 함**
> (optimum-intel 이슈 #1628). 반면 **llama.cpp 는 Qwen3.5 지원**(Unsloth GGUF 존재) + LoRA 적용 가능.
> llama-server 를 **별도 프로세스**로 띄우고 우리 서빙은 HTTP 로만 호출 → torch/OV segfault 회피.

대상: 삼성 갤럭시북5 Pro / Intel Arc 140V iGPU(16GB) / Windows 11.

---

## 0) llama.cpp 받기 (택1)

**A. 프리빌트(가장 빠름)** — GitHub Releases 의 Windows Vulkan 빌드
`llama-bXXXX-bin-win-vulkan-x64.zip` 다운로드 → `C:\tools\llama.cpp\` 에 압축 해제.
(GGUF 변환/양자화 파이썬 스크립트가 필요하면 아래 B 의 소스도 같이 clone)

**B. 소스(변환 스크립트 포함)**
```powershell
git clone https://github.com/ggml-org/llama.cpp C:\tools\llama.cpp
pip install -r C:\tools\llama.cpp\requirements.txt   # gguf, convert_*.py 의존성
```

---

## 1) 모델 준비 (택1)

### ★ 빠른 길 — base GGUF + 런타임 LoRA (9GB base 다운로드 회피, 권장)

1. Qwen3.5-4B **base GGUF**(Q4) 받기(~2.5GB):
```powershell
huggingface-cli download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir models_gguf
```
2. 우리 **LoRA 어댑터 → GGUF** 변환(어댑터만, 작음. base config 만 추가로 받음):
```powershell
python C:\tools\llama.cpp\convert_lora_to_gguf.py `
    models\llm_phone\A\checkpoint-90 `
    --base Qwen/Qwen3.5-4B `
    --outfile models_gguf\mom-A-lora-f16.gguf
```
> `checkpoint-90` 위치는 실제 어댑터 폴더로(`adapter_config.json` 있는 곳). B 화자는 `llm_phone\B`.

### 정석 길 — 병합본 단일 GGUF (품질 최고, base 9GB 필요)
```powershell
python scripts\make_gguf.py --speaker A --llama-cpp C:\tools\llama.cpp --quant Q4_K_M
# → models_gguf\qwen3.5-4b-A-Q4_K_M.gguf (LoRA 이미 병합됨, --lora 불필요)
```

---

## 2) llama-server 띄우기 (별도 터미널, 계속 켜둠)

**Arc GPU (Vulkan) — TDR 크래시 회피로 coopmat 끄기:**
```powershell
$env:GGML_VK_DISABLE_COOPMAT = "1"
# 빠른 길(base + 런타임 LoRA):
C:\tools\llama.cpp\llama-server.exe `
    -m models_gguf\Qwen3.5-4B-Q4_K_M.gguf `
    --lora models_gguf\mom-A-lora-f16.gguf `
    -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
# 정석 길(병합본): --lora 빼고 -m 을 병합 GGUF 로
```
- `-ngl 99` = 전 레이어 GPU 오프로드(Arc). GPU 불안정하면 `-ngl 0`(CPU) 로.
- 첫 실행 후 `http://127.0.0.1:8080/health` 가 `{"status":"ok"}` 면 준비됨.

**CPU 폴백(드라이버 문제 시, 안정적이나 4B 라 약간 느림):**
```powershell
C:\tools\llama.cpp\llama-server.exe -m models_gguf\Qwen3.5-4B-Q4_K_M.gguf `
    --lora models_gguf\mom-A-lora-f16.gguf -c 4096 --host 127.0.0.1 --port 8080
```

> Arc 가 더 빠른 SYCL 빌드를 원하면 Intel IPEX-LLM 의 llama.cpp 포터블 zip 사용 가능
> (oneAPI 내장). 우리 서빙 코드는 **포트만 같으면(8080) 그대로 동작**.

---

## 3) callone 서빙 붙이기

`configs/serve.yaml` 의 `llm.backend: auto`(기본) 면 8080 의 llama-server 를 자동 감지해 사용.
서버가 안 떠 있으면 OpenVINO base → PersonaLLM 으로 자동 폴백.

```powershell
$env:CALLONE_SPEAKER = "A"        # 화자 A 클론
callone-serve                      # 또는 python -m callone.serve.app
```

페르소나 카드(화자 A 경상도 말투) + RAG(화자 A 실제 발화 17,296개) 는 우리 서빙이 system 프롬프트로
주입한다 → **LoRA(내재화) + RAG(사실 기억) + 페르소나(말투) 3중 반영**.

---

## 빠른 점검(서버만 단독 테스트)
```powershell
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d `
 '{\"messages\":[{\"role\":\"user\",\"content\":\"화자 A 나 왔어\"}],\"max_tokens\":64}'
```
경상도 말투("마", "~가/~노")가 나오면 LoRA 적용 성공.
