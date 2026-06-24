# callone 새 인스턴스 원샷 세팅 (A100/4090)

## TL;DR — 이 두 줄로 끝

```bash
git clone <리포URL> callone && cd callone
bash install.sh
```

llama(EXAONE) → CosyVoice3(TTS) → Ditto(avatar) → UI dev server 순서로 자동 설치·기동.  
재기동만 필요하면: `bash scripts/run_all.sh`  
단계별 수동 제어: `SKIP_COSYVOICE=1 SKIP_AVATAR=1 bash install.sh`

---

인스턴스를 지웠다 다시 만들었을 때 **이 순서대로만** 하면 음성+영상 통화까지 바로 된다.
모든 함정(드라이버·의존성·모델경로)은 스크립트에 자가치유로 박혀 있다. 게이트 모델 없음(HF 토큰 불필요).

> 이 문서 = **제로샷 방식**(5~10초 음성+사진 업로드 → 즉시 복제, 학습 없음). GPU는 **A100/H100·RTX 4090/3090 둘 다** 지원
> (Ditto TensorRT 차이는 맨 아래 "알아둘 것" 참고). 긴 녹음(1시간+)으로 화자를 **풀 파인튜닝**하는 고급 경로는 [docs/README.md](README.md) 참고.

## 구성 (4개 서비스, 각자 독립 venv/프로세스)
| 서비스 | 포트 | venv/env | 역할 |
|---|---|---|---|
| llama-server | 8090 | (바이너리) | LLM **EXAONE-4.0-32B**(A100/H100) / **3.5-7.8B**(4090/3090) |
| cosyvoice-server | 8092 | conda `cosyvoice` | TTS **CosyVoice3-0.5B**(제로샷 음색복제, 안정) |
| avatar-server | 8091 | `.venv-avatar` | 토킹헤드 **Ditto**(TensorRT, RTF<1) |
| callone-serve | 8000 | `.venv-serve` | 오케스트레이터(ASR+LLM+TTS+아바타 조립, WS) |

우선순위: ①목소리 ②한국어 ③얼굴 ④속도.

---

## 0) 리포 클론
```bash
cd ~   # /workspace 있으면 거기(자동감지)
git clone <리포URL> callone && cd callone
```

## 1) 서빙 스택 (llama EXAONE + .venv-serve + TTS 폴백)  ~10–20분
```bash
PORT=8090 bash scripts/bootstrap_gpu.sh
```
- `.venv-serve` 생성 + `pip install -e .`
- llama-server 바이너리(프리빌트 다운, 드라이버 안 맞으면 CUDA12.4 자동 재빌드)
- GPU VRAM 자동 감지 후 **EXAONE GGUF Q6_K** 다운로드: 40GB 이상은 4.0-32B, 24GB는 3.5-7.8B
  환경변수 `LLM_REPO`/`LLM_GLOB`/`LLM_DIR`로 명시적 재정의 가능.
- llama 기동 + GPU 검증. 끝에 `LLM 템플릿: --jinja (GGUF=EXAONE...)` 확인.
- 검증: `curl -s 127.0.0.1:8090/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"안녕"}],"max_tokens":32}'`

## 2) CosyVoice3 음색 백엔드  ~수십분(9.75GB 모델)
```bash
bash scripts/setup_cosyvoice_gpu.sh
```
- conda env `cosyvoice`(py3.10, torch2.6 cu124) + CosyVoice 클론 + 모델
- 함정 자동처리: openai-whisper(=`import whisper`) 설치 → **그 뒤** torch 2.6 재고정(whisper가 torch 끌어내림 방지)

## 3) 아바타(Ditto, TensorRT)  ~수십분(체크포인트 6.5GB)
```bash
bash scripts/setup_avatar_gpu.sh
```
- `.venv-avatar` + Ditto repo + 체크포인트(pytorch/onnx/**trt_Ampere_Plus**/cfg)
- 함정 자동처리:
  - 시스템 libs(GL/X11/오디오: `libGLESv2.so.2` 등) `sudo apt`
  - Ditto import 스캔 → 누락 파이썬 모듈 일괄 설치(filetype·cython·onnxruntime·cuda-python 등)
  - **TensorRT 8.6.1 고정**(최신 TRT는 드라이버535에 `CUDA error 35` + 프리빌트 엔진 비호환) + **cuDNN8**(TRT8.6용, 토치 cuDNN9와 soname 달라 공존)
  - **DITTO_* env**(`~/.bashrc`): TRT(Ampere+trt_online) 우선, 없으면 PyTorch
  - ⚠️ **Ampere(A100/3090 등)면 얼굴이 움직임(TRT). 4090(Ada)은 PyTorch가 0프레임(미동작)** → 아래 "알아둘 것" 참고,
    4090에선 `AVATAR_BACKEND=static`(음성+정지사진)으로

## 4) 전체 기동
```bash
source ~/.bashrc                 # DITTO_* env 로드(중요 — avatar-server 가 필요)
bash scripts/run_all.sh          # llama·cosy·avatar(ditto)·serve 한 방 + health 4줄
```
health 4개 다 `ok`(+`avatar: backend:ditto`, `cosy: ok`) 확인. cosy 모델로드 ~30s.

## 5) UI (별 터미널, 인스턴스 안에서)
```bash
cd ~/callone/ui
# Node 없으면 설치. root 컨테이너(RunPod)면 sudo 없음 → 그냥 실행, Elice 등 sudo 있으면 sudo.
command -v npm >/dev/null || { SUDO=""; command -v sudo >/dev/null && SUDO="sudo -E"; \
  curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO bash - && ${SUDO:+sudo }apt-get install -y nodejs; }
npm install && npm run dev       # 평범하게(VITE_BASE 붙이지 말 것). Local: http://localhost:5173/ 떠야 정상.
```
⚠️ **`VITE_BASE=/proxy/5173/` 붙이지 마라** — SSH 터널로 접속하면 localhost 직결이라 그 prefix 가 라우팅을 깨서 빈 화면 나온다. 그냥 `npm run dev`.

## 6) 브라우저 (내 노트북) — SSH 터널로 localhost 매핑
마이크는 HTTPS/localhost 보안컨텍스트 필요 → **내 PC에서 SSH 터널**(`-L 5173:localhost:5173`):

**RunPod**: Connect→Direct TCP 의 host/port 로
```powershell
ssh root@<IP> -p <포트> -i ~/.ssh/id_ed25519 -L 5173:localhost:5173
```
**Elice**: 받은 `.pem` 키가 있는 폴더에서 터미널 열고(또는 -i 에 풀경로), Connect 의 ssh 명령에 `-L` 만 추가
```powershell
ssh -i elice-cloud-ondemand-XXXX.pem elicer@central-01.tcp.tunnel.elice.io -p <포트> -L 5173:localhost:5173
```
→ 그 창은 그대로 둠(터널). 브라우저: **`http://localhost:5173/call/me`** (← `/proxy/5173/` 같은 거 붙이지 말 것)
→ 음성·사진 업로드 → 통화시작 → 말 → **응답 전송**.

---

## 재시작(인스턴스 안 지우고 멈췄다 켬)
모델·venv 다 남아 있으니:
```bash
cd ~/callone && source ~/.bashrc
pkill -9 -f callone; sleep 2
bash scripts/run_all.sh          # 다 떠 있으면 health만, 죽은 것만 기동
# 별 터미널: cd ui && npm run dev
```
종료: `bash scripts/run_all.sh stop`

## 알아둘 것
- **첫 턴 영상은 콜드(~30s, TRT 첫 추론)** — 이후 턴은 RTF<1로 빠름. 정상.
- **A/V 동기**: 음성이 영상 생성을 기다렸다 같이 재생(입싱크 우선). 음성전용(사진 미업로드)이면 지연 없음.
- **프라이버시**: 음성·사진·대화는 프론트(브라우저) 소유. 서버는 인메모리(/dev/shm)만, 통화 끝나면 폐기. 디스크/로그에 본문 0.
- **움직이는 얼굴(Ditto) — GPU별 경로(`setup_avatar_gpu.sh` 자동 처리)**:
  - **A100/H100/3090/3090Ti/A5000/A6000 등 Ampere(cc 8.0/8.6)** → 프리빌트 TRT 엔진(`ditto_trt_Ampere_Plus`)으로
    **얼굴 움직임**(립싱크·표정, RTF<1). ✅ 검증된 경로(빌드 없음, 바로 됨).
  - **RTX 4090/L40 등 Ada(cc 8.9)** → 프리빌트 비호환이라, setup 이 **이 GPU용 TRT 엔진을 ONNX에서 자동 빌드**
    (`cvt_onnx_to_trt` → `ditto_trt_custom`, ~수십분 1회). 성공하면 **4090도 얼굴 움직임**. ⚠️ TRT↔드라이버 버전이
    안 맞아 빌드가 깨질 수 있음(미실측) — 그 경우 PyTorch는 0프레임이라 **`AVATAR_BACKEND=static`(음성+정지사진)** 으로:
    ```bash
    echo 'export AVATAR_BACKEND=static' >> ~/.bashrc && source ~/.bashrc && bash scripts/run_all.sh
    ```
    가장 확실한 움직이는 얼굴은 여전히 **Ampere GPU**(빌드 불필요).
  - setup 끝의 `Ditto 백엔드:` 줄로 확인 — `TensorRT(Ampere 프리빌트)` / `TensorRT(custom 빌드)` = 움직임 O, `PyTorch(폴백)` = static 권장.
  - **음성·한국어·목소리복제·정지사진은 모든 GPU 동일 동작**(LLM EXAONE·TTS CosyVoice3·ASR).
- **로그**: `~/serve.log` `~/avatar.log` `~/cosyvoice.log` `~/llama.log` (본문 없음 — 길이·지연만)
- **단계별 시간**: `grep -E "ASR 완료|LLM 완료|아바타" ~/serve.log`
