# 갤럭시북5(Lunar Lake) 버전 — CPU 최적화 (실측 확정)

갤럭시북5 Pro(Intel Core Ultra 7 258V, Arc 140V iGPU, NVIDIA 없음)에 맞춘 속도 최적화판입니다.

> **실측 결론 (2026-06):** Arc 140V iGPU(XPU)도 시도해봤으나, CosyVoice 같은 **자기회귀
> TTS**는 작은 iGPU + WSL2 다리에선 **오히려 더 느림**(CPU 8스레드 ~31초 vs iGPU 100초+ 미완료).
> → **CPU 8스레드가 이 노트북에서 최速.** run.sh 기본값을 그렇게 박아뒀습니다.
>
> 31초/3초오디오(RTF~10)가 0.5B 모델 + 이 CPU의 현실적 바닥입니다. 더 빠르게(문장당 수 초)는
> NVIDIA GPU 가 필요 → 같은 앱·같은 클론이 도는 `GPU_A100/` 버전을 쓰세요.

앱에는 XPU(iGPU) 코드 경로도 남아 있어 원하면 실험할 수 있습니다(맨 아래 **부록** 참고). 단 더 느립니다.
**검증된 순수 CPU 원본 백업**은 `_backup_cpu_original/`.

## 0. WSL2 준비 (처음이면)
Windows PowerShell(관리자)에서:
```powershell
wsl --install
```
재부팅 후 Ubuntu 계정을 만듭니다. 이후 시작 메뉴의 **Ubuntu** 가 리눅스 터미널입니다.
(주의: 반드시 일반 Ubuntu여야 합니다. 프롬프트가 `pjo12346@...:~$` 형태면 정상.)

## 1. 이 폴더를 WSL 홈으로 옮기기
압축을 `C:\Users\pjo12\Downloads\` 에 풀었다면, Ubuntu 터미널에서:
```bash
cp -r /mnt/c/Users/pjo12/Downloads/CPU_laptop_WSL2 ~/vc
cd ~/vc
```
(경로의 `pjo12`/`Downloads` 는 본인 것에 맞게 바꾸세요.)

## 2. 설치 (한 번, 수십 분)
```bash
bash setup.sh
```
- Miniconda → CosyVoice 클론 → 환경 → 의존성 → **9.75GB 모델 다운로드** → CPU dtype 보정 →
  앱 배치까지 전부 자동입니다.
- (선택) 모델 다운로드가 너무 느리면 HF 토큰을 먼저 주세요:
  `export HF_TOKEN='hf_...'` 후 `bash setup.sh`.

## 3. 실행
```bash
cd ~/CosyVoice
bash run.sh
```
콘솔에 `Running on local URL: http://0.0.0.0:50000` 이 뜨면 됩니다.

## 4. 접속
Windows 브라우저에서:
```
http://localhost:50000
```
(WSL2는 localhost가 Windows로 자동 연결됩니다.)

## 5. 사용
- ① 만들기: WAV 업로드 → 🔎 자동 전사(틀린 글자 수정) → 🧪 후보 생성 →
  가장 본인 같은 후보 선택 → 이름 적고 💾 저장.
- ② 사용하기: 저장한 목소리 선택 → 텍스트 입력 → ▶️ 합성 → 결과 ⤓ 다운로드.
- CPU라 ②의 합성은 문장당 ~30초(3초 오디오 기준, RTF~10). 이게 이 노트북의 현실적 속도입니다.
- (저장된 클론을 앱에서 보려면 `~/CosyVoice/voices/` 안에 `이름/ref_16k.wav`+`meta.json` 이 있어야 함.)

## 속도 스위치 (run.sh 안)
- `COSYVOICE_DEVICE=cpu` : **기본·최速**. (iGPU 시험하려면 `=xpu` 로 — 단 더 느림)
- `TTS_THREADS=8` : 스레드 수. 실측상 8(31초)이 4(35초)보다 살짝 빠름.
- `WHISPER_MODEL=large-v3-turbo` : 전사 모델(②합성만 쓰면 안 돎).

## 콘솔에서 확인
- `[device] 사용: cpu  (threads=8, fp16=False)` → 정상(최速 설정).
- 합성 끝나면 `rtf 9.x` 정도 + 문장당 ~30초(3초 오디오 기준).

## ⛑️ 앱이 꼬이면 원본으로 복귀
`setup.sh` 가 이미 검증된 CPU 휠(torch 2.6.0+cpu)을 깔아주므로 보통은 손댈 일 없습니다.
앱 파일만 꼬였다면 백업으로 되돌리세요:
```bash
cd ~/vc                         # 이 폴더(setup.sh 있는 곳)
cp _backup_cpu_original/app_studio.py ~/CosyVoice/app_studio.py
cp _backup_cpu_original/run.sh        ~/CosyVoice/run.sh
cd ~/CosyVoice && bash run.sh
```

## 왜 이렇게 설치하나 (이전 실패 방지 포인트)
- **AutoModel 로 로드** → CosyVoice3 코드 경로 자동 선택(외계어 원인 제거).
- **torch/torchaudio 2.6.0 CPU 고정** → torchcodec 경로 회피(파일 로딩 깨짐 방지). 실측상 CPU 가 최速.
- **스레드 8(P4+E4) + 경량 Whisper(turbo)** → CPU 경로 튜닝 (8→31초, 4→35초).
- **tensorrt/onnxruntime-gpu/deepspeed 제외** → 빌드 멈춤 방지, onnxruntime 사용.
- **setuptools<81 먼저** → openai-whisper 빌드 충돌 방지.
- **config.json torch_dtype=float32** → CPU 에서 bfloat16 dtype 불일치 방지.
> CosyVoice 소스 파일(llm.py, model.py 등)은 **건드리지 않습니다.**

## 막히면
실행 중 새 에러가 뜨면 그 **콘솔 메시지 한 줄**을 그대로 보내주세요.
또는 더 확실하고 30배 빠른 GPU(A100) 버전으로 갈아타셔도 됩니다 — 같은 앱·같은 클론이 그대로 돌아갑니다.

---

## 부록 — (선택) Arc iGPU(XPU) 실험: 권장 안 함, 더 느림
실측상 iGPU 가 CPU 보다 느렸지만, 그래도 직접 돌려보고 싶다면:
```bash
conda activate cosyvoice
# 1) torch 를 +xpu 빌드로 교체 (버전 2.6.0 고정 — CosyVoice 의 torch==2.6.0 핀과 충돌 회피)
pip install --force-reinstall torch==2.6.0+xpu torchaudio==2.6.0+xpu \
  --index-url https://download.pytorch.org/whl/xpu
pip install "markupsafe>=2.1,<3" "fsspec[http]>=2022.5.0,<2025.0"   # 충돌 핀 되돌리기

# 2) Intel GPU 드라이버(Level-Zero) 설치 — 우분투 코드네임에 repo 없으면 noble 로 대체
. /etc/os-release
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu.list
sudo apt update
sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-ocloc libze-dev

# 3) 확인
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"   # 2.6.0+xpu True 면 OK

# 4) run.sh 에서 COSYVOICE_DEVICE=cpu → xpu 로 바꿔 실행
```
주의: XPU 에선 **fp32 만** 쓰세요(`TTS_FP16=1` 금지). CosyVoice 가 fp16=True 면 `torch.cuda.amp.autocast`(CUDA 전용)를 켜서 XPU 와 dtype 충돌이 납니다.
다시 CPU 로: `pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu` 후 run.sh 를 `=cpu` 로.
