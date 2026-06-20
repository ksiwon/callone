#!/usr/bin/env bash
# =============================================================================
# [CPU 버전] 처음부터 설치 — 갤럭시북5(258V/Arc 140V) + Windows + WSL2(우분투)
# 실행:  bash setup.sh
# (Miniconda → 클론 → 환경 → 의존성 → 9.75GB 모델 → 앱 배치까지 자동, 수십 분)
# 실측: 이 노트북에선 CPU 8스레드가 최速. iGPU(XPU)는 자기회귀 TTS 에 더 느려서 안 씀.
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> [1/8] 시스템 패키지"
sudo apt-get update
sudo apt-get install -y git git-lfs sox libsox-dev ffmpeg wget build-essential
git lfs install || true

echo "==> [2/8] Miniconda (없으면 설치)"
if ! command -v conda >/dev/null 2>&1 && [ ! -d "$HOME/miniconda3" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
source "$(conda info --base 2>/dev/null || echo $HOME/miniconda3)/etc/profile.d/conda.sh"
# 아나콘다 약관 자동 동의 (비대화형 설치 차단 방지)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "==> [3/8] CosyVoice 클론"
cd ~
[ -d CosyVoice ] || git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

echo "==> [4/8] 환경 생성"
conda create -n cosyvoice -y python=3.10 || true
conda activate cosyvoice
pip install --upgrade pip
pip install "setuptools<81" wheel

echo "==> [5/8] 의존성 (빌드지옥/GPU 패키지 제외)"
grep -vE 'tensorrt|onnxruntime-gpu|deepspeed|openai-whisper' requirements.txt > requirements_cpu.txt
pip install -r requirements_cpu.txt
pip install onnxruntime
pip install openai-whisper==20231117 --no-build-isolation || pip install openai-whisper --no-build-isolation || true

echo "==> [6/8] torch/torchaudio CPU 고정 (실측상 이 노트북에선 CPU 가 최速)"
# 실측: 갤럭시북5(258V/Arc 140V)에서 CosyVoice 자기회귀 TTS 는 CPU 8스레드가 가장 빠름
# (iGPU/XPU 는 오히려 더 느림). 그래서 검증된 CPU 휠 2.6.0 으로 고정 (torchcodec 경로 회피).
# (XPU 를 실험하고 싶으면 README 의 "선택: Arc iGPU(XPU) 실험" 부록 참고 — 더 느림 주의.)
pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install gradio pydub faster-whisper noisereduce soundfile huggingface_hub

echo "==> [7/8] 모델 다운로드 (Fun-CosyVoice3-0.5B-2512)"
mkdir -p pretrained_models
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
print("모델 다운로드 완료")
PY

echo "==> [8/8] CPU용 dtype 보정 (Qwen2 LLM 을 float32 로) + 앱 배치"
python - <<'PY'
import json, os
cfg = "pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN/config.json"
if os.path.exists(cfg):
    try:
        d = json.load(open(cfg, encoding="utf-8"))
        d["torch_dtype"] = "float32"
        json.dump(d, open(cfg, "w", encoding="utf-8"), indent=2)
        print("config.json torch_dtype=float32")
    except Exception as e:
        print("[warn] config.json 수정 실패:", e)
PY
cp -f "$SCRIPT_DIR/app_studio.py" ./app_studio.py
cp -f "$SCRIPT_DIR/run.sh" ./run.sh 2>/dev/null || true

echo ""
echo "============================================================"
echo " 설치 완료!(CPU 최速 설정)  실행:  cd ~/CosyVoice && bash run.sh"
echo " 그다음 Windows 브라우저에서  http://localhost:50000"
echo "============================================================"
