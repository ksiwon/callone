#!/usr/bin/env bash
# =============================================================================
# [GPU 버전] 처음부터 설치 — A100 등 CUDA 인스턴스 (Elice 등)
# 실행:  bash setup.sh
# (클론 → 환경 → 의존성 → 9.75GB 모델 다운로드 → 앱 배치까지 자동, 수십 분 소요)
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> [1/7] 시스템 패키지"
sudo apt-get update && sudo apt-get install -y git git-lfs sox libsox-dev ffmpeg wget build-essential || \
  echo "[info] apt 생략(이미 있거나 권한 없음) — 계속 진행"
git lfs install || true

echo "==> [2/7] conda 확인/설치 + CosyVoice 클론"
# conda 가 없으면 Miniconda 를 홈에 설치 (sudo 불필요)
if ! command -v conda >/dev/null 2>&1 && [ ! -d "$HOME/miniconda3" ]; then
  echo "  conda 없음 → Miniconda 설치"
  ( wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    || curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh )
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
cd ~
[ -d CosyVoice ] || git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

echo "==> [3/7] 환경 생성"
conda create -n cosyvoice -y python=3.10 || true
conda activate cosyvoice
pip install --upgrade pip
pip install "setuptools<81" wheel

echo "==> [4/7] 의존성 (빌드지옥 패키지 제외)"
grep -vE 'tensorrt|onnxruntime-gpu|deepspeed|openai-whisper' requirements.txt > requirements_clean.txt
pip install -r requirements_clean.txt
pip install onnxruntime    # 작은 토크나이저 onnx는 CPU로 충분(CUDA 버전지옥 회피)
pip install openai-whisper==20231117 --no-build-isolation || pip install openai-whisper --no-build-isolation || true

echo "==> [5/7] CUDA용 torch/torchaudio 고정 (torchcodec 경로 회피, cu124)"
pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install gradio pydub faster-whisper noisereduce soundfile huggingface_hub

echo "==> [6/7] 모델 다운로드 (Fun-CosyVoice3-0.5B-2512)"
mkdir -p pretrained_models
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
print("모델 다운로드 완료")
PY

echo "==> [7/7] 앱 배치"
cp -f "$SCRIPT_DIR/app_studio.py" ./app_studio.py
cp -f "$SCRIPT_DIR/run.sh" ./run.sh 2>/dev/null || true

echo ""
echo "============================================================"
echo " GPU 설치 완료!  실행:  cd ~/CosyVoice && bash run.sh"
echo " 외부 접속이 막히면 app_studio.py 의 share=False 를 True 로."
echo "============================================================"
