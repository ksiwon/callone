#!/usr/bin/env bash
# GPU 버전 실행 — CosyVoice 폴더에서 실행
set -e
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate cosyvoice
cd ~/CosyVoice
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PORT=50000
python app_studio.py
