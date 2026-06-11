#!/usr/bin/env bash
# 화자 A 음색 Piper 학습용 GPU 환경 셋업 (Ubuntu + CUDA GPU, 예: Elice).
# Piper 학습 코드는 Python 3.10 + numpy<2 + cython 빌드라 까다롭다 → 이 스크립트로 한 번에.
#
# 사용:
#   bash scripts/setup_piper_gpu.sh
#   그다음: source ~/piper-train/bin/activate  하고 train_piper.md 의 preprocess→train→export
set -e

echo "== 0) 시스템 의존성 (espeak-ng = 한국어 음소화) =="
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev \
    espeak-ng libespeak-ng-dev build-essential git wget

echo "== 1) venv (Python 3.10) =="
python3.10 -m venv ~/piper-train
source ~/piper-train/bin/activate
# ⚠️ pip 24.1+ 는 piper 가 쓰는 pytorch-lightning 1.7.x 의 옛 메타데이터를 거부한다 → pip<24.1 고정.
pip install "pip<24.1" wheel

echo "== 2) Piper 학습 코드 =="
[ -d ~/piper ] || git clone https://github.com/rhasspy/piper ~/piper
cd ~/piper/src/python
pip install -e .
pip install -r requirements.txt
pip install -r requirements_train.txt
pip install "numpy<2" "torchmetrics==0.10.3" "huggingface_hub[cli]"   # PL1.7↔torchmetrics0.10 핀(_compare_version)

echo "== 3) monotonic_align (cython 빌드) =="
bash build_monotonic_align.sh

echo "== 4) lessac-medium 베이스 체크포인트(파인튜닝 시작점) =="
mkdir -p ~/ckpt
hf download rhasspy/piper-checkpoints --repo-type dataset \
    --include "en/en_US/lessac/medium/*.ckpt" \
    --local-dir ~/ckpt || \
  echo "⚠ lessac ckpt 자동 다운 실패 — https://huggingface.co/datasets/rhasspy/piper-checkpoints 에서 en_US-lessac-medium .ckpt 수동 다운로드해 ~/ckpt 에 둬라"

echo ""
echo "== 완료 =="
echo "  venv:  source ~/piper-train/bin/activate"
echo "  ckpt:  $(find ~/ckpt -name '*.ckpt' 2>/dev/null | head -1)"
echo "  다음:  데이터 업로드(piper_ds/A) → preprocess → train → export (train_piper.md)"
