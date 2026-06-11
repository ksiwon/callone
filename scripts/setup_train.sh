#!/usr/bin/env bash
# 학습 전용 venv (LLM LoRA). 데이터 파이프라인 venv(.venv)와 분리 —
# pyannote 가 묶어둔 transformers 4.44 / torch 2.5 와 충돌하지 않게.
#
# 사용:  bash scripts/setup_train.sh        # 환경 설치
#        source .venv-train/bin/activate    # 그 뒤 활성화
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "=== [1/3] 학습 venv ==="
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip >/dev/null

echo "=== [2/3] torch + 학습 deps (최신 transformers) ==="
pip install --index-url "$CUDA_INDEX" torch
pip install -e .                                  # callone 코어(pydantic/yaml/librosa 등)
pip install "transformers>=4.57" peft trl bitsandbytes accelerate datasets sentencepiece

echo "=== [3/3] 확인 ==="
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
echo "준비 끝. 다음:"
echo "  source .venv-train/bin/activate"
echo "  callone-llm-train --config llm_phone --speakers A B"
