#!/usr/bin/env bash
# llama.cpp CUDA 빌드 (RunPod RTX 4090 / Linux). llama-server 바이너리 생성.
# LLM 은 이 별도 바이너리로 HTTP 서빙 → 서빙 venv 와 파이썬 의존성 0(충돌 0).
#
# 사용:  bash scripts/build_llama_cuda.sh           # /workspace/llama.cpp 에 빌드
#        DEST=/root/llama.cpp bash scripts/build_llama_cuda.sh
set -euo pipefail

DEST="${DEST:-/workspace/llama.cpp}"

echo "=== llama.cpp CUDA 빌드 → $DEST ==="
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$DEST"
fi
cd "$DEST"
git pull --ff-only || true

# CUDA 빌드(cmake). 빌드 도구 없으면 설치.
command -v cmake >/dev/null 2>&1 || (apt-get update -y && apt-get install -y cmake build-essential) || true

cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j"$(nproc)" --target llama-server

BIN="$DEST/build/bin/llama-server"
if [ -x "$BIN" ]; then
  echo "✅ 빌드 완료: $BIN"
  echo "   기동 예:  $BIN -m <gguf> --host 0.0.0.0 --port 8080 -c 8192 -n 512 --n-gpu-layers 99 --flash-attn"
else
  echo "⚠️ llama-server 바이너리를 못 찾음 → build/bin 확인"; exit 1
fi
