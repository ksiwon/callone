#!/usr/bin/env bash
# llama.cpp CUDA 빌드 (클라우드 GPU / Linux). llama-server 바이너리 생성.
# RunPod(RTX 3090/4090) · Elice(A100/H100) 모두 동작. LLM 은 이 별도 바이너리로
# HTTP 서빙 → 서빙 venv 와 파이썬 의존성 0(충돌 0).
#
# 빌드 위치(영속 폴더) 자동 선택: RunPod=/workspace(쓰기가능) / 그 외(Elice 등)=$HOME.
#   CALLONE_HOME 또는 DEST 로 강제 가능:
#   사용:  bash scripts/build_llama_cuda.sh
#          DEST=$HOME/llama.cpp bash scripts/build_llama_cuda.sh
#          CALLONE_HOME=/data bash scripts/build_llama_cuda.sh   # → /data/llama.cpp
set -euo pipefail

if [ -z "${DEST:-}" ]; then
  BASE="${CALLONE_HOME:-}"
  if [ -z "$BASE" ]; then
    # /workspace 가 쓰기 가능하면(RunPod) 거기, 아니면(Elice 등 non-root) 홈.
    if [ -d /workspace ] && [ -w /workspace ]; then BASE=/workspace; else BASE="$HOME"; fi
  fi
  DEST="$BASE/llama.cpp"
fi

echo "=== llama.cpp CUDA 빌드 → $DEST ==="
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$DEST"
fi
cd "$DEST"
git pull --ff-only || true

# CUDA 빌드(cmake). 빌드 도구 없으면 설치(root/sudo 환경에서만 — Elice non-root 면 이미 깔려있음).
if ! command -v cmake >/dev/null 2>&1; then
  (sudo apt-get update -y && sudo apt-get install -y cmake build-essential) 2>/dev/null \
    || (apt-get update -y && apt-get install -y cmake build-essential) 2>/dev/null \
    || echo "⚠️ cmake 자동설치 실패 — 'conda install -y cmake' 또는 패키지매니저로 직접 설치 후 재실행"
fi

# CUDA 아키텍처를 현재 GPU 것만 빌드 → 안 쓸 옛 아키 중복 컴파일 제거로 빌드 대폭 단축.
#   (default 는 여러 아키 리스트라 nvcc 가 커널마다 N배 컴파일 = 40~60분).
#   ⚠️ "native" 는 CMake>=3.24 만 지원 → 구버전(예 3.22)에선 빈 값→nvcc fatal. 그래서 nvidia-smi 로
#      **compute capability 숫자를 직접 감지**(A100 8.0→80, 4090 8.9→89, H100 9.0→90)해 넘긴다.
#   강제: CUDA_ARCH=80 bash ...   (자동감지 실패 시 안전 리스트로 폴백)
CUDA_ARCH="${CUDA_ARCH:-auto}"
if [ "$CUDA_ARCH" = "auto" ] || [ "$CUDA_ARCH" = "native" ]; then
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
  if printf '%s' "$cap" | grep -qE '^[0-9]+$'; then
    CUDA_ARCH="$cap"
  else
    echo "⚠️ GPU arch 자동감지 실패(nvidia-smi) → 안전 리스트(80;86;89;90)로 빌드"
    CUDA_ARCH="80;86;89;90"
  fi
fi
echo "=== CUDA 아키텍처: $CUDA_ARCH (jobs=$(nproc)) ==="
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build --config Release -j"$(nproc)" --target llama-server

BIN="$DEST/build/bin/llama-server"
if [ -x "$BIN" ]; then
  echo "✅ 빌드 완료: $BIN"
  echo "   기동 예:  $BIN -m <gguf> --host 127.0.0.1 --port 8080 -c 8192 -n 512 --n-gpu-layers 99"
  echo "   (속도옵션은 잘 뜬 뒤 --flash-attn on 추가. bare --flash-attn 은 최신 빌드서 에러)"
else
  echo "⚠️ llama-server 바이너리를 못 찾음 → build/bin 확인"; exit 1
fi
