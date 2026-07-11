#!/usr/bin/env bash
# Qwen3-TTS 서버(:8093) 셋업/기동 — cosyvoice_server 와 동일 패턴(별 venv, idempotent).
#
# 사용:
#   bash scripts/setup_qwen_tts_gpu.sh          # 설치만(.venv-qwentts + 모델 다운)
#   bash scripts/setup_qwen_tts_gpu.sh run      # (설치돼 있으면 스킵 후) 서버 기동
#   QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base bash scripts/setup_qwen_tts_gpu.sh  # ultra 티어
#
# 티어 기본값(callone.common.hardware): high=0.6B, ultra=1.7B — run_all.sh 가 넘겨준다.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d /workspace ] && [ -w /workspace ]; then export CALLONE_HOME="${CALLONE_HOME:-/workspace}"
else export CALLONE_HOME="${CALLONE_HOME:-$HOME}"; fi
export HF_HOME="${HF_HOME:-$CALLONE_HOME/hf_cache}"
MODEL="${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
PORT="${QWEN_TTS_PORT:-8093}"
VENV=.venv-qwentts

# ── 1. venv (있으면 스킵) ─────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo "[qwen-tts 1/3] venv 생성 + 패키지 설치..."
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install -q --upgrade pip
  # torch 는 이미지 CUDA 에 맞는 휠 자동(RunPod 기본 인덱스 OK). flash-attn 은 선택(실패 무해).
  pip install -q torch qwen-tts "fastapi" "uvicorn" soundfile numpy "huggingface_hub[cli]" hf_transfer
  pip install -q flash-attn --no-build-isolation 2>/dev/null \
    || echo "  (flash-attn 빌드 스킵 — 기본 attention 으로 동작, 서버가 자동 폴백)"
else
  echo "[qwen-tts 1/3] $VENV 있음 → 스킵"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# ── 2. 모델 프리다운로드 (있으면 스킵 — HF 캐시 기준) ──────────────────────
echo "[qwen-tts 2/3] 모델 확인: $MODEL"
HF_HUB_ENABLE_HF_TRANSFER=1 python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL")
print("  모델 준비 완료")
PY

# ── 3. 기동 (run 인자일 때만) ─────────────────────────────────────────────
if [ "${1:-}" = "run" ]; then
  echo "[qwen-tts 3/3] 서버 기동(:$PORT, model=$MODEL)..."
  exec env QWEN_TTS_MODEL="$MODEL" PORT="$PORT" python qwen_tts_server/app.py
else
  echo "[qwen-tts 3/3] 설치 완료. 기동: bash scripts/setup_qwen_tts_gpu.sh run  (또는 run_all.sh)"
fi
