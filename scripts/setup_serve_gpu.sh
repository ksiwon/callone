#!/usr/bin/env bash
# callone 서버(RTX 4090) 실시간 통화 — 서빙 전용 환경 한방 설치.
#
# ⚠️ setup_server.sh(=.[heavy], 데이터 파이프라인)와 다르다. 이건 "서빙만".
#    heavy 의 transformers<4.50 ↔ Qwen3-TTS 의 transformers==4.57.3 충돌을 피하려고
#    별도 venv(.venv-serve)에 깐다. LLM 은 llama-server(별 바이너리)라 여기 없음.
#
# 사용:  bash scripts/setup_serve_gpu.sh
#   CUDA 버전 다르면:  CUDA_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/setup_serve_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
VENV="${VENV:-.venv-serve}"

echo "=== [0] ffmpeg (브라우저 마이크 opus/webm 변환) ==="
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    (sudo apt-get update -y && sudo apt-get install -y ffmpeg) 2>/dev/null \
      || (apt-get update -y && apt-get install -y ffmpeg) 2>/dev/null || true
  fi
fi
command -v ffmpeg >/dev/null 2>&1 && echo "ffmpeg OK" || echo "⚠️ ffmpeg 없음 — wav 업로드만 동작"

echo "=== [1] 서빙 전용 venv ($VENV) ==="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip >/dev/null

echo "=== [2] torch+torchaudio (CUDA: $CUDA_INDEX) — faster-qwen3-tts 가 torch>=2.5.1 요구 ==="
pip install --index-url "$CUDA_INDEX" torch torchaudio

echo "=== [3] callone 코어(폴백 배관, heavy 아님) ==="
pip install -e .

echo "=== [4] 서빙 의존성(faster-whisper + faster-qwen3-tts + studio) ==="
pip install -r requirements-serve-gpu.txt

echo "=== [5] numpy 핀(librosa/numba — 반드시 마지막) ==="
pip install "numpy<2.4"

echo "=== [6] 충돌 점검 ==="
pip check || echo "  ⚠️ pip check 경고 — 위 메시지 확인(transformers 가 4.57.3 인지 봐라)"
python - <<'PY'
import importlib.metadata as m
for p in ("torch","transformers","faster-qwen3-tts","qwen-tts","faster-whisper","numpy","gradio"):
    try:
        print(f"  {p:18s} {m.version(p)}")
    except Exception:
        print(f"  {p:18s} (미설치)")
PY

cat <<'EOF'

=== 준비 완료 — 다음 순서 (자세히는 docs/RUNPOD_RUN.md) ===
  ※ $CALLONE_HOME = 영속 폴더(RunPod=/workspace, Elice 등=$HOME). 먼저 잡아라:
    if [ -w /workspace ]; then export CALLONE_HOME=/workspace; else export CALLONE_HOME=$HOME; fi
  1) LLM 모델:  huggingface-cli download HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive \
                  --local-dir $CALLONE_HOME/models/llm_A --include "*Q4_K_M*.gguf"
  2) TTS 모델:  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
                  --local-dir $CALLONE_HOME/models/qwen3_tts     # Apache2.0, 비게이트(토큰 불필요)
              export CALLONE_TTS_MODEL=$CALLONE_HOME/models/qwen3_tts
  3) 화자 참조:  data/speakers/A/ref_24k.wav (7~10초·24kHz·깨끗) — Qwen3-TTS 클론 필수입력
  4) llama.cpp:  bash scripts/build_llama_cuda.sh    # → $CALLONE_HOME/llama.cpp
  5) llama-server:  $CALLONE_HOME/llama.cpp/build/bin/llama-server -m <위 Q4_K_M.gguf> \
                      --host 127.0.0.1 --port 8080 -c 8192 -n 512 --n-gpu-layers 99
  6) 앱:  source .venv-serve/bin/activate && (callone-serve  |  python -m studio)
EOF
