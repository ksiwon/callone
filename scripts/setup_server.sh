#!/usr/bin/env bash
# callone 서버 한방 설치+실행 스크립트 (엘리스 H100 / Linux GPU)
#
# 사용법 (callone 폴더 안에서):
#   bash scripts/setup_server.sh           # 환경 설치만
#   bash scripts/setup_server.sh pilot      # 설치 + 50통 파일럿(A)
#   bash scripts/setup_server.sh full       # 설치 + 1000+ 전량(A,B)
#
# CUDA 버전 다르면:  CUDA_INDEX=https://download.pytorch.org/whl/cu124 bash scripts/setup_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # callone 루트로

RUN_MODE="${1:-none}"            # none | pilot | full
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "=== [0] ffmpeg 확인/설치 (S0 변환 필수) ==="
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg 없음 → 설치 시도"
  if command -v apt-get >/dev/null 2>&1; then
    (sudo apt-get update -y && sudo apt-get install -y ffmpeg) 2>/dev/null \
      || (apt-get update -y && apt-get install -y ffmpeg) 2>/dev/null || true
  fi
  if ! command -v ffmpeg >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
    conda install -y -c conda-forge ffmpeg || true
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "  apt/conda 실패 → 정적 바이너리 설치(~/bin)"
    cd ~ && wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
      && tar xf ffmpeg-release-amd64-static.tar.xz && mkdir -p ~/bin \
      && cp ffmpeg-*-amd64-static/ffmpeg ffmpeg-*-amd64-static/ffprobe ~/bin/ \
      && export PATH="$HOME/bin:$PATH" \
      && grep -q 'HOME/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
    cd - >/dev/null
  fi
fi
command -v ffmpeg >/dev/null 2>&1 && echo "ffmpeg OK: $(ffmpeg -version 2>/dev/null | head -1)" || echo "⚠️ ffmpeg 설치 실패 — 수동 설치 필요"

echo "=== [1/6] 가상환경 ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip >/dev/null

echo "=== [2/6] torch 3종 (CUDA: $CUDA_INDEX) ==="
pip install --index-url "$CUDA_INDEX" torch torchvision torchaudio

echo "=== [3/6] callone + 무거운 의존성 ==="
pip install Cython               # hdbscan 등 소스빌드 패키지용
pip install -e ".[heavy]"
pip install -r requirements-heavy.txt
# hdbscan 은 소스빌드 필요(wheel 없을 때) → 별도, 실패해도 비치명(KMeans 폴백)
pip install "hdbscan>=0.8" --no-build-isolation || echo "  hdbscan 건너뜀 → KMeans 폴백 사용"
pip install "numpy<2.4"          # numba/librosa 호환(반드시 마지막)

echo "=== [4/6] .env 설정 (DEVICE=cuda) ==="
[ -f .env ] || cp .env.example .env
python - <<'PY'
import re, pathlib
p = pathlib.Path(".env"); t = p.read_text(encoding="utf-8")
t = re.sub(r'^DEVICE=.*$', 'DEVICE=cuda', t, flags=re.M) if re.search(r'^DEVICE=', t, re.M) else t + "\nDEVICE=cuda\n"
p.write_text(t, encoding="utf-8")
print("DEVICE=cuda 설정됨")
PY
if ! grep -q '^HF_TOKEN=hf' .env 2>/dev/null; then
  echo "  ⚠️  .env 에 HF_TOKEN 이 없음 → pyannote/Gemma 다운로드 전에 넣을 것"
fi

echo "=== [5/6] 하드웨어/티어 확인 ==="
python -c "from callone.common.hardware import describe; print(describe())"

echo "=== [6/6] 준비 완료 ==="
echo "raw m4a 개수: $(ls data/raw/*.m4a 2>/dev/null | wc -l)"

case "$RUN_MODE" in
  pilot)
    echo ">>> 50통 파일럿 시작 (A)"
    bash scripts/run_pilot.sh 50 A
    ;;
  full)
    echo ">>> 전량 파이프라인 시작 (A,B)"
    bash scripts/run_full.sh
    ;;
  *)
    echo "환경만 설치함. 실행하려면:"
    echo "  bash scripts/run_pilot.sh 50 A     # 먼저 50통 확인"
    echo "  bash scripts/run_full.sh           # 1000+ 전량"
    echo "  그 뒤 학습: callone-asr-train / setup_piper_gpu.sh / callone-llm-train"
    ;;
esac
