#!/usr/bin/env bash
# avatar-server(토킹헤드) 설치 — **별 venv `.venv-avatar`** (callone 서빙과 분리).
# 기본 = Ditto **PyTorch 백엔드**(TensorRT 버전지옥 회피 — 우리 드라이버 호환 전략과 일관).
#   머리·표정·고개까지 사진 1장 실시간. static(정지) 폴백은 AVATAR_BACKEND=static 로 무모델 검증.
#
# 사용:  bash scripts/setup_avatar_gpu.sh              # Ditto PyTorch 풀셋업(repo+checkpoints+deps)
#        AVATAR_SKIP_DITTO=1 bash scripts/setup_avatar_gpu.sh   # 서버 골격만(static, CPU 검증)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d /workspace ] && [ -w /workspace ]; then export CALLONE_HOME="${CALLONE_HOME:-/workspace}"
else export CALLONE_HOME="${CALLONE_HOME:-$HOME}"; fi
VENV="${VENV:-.venv-avatar}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
DITTO_REPO="$CALLONE_HOME/ditto-talkinghead"

echo "=== [1] avatar 전용 venv ($VENV) ==="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip >/dev/null
pip install -r requirements-avatar-gpu.txt
pip install hf_transfer >/dev/null 2>&1 || true

if [ "${AVATAR_SKIP_DITTO:-0}" = "1" ]; then
  echo "=== Ditto 스킵 — static 백엔드만. 기동: AVATAR_BACKEND=static python -m avatar_server --port 8091"
  exit 0
fi

echo "=== [2] Ditto repo + checkpoints(PyTorch) ==="
[ -d "$DITTO_REPO/.git" ] || git clone https://github.com/antgroup/ditto-talkinghead "$DITTO_REPO"
if [ ! -d "$DITTO_REPO/checkpoints/ditto_pytorch" ]; then
  huggingface-cli download digital-avatar/ditto-talkinghead --local-dir "$DITTO_REPO/checkpoints"
fi

echo "=== [3] Ditto PyTorch 의존성(드라이버 호환 torch) ==="
# torch 는 드라이버 호환 CUDA 로(서빙 venv 와 동일 전략). 드라이버 구버전이면 import 후 동작검증.
pip install --index-url "$CUDA_INDEX" torch torchaudio
pip install librosa opencv-python-headless imageio imageio-ffmpeg scikit-image tqdm "numpy<2.4"
# Ditto repo 자체 requirements 가 있으면 추가(저자 의도 — 버전 충돌나면 위 핀 유지).
[ -f "$DITTO_REPO/requirements.txt" ] && pip install -r "$DITTO_REPO/requirements.txt" || true

# env 고정 — DittoModel 이 이걸로 SDK 로드.
DATA_ROOT="$DITTO_REPO/checkpoints/ditto_pytorch/models"
# cfg pkl 은 하위 폴더(ditto_cfg/)에 있음 → 재귀 검색(hubert_cfg_pytorch 우선).
CFG_PKL="$(find "$DITTO_REPO/checkpoints" -name '*cfg*pytorch*.pkl' 2>/dev/null | grep -i hubert | head -1)"
[ -z "$CFG_PKL" ] && CFG_PKL="$(find "$DITTO_REPO/checkpoints" -name '*pytorch*.pkl' 2>/dev/null | head -1)"
{ echo "export DITTO_REPO=$DITTO_REPO"
  echo "export DITTO_DATA_ROOT=$DATA_ROOT"
  echo "export DITTO_CFG_PKL=$CFG_PKL"; } >> ~/.bashrc
export DITTO_REPO DITTO_DATA_ROOT="$DATA_ROOT" DITTO_CFG_PKL="$CFG_PKL"

echo "=== [4] torch GPU 동작 검증(드라이버 호환) ==="
python - <<'PY' || echo "⚠️ torch GPU 실패 — 드라이버 호환 CUDA 로 재설치 필요(서빙과 동일 이슈)"
import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "OK")
PY

cat <<EOF

=== avatar-server 준비 완료(Ditto PyTorch) ===
  DITTO_DATA_ROOT=$DATA_ROOT
  DITTO_CFG_PKL=$CFG_PKL
  기동:   AVATAR_BACKEND=ditto python -m avatar_server --port 8091
          curl http://127.0.0.1:8091/health   → {"status":"ok","backend":"ditto"}
  callone 연동: serve.yaml avatar.enabled=true (base_url=http://127.0.0.1:8091),
               증명사진 data/speakers/<화자>/portrait.jpg|png 두기.
  ⚠️ DittoModel.frames 의 [V] 검증지점(setup_Nd/청크크기/writer attr)을 repo 코드로 맞춰라.
     안 맞으면 _pick_model 이 static(정지사진)으로 폴백 — 통화·음성은 정상.
EOF
