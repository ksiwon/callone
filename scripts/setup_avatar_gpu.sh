#!/usr/bin/env bash
# avatar-server(토킹헤드) 전용 환경 설치 — **별 venv `.venv-avatar`** (callone 서빙과 분리).
#
# 단계: ① 서버 골격(static, CPU 도 됨) 먼저 → /health 확인 ② GPU 백엔드(Ditto/MuseTalk)는 repo+
#       체크포인트 받아 어댑터(avatar_server/backends/*) 채운 뒤 AVATAR_BACKEND 로 전환.
#
# 사용:  bash scripts/setup_avatar_gpu.sh           # 골격 venv (static 검증까지)
#        AVATAR_WITH_DITTO=1 bash scripts/setup_avatar_gpu.sh   # + Ditto repo/checkpoints (TODO 채우기 전제)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d /workspace ] && [ -w /workspace ]; then export CALLONE_HOME="${CALLONE_HOME:-/workspace}"
else export CALLONE_HOME="${CALLONE_HOME:-$HOME}"; fi
VENV="${VENV:-.venv-avatar}"

echo "=== [1] avatar 전용 venv ($VENV) ==="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip >/dev/null
pip install -r requirements-avatar-gpu.txt

echo "=== [2] static 백엔드 기동 검증(CPU 도 OK) ==="
echo "  AVATAR_BACKEND=static python -m avatar_server --port 8091"
echo "  → curl http://127.0.0.1:8091/health  → {\"status\":\"ok\",\"backend\":\"static\"}"

if [ "${AVATAR_WITH_DITTO:-0}" = "1" ]; then
  echo "=== [3] Ditto repo + checkpoints (GPU) ==="
  DITTO_DIR="$CALLONE_HOME/ditto-talkinghead"
  [ -d "$DITTO_DIR/.git" ] || git clone https://github.com/antgroup/ditto-talkinghead "$DITTO_DIR"
  if [ ! -d "$DITTO_DIR/checkpoints" ]; then
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead "$DITTO_DIR/checkpoints"
  fi
  # torch 는 드라이버 호환 CUDA 로(callone 처럼 torch.version.cuda 맞춤). TensorRT 는 Ampere(A100) 권장.
  echo "  torch(cu124)+librosa+opencv+imageio+tensorrt 는 requirements 주석 해제 후 설치."
  echo "  그담 avatar_server/backends/ditto_model.py 의 TODO(stream_pipeline_online 연결) 를 채워라."
  echo "  실행:  DITTO_DATA_ROOT=$DITTO_DIR/checkpoints/ditto_trt_Ampere_Plus \\"
  echo "         DITTO_CFG_PKL=$DITTO_DIR/checkpoints/.../v0.4_hubert_cfg_trt.pkl \\"
  echo "         AVATAR_BACKEND=ditto python -m avatar_server --port 8091"
fi

cat <<EOF

=== avatar-server 준비 ===
  골격(static):  AVATAR_BACKEND=static python -m avatar_server --port 8091
  callone 연동:  serve.yaml avatar.enabled=true, base_url=http://127.0.0.1:8091
                 (callone-serve 와 동시 기동 → 통화 시 ("frame") 이벤트로 영상)
  GPU 백엔드:    Ditto/MuseTalk repo+checkpoints 후 backends/*.py TODO 채우고 AVATAR_BACKEND 전환.
EOF
