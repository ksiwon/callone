#!/usr/bin/env bash
# 통화 서비스 3개 한 방 기동: llama-server(:8090) + avatar-server(:8091) + callone-serve(:8000).
# 각자 다른 venv·백그라운드(nohup)라 한 터미널서 딸깍. 이미 떠있는 건 재시작(idempotent).
#
# 사용:  bash scripts/run_all.sh                 # 음성+정지사진(static)
#        AVATAR_BACKEND=ditto bash scripts/run_all.sh   # 움직이는 얼굴(Ditto, setup 후)
#        bash scripts/run_all.sh stop            # 셋 다 종료
#   그 뒤 UI 는 별 터미널:  cd ui && npm run dev   (브라우저 localhost:5173/call/me)
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -d /workspace ] && [ -w /workspace ]; then export CALLONE_HOME="${CALLONE_HOME:-/workspace}"
else export CALLONE_HOME="${CALLONE_HOME:-$HOME}"; fi
PORT_LLM="${PORT:-8090}"
AVATAR_BACKEND="${AVATAR_BACKEND:-ditto}"   # 턴키: 기본 Ditto(영상). 로드 실패 시 _pick_model 이 static 폴백
LOG="${LOGDIR:-$HOME}"

if [ "${1:-}" = "stop" ]; then
  pkill -f "llama-server" 2>/dev/null; pkill -f "avatar_server" 2>/dev/null
  pkill -f "callone-serve" 2>/dev/null; pkill -f "cosyvoice_server/app.py" 2>/dev/null
  echo "서비스 종료."; exit 0
fi

# ① llama-server (안 떠있으면 bootstrap 으로 — LD_LIBRARY_PATH/포트 동기 처리)
if curl -s "http://127.0.0.1:$PORT_LLM/health" 2>/dev/null | grep -q '"status"'; then
  echo "[1/3] llama-server 이미 :$PORT_LLM ✅"
else
  echo "[1/3] llama-server 기동(:$PORT_LLM)..."
  PORT="$PORT_LLM" bash scripts/bootstrap_gpu.sh
fi

# ①.5 cosyvoice-server (conda env, :8092) — 음색 안정 TTS. env 있을 때만(없으면 Qwen 폴백).
if curl -s "http://127.0.0.1:8092/health" 2>/dev/null | grep -q '"status"'; then
  echo "[cosy] cosyvoice-server 이미 :8092 ✅"
elif [ -d "$HOME/CosyVoice" ] && (source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh" 2>/dev/null && conda env list 2>/dev/null | grep -q '^cosyvoice\b'); then
  echo "[cosy] cosyvoice-server 기동(:8092, 모델로드 ~30s)..."
  nohup bash scripts/setup_cosyvoice_gpu.sh run > "$LOG/cosyvoice.log" 2>&1 &
else
  echo "[cosy] cosyvoice env 없음 → TTS 는 Qwen3-TTS 폴백(scripts/setup_cosyvoice_gpu.sh 로 설치 권장)."
fi

# ② avatar-server (.venv-avatar)
pkill -f "avatar_server" 2>/dev/null; sleep 1
if [ -d .venv-avatar ]; then
  echo "[2/3] avatar-server($AVATAR_BACKEND) :8091..."
  # TRT 백엔드: cuDNN8(TRT8.6)·tensorrt_libs·cuBLAS 를 LD_LIBRARY_PATH 에(libcudnn.so.8 등 못 찾는 것 방지).
  AV_SP="$(.venv-avatar/bin/python -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)"
  AV_LD="$AV_SP/nvidia/cudnn/lib:$AV_SP/tensorrt_libs:$AV_SP/nvidia/cublas/lib:$AV_SP/nvidia/cuda_runtime/lib"
  ( source .venv-avatar/bin/activate && AVATAR_BACKEND="$AVATAR_BACKEND" \
      LD_LIBRARY_PATH="$AV_LD:${LD_LIBRARY_PATH:-}" \
      nohup python -m avatar_server --port 8091 > "$LOG/avatar.log" 2>&1 & )
else
  echo "[2/3] .venv-avatar 없음 → 영상 생략(scripts/setup_avatar_gpu.sh 먼저). 음성은 정상."
fi

# ③ callone-serve (.venv-serve)
pkill -f "callone-serve" 2>/dev/null; sleep 1
echo "[3/3] callone-serve :8000..."
( source .venv-serve/bin/activate && nohup callone-serve > "$LOG/serve.log" 2>&1 & )

sleep 5
echo "--- health ---"
echo -n "llama  : "; curl -s "127.0.0.1:$PORT_LLM/health" || echo "(아직)"; echo
echo -n "avatar : "; curl -s "127.0.0.1:8091/health" || echo "(없음/static 생략)"; echo
echo -n "serve  : "; curl -s "127.0.0.1:8000/api/health" || echo "(아직)"; echo
echo -n "cosy   : "; curl -s "127.0.0.1:8092/health" || echo "(로딩중/없음 → Qwen 폴백)"; echo
cat <<EOF
--- 다음 ---
  UI:  cd ui && npm run dev        (별 터미널)  → 브라우저 localhost:5173/call/me
  로그: tail -f $LOG/serve.log  /  $LOG/avatar.log  /  $CALLONE_HOME/llama.log
  종료: bash scripts/run_all.sh stop
EOF
