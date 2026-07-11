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
  pkill -f "qwen_tts_server/app.py" 2>/dev/null
  echo "서비스 종료."; exit 0
fi

# ① llama-server (안 떠있으면 bootstrap 으로 — LD_LIBRARY_PATH/포트 동기 처리)
if curl -s "http://127.0.0.1:$PORT_LLM/health" 2>/dev/null | grep -q '"status"'; then
  echo "[1/3] llama-server 이미 :$PORT_LLM ✅"
else
  echo "[1/3] llama-server 기동(:$PORT_LLM)..."
  PORT="$PORT_LLM" bash scripts/bootstrap_gpu.sh
fi

# ①.5 cosyvoice-server (conda env, :8092) — 음색 안정 TTS(기본 백엔드). env 있을 때만.
if curl -s "http://127.0.0.1:8092/health" 2>/dev/null | grep -q '"status"'; then
  echo "[cosy] cosyvoice-server 이미 :8092 ✅"
elif [ -d "$HOME/CosyVoice" ] && (source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh" 2>/dev/null && conda env list 2>/dev/null | grep -q '^cosyvoice\b'); then
  echo "[cosy] cosyvoice-server 기동(:8092, 모델로드 ~30s)..."
  nohup bash scripts/setup_cosyvoice_gpu.sh run > "$LOG/cosyvoice.log" 2>&1 &
else
  echo "[cosy] cosyvoice env 없음 → TTS 는 qwen3tts/Piper 폴백(scripts/setup_cosyvoice_gpu.sh 로 설치 권장)."
fi

# ①.6 qwen-tts-server (.venv-qwentts, :8093) — v2 저지연 스트리밍 TTS(게이트 통과 전엔 벤치/A-B용).
#   티어 자동: VRAM ≥70GB(H100/A100-80) → 1.7B, 아니면 0.6B (QWEN_TTS_MODEL env 로 강제).
if curl -s "http://127.0.0.1:8093/health" 2>/dev/null | grep -q '"status"'; then
  echo "[qwen] qwen-tts-server 이미 :8093 ✅"
elif [ -d .venv-qwentts ]; then
  VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}')"
  if [ -z "${QWEN_TTS_MODEL:-}" ] && [ "${VRAM_GB:-0}" -ge 70 ]; then
    export QWEN_TTS_MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  fi
  echo "[qwen] qwen-tts-server 기동(:8093, ${QWEN_TTS_MODEL:-0.6B 기본})..."
  nohup bash scripts/setup_qwen_tts_gpu.sh run > "$LOG/qwentts.log" 2>&1 &
else
  echo "[qwen] .venv-qwentts 없음 → qwen3tts 생략(scripts/setup_qwen_tts_gpu.sh 로 설치)."
fi

# DITTO_* env 자가치유: run_all 을 `source ~/.bashrc` 없이 돌려도 Ditto 가 뜨게 한다(=static 폴백의
#   1위 원인 제거). 비대화형 셸이라 `source ~/.bashrc` 는 PS1 가드에 막히므로 DITTO_ 줄만 뽑아
#   set -a(자동 export)로 eval → avatar-server 서브셸이 상속. 이미 env 에 있으면 건너뜀.
if [ "$AVATAR_BACKEND" = "ditto" ] && [ -z "${DITTO_REPO:-}" ] && [ -f "$HOME/.bashrc" ]; then
  set -a
  eval "$(grep -E '^[[:space:]]*(export[[:space:]]+)?DITTO_[A-Za-z_]+=' "$HOME/.bashrc" 2>/dev/null | sed -E 's/^[[:space:]]*export[[:space:]]+//')"
  set +a
  if [ -n "${DITTO_REPO:-}" ]; then echo "[2/3] DITTO_* env 자동 로드(.bashrc) — Ditto 준비"
  else echo "[2/3] ⚠️ DITTO_* env 못 찾음 → static 폴백 예상. setup_avatar_gpu.sh 먼저 돌려라."; fi
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

# health poll — 서비스별 최대 대기(모델 로드가 5s 초과 가능, 특히 32B GGUF).
#   _wait SVC URL MAX_WAIT_S: MAX_WAIT_S 초 안에 200 응답이 오면 ✅, 초과면 경고만(exit 0 유지).
_wait() {
  local name="$1" url="$2" max="${3:-30}" waited=0
  printf "%-8s: " "$name"
  while true; do
    if curl -sf "$url" >/dev/null 2>&1; then
      curl -s "$url"; echo " ✅"; return 0
    fi
    if [ "$waited" -ge "$max" ]; then
      echo "(${max}s 내 응답 없음 — 백그라운드서 계속 로드 중)"; return 0
    fi
    sleep 3; waited=$((waited + 3))
  done
}
echo "--- health (서비스별 최대 30~60s 대기) ---"
_wait "llama"   "http://127.0.0.1:$PORT_LLM/health" 60
_wait "avatar"  "http://127.0.0.1:8091/health"       30
# Ditto 요청했는데 static 으로 떴으면(움직임 없음) 크게 경고 + 고침 안내(조용히 폴백해 헷갈리는 것 방지).
if [ "$AVATAR_BACKEND" = "ditto" ] && curl -s "http://127.0.0.1:8091/health" 2>/dev/null | grep -q '"backend":"static"'; then
  echo "  ⚠️⚠️ avatar 가 static 으로 떴다(얼굴 안 움직임). 원인=DITTO_* env 미로드. 고침:"
  echo "        source ~/.bashrc && pkill -9 -f avatar_server && bash scripts/run_all.sh"
  echo "        그래도 static 이면: bash scripts/setup_avatar_gpu.sh (env/체크포인트 재설정) — 로그: grep 'Ditto 로드 실패' $LOG/avatar.log"
fi
_wait "serve"   "http://127.0.0.1:8000/api/health"   30
_wait "cosy"    "http://127.0.0.1:8092/health"       60
[ -d .venv-qwentts ] && _wait "qwentts" "http://127.0.0.1:8093/health" 90
cat <<EOF
--- 다음 ---
  UI:  cd ui && npm run dev        (별 터미널)  → 브라우저 localhost:5173/call/me
  로그: tail -f $LOG/serve.log  /  $LOG/avatar.log  /  $CALLONE_HOME/llama.log
  종료: bash scripts/run_all.sh stop
EOF
