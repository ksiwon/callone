#!/usr/bin/env bash
# callone 원샷 설치·기동 — GPU 서버에서 git clone 후 이 한 줄로 음성+영상 통화까지 끝.
#
#   bash install.sh
#
# 이미 설치된 단계는 스킵(멱등). 재기동만 필요하면:
#   bash scripts/run_all.sh
#
# 환경변수로 동작 제어(선택):
#   PORT=8090              llama-server 포트 (기본 8090, serve.yaml 과 동기)
#   SKIP_COSYVOICE=1       TTS 셋업 건너뜀 (Piper/Kokoro 폴백)
#   SKIP_AVATAR=1          Ditto 셋업 건너뜀 (정지사진 폴백)
#   SKIP_UI=1              UI npm 빌드 건너뜀

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"   # repo 루트

PORT="${PORT:-8090}"
SKIP_COSYVOICE="${SKIP_COSYVOICE:-0}"
SKIP_AVATAR="${SKIP_AVATAR:-0}"
SKIP_UI="${SKIP_UI:-0}"

# 영속폴더(RunPod=/workspace 있으면, 없으면 $HOME)
if [ -d /workspace ] && [ -w /workspace ]; then
  export CALLONE_HOME="${CALLONE_HOME:-/workspace}"
else
  export CALLONE_HOME="${CALLONE_HOME:-$HOME}"
fi
LOG="$CALLONE_HOME"

# ── 헬퍼 ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
_ok()   { echo -e "${GREEN}[OK]  $*${NC}"; }
_warn() { echo -e "${YELLOW}[!!]  $*${NC}"; }
_err()  { echo -e "${RED}[ERR] $*${NC}"; }
_hdr()  { echo; echo -e "${GREEN}=== $* ===${NC}"; }

# 단계 결과 (ok / skip / fail)
R_LLM=""; R_TTS=""; R_AVATAR=""; R_UI=""

# ── 사전 체크 ───────────────────────────────────────────────────────────────
_hdr "사전 체크"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  _err "nvidia-smi 없음 — GPU 서버에서 실행해야 함"; exit 1
fi

VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
           | head -1 | awk '{print int($1/1024)}')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  GPU : $GPU_NAME"
echo "  VRAM: ${VRAM_GB:-?}GB"

FREE_GB="$(df -BG "$CALLONE_HOME" 2>/dev/null | tail -1 | awk '{gsub("G",""); print $4}')"
if [ "${FREE_GB:-99}" -lt 25 ]; then
  _warn "디스크 여유 ${FREE_GB:-?}GB — 25GB 이상 권고(모델 합계 ~22GB). 다운로드 도중 실패 가능."
else
  echo "  Disk: ${FREE_GB:-?}GB 여유"
fi

# ── 1. LLM + 서빙 venv ──────────────────────────────────────────────────────
# 이 단계가 실패하면 이후 단계 무의미 → 즉시 종료
_hdr "1/4  LLM(EXAONE GGUF) + .venv-serve"
if PORT="$PORT" bash scripts/bootstrap_gpu.sh; then
  R_LLM="ok"; _ok "LLM 스택 완료"
else
  R_LLM="fail"
  _err "LLM 스택 실패 — 로그: $CALLONE_HOME/llama.log"
  exit 1
fi

# ── 2. CosyVoice3 TTS ────────────────────────────────────────────────────────
_hdr "2/4  CosyVoice3 TTS(음색복제)"
if [ "$SKIP_COSYVOICE" = "1" ]; then
  R_TTS="skip"; _warn "SKIP_COSYVOICE=1 — Piper/Kokoro 폴백"
elif bash scripts/setup_cosyvoice_gpu.sh; then
  R_TTS="ok"; _ok "CosyVoice3 완료"
else
  R_TTS="fail"; _warn "CosyVoice 실패 — Piper/Kokoro 폴백(음색 유사도 낮아짐)"
fi

# ── 3. Avatar(Ditto) ─────────────────────────────────────────────────────────
_hdr "3/4  Avatar Ditto(토킹헤드)"
if [ "$SKIP_AVATAR" = "1" ]; then
  R_AVATAR="skip"; _warn "SKIP_AVATAR=1 — 정지사진 폴백"
elif bash scripts/setup_avatar_gpu.sh; then
  R_AVATAR="ok"; _ok "Ditto 완료"
else
  R_AVATAR="fail"; _warn "Ditto 실패 — 정지사진 폴백(음성 통화는 정상)"
fi

# ── 4. UI ────────────────────────────────────────────────────────────────────
_hdr "4/4  UI (npm)"
if [ "$SKIP_UI" = "1" ]; then
  R_UI="skip"; _warn "SKIP_UI=1 — 나중에 직접: cd ui && npm run dev"
else
  # npm 없으면 Node.js 설치
  if ! command -v npm >/dev/null 2>&1; then
    echo "  npm 없음 — Node.js 20 설치 시도..."
    SUDO=""; command -v sudo >/dev/null 2>&1 && SUDO="sudo -E"
    curl -fsSL https://deb.nodesource.com/setup_20.x | ${SUDO:-bash} bash - 2>/dev/null \
      && ${SUDO:+sudo }apt-get install -y nodejs 2>/dev/null \
      || _warn "Node.js 자동 설치 실패. 수동: apt-get install nodejs"
  fi
  if command -v npm >/dev/null 2>&1; then
    echo "  npm $(npm --version) — 의존성 설치 중..."
    if ( cd ui && npm install --prefer-offline 2>&1 | tail -3 ); then
      R_UI="ok"; _ok "UI 의존성 완료 (실행: cd ui && npm run dev)"
    else
      R_UI="fail"; _warn "npm install 실패 — 수동: cd ui && npm install && npm run dev"
    fi
  else
    R_UI="fail"; _warn "npm 없음 — UI 스킵"
  fi
fi

# ── 5. 전체 서비스 기동 ───────────────────────────────────────────────────────
_hdr "서비스 기동"
# bashrc 에 들어간 env(DITTO_*, LD_LIBRARY_PATH 등) 적용
# shellcheck disable=SC1090
source ~/.bashrc 2>/dev/null || true
bash scripts/run_all.sh

# UI dev server — 백그라운드 기동(의존성 설치 성공 시)
if [ "$R_UI" = "ok" ]; then
  if ! curl -sf "http://127.0.0.1:5173" >/dev/null 2>&1; then
    echo "  UI dev server 백그라운드 기동..."
    ( cd ui && nohup npm run dev -- --host > "$LOG/ui.log" 2>&1 & )
    # 최대 15초 대기
    for i in $(seq 1 5); do
      sleep 3
      curl -sf "http://127.0.0.1:5173" >/dev/null 2>&1 && break
    done
    curl -sf "http://127.0.0.1:5173" >/dev/null 2>&1 \
      && _ok "UI dev server :5173" \
      || _warn "UI 아직 로딩 중 — tail -f $LOG/ui.log"
  else
    _ok "UI dev server 이미 :5173"
  fi
fi

# ── 최종 요약 ────────────────────────────────────────────────────────────────
_status() {
  case "$1" in
    ok)   echo "OK " ;;
    skip) echo "SKIP" ;;
    fail) echo "FAIL" ;;
    *)    echo "?   " ;;
  esac
}
SERVE_HEALTH="$(curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1 && echo ok || echo wait)"

echo
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  callone 설치 완료${NC}"
echo -e "${GREEN}================================================================${NC}"
echo
echo "  단계별 결과:"
printf "  [%s] LLM (EXAONE GGUF :%s)\n"   "$(_status "$R_LLM")"      "$PORT"
printf "  [%s] CosyVoice3 TTS  (:8092)\n" "$(_status "$R_TTS")"
printf "  [%s] Avatar Ditto    (:8091)\n"  "$(_status "$R_AVATAR")"
printf "  [%s] UI npm deps\n"              "$(_status "$R_UI")"
printf "  [%s] callone-serve   (:8000)\n"  "$(_status "$SERVE_HEALTH")"
echo
echo "  FAIL = 폴백으로 동작 / SKIP = 환경변수로 건너뜀"
echo
echo "  ── 브라우저 접속 (내 노트북 터미널에서 SSH 터널) ──"
echo
echo "  # RunPod:"
echo "  ssh root@<IP> -p <포트> -i ~/.ssh/id_ed25519 -L 5173:localhost:5173"
echo
echo "  # Elice:"
echo "  ssh -i elice-cloud-ondemand-XXXX.pem elicer@central-01.tcp.tunnel.elice.io -p <포트> -L 5173:localhost:5173"
echo
echo "  → 터미널 열어둔 채로:  http://localhost:5173/call/me"
echo
echo "  ── 재기동 ──"
echo "  cd $(pwd) && source ~/.bashrc && bash scripts/run_all.sh"
echo "  (별 터미널) cd ui && npm run dev"
echo
echo "  ── 로그 ──"
echo "  tail -f $LOG/serve.log $LOG/avatar.log $LOG/cosyvoice.log $LOG/llama.log"
