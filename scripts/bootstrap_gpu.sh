#!/usr/bin/env bash
# callone GPU 원샷 부트스트랩 — 새 클라우드 인스턴스(RunPod·Elice)서 통화 준비 끝까지 한 방.
#
# 인스턴스를 자주 지웠다 새로 만드는 경우를 위해 **전부 idempotent**:
#   이미 깔린 venv / 빌드된 llama-server / 받아둔 모델은 건너뛴다(중복작업 0).
#
# 사용(클론 직후):
#   bash scripts/bootstrap_gpu.sh                 # 전부 자동(필요 시 컴파일)
#   SPK=mom bash scripts/bootstrap_gpu.sh         # 화자 ID 지정(ref 체크용)
#   LLAMA_SERVER_URL=https://.../llama-bin.tgz bash scripts/bootstrap_gpu.sh
#                                                 # ↑ 한 번 빌드해 올려둔 바이너리 재사용(컴파일 스킵)
#   NO_SERVE=1 bash scripts/bootstrap_gpu.sh      # llama-server 자동기동 생략
set -euo pipefail
cd "$(dirname "$0")/.."                            # repo 루트

# ── 0. 영속폴더/환경 (RunPod=/workspace, Elice 등=$HOME 자동) ──────────────
if [ -d /workspace ] && [ -w /workspace ]; then export CALLONE_HOME="${CALLONE_HOME:-/workspace}"
else export CALLONE_HOME="${CALLONE_HOME:-$HOME}"; fi
export HF_HOME="${HF_HOME:-$CALLONE_HOME/hf_cache}"
export CALLONE_TIER="${CALLONE_TIER:-server_gpu}"
export CALLONE_TTS_MODEL="${CALLONE_TTS_MODEL:-$CALLONE_HOME/models/qwen3_tts}"
SPK="${SPK:-sis}"
LLM_REPO="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
TTS_REPO="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LBIN="$CALLONE_HOME/llama.cpp/build/bin/llama-server"

echo "== callone GPU bootstrap =="
echo "   CALLONE_HOME=$CALLONE_HOME  speaker=$SPK"

# bashrc 에 env 고정(중복 안 쌓이게 idempotent)
for kv in "CALLONE_HOME=$CALLONE_HOME" "HF_HOME=$HF_HOME" "CALLONE_TIER=$CALLONE_TIER" \
          "CALLONE_TTS_MODEL=$CALLONE_TTS_MODEL"; do
  grep -q "export ${kv%%=*}=" ~/.bashrc 2>/dev/null || echo "export $kv" >> ~/.bashrc
done

# ── 1. 서빙 venv (있으면 스킵) ─────────────────────────────────────────────
if [ ! -d .venv-serve ]; then
  echo "[1/5] 서빙 venv 설치..."; bash scripts/setup_serve_gpu.sh
else echo "[1/5] .venv-serve 있음 → 스킵"; fi
# shellcheck disable=SC1091
source .venv-serve/bin/activate
command -v huggingface-cli >/dev/null || pip install -q "huggingface_hub[cli]" || true
pip install -q hf_transfer 2>/dev/null || true

# ── 2. llama-server: 있으면 스킵 / URL 주면 다운 / 아니면 native 컴파일 ────
if [ -x "$LBIN" ]; then
  echo "[2/5] llama-server 있음 → 빌드 스킵 ($LBIN)"
elif [ -n "${LLAMA_SERVER_URL:-}" ]; then
  echo "[2/5] 프리빌트 다운로드: $LLAMA_SERVER_URL (컴파일 스킵)"
  mkdir -p "$(dirname "$LBIN")"
  curl -fSL "$LLAMA_SERVER_URL" -o /tmp/llama-bin.tgz
  tar -xzf /tmp/llama-bin.tgz -C "$(dirname "$LBIN")"
  chmod +x "$LBIN" 2>/dev/null || true
  [ -x "$LBIN" ] || { echo "⚠️ 다운본에 llama-server 없음 → 컴파일로 폴백"; bash scripts/build_llama_cuda.sh; }
else
  echo "[2/5] llama-server 컴파일(native 아키)... 한 번만. 다음 인스턴스 위해 끝나고 pack 안내 출력"
  bash scripts/build_llama_cuda.sh
fi

# ── 3. 모델 (받아둔 거 있으면 스킵) ────────────────────────────────────────
GGUF="$(ls "$CALLONE_HOME"/models/llm_A/*Q4_K_M*.gguf 2>/dev/null | head -1 || true)"
if [ -z "$GGUF" ]; then
  echo "[3/5] LLM GGUF 다운로드(5.3GB)..."
  huggingface-cli download "$LLM_REPO" --include "*Q4_K_M*.gguf" --local-dir "$CALLONE_HOME/models/llm_A"
  GGUF="$(ls "$CALLONE_HOME"/models/llm_A/*Q4_K_M*.gguf 2>/dev/null | head -1)"
else echo "[3/5] LLM GGUF 있음 → 스킵 ($GGUF)"; fi
if [ ! -f "$CALLONE_HOME/models/qwen3_tts/config.json" ]; then
  echo "[3/5] TTS 모델 다운로드(4.5GB)..."
  huggingface-cli download "$TTS_REPO" --local-dir "$CALLONE_HOME/models/qwen3_tts"
else echo "[3/5] TTS 모델 있음 → 스킵"; fi

# ── 4. 화자 참조음성 체크(경고만 — 개인음성이라 자동다운 불가) ───────────
if [ -f "data/speakers/$SPK/ref_24k.wav" ] && [ -f "data/speakers/$SPK/ref_text.txt" ]; then
  echo "[4/5] 화자 참조 OK (data/speakers/$SPK/)"
else
  echo "[4/5] ⚠️ data/speakers/$SPK/ref_24k.wav (+ref_text.txt) 없음 → RUNPOD_RUN.md §5 로 만들어라."
  echo "       (없으면 TTS 가 Piper 로 폴백한다)"
fi

# ── 5. llama-server 기동 + health (NO_SERVE=1 이면 생략) ───────────────────
if [ "${NO_SERVE:-0}" != "1" ]; then
  if curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status"'; then
    echo "[5/5] llama-server 이미 떠있음"
  elif [ -n "$GGUF" ] && [ -x "$LBIN" ]; then
    echo "[5/5] llama-server 기동(:8080)..."
    nohup "$LBIN" -m "$GGUF" --host 127.0.0.1 --port 8080 \
      -c 8192 -n 512 --n-gpu-layers 99 > "$CALLONE_HOME/llama.log" 2>&1 &
    for i in $(seq 1 30); do
      curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q '"status"' && break; sleep 1; done
  fi
  curl -s http://127.0.0.1:8080/health ; echo
fi

cat <<EOF

== 부트스트랩 완료 ==
  llama-server: $LBIN   GGUF: ${GGUF:-(없음)}
  다음:
    callone-bench --speaker $SPK --turns 5 --text "여보세요, 밥은 먹었어?"   # 지연 실측
    GRADIO_SHARE=1 python -m studio                                          # 브라우저 통화

  ※ 인스턴스를 또 지울 거면 — 이번에 컴파일한 바이너리를 한 번만 올려두면 다음부턴 컴파일 0:
    tar czf /tmp/llama-bin.tgz -C "$(dirname "$LBIN")" .
    # 이 tgz 를 본인 저장소(HF/S3/깃릴리스 등)에 올린 URL 을 다음 인스턴스서:
    #   LLAMA_SERVER_URL=<그 URL> bash scripts/bootstrap_gpu.sh
EOF
