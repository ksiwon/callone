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
#   PORT=8090 bash scripts/bootstrap_gpu.sh       # llama-server 포트(기본 8080)
set -euo pipefail
cd "$(dirname "$0")/.."                            # repo 루트
PORT="${PORT:-8080}"

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
# 프리빌트 llama-server(미리 빌드해 HF 에 올려둔 것). 기본으로 받아 **컴파일 생략**.
#   다른 이미지(CUDA/glibc 불일치)면 실행검사 실패 → 자동 컴파일 폴백(아래 [2/5]).
#   강제로 직접 컴파일하려면:  LLAMA_SERVER_URL= bash scripts/bootstrap_gpu.sh  (빈 값)
LLAMA_SERVER_URL="${LLAMA_SERVER_URL-https://huggingface.co/ksiwon/callone-llama-bin/resolve/main/llama-bin.tgz}"

echo "== callone GPU bootstrap =="
echo "   CALLONE_HOME=$CALLONE_HOME  speaker=$SPK"

# bashrc 에 env 고정(중복 안 쌓이게 idempotent)
for kv in "CALLONE_HOME=$CALLONE_HOME" "HF_HOME=$HF_HOME" "CALLONE_TIER=$CALLONE_TIER" \
          "CALLONE_TTS_MODEL=$CALLONE_TTS_MODEL"; do
  grep -q "export ${kv%%=*}=" ~/.bashrc 2>/dev/null || echo "export $kv" >> ~/.bashrc
done

# llama-server 포트($PORT)를 serve.yaml llm.base_url 에만 반영(llama-server 주석 달린 줄).
# ⚠️ 전역 치환하면 avatar.base_url(8091)까지 덮어써 아바타가 llama 를 보는 버그 → 줄 스코프 필수.
sed -i -E "/llama-server/ s#(127\.0\.0\.1:)[0-9]+#\1$PORT#" configs/serve.yaml 2>/dev/null || true

# ── 1. 서빙 venv (있으면 스킵) ─────────────────────────────────────────────
if [ ! -d .venv-serve ]; then
  echo "[1/5] 서빙 venv 설치..."; bash scripts/setup_serve_gpu.sh
else echo "[1/5] .venv-serve 있음 → 스킵"; fi
# shellcheck disable=SC1091
source .venv-serve/bin/activate
command -v huggingface-cli >/dev/null || pip install -q "huggingface_hub[cli]" || true
pip install -q hf_transfer 2>/dev/null || true

# llama-server 가 **torch 번들 CUDA 런타임**(드라이버 호환 버전, 보통 12.4)을 쓰게 LD_LIBRARY_PATH 설정.
# 구드라이버(Elice 실측: 드라이버 12.2) 박스서도 12.4 빌드 바이너리가 돌게 하는 핵심(torch 와 동일 런타임).
TORCH_LIBS="$(python - <<'PY' 2>/dev/null
import os, glob
try:
    import nvidia
    b = os.path.dirname(nvidia.__file__)
    print(":".join(sorted({os.path.dirname(p) for p in glob.glob(b + "/*/lib/*.so*")})))
except Exception:
    pass
PY
)"
if [ -n "$TORCH_LIBS" ]; then
  export LD_LIBRARY_PATH="$TORCH_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  # 새 셸(수동 기동)에서도 되게 bashrc 에 한 줄(idempotent)
  grep -q 'callone: torch CUDA libs' ~/.bashrc 2>/dev/null || \
    echo "export LD_LIBRARY_PATH=\"$TORCH_LIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\"  # callone: torch CUDA libs" >> ~/.bashrc
fi

# ── 2. llama-server: 있으면 스킵 / URL 주면 다운 / 아니면 컴파일 ───────────
#   ⚠️ --version 통과 ≠ GPU 동작(커널이미지 호환은 실제 모델 로드해야 드러남) → step5 에서 GPU
#      검증 후 실패 시(다운본일 때만) 자동 재빌드. DL_FROM_URL 로 출처 추적.
DL_OK=0; DL_FROM_URL=0
if [ -x "$LBIN" ] && "$LBIN" --version >/dev/null 2>&1; then
  echo "[2/5] llama-server 있음 → 빌드 스킵 ($LBIN)"; DL_OK=1
elif [ -n "$LLAMA_SERVER_URL" ]; then
  echo "[2/5] 프리빌트 다운로드 시도: $LLAMA_SERVER_URL"
  mkdir -p "$(dirname "$LBIN")"
  if curl -fSL "$LLAMA_SERVER_URL" -o /tmp/llama-bin.tgz \
     && tar -xzf /tmp/llama-bin.tgz -C "$(dirname "$LBIN")"; then
    chmod +x "$LBIN" 2>/dev/null || true
    if [ -x "$LBIN" ] && "$LBIN" --version >/dev/null 2>&1; then
      echo "   프리빌트 실행 가능 → GPU 동작은 기동 시 검증(실패하면 자동 재빌드)"; DL_OK=1; DL_FROM_URL=1
    else
      echo "   ⚠️ 프리빌트 실행 불가(glibc 불일치) → 컴파일로 폴백"
    fi
  else
    echo "   ⚠️ 다운로드/해제 실패 → 컴파일로 폴백"
  fi
fi
if [ "$DL_OK" != "1" ]; then
  echo "[2/5] llama-server 드라이버호환 컴파일... (torch CUDA 버전에 맞춤)"
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

# ── 5. llama-server 기동 + GPU 검증 (NO_SERVE=1 이면 생략) ─────────────────
_health() { curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status"'; }
_start_verify() {   # 0=정상 / 2=CUDA에러 / 1=기타실패
  pkill -f llama-server 2>/dev/null; sleep 2
  # --jinja: 모델 jinja 템플릿 사용(thinking 토글/chat_template_kwargs 적용에 필수).
  # --reasoning-budget 0: Qwen3.5 thinking OFF. 둘 다 없으면 출력이 reasoning_content 로만 가고
  #   content 가 비어 **빈 응답**(실측: LLM 완료 0자, reasoning_content 만 참). 둘 다 줘야 content 참.
  nohup "$LBIN" -m "$GGUF" --host 127.0.0.1 --port "$PORT" \
    -c 8192 -n 512 --n-gpu-layers 99 --jinja --reasoning-budget 0 > "$CALLONE_HOME/llama.log" 2>&1 &
  for _ in $(seq 1 90); do
    _health && return 0
    grep -qiE 'CUDA error|kernel image is invalid|out of memory' "$CALLONE_HOME/llama.log" && return 2
    sleep 1
  done
  return 1
}
if [ "${NO_SERVE:-0}" != "1" ]; then
  if _health; then
    echo "[5/5] llama-server 이미 :$PORT 에 떠있음"
  elif [ -n "$GGUF" ] && [ -x "$LBIN" ]; then
    echo "[5/5] llama-server 기동(:$PORT) + GPU 검증..."
    rc=0; _start_verify || rc=$?
    if [ "$rc" = "2" ] && [ "$DL_FROM_URL" = "1" ]; then
      echo "   ⚠️ 프리빌트가 이 드라이버서 GPU 동작 실패(CUDA error) → 드라이버 호환 재빌드 후 재기동"
      rm -f "$LBIN"; bash scripts/build_llama_cuda.sh; DL_FROM_URL=0
      rc=0; _start_verify || rc=$?
    fi
    if _health; then echo "   ✅ llama-server OK (:$PORT)"
    else echo "   ⚠️ llama-server 안 뜸(rc=$rc) — 로그:"; tail -8 "$CALLONE_HOME/llama.log"; fi
  fi
fi

cat <<EOF

== 부트스트랩 완료 ==
  llama-server: $LBIN   GGUF: ${GGUF:-(없음)}
  다음:
    callone-bench --speaker $SPK --turns 5 --text "여보세요, 밥은 먹었어?"   # 지연 실측
    GRADIO_SHARE=1 python -m studio                                          # 브라우저 통화

  ※ 새로 컴파일됐다면(프리빌트 폴백 등) — 이 바이너리를 올려두면 다음 인스턴스부턴 컴파일 0:
    tar czf /tmp/llama-bin.tgz -C "$(dirname "$LBIN")" .
    hf upload ksiwon/callone-llama-bin /tmp/llama-bin.tgz llama-bin.tgz   # HF write 토큰 로그인 필요
    # (LLAMA_SERVER_URL 기본값이 이 repo → 다음 인스턴스는 자동 다운. 드라이버 안 맞으면 자동 재빌드.)
EOF
