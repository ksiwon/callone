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

echo "=== [0] 시스템 라이브러리(GL/X11/오디오) — Ditto 렌더·cv2 가 요구(libGLESv2.so.2 등) ==="
# root 컨테이너(RunPod)는 sudo 없음 → apt-get 직접. 이게 스킵되면 libGLESv2.so.2 누락으로
# Ditto 가 로드 실패→static 폴백(얼굴 안 움직임). sudo 있으면 sudo, 둘 다 없으면 안내.
if command -v sudo >/dev/null 2>&1; then APT="sudo apt-get"; elif command -v apt-get >/dev/null 2>&1; then APT="apt-get"; else APT=""; fi
if [ -n "$APT" ]; then
  $APT update 2>/dev/null || true
  # ⚠️ libGLESv2.so.2 는 libgles2 가 아니라 **libgles2-mesa** 가 제공한다(실측). 둘 다 넣는다.
  $APT install -y libgl1 libglx-mesa0 libgles2-mesa libgles2 libegl1 libglvnd0 libglib2.0-0 \
    libsm6 libxext6 libxrender1 libxi6 libxrandr2 libxfixes3 libgomp1 libsndfile1 ffmpeg \
    || echo "[info] apt 일부 실패 — libGLESv2.so.2(libgles2-mesa) 없으면 Ditto 가 static 으로 폴백"
  ldconfig 2>/dev/null || true   # 새 .so 캐시 갱신(설치 후 곧바로 로더가 찾게)
else
  echo "[info] apt 없음 — GL 라이브러리 수동 설치 필요(없으면 Ditto static 폴백)"
fi

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
# Ditto environment.yaml pip 의존성 전체(PyTorch 백엔드 기준 — TRT 전용 tensorrt/polygraphy 제외).
pip install librosa opencv-python-headless imageio imageio-ffmpeg scikit-image tqdm "numpy<2.4" \
  filetype cython cuda-python colored numba scikit-learn scipy audioread soxr pooch \
  lazy-loader joblib msgpack tifffile decorator llvmlite onnxruntime
# Ditto repo 자체 requirements 가 있으면 추가(저자 의도 — 버전 충돌나면 위 핀 유지).
[ -f "$DITTO_REPO/requirements.txt" ] && pip install -r "$DITTO_REPO/requirements.txt" || true

# 자가치유: Ditto repo 의 모든 import 를 스캔 → venv 에 없는 모듈 전부 자동 설치(연쇄 누락 한방 해결).
echo "=== [3.5] Ditto import 스캔 → 누락 모듈 자동 설치 ==="
( cd "$DITTO_REPO" && python - <<'PY'
import ast, os, importlib.util, subprocess, sys
PIPMAP = {'cv2':'opencv-python-headless','PIL':'pillow','skimage':'scikit-image',
          'sklearn':'scikit-learn','yaml':'pyyaml'}
mods = set()
for root, _, files in os.walk('.'):
    if '/.git' in root or '/.pyxbld' in root:
        continue
    for f in files:
        if not f.endswith('.py'):
            continue
        try:
            t = ast.parse(open(os.path.join(root, f), encoding='utf-8', errors='ignore').read())
        except Exception:
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                for a in n.names:
                    mods.add(a.name.split('.')[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module and n.level == 0:
                    mods.add(n.module.split('.')[0])
missing = [m for m in sorted(mods) if importlib.util.find_spec(m) is None]
pips = sorted({PIPMAP.get(m, m) for m in missing})
print("누락:", missing, "→ 설치:", pips)
# tensorrt 계열은 PyPI 소스스텁이 빌드 깨짐 → NVIDIA prebuilt 인덱스 추가(전체에 줘도 무해).
NV = ['--extra-index-url', 'https://pypi.nvidia.com']
for p in pips:
    subprocess.run([sys.executable, '-m', 'pip', 'install', p, *NV])
PY
) || echo "[warn] import 스캔 생략(계속)"

# ── GPU 아키텍처 분기(중요) ───────────────────────────────────────────────
# 얼굴 움직임 = **TRT 경로**(A100 검증). PyTorch 온라인 경로는 비-Ampere에서 0프레임(실측)이라 안 씀.
#   - Ampere(A100·3090·3090Ti, cc 8.0/8.6): 프리빌트 엔진 ditto_trt_Ampere_Plus 그대로.
#   - Ada(4090, 8.9)·Hopper(9.0): 프리빌트 비호환 → **cvt_onnx_to_trt 로 이 GPU용 custom 엔진 빌드**(repo 공식 방법).
# 둘 다 TRT 8.6.1 + cuDNN8(torch cuDNN9 와 .so 버전 달라 공존, 추론은 TRT). cuda-python<12.9(import 호환).
CK="$DITTO_REPO/checkpoints"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
USE_TRT=0; BUILD_TRT=0
case "$CC" in
  8.0|8.6) USE_TRT=1 ;;              # Ampere → 프리빌트 엔진
  8.9|9.0) USE_TRT=1; BUILD_TRT=1 ;; # Ada(4090)/Hopper → 프리빌트 비호환, custom 빌드
esac
pip install --upgrade pip >/dev/null
pip install "cuda-python<12.9" --extra-index-url https://pypi.nvidia.com   # 공통(import 호환)
if [ "$USE_TRT" = 1 ]; then
  # tensorrt 메타패키지는 격리빌드서 깨짐 → libs/bindings 직접 + 메타 --no-build-isolation.
  pip uninstall -y tensorrt tensorrt-libs tensorrt-bindings tensorrt_cu12 tensorrt_cu12_libs tensorrt_cu12_bindings >/dev/null 2>&1 || true
  pip install tensorrt-libs==8.6.1 tensorrt-bindings==8.6.1 nvidia-cudnn-cu12==8.9.7.29 --extra-index-url https://pypi.nvidia.com
  pip install tensorrt==8.6.1 --no-build-isolation --extra-index-url https://pypi.nvidia.com || true
  [ "$BUILD_TRT" = 1 ] \
    && echo "  [GPU] 비-Ampere(cc=$CC) → custom TRT 엔진 빌드 예정(cvt_onnx_to_trt). 프리빌트는 Ampere 전용." \
    || echo "  [GPU] Ampere(cc=$CC) → 프리빌트 TRT 8.6.1 + cuDNN8 엔진."
else
  echo "  [GPU] cc=${CC:-?} 미인식 → PyTorch 폴백(애니 0프레임 가능). AVATAR_BACKEND=static 권장."
fi

# 비-Ampere: 이 GPU용 TRT 엔진을 일반 ONNX 에서 빌드(프리빌트 대체, repo scripts/cvt_onnx_to_trt.py). ~수십분, 1회.
if [ "$BUILD_TRT" = 1 ] && [ ! -d "$CK/ditto_trt_custom" ]; then
  if [ -d "$CK/ditto_onnx" ] && [ -f "$DITTO_REPO/scripts/cvt_onnx_to_trt.py" ]; then
    pip install onnx >/dev/null 2>&1 || true   # cvt 가 onnx 파서 사용(import 스캔서 빠질 수 있어 보강)
    echo "=== [GPU] custom TRT 엔진 빌드 중 (cvt_onnx_to_trt, 수십분) ... ==="
    ( cd "$DITTO_REPO" && python scripts/cvt_onnx_to_trt.py --onnx_dir "$CK/ditto_onnx" --trt_dir "$CK/ditto_trt_custom" ) \
      && echo "  ✅ custom TRT 엔진 완료: $CK/ditto_trt_custom" \
      || { echo "  ⚠️ custom TRT 빌드 실패 — TRT 버전/드라이버 호환 확인 필요. 우선 AVATAR_BACKEND=static(음성+정지사진)."; rm -rf "$CK/ditto_trt_custom" 2>/dev/null || true; }
  else
    echo "  ⚠️ ditto_onnx 또는 cvt_onnx_to_trt.py 없음 → custom 빌드 스킵. AVATAR_BACKEND=static 권장."
  fi
fi

# env 고정 — DATA_ROOT 우선순위: Ampere 프리빌트 > custom(비-Ampere) > PyTorch(폴백). 모두 TRT online cfg.
TRT_CFG="$CK/ditto_cfg/v0.4_hubert_cfg_trt_online.pkl"
if [ "$USE_TRT" = 1 ] && [ "$BUILD_TRT" != 1 ] && [ -d "$CK/ditto_trt_Ampere_Plus" ] && [ -f "$TRT_CFG" ]; then
  DATA_ROOT="$CK/ditto_trt_Ampere_Plus"; CFG_PKL="$TRT_CFG"   # Ampere(USE_TRT=1, 빌드 안 함) 만
  BACKEND_LABEL="Ditto TensorRT(Ampere 프리빌트, cc=$CC) — 움직임 O"
elif [ -d "$CK/ditto_trt_custom" ] && [ -f "$TRT_CFG" ]; then
  DATA_ROOT="$CK/ditto_trt_custom"; CFG_PKL="$TRT_CFG"
  BACKEND_LABEL="Ditto TensorRT(custom 빌드, cc=$CC) — 움직임 O"
else
  # PyTorch 최후 폴백(비-Ampere custom 빌드 실패 시). 0프레임 가능 → static 권장.
  DATA_ROOT="$(find "$CK" -name det_10g.onnx 2>/dev/null | head -1 | xargs -r dirname | xargs -r dirname)"
  [ -z "$DATA_ROOT" ] && DATA_ROOT="$CK/ditto_pytorch"
  CFG_PKL="$(find "$CK" -name '*cfg*pytorch*.pkl' 2>/dev/null | grep -i hubert | head -1)"
  [ -z "$CFG_PKL" ] && CFG_PKL="$(find "$CK" -name '*pytorch*.pkl' 2>/dev/null | head -1)"
  BACKEND_LABEL="Ditto PyTorch(폴백, cc=${CC:-?}) — ⚠️ 비-Ampere 애니 0프레임 가능, AVATAR_BACKEND=static 권장"
fi
echo "  Ditto 백엔드: $BACKEND_LABEL"
sed -i '/export DITTO_REPO=/d;/export DITTO_DATA_ROOT=/d;/export DITTO_CFG_PKL=/d' ~/.bashrc
{ echo "export DITTO_REPO=$DITTO_REPO"
  echo "export DITTO_DATA_ROOT=$DATA_ROOT"
  echo "export DITTO_CFG_PKL=$CFG_PKL"; } >> ~/.bashrc
export DITTO_REPO DITTO_DATA_ROOT="$DATA_ROOT" DITTO_CFG_PKL="$CFG_PKL"

echo "=== [4] torch GPU 동작 검증(드라이버 호환) ==="
python - <<'PY' || echo "⚠️ torch GPU 실패 — 드라이버 호환 CUDA 로 재설치 필요(서빙과 동일 이슈)"
import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "OK")
PY

cat <<EOF

=== avatar-server 준비 완료($BACKEND_LABEL) ===
  DITTO_DATA_ROOT=$DATA_ROOT
  DITTO_CFG_PKL=$CFG_PKL
  기동:   AVATAR_BACKEND=ditto python -m avatar_server --port 8091
          curl http://127.0.0.1:8091/health   → {"status":"ok","backend":"ditto"}
  callone 연동: serve.yaml avatar.enabled=true (base_url=http://127.0.0.1:8091),
               증명사진 data/speakers/<화자>/portrait.jpg|png 두기.
  ⚠️ DittoModel.frames 의 [V] 검증지점(setup_Nd/청크크기/writer attr)을 repo 코드로 맞춰라.
     안 맞으면 _pick_model 이 static(정지사진)으로 폴백 — 통화·음성은 정상.
EOF
