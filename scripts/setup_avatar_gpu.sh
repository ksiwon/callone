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
sudo apt-get update 2>/dev/null && sudo apt-get install -y \
  libgl1 libglx-mesa0 libgles2 libegl1 libglvnd0 libglib2.0-0 \
  libsm6 libxext6 libxrender1 libxi6 libxrandr2 libxfixes3 libgomp1 libsndfile1 ffmpeg \
  2>/dev/null || echo "[info] apt 생략(권한 없음/이미 있음) — 없으면 Ditto 로드 시 libGLESv2.so.2 등 에러"

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

# TensorRT/cuda-python 버전 고정(중요): 스캔이 최신 TRT(11.x)를 깔면 Elice 드라이버(535=CUDA12.2)엔
# 너무 최신이라 'CUDA error 35(insufficient driver)'. 프리빌트 Ampere 엔진도 TRT 8.6 빌드라 8.6.1 로 맞춤.
# cuda-python 은 옛 API('from cuda import cuda') 필요 → <12.9.
pip uninstall -y tensorrt tensorrt-libs tensorrt-bindings tensorrt_cu12 tensorrt_cu12_libs tensorrt_cu12_bindings >/dev/null 2>&1 || true
# tensorrt 메타패키지는 빌드 중 내부에서 pip 호출하다 격리env 서 깨짐 → libs/bindings 직접 설치 후
# 메타는 --no-build-isolation 으로(이미 만족된 의존성 재호출 안 함).
pip install --upgrade pip >/dev/null
pip install tensorrt-libs==8.6.1 tensorrt-bindings==8.6.1 "cuda-python<12.9" \
  nvidia-cudnn-cu12==8.9.7.29 --extra-index-url https://pypi.nvidia.com   # TRT8.6 은 cuDNN8 필요(토치는 cuDNN9)
pip install tensorrt==8.6.1 --no-build-isolation --extra-index-url https://pypi.nvidia.com || true  # 메타 실패해도 모듈 있으면 OK

# env 고정 — DittoModel 이 이걸로 SDK 로드.
# DATA_ROOT 는 aux_models/ 와 models/ 를 품은 디렉토리(=ditto_pytorch). aux_models(det_10g.onnx)의
# 부모로 자동 탐지(고정 경로 'ditto_pytorch/models' 는 한 단계 깊어 det_10g 못 찾음 → 실측 버그).
DATA_ROOT="$(find "$DITTO_REPO/checkpoints" -name det_10g.onnx 2>/dev/null | head -1 | xargs -r dirname | xargs -r dirname)"
[ -z "$DATA_ROOT" ] && DATA_ROOT="$DITTO_REPO/checkpoints/ditto_pytorch"
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
