#!/usr/bin/env bash
# =============================================================================
# CosyVoice3 TTS 서버 셋업(GPU) — callone 음색 안정 백엔드(:8092)
# 별 conda env(cosyvoice) + CosyVoice 클론 + 9.75GB 모델 + fastapi 서버.
# 실행:  bash scripts/setup_cosyvoice_gpu.sh           (최초 1회, 수십 분)
#        bash scripts/setup_cosyvoice_gpu.sh run       (서버만 기동)
# 멱등: 이미 된 단계는 건너뜀. 인스턴스 갈아엎어도 이거 한 줄.
#
# ⚠️ env 격리: 'conda activate' 가 스크립트(특히 RunPod base 자동활성) 컨텍스트에서 종종 base 로
#    새서, 의존성이 cosyvoice 가 아니라 base(py3.11)에 깔리고 base 의 깨진 flash_attn 까지 끌려오는
#    문제가 있었다 → **conda run -n cosyvoice 로 env 를 못 박는다**(activate 미사용).
# =============================================================================
set -e
CALLONE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSY_DIR="$HOME/CosyVoice"
MODEL_DIR="$COSY_DIR/pretrained_models/Fun-CosyVoice3-0.5B"
PORT="${PORT:-8092}"

_conda() { source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"; }
# cosyvoice env 에서 명령 실행(activate 안 씀 — env 못 박기). --no-capture-output: 긴 설치 로그 실시간 표시.
CRUN() { conda run -n cosyvoice --no-capture-output "$@"; }

run_server() {
  _conda
  cd "$COSY_DIR"
  export PYTHONPATH="$COSY_DIR/third_party/Matcha-TTS:$PYTHONPATH"
  export COSYVOICE_MODEL_DIR="pretrained_models/Fun-CosyVoice3-0.5B"
  export PORT="$PORT"
  echo "==> cosyvoice_server :$PORT 기동 (모델=$COSYVOICE_MODEL_DIR)"
  exec conda run -n cosyvoice --no-capture-output python "$CALLONE_DIR/cosyvoice_server/app.py"
}

if [ "$1" = "run" ]; then run_server; fi

echo "==> [1/6] 시스템 패키지(sox/ffmpeg/git-lfs)"
# root(컨테이너)면 sudo 없음 → apt-get 직접. sudo 있으면 sudo. 둘 다 없으면 생략.
if command -v sudo >/dev/null 2>&1; then APT="sudo apt-get"; elif command -v apt-get >/dev/null 2>&1; then APT="apt-get"; else APT=""; fi
if [ -n "$APT" ]; then
  $APT update && $APT install -y git git-lfs sox libsox-dev ffmpeg wget build-essential || \
    echo "[info] apt 일부 실패 — git-lfs/모델은 huggingface_hub 로 받으므로 보통 무방"
else
  echo "[info] apt 없음 — 생략(모델은 huggingface_hub 로 받음)"
fi
git lfs install 2>/dev/null || true

echo "==> [2/6] conda + CosyVoice 클론"
if ! command -v conda >/dev/null 2>&1 && [ ! -d "$HOME/miniconda3" ]; then
  ( wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    || curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh )
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
_conda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
[ -d "$COSY_DIR" ] || git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSY_DIR"
cd "$COSY_DIR"; git submodule update --init --recursive

echo "==> [3/6] conda env(cosyvoice, py3.10)"
# 이미 있으면 스킵. 없으면 conda-forge 명시로 생성(에러를 숨기지 않는다 — 실패 시 원인이 보여야 함).
if ! conda env list | grep -qE '/cosyvoice$|^cosyvoice[[:space:]]'; then
  conda create -n cosyvoice -y -c conda-forge python=3.10 \
    || { echo "❌ conda env 생성 실패 — 위 에러 확인(흔한 원인: 컨테이너 디스크 부족 'df -h /', 네트워크/SSL)."; exit 1; }
fi
CRUN python -V \
  || { echo "❌ cosyvoice env 사용 불가(conda run) — env 생성 확인."; exit 1; }
CRUN pip install --upgrade pip
CRUN pip install "setuptools<81" wheel

echo "==> [4/6] 의존성(빌드지옥 제외) + fastapi 서버"
# ⚠️ pynini/openfst 는 pip 빌드가 OpenFst 컴파일에서 잘 깨진다 → conda-forge 프리빌트로 cosyvoice env 에 먼저.
#    이게 안 깔리면 'pip install -r requirements' 가 pynini 에서 set -e 로 죽어 hyperpyyaml/conformer
#    등 CosyVoice 본체 의존성이 통째로 안 깔리고 → 서버가 import 단계서 하나씩 죽는다(근본원인).
conda install -n cosyvoice -y -c conda-forge "pynini==2.1.5" openfst 2>/dev/null \
  || CRUN pip install pynini==2.1.5 2>/dev/null \
  || echo "[warn] pynini 설치 실패 — 텍스트정규화 일부 제한(서버 기동엔 보통 무방)"
# requirements 에서: 빌드지옥(tensorrt/gpu/deepspeed) · 별도설치(openai-whisper) · conda로 깐 pynini 제외
grep -vE 'tensorrt|onnxruntime-gpu|deepspeed|openai-whisper|^pynini' requirements.txt > requirements_clean.txt
CRUN pip install -r requirements_clean.txt
CRUN pip install onnxruntime
CRUN pip install soundfile huggingface_hub hf_transfer faster-whisper noisereduce pydub  # hf_transfer: RunPod HF_HUB_ENABLE_HF_TRANSFER=1 대비
# CosyVoice frontend.py 가 'import whisper'(openai-whisper) 필요 — requirements_clean 에서 제외됐으므로 별도 설치.
# ⚠️ openai-whisper 가 torch 를 끌어내리므로(2.6→2.3) **반드시 torch 고정 전에** 깐다.
CRUN pip install openai-whisper==20231117 --no-build-isolation || CRUN pip install openai-whisper --no-build-isolation || CRUN pip install openai-whisper
CRUN pip install fastapi uvicorn pydantic           # callone cosyvoice_server 용
# flash_attn: RunPod 베이스 등에서 끌려와 ABI 깨짐(undefined symbol)으로 transformers import 를 죽인다.
# CosyVoice 엔 불필요(없으면 표준 attention) → cosyvoice env 에 있으면 제거.
CRUN pip uninstall -y flash-attn flash_attn 2>/dev/null || true
# torch 2.6.0 cu124 고정 — whisper 가 깎은 torch 복구. ⚠️ --no-deps 는 쓰지 않는다:
#   쓰면 CUDA 런타임 lib 이 옛 cu121 버전(cublas 12.1·cudnn 8.9·cusparselt 등)으로 남아 torch
#   2.6(cu124)이 'libcusparseLt.so.0 없음' 등으로 죽는다 → deps 포함 force-reinstall 로 cu124
#   nvidia-* lib(cublas 12.4·cudnn 9.1·cusparselt 0.6.2·triton 3.2 …) 셋을 통째로 정합.
CRUN pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
CRUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, '· cuda_avail', torch.cuda.is_available())" \
  || { echo "❌ torch 로드 실패 — 위 ImportError 의 CUDA lib 확인."; exit 1; }
# ★의존성 완전성 게이트★ — cosyvoice env 에서 AutoModel import 되면 서버 기동 보장. 안 되면 여기서 멈춰 원인 표시.
export PYTHONPATH="$COSY_DIR/third_party/Matcha-TTS"
( cd "$COSY_DIR" && CRUN python -c "from cosyvoice.cli.cosyvoice import AutoModel" ) \
  && echo "  ✅ CosyVoice import 체인 OK — 서버 기동 가능" \
  || { echo "❌ CosyVoice import 실패 — 위 마지막 에러의 모듈 확인(보통 pynini/hyperpyyaml/conformer). 그 모듈 'conda run -n cosyvoice pip install <모듈>' 후 재실행."; exit 1; }

# RunPod 이미지가 HF_HUB_ENABLE_HF_TRANSFER=1 을 전역 설정 → hf_transfer 패키지 없으면 다운로드가 크래시.
# 설치돼 있으면 빠른 다운로드 유지, 없으면 비활성(일반 다운로드로 폴백). 설치 의존성 없이 확실히 동작.
CRUN python -c "import hf_transfer" 2>/dev/null || { echo "[info] hf_transfer 없음 → HF_HUB_ENABLE_HF_TRANSFER=0(일반 다운로드)"; export HF_HUB_ENABLE_HF_TRANSFER=0; }

echo "==> [5/6] 모델 다운로드(Fun-CosyVoice3-0.5B-2512, ~9.75GB)"
# 존재 판정: 실제 모델은 cosyvoice3.yaml + config.json 을 갖는다(구버전 cosyvoice.yaml 도 허용).
if [ ! -f "$MODEL_DIR/cosyvoice3.yaml" ] && [ ! -f "$MODEL_DIR/cosyvoice.yaml" ] && [ ! -f "$MODEL_DIR/config.json" ]; then
  mkdir -p "$COSY_DIR/pretrained_models"
  CRUN python - <<PY
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='$MODEL_DIR')
print("모델 다운로드 완료")
PY
else
  echo "  모델 이미 있음 — 스킵"
fi

echo "==> [6/6] 완료"
echo "============================================================"
echo " CosyVoice3 서버 기동:  bash scripts/setup_cosyvoice_gpu.sh run"
echo " 또는 백그라운드:       nohup bash scripts/setup_cosyvoice_gpu.sh run > ~/cosyvoice.log 2>&1 &"
echo " callone 은 serve.yaml tts.backend: cosyvoice3 로 :$PORT 호출"
echo "============================================================"
