#!/usr/bin/env bash
# Arc(XPU) 우선 실행 — XPU 안 잡히거나 합성 중 터지면 앱이 CPU 로 자동 폴백.
# 순수 CPU 로만 돌리려면 아래 COSYVOICE_DEVICE 줄을 주석처리(#) 하세요.
set -e
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate cosyvoice
cd ~/CosyVoice
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PORT=50000

# ===== 속도 튜닝 (실측으로 확정) =====
# (1) 디바이스: CPU 가 최速. Intel Arc 140V iGPU 는 자기회귀 TTS 에 오히려 더 느려서 끔.
#     (실측: CPU 8스레드 31초 vs iGPU 100초+ 미완료)
#     iGPU 실험은 README "선택: Arc iGPU(XPU) 실험" 부록대로 +xpu 휠/드라이버 먼저 깔고
#     COSYVOICE_DEVICE=xpu 로 바꿔야 함. (느리니 권장 안 함)
export COSYVOICE_DEVICE=cpu
# (2) 스레드 8개(P4+E4)가 4개보다 살짝 빠름 (실측: 8→31초, 4→35초)
export TTS_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export KMP_BLOCKTIME=1
# (3) 전사용 Whisper 경량 모델(②합성만 쓰면 안 돎). 정확도 더 원하면 large-v3 로.
export WHISPER_MODEL=large-v3-turbo

python app_studio.py
