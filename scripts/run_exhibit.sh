#!/usr/bin/env bash
# 전시 원버튼 기동 (EXHIBIT_PLAN §6-9) — 아침에 이 스크립트 하나.
# run_all.sh(모델 서버들 + serve) 띄우고, 키오스크/업로드 주소를 출력한다.
# GPIO 브리지는 라즈베리파이에서 별도: python scripts/exhibit_gpio_bridge.py --server http://<box>:8000
set -e
cd "$(dirname "$0")/.."

bash scripts/run_all.sh

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
IP=${IP:-127.0.0.1}
echo
echo "================ call:one 전시 모드 ================"
echo "  키오스크(부스 브라우저 전체화면):  http://${IP}:8000/kiosk"
echo "  폰 업로드(트랙② QR 목적지):       http://${IP}:8000/upload"
echo "  소멸 카운터(벽면 표시용 API):      http://${IP}:8000/api/exhibit/count"
echo "  통화 시간 조정: 키오스크 콘솔에서 localStorage.callone_kiosk_limit='110'"
echo "===================================================="
