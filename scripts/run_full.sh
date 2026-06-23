#!/usr/bin/env bash
# 전량 처리 (§18 M8) — 1,000통 전체 + 두 화자(A,B)
set -euo pipefail

echo "[callone] 전량 처리 시작 (limit 없음)"

# ⚠️ 순서: 전사(transcribe) → 연결(link) → 프로필. link 가 먼저면 parquet text 가 빔.
# S1(복원)은 denoise 모델 미설치 시 무용 → 제외 (필요시 callone-restore 추가).
callone-ingest
callone-diarize
callone-transcribe                  # diarized 에 text 채움
callone-link                        # parquet 에 text 복사 + A/B 귀속
callone-profile     --speakers A B  # 방언 자동측정
callone-build-tts   --speakers A B
callone-build-dlg   --speakers A B
callone-llm-sft     --speakers A B

echo "[callone] 전량 데이터셋 완료. 화자별 학습:"
echo "  callone-asr-train"
echo "  bash scripts/setup_piper_gpu.sh   # 목소리 학습(Piper)"
echo "  callone-llm-train  --config llm_server --speakers A B"
