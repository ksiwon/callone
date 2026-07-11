#!/usr/bin/env bash
# 파일럿: 50통 + 한 사람(A) 수직 관통 (§18 M1~M4 데이터 파이프라인)
set -euo pipefail

N="${1:-50}"
SPK="${2:-A}"

echo "[callone] 파일럿 시작 N=$N speaker=$SPK"

# ⚠️ 순서: 전사(transcribe)가 link 보다 먼저여야 global_assignment 에 text 가 들어간다.
# S1(복원)은 denoise 모델 미설치 시 무용 → 제외 (필요시 callone-restore 추가).
callone-ingest      --limit "$N"
callone-diarize     --limit "$N"
callone-transcribe  --limit "$N"   # diarized JSON 에 한국어 text 채움
callone-link        --limit "$N"   # parquet 에 text 복사 + A/B 귀속
callone-profile     --speakers "$SPK"   # 방언 자동측정(텍스트 필요)
callone-build-tts   --speakers "$SPK"
callone-build-dlg   --speakers "$SPK"
callone-llm-sft     --speakers "$SPK"

echo "[callone] 데이터 파이프라인 완료. 다음: 학습 단계(H100)"
echo "  callone-correct --hours 3   # ASR 교정셋 → 수동 교정 → callone-asr-train"
echo "  callone-llm-train  --config llm_server --speakers $SPK   # 말투 LoRA(목소리는 제로샷)"
echo "  callone-serve                # 실시간 서버"
