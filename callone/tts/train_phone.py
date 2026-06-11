"""S4 폰 TTS 학습 (§14) — per-speaker 소형 단일화자 모델.

Piper/VITS(초경량) · MeloTTS(한국어 CPU 실시간) · GPT-SoVITS · Kokoro+KokoClone.
동일 정제 코퍼스(metadata.csv) 재사용. → models/tts_phone/{spk}.
온디바이스 TTS 클론이 가장 까다로움 → 데이터 풍부 살린 전용 모델이 정답(§17.2).

백엔드 repo 무거움 → 미설치 시 학습 레시피 안내(폴백).

사용:
  callone-tts-phone --speakers A
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..common.io import data_dir, load_config
from ..common.logging import get_logger

log = get_logger("tts_phone")


def run(cfg: dict, speakers: list[str]) -> None:
    backend = cfg.get("backend", "piper")
    out_base = Path(cfg.get("output_dir", "models/tts_phone"))
    for spk in speakers:
        csv = data_dir() / "datasets" / spk / "tts" / "metadata.csv"
        if not csv.exists():
            log.error("TTS셋 없음: %s — callone-build-tts 먼저", csv)
            continue
        out = out_base / spk
        out.mkdir(parents=True, exist_ok=True)
        log.info(
            "[폰 TTS 학습 레시피 backend=%s spk=%s]\n"
            "  데이터: %s (%dHz 재샘플 필요할 수 있음)\n"
            "  Piper: preprocess → train (espeak-ng phonemizer, 단일화자)\n"
            "  MeloTTS: 한국어 베이스에서 per-speaker FT, CPU 실시간\n"
            "  출력: %s → 폰(LiteRT/MediaPipe/ggml) 변환\n"
            "  H100 또는 로컬 GPU 에서 repo 스크립트 실행.",
            backend, spk, csv, cfg.get("sample_rate", 22050), out,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 폰 TTS 학습")
    ap.add_argument("--config", default="tts_phone")
    ap.add_argument("--speakers", nargs="+", default=["A"])
    args = ap.parse_args()
    run(load_config(args.config), args.speakers)


if __name__ == "__main__":
    main()
