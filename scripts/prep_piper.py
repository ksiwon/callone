"""화자 A TTS 데이터셋 → Piper 학습 형식 변환 (오프라인).

입력:  data/datasets/{spk}/tts/metadata.csv  ('wav경로|텍스트|길이|SNR')
       + data/datasets/{spk}/tts/wavs/*.wav  (48kHz 모노)
출력:  piper_ds/{spk}/wav/{id}.wav  (22050Hz 모노)  +  piper_ds/{spk}/metadata.csv ('id|text')

SNR/길이로 노이즈·너무 짧/긴 클립 거른다(음질↑). 22050 = Piper medium 표준.

사용:
  python scripts/prep_piper.py --speaker A
  python scripts/prep_piper.py --speaker A --min-snr 12 --min-dur 1.0 --max-dur 12
"""
from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callone.common.io import data_dir  # noqa: E402
from callone.common.logging import get_logger  # noqa: E402

log = get_logger("prep_piper")
TARGET_SR = 22050


def _resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y
    try:
        import soxr  # 고품질, numba 무관

        return soxr.resample(y, sr, target)
    except Exception:  # noqa: BLE001
        from scipy.signal import resample_poly  # type: ignore

        g = gcd(sr, target)
        return resample_poly(y, target // g, sr // g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="화자 A TTS셋 → Piper 형식")
    ap.add_argument("--speaker", default="A")
    ap.add_argument("--out", default="piper_ds")
    ap.add_argument("--min-snr", type=float, default=10.0)
    ap.add_argument("--min-dur", type=float, default=0.8)
    ap.add_argument("--max-dur", type=float, default=13.0)
    args = ap.parse_args()

    src = data_dir() / "datasets" / args.speaker / "tts"
    meta = src / "metadata.csv"
    if not meta.exists():
        log.error("메타데이터 없음: %s", meta)
        sys.exit(1)

    out = Path(args.out) / args.speaker
    wav_out = out / "wav"
    wav_out.mkdir(parents=True, exist_ok=True)

    kept, skipped, total_sec = 0, 0, 0.0
    rows_out = []
    for ln in meta.read_text(encoding="utf-8").splitlines():
        p = ln.split("|")
        if len(p) < 2:
            continue
        wav_path, text = p[0], p[1].strip()
        dur = float(p[2]) if len(p) > 2 and p[2] else 0.0
        snr = float(p[3]) if len(p) > 3 and p[3] else 0.0
        wp = Path(wav_path)
        if not wp.is_absolute() and not wp.exists():
            wp = src / "wavs" / wp.name           # 경로 보정
        if not wp.exists() or not text:
            skipped += 1
            continue
        if (dur and not (args.min_dur <= dur <= args.max_dur)) or snr < args.min_snr:
            skipped += 1
            continue
        try:
            y, sr = sf.read(str(wp), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)               # 모노
            y = _resample(y, sr, TARGET_SR)
            peak = float(np.max(np.abs(y))) or 1.0
            y = (y / peak * 0.95).astype(np.float32)   # 피크 정규화
            wid = wp.stem
            sf.write(str(wav_out / f"{wid}.wav"), y, TARGET_SR, subtype="PCM_16")
            rows_out.append(f"{wid}|{text}")
            kept += 1
            total_sec += len(y) / TARGET_SR
        except Exception as e:  # noqa: BLE001
            log.warning("스킵 %s (%s)", wp.name, e)
            skipped += 1

    (out / "metadata.csv").write_text("\n".join(rows_out) + "\n", encoding="utf-8")
    log.info("완료: %d 클립(%.1f분) → %s | 거름 %d", kept, total_sec / 60, out, skipped)
    log.info("다음: scripts/train_piper.md 의 preprocess→train→export 진행")


if __name__ == "__main__":
    main()
