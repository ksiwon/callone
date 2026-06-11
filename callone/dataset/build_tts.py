"""S3(a) TTS 학습셋 빌드 (§12a, §7.5).

clean=true & overlap=false & snr≥임계 세그먼트만, 3~15초 컷 →
data/datasets/{spk}/tts/wavs/*.wav + metadata.csv (LJSpeech 류).
한국어 텍스트 정규화(숫자/기호→한글). PII 마스킹 적용.

사용:
  callone-build-tts --speakers A B
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..common import db
from ..common.audio import load_wav, save_wav
from ..common.io import data_dir, load_config, read_json
from ..common.logging import get_logger
from ..common.schemas import TTSRow
from ..asr.pii import mask_text

log = get_logger("build_tts")

_NUM = {"0": "영", "1": "일", "2": "이", "3": "삼", "4": "사",
        "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구"}


def normalize_ko(text: str) -> str:
    """간이 텍스트 정규화: 숫자→한글 음독, 기호 정리. (정밀 정규화는 g2pk 등 확장)"""
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 자리수 무시한 간이 음독 (TTS 코퍼스 텍스트 안정화용)
    text = re.sub(r"\d", lambda m: _NUM[m.group()], text)
    return text


def _assignments():
    ga = data_dir() / "speakers" / "global_assignment.parquet"
    if ga.exists():
        import pandas as pd

        return pd.read_parquet(ga).to_dict("records")
    if ga.with_suffix(".json").exists():
        return read_json(ga.with_suffix(".json"))
    return []


def run(cfg: dict, speakers: list[str]) -> None:
    tcfg = cfg.get("tts", {})
    min_snr = tcfg.get("min_snr_db", 14)
    min_s, max_s = tcfg.get("min_sec", 3.0), tcfg.get("max_sec", 15.0)
    pii_tokens = cfg.get("pii", {}).get("mask_tokens")

    con = db.connect()
    call_wav = {c.call_id: (c.restored_path or c.wav16k_path) for c in db.all_calls(con)}
    rows = _assignments()

    for spk in speakers:
        out_dir = data_dir() / "datasets" / spk / "tts"
        wav_dir = out_dir / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        meta, total_dur, idx = [], 0.0, 0

        for r in rows:
            if r["global_speaker"] != spk:
                continue
            if tcfg.get("require_clean", True) and not r.get("clean", False):
                continue
            if r.get("is_overlap"):
                continue
            dur = r["end"] - r["start"]
            if not (min_s <= dur <= max_s):
                continue
            snr = r.get("snr_db", 0.0)
            if snr and snr < min_snr:
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            wav = call_wav.get(r["call_id"])
            if not wav or not Path(wav).exists():
                continue
            try:
                y, sr = load_wav(wav, sr=48000)  # TTS는 복원본(고sr) 기준
                clip = y[int(r["start"] * sr): int(r["end"] * sr)]
                idx += 1
                cp = wav_dir / f"{spk}_{idx:06d}.wav"
                save_wav(cp, clip, sr)
            except Exception:  # noqa: BLE001
                continue
            norm = normalize_ko(text)
            if cfg.get("pii", {}).get("enabled", True):
                norm = mask_text(norm, pii_tokens)
            meta.append(TTSRow(wav_path=str(cp), text=norm, duration=round(dur, 2),
                               snr=round(snr, 1)))
            total_dur += dur

        csv_path = out_dir / "metadata.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            for m in meta:
                f.write(m.to_csv_line() + "\n")
        log.info("화자 %s TTS셋: %d clip / %.1f분 → %s",
                 spk, len(meta), total_dur / 60, csv_path)
        lo, hi = tcfg.get("target_hours", [5, 20])
        if total_dur / 3600 < lo:
            log.warning("  ⚠️ 목표 %dh 미달 — 데이터 더 필요할 수 있음", lo)


def main() -> None:
    ap = argparse.ArgumentParser(description="S3(a) TTS 학습셋 빌드")
    ap.add_argument("--config", default="s3_dataset")
    ap.add_argument("--speakers", nargs="+", default=["A", "B"])
    args = ap.parse_args()
    run(load_config(args.config), args.speakers)


if __name__ == "__main__":
    main()
