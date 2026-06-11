"""S13 수동교정셋 생성 (§13).

본인 통화에서 2~5h 분량 세그먼트를 다양성(화자/주제) 기준 샘플링 →
사람이 전사 수동 교정할 CSV 생성. 교정 완료 CSV 가 whisper_finetune 입력.

출력: data/datasets/asr_correction/to_correct.csv
   컬럼: seg_uid, call_id, start, end, wav_clip, asr_text, corrected_text(빈칸)

사용:
  callone-correct --hours 3
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from ..common.audio import load_wav, save_wav
from ..common.io import data_dir, load_config, read_json
from ..common.logging import get_logger

log = get_logger("correct")


def _all_clean_segments():
    ga = data_dir() / "speakers" / "global_assignment.parquet"
    rows = []
    if ga.exists():
        import pandas as pd

        df = pd.read_parquet(ga)
        rows = df[df["clean"]].to_dict("records")
    elif ga.with_suffix(".json").exists():
        rows = [r for r in read_json(ga.with_suffix(".json")) if r.get("clean")]
    return rows


def run(cfg: dict, hours: float) -> None:
    rows = _all_clean_segments()
    if not rows:
        log.warning("세그먼트 없음 — S2 먼저 실행")
        return
    random.seed(42)
    random.shuffle(rows)

    cset = cfg.get("correction_set", {})
    min_s, max_s = cset.get("min_seg_sec", 3.0), cset.get("max_seg_sec", 15.0)
    budget = hours * 3600
    out_dir = data_dir() / "datasets" / "asr_correction"
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    con_calls = {}
    selected, acc = [], 0.0
    from ..common import db

    con = db.connect()
    for c in db.all_calls(con):
        con_calls[c.call_id] = c.restored_path or c.wav16k_path

    for r in rows:
        dur = r["end"] - r["start"]
        if not (min_s <= dur <= max_s):
            continue
        wav = con_calls.get(r["call_id"])
        if not wav or not Path(wav).exists():
            continue
        try:
            y, sr = load_wav(wav, sr=16000)
            clip = y[int(r["start"] * sr): int(r["end"] * sr)]
            cp = clip_dir / f"{r['segment_uid'].replace('#', '_')}.wav"
            save_wav(cp, clip, sr)
        except Exception:  # noqa: BLE001
            continue
        selected.append({**r, "wav_clip": str(cp)})
        acc += dur
        if acc >= budget:
            break

    csv_path = out_dir / "to_correct.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seg_uid", "call_id", "start", "end", "wav_clip", "asr_text", "corrected_text"])
        for r in selected:
            w.writerow([r["segment_uid"], r["call_id"], r["start"], r["end"],
                        r["wav_clip"], r.get("text", ""), ""])
    log.info("교정셋 %d 세그먼트 (%.1f분) → %s (corrected_text 칸 채우기)",
             len(selected), acc / 60, csv_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="ASR 수동교정셋 생성")
    ap.add_argument("--config", default="asr_adapt")
    ap.add_argument("--hours", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    hours = args.hours or cfg.get("correction_set", {}).get("hours", 3.0)
    run(cfg, hours)


if __name__ == "__main__":
    main()
