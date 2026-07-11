#!/usr/bin/env python3
"""pick_ref_clip — 긴 음성에서 **제로샷 클론에 최적인 레퍼런스 클립을 자동 추출**.

왜?  제로샷 품질은 레퍼런스 클립이 좌우한다(깨끗·단일화자·6~12초·전사 있음).
     긴 녹음/통화본에서 사람이 귀로 찾는 대신, 파이프라인 산출물(화자분리+SNR+전사)
     또는 에너지 VAD 로 후보를 점수화해 최고 클립을 뽑는다.

두 가지 입력 모드:
  ① 통화본(두 화자) — 파이프라인 산출물 사용(화자분리 완료 상태여야 함):
       callone-pilot --stages s0 s2 s3 s2b     # (최초 1회) 분리+전사
       python scripts/pick_ref_clip.py --speaker A --name mom
     → global_assignment.parquet 에서 화자 A 의 clean(비겹침·SNR≥) 세그먼트만 후보.
       전사(text)도 parquet 에서 그대로 가져와 ref_text 로 저장(ICL 유사도↑).

  ② 단일 화자 긴 파일 — 파이프라인 불필요:
       python scripts/pick_ref_clip.py --wav long_recording.m4a --name mom
     → 에너지 VAD 로 발화 구간을 찾아 6~12초 창 후보 생성, SNR+길이 점수화.
       faster-whisper 있으면 전사도 자동(--no-transcribe 로 끔).

출력: data/voice_presets/<name>.wav (+ <name>.txt 전사)
  → UI '준비된 목소리' 드롭다운에 **자동 노출**(voice_presets 레지스트리), 통화에서 바로 선택.
  --top N 이면 <name>.wav, <name>_2.wav ... 후보 N개 저장(들어보고 고르기).

점수 = SNR(60%) + 길이 적합도(40%, 9초 정점). 겹침발화·짧은/긴 구간 제외.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callone.common.audio import estimate_snr_db, load_wav, save_wav  # noqa: E402
from callone.common.io import data_dir  # noqa: E402

OUT_SR = 24000          # 프리셋 저장 sr(서빙이 어차피 16k 리샘플, 원음 보존 겸 24k)
IDEAL_SEC = 9.0         # 제로샷 레퍼런스 최적 길이(6~12s 중앙)


def _score(snr_db: float, dur: float, min_s: float, max_s: float) -> float:
    if not (min_s <= dur <= max_s):
        return -1.0
    snr_n = min(max(snr_db, 0.0), 25.0) / 25.0          # 25dB 이상은 동급
    dur_n = 1.0 - min(abs(dur - IDEAL_SEC) / IDEAL_SEC, 1.0)
    return 0.6 * snr_n + 0.4 * dur_n


# ── 모드 ①: 파이프라인 산출물(통화본, 화자분리 완료) ──────────────────────────
def candidates_from_pipeline(speaker: str, min_s: float, max_s: float) -> list[dict]:
    ga = data_dir() / "speakers" / "global_assignment.parquet"
    if not ga.exists():
        raise SystemExit(f"파이프라인 산출물 없음: {ga}\n먼저: callone-pilot --stages s0 s2 s3 s2b")
    import pandas as pd

    from callone.common import db

    con = db.connect()
    call_wav = {c.call_id: (c.restored_path or c.wav16k_path) for c in db.all_calls(con)}
    rows = pd.read_parquet(ga).to_dict("records")
    out = []
    for r in rows:
        if r["global_speaker"] != speaker or not r.get("clean", False) or r.get("is_overlap"):
            continue                                     # 단일화자·비겹침·SNR 통과분만
        dur = r["end"] - r["start"]
        sc = _score(r.get("snr_db", 0.0), dur, min_s, max_s)
        if sc < 0:
            continue
        wav = call_wav.get(r["call_id"])
        if not wav or not Path(wav).exists():
            continue
        out.append({"wav": wav, "start": r["start"], "end": r["end"],
                    "text": (r.get("text") or "").strip(), "score": sc,
                    "snr": r.get("snr_db", 0.0), "dur": dur})
    return out


# ── 모드 ②: 단일 화자 긴 파일(VAD 창 후보) ───────────────────────────────────
def candidates_from_wav(path: str, min_s: float, max_s: float) -> list[dict]:
    import librosa

    y, sr = load_wav(path, sr=16000)
    intervals = librosa.effects.split(y, top_db=30)      # 발화 구간(에너지 VAD)
    # 인접 구간(무음 <0.5s)을 이어붙여 min_s~max_s 창 후보 생성
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        st, en = s / sr, e / sr
        if merged and st - merged[-1][1] < 0.5:
            merged[-1] = (merged[-1][0], en)
        else:
            merged.append((st, en))
    out = []
    for st, en in merged:
        if en - st < min_s:
            continue
        # 긴 구간은 max_s 창으로 쪼개 후보 여러 개(2초 간격 슬라이드)
        starts = np.arange(st, max(st + 0.01, en - min_s), 2.0)
        for w0 in starts:
            w1 = min(w0 + max_s, en)
            dur = w1 - w0
            clip = y[int(w0 * sr): int(w1 * sr)]
            sc = _score(estimate_snr_db(clip), dur, min_s, max_s)
            if sc >= 0:
                out.append({"wav": path, "start": float(w0), "end": float(w1),
                            "text": "", "score": sc,
                            "snr": estimate_snr_db(clip), "dur": dur})
    return out


def _transcribe(y: np.ndarray, sr: int) -> str:
    """전사(선택) — faster-whisper 있으면. 실패 시 빈 문자열(서버가 자동전사 폴백)."""
    try:
        from faster_whisper import WhisperModel

        m = WhisperModel("large-v3-turbo" if _has_cuda() else "small",
                         device="cuda" if _has_cuda() else "cpu",
                         compute_type="float16" if _has_cuda() else "int8")
        segs, _ = m.transcribe(y, language="ko", beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segs).strip()
    except Exception as e:  # noqa: BLE001
        print(f"  (전사 스킵: {e} — 서버가 통화 시작 시 자동전사)")
        return ""


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="제로샷 레퍼런스 클립 자동 추출")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--speaker", help="파이프라인 화자(A/B) — 통화본 모드(분리 완료 필요)")
    src.add_argument("--wav", help="단일 화자 긴 파일 — VAD 모드(m4a/mp3/wav)")
    ap.add_argument("--name", required=True, help="프리셋 이름(voice_presets/<name>.wav)")
    ap.add_argument("--top", type=int, default=1, help="상위 N개 저장(청취 비교용)")
    ap.add_argument("--min-sec", type=float, default=6.0)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--no-transcribe", action="store_true")
    args = ap.parse_args()

    cands = (candidates_from_pipeline(args.speaker, args.min_sec, args.max_sec)
             if args.speaker else
             candidates_from_wav(args.wav, args.min_sec, args.max_sec))
    if not cands:
        raise SystemExit("후보 없음 — 파일이 너무 짧거나(<6s) 전부 잡음/겹침. "
                         "--min-sec 를 낮추거나 더 깨끗한 구간이 있는 녹음을 써라.")
    cands.sort(key=lambda c: c["score"], reverse=True)

    out_dir = data_dir() / "voice_presets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(cands[:max(1, args.top)], 1):
        y, sr = load_wav(c["wav"], sr=OUT_SR)
        clip = y[int(c["start"] * sr): int(c["end"] * sr)]
        stem = args.name if i == 1 else f"{args.name}_{i}"
        wav_path = out_dir / f"{stem}.wav"
        save_wav(wav_path, clip, sr)
        text = c["text"]
        if not text and not args.no_transcribe:
            y16, _ = load_wav(str(wav_path), sr=16000)
            text = _transcribe(y16, 16000)
        if text:
            (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        print(f"✓ {stem}.wav  {c['dur']:.1f}s  SNR {c['snr']:.1f}dB  score {c['score']:.3f}"
              + (f"  '{text[:30]}…'" if text else "  (전사 없음 — 서버 자동전사)"))

    print(f"\n→ UI '준비된 목소리' 드롭다운에 자동 표시됨({out_dir}). "
          f"미리듣기로 확인 후 통화에서 선택.")


if __name__ == "__main__":
    main()
