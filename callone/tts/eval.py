"""S4 TTS 평가 (§14, §19 test_s4).

SECS(화자검증 임베딩 코사인) > 0.70, 자가 ASR 재인식 WER < 10%, MOS(청취).
합성 음성 vs 화자 원본 임베딩 코사인 + 합성→ASR→원문 WER.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..common.audio import cosine, load_wav
from ..common.io import data_dir
from ..common.logging import get_logger

log = get_logger("tts_eval")


def secs(synth_wav: str, ref_wav: str) -> float:
    """화자 유사도: 합성 vs 원본 임베딩 코사인."""
    from ..diarize.embeddings import embed_file

    return cosine(embed_file(synth_wav), embed_file(ref_wav))


def self_wer(synth_wav: str, target_text: str, asr_cfg: dict | None = None) -> float:
    """합성음을 ASR 로 재인식 → 원문과 WER."""
    try:
        import jiwer  # type: ignore

        from ..asr.s3_transcribe import transcribe_file
        from ..common.io import load_config

        segs = transcribe_file(synth_wav, asr_cfg or load_config("asr"))
        hyp = " ".join(s["text"] for s in segs)
        return float(jiwer.wer(target_text, hyp))
    except Exception as e:  # noqa: BLE001
        log.warning("self-WER 측정 불가(%s)", e)
        return float("nan")


def evaluate(speaker: str, cfg: dict) -> dict:
    """TTS셋 일부로 합성 → SECS/WER 집계."""
    from .infer import TTSEngine

    eng = TTSEngine(speaker, cfg)
    csv = data_dir() / "datasets" / speaker / "tts" / "metadata.csv"
    if not csv.exists():
        return {"error": "TTS셋 없음"}
    lines = csv.read_text(encoding="utf-8").strip().splitlines()[:10]
    tmp = data_dir().parent / "reports" / "tts_eval_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    from ..common.audio import save_wav

    secs_list, wer_list = [], []
    for i, ln in enumerate(lines):
        parts = ln.split("|")
        if len(parts) < 2:
            continue
        ref_wav, text = parts[0], parts[1]
        y, sr = eng.synth(text)
        sp = tmp / f"{speaker}_{i}.wav"
        save_wav(sp, y, sr)
        if Path(ref_wav).exists():
            try:
                secs_list.append(secs(str(sp), ref_wav))
            except Exception:  # noqa: BLE001
                pass
        w = self_wer(str(sp), text)
        if not np.isnan(w):
            wer_list.append(w)

    res = {
        "speaker": speaker,
        "secs_mean": round(float(np.mean(secs_list)), 3) if secs_list else None,
        "self_wer_mean": round(float(np.mean(wer_list)), 3) if wer_list else None,
        "secs_pass": (np.mean(secs_list) > 0.70) if secs_list else None,
        "wer_pass": (np.mean(wer_list) < 0.10) if wer_list else None,
    }
    log.info("TTS 평가 %s: %s", speaker, res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 TTS 평가")
    ap.add_argument("--config", default="tts_server")
    ap.add_argument("--speaker", default="A")
    args = ap.parse_args()
    from ..common.io import load_config, write_json

    res = evaluate(args.speaker, load_config(args.config))
    write_json(data_dir().parent / "reports" / f"tts_eval_{args.speaker}.json", res)


if __name__ == "__main__":
    main()
