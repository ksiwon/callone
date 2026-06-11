"""S3 전사 CLI (§12).

방언 적응 Whisper(있으면) 또는 large-v3 로 전사. 사투리 어형 보존
(표준어 교정 금지). 단어 타임스탬프 정렬. diarized json 의 text 채움.

faster-whisper 미설치 시: 기존 text 유지(더미 분리 경로) + 경고.

사용:
  callone-transcribe [--limit 50]
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from ..common import db
from ..common.io import data_dir, load_config, read_json, write_json
from ..common.logging import get_logger
from ..common.schemas import DiarizedCall, Segment, Word

log = get_logger("s3")


@lru_cache(maxsize=2)
def _load_whisper(model: str, compute_type: str | None = None):
    from faster_whisper import WhisperModel  # type: ignore

    from ..common.io import compute_type_for, resolve_device

    dev = resolve_device()           # cuda 없으면 자동 cpu
    ct = compute_type_for(dev)       # GPU=float16, CPU=int8
    log.info("Whisper 로드: model=%s device=%s compute=%s", model, dev, ct)
    return WhisperModel(model, device=dev, compute_type=ct)


def _pick_model(cfg: dict) -> str:
    adapted = cfg.get("adapted_model_dir", "models/asr_dialect")
    if Path(adapted).exists() and any(Path(adapted).iterdir()):
        log.info("방언 적응 ASR 사용: %s", adapted)
        return adapted
    model = cfg.get("model", "auto")
    if model == "auto":
        # 티어 자동: GPU=large-v3(정확), CPU=small(가벼움)
        from ..common.hardware import detect_tier

        model = "large-v3" if detect_tier() == "server_gpu" else "small"
        log.info("ASR 모델 자동선택: %s", model)
    return model


def transcribe_file(wav: str, cfg: dict) -> list[dict]:
    model = _load_whisper(_pick_model(cfg))
    segments, _ = model.transcribe(
        wav, language=cfg.get("language", "ko"),
        beam_size=cfg.get("beam_size", 5),
        word_timestamps=cfg.get("word_timestamps", True),
        vad_filter=cfg.get("vad_filter", True),
    )
    out = []
    for s in segments:
        words = [{"word": w.word, "start": w.start, "end": w.end, "score": getattr(w, "probability", 0.0)}
                 for w in (s.words or [])]
        out.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                    "asr_conf": getattr(s, "avg_logprob", 0.0), "words": words})
    return out


def _align_to_diarized(dc: dict, asr_segs: list[dict]) -> DiarizedCall:
    """ASR 세그먼트 텍스트를 diarized 세그먼트에 시간 겹침으로 정렬."""
    segs = []
    for seg in dc["segments"]:
        s, e = seg["start"], seg["end"]
        texts, words, confs = [], [], []
        for a in asr_segs:
            # 겹침 비율
            ov = max(0.0, min(e, a["end"]) - max(s, a["start"]))
            if ov > 0.3 * (a["end"] - a["start"]):
                texts.append(a["text"])
                confs.append(a.get("asr_conf", 0.0))
                words.extend(Word(**w) for w in a.get("words", []) if s <= w["start"] <= e)
        seg_obj = Segment(**{**seg, "text": " ".join(texts).strip() or seg.get("text", ""),
                             "words": words,
                             "asr_conf": (sum(confs) / len(confs)) if confs else seg.get("asr_conf", 0.0)})
        segs.append(seg_obj)
    return DiarizedCall(call_id=dc["call_id"], segments=segs)


def run(cfg: dict, limit: int | None = None) -> None:
    con = db.connect()
    calls = [c for c in db.all_calls(con) if c.status == "ok"]
    if limit:
        calls = calls[:limit]
    diar_dir = data_dir() / "diarized"

    total = len(calls)
    for i, c in enumerate(calls, 1):
        jp = diar_dir / f"{c.call_id}.json"
        if not jp.exists():
            continue
        wav = c.restored_path or c.wav16k_path
        dc = read_json(jp)
        try:
            asr_segs = transcribe_file(wav, cfg)
            new_dc = _align_to_diarized(dc, asr_segs)
            write_json(jp, new_dc)
            log.info("전사 %s: %d 세그먼트 [%d/%d]", c.call_id, len(asr_segs), i, total)
        except Exception as e:  # noqa: BLE001
            log.warning("전사 불가 %s (%s) [%d/%d] — 기존 text 유지", c.call_id, e, i, total)

    log.info("S3 전사 완료: 총 %d 통화", total)


def main() -> None:
    ap = argparse.ArgumentParser(description="S3 전사")
    ap.add_argument("--config", default="asr")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
