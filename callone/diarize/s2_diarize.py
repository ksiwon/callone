"""S2 (a) 통화별 화자 분리 CLI (§10a).

WhisperX(Whisper + pyannote) 또는 pyannote diarization.
모노믹스 → 2화자(+제3자 대비 max 3) 세그먼트 + 단어 타임스탬프 + SNR/overlap.
출력: data/diarized/{call_id}.json (§7.2).

무거운 모델 미설치 시: 에너지 기반 VAD 더미 분리로 폴백(파이프라인 검증).

사용:
  callone-diarize [--limit 50]
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..common import db
from ..common.audio import estimate_snr_db, load_wav
from ..common.io import data_dir, load_config, write_json
from ..common.logging import get_logger
from ..common.schemas import DiarizedCall, Segment

log = get_logger("s2")


def _source_wav(call) -> str | None:
    # 복원본 우선, 없으면 16k
    return call.restored_path or call.wav16k_path


def diarize_whisperx(wav: str, cfg: dict) -> list[Segment]:
    """WhisperX 경로 (선택 설치)."""
    import whisperx  # type: ignore

    from ..common.io import compute_type_for, resolve_device

    device = resolve_device()        # cuda 없으면 cpu 자동
    model_name = cfg.get("asr_model", "large-v3")
    if Path(model_name).exists() is False and model_name.startswith("models/"):
        model_name = "large-v3"
    model = whisperx.load_model(model_name, device, compute_type=compute_type_for(device),
                                language=cfg.get("language", "ko"))
    audio = whisperx.load_audio(wav)
    result = model.transcribe(audio, batch_size=16)

    align_model, meta = whisperx.load_align_model(language_code="ko", device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device)

    import os

    dia = whisperx.DiarizationPipeline(use_auth_token=os.environ.get("HF_TOKEN"), device=device)
    diar = dia(audio, min_speakers=cfg.get("min_speakers", 2),
               max_speakers=cfg.get("max_speakers", 3))
    result = whisperx.assign_word_speakers(diar, result)

    segs = []
    for s in result["segments"]:
        segs.append(Segment(
            start=s["start"], end=s["end"],
            local_speaker=s.get("speaker", "SPK_00"),
            text=s.get("text", "").strip(),
            asr_conf=float(np.mean([w.get("score", 0) for w in s.get("words", [])]) if s.get("words") else 0.0),
        ))
    return segs


@lru_cache(maxsize=1)
def _load_pyannote(diarizer: str, fallback: str | None):
    """pyannote 파이프라인 1회 로드 후 캐시 (매 통화 재로드 방지)."""
    import functools
    import os

    import torch  # type: ignore

    # torch>=2.6 은 torch.load 기본 weights_only=True → pyannote 구 체크포인트 로딩 실패.
    # pyannote 공식 모델은 신뢰원이므로 weights_only=False 강제(로딩 호환).
    if not getattr(torch, "_callone_load_patched", False):
        _orig_load = torch.load

        @functools.wraps(_orig_load)
        def _patched_load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_load(*a, **k)

        torch.load = _patched_load
        torch._callone_load_patched = True

    from pyannote.audio import Pipeline  # type: ignore

    from ..common.io import resolve_device

    token = os.environ.get("HF_TOKEN")
    last_err = None
    for model_id in (diarizer, fallback):
        if not model_id:
            continue
        # pyannote 4.x = token=, 3.x = use_auth_token= → 둘 다 시도
        for kw in ("token", "use_auth_token"):
            try:
                pipe = Pipeline.from_pretrained(model_id, **{kw: token})
                pipe.to(torch.device(resolve_device()))
                log.info("pyannote 로드(1회 캐시): %s (%s)", model_id, resolve_device())
                return pipe
            except TypeError:
                continue
            except Exception as e:  # noqa: BLE001
                log.warning("pyannote 모델 로드 실패 %s: %s", model_id, e)
                last_err = e
                break
    raise RuntimeError(f"pyannote 파이프라인 로드 불가: {last_err}")


def overlap_flags(turns: list[tuple[float, float, str]], min_ov: float = 0.15) -> list[bool]:
    """턴별 겹침발화 여부 — **다른 화자** 턴과 min_ov 초 이상 겹치면 True.

    겹침발화는 화자 임베딩/TTS 학습셋/제로샷 ref 의 오염원이라 하류(s2b clean 플래그,
    build_tts, pick_ref_clip)가 이 플래그로 거른다. (v2 전엔 미계산 → 필터가 무동작이었음.)"""
    out = []
    for i, (s, e, spk) in enumerate(turns):
        out.append(any(j != i and o_spk != spk and min(e, o_e) - max(s, o_s) >= min_ov
                       for j, (o_s, o_e, o_spk) in enumerate(turns)))
    return out


def diarize_pyannote(wav: str, cfg: dict) -> list[Segment]:
    """pyannote.audio 직접 경로 (whisperx 불필요, 로컬 CPU 가능).

    config diarizer(community-1) 우선, 실패 시 diarizer_fallback(3.1).
    HF_TOKEN + 게이트 동의 필요. text 는 비우고 S3 에서 채움. 모델은 캐시.
    """
    import torch  # type: ignore

    pipe = _load_pyannote(cfg.get("diarizer"), cfg.get("diarizer_fallback"))

    # 파일 경로 대신 메모리 파형 전달 → pyannote 4.x 의 torchcodec 파일로더 우회
    # (Windows 에서 libtorchcodec DLL 부재 회피). waveform=(channel, time).
    y, sr = load_wav(wav, sr=16000)
    wf = torch.from_numpy(y).float().unsqueeze(0)
    diar = pipe({"waveform": wf, "sample_rate": sr},
                min_speakers=cfg.get("min_speakers", 2),
                max_speakers=cfg.get("max_speakers", 3))

    # pyannote 4.x 는 DiarizeOutput 반환 → .speaker_diarization(Annotation). 3.x 는 Annotation 직접.
    annotation = getattr(diar, "speaker_diarization", diar)
    turns = [(float(t.start), float(t.end), spk)
             for t, _, spk in annotation.itertracks(yield_label=True)]
    ov = overlap_flags(turns)

    segs = []
    for (s, e, spk), is_ov in zip(turns, ov):
        if e - s < 0.4:
            continue
        clip = y[int(s * sr): int(e * sr)]
        segs.append(Segment(
            start=round(s, 3), end=round(e, 3),
            local_speaker=spk,                     # SPEAKER_00 / SPEAKER_01
            text="", snr_db=round(estimate_snr_db(clip), 1),
            overlap=is_ov,
        ))
    log.info("pyannote 분리 완료: %d 세그먼트(겹침 %d)", len(segs),
             sum(1 for x in segs if x.overlap))
    return segs


def diarize_dummy(wav: str, cfg: dict) -> list[Segment]:
    """폴백: 에너지 VAD 로 발화 구간 추출 + 화자 라벨 교번(파이프라인 검증용)."""
    import librosa

    y, sr = load_wav(wav, sr=16000)
    intervals = librosa.effects.split(y, top_db=30)
    segs = []
    for i, (s, e) in enumerate(intervals):
        st, en = s / sr, e / sr
        if en - st < 0.4:
            continue
        seg_y = y[s:e]
        segs.append(Segment(
            start=round(st, 3), end=round(en, 3),
            local_speaker=f"SPK_{i % 2:02d}",
            text="", snr_db=round(estimate_snr_db(seg_y), 1),
        ))
    log.warning("더미 분리 사용 — %d 세그먼트 (실모델 설치 권장)", len(segs))
    return segs


def diarize_one(wav: str, cfg: dict) -> list[Segment]:
    # 1순위 pyannote 직접(가벼움) → 2순위 whisperx → 최후 더미 폴백
    try:
        return diarize_pyannote(wav, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("pyannote 사용 불가(%s) — whisperx 시도", e)
    try:
        return diarize_whisperx(wav, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("WhisperX 사용 불가(%s) — 더미 폴백(품질 낮음)", e)
        return diarize_dummy(wav, cfg)


def _already_done(out_dir: Path, call_id: str) -> bool:
    """이미 분리된 비어있지 않은 결과가 있으면 True (재개용)."""
    p = out_dir / f"{call_id}.json"
    if not p.exists():
        return False
    try:
        import json

        return len(json.loads(p.read_text(encoding="utf-8")).get("segments", [])) > 0
    except Exception:
        return False


def run(cfg: dict, limit: int | None = None, skip_existing: bool = True) -> None:
    con = db.connect()
    calls = [c for c in db.all_calls(con) if c.status == "ok"]
    if limit:
        calls = calls[:limit]
    out_dir = data_dir() / "diarized"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for i, c in enumerate(calls, 1):
        if skip_existing and _already_done(out_dir, c.call_id):
            skipped += 1
            continue
        wav = _source_wav(c)
        if not wav or not Path(wav).exists():
            log.warning("wav 없음 %s", c.call_id)
            continue
        segs = diarize_one(wav, cfg)
        dc = DiarizedCall(call_id=c.call_id, segments=segs)
        write_json(out_dir / f"{c.call_id}.json", dc)
        done += 1
        log.info("분리 %s: %d 세그먼트 [%d/%d]", c.call_id, len(segs), i, len(calls))

    log.info("S2(a) 완료: 신규 %d, 건너뜀 %d, 총 %d 통화", done, skipped, len(calls))


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 화자 분리")
    ap.add_argument("--config", default="s2_diarize")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skip", action="store_true", help="이미 한 것도 다시")
    args = ap.parse_args()
    run(load_config(args.config), limit=args.limit, skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
