"""겹침발화 플래그(s2_diarize.overlap_flags) + 제로샷 ref 자동추출(pick_ref_clip) 검증."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from callone.diarize.s2_diarize import overlap_flags


def _load_pick_ref():
    p = Path(__file__).resolve().parents[1] / "scripts" / "pick_ref_clip.py"
    spec = importlib.util.spec_from_file_location("pick_ref_clip", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ----- overlap_flags ----------------------------------------------------------
def test_overlap_marks_cross_speaker_only():
    """다른 화자와 겹칠 때만 True — 같은 화자 연속/인접 턴은 겹침 아님."""
    turns = [
        (0.0, 5.0, "S0"),     # S1(4.5~8)과 0.5s 겹침 → True
        (4.5, 8.0, "S1"),     # 위와 겹침 → True
        (8.2, 12.0, "S0"),    # 아무와도 안 겹침 → False
        (11.9, 13.0, "S0"),   # S0 끼리 0.1s 겹침(같은 화자) → False
    ]
    assert overlap_flags(turns) == [True, True, False, False]


def test_overlap_threshold():
    """겹침이 0.15s 미만이면 무시(경계 접촉은 겹침 아님)."""
    turns = [(0.0, 5.0, "S0"), (4.9, 8.0, "S1")]      # 0.1s 겹침
    assert overlap_flags(turns) == [False, False]
    turns = [(0.0, 5.0, "S0"), (4.8, 8.0, "S1")]      # 0.2s 겹침
    assert overlap_flags(turns) == [True, True]


# ----- pick_ref_clip 점수/후보 --------------------------------------------------
def test_score_prefers_ideal_duration_and_snr():
    m = _load_pick_ref()
    assert m._score(20.0, 3.0, 6.0, 12.0) < 0            # 범위 밖 → 탈락
    assert m._score(20.0, 20.0, 6.0, 12.0) < 0
    ideal = m._score(25.0, 9.0, 6.0, 12.0)               # 최적: 9s + 고SNR
    assert ideal > m._score(25.0, 6.0, 6.0, 12.0)        # 같은 SNR, 길이 덜 이상적
    assert ideal > m._score(10.0, 9.0, 6.0, 12.0)        # 같은 길이, SNR 낮음


def test_candidates_from_wav_picks_clean_segment(tmp_path):
    """합성 오디오: [잡음 5s][깨끗한 발화 9s][무음 3s] → 후보 최고점이 발화 구간을 잡는다."""
    m = _load_pick_ref()
    import soundfile as sf

    sr = 16000
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(5 * sr) * 0.02).astype(np.float32)          # 저레벨 잡음
    t = np.arange(9 * sr) / sr
    speech = (0.5 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
              ).astype(np.float32)                                            # 변조 사인 = 발화 흉내
    silence = np.zeros(3 * sr, dtype=np.float32)
    wav = tmp_path / "long.wav"
    sf.write(wav, np.concatenate([noise, speech, silence]), sr)

    cands = m.candidates_from_wav(str(wav), 6.0, 12.0)
    assert cands, "후보가 나와야 함"
    best = max(cands, key=lambda c: c["score"])
    # 최고 후보는 발화 구간(5s~14s) 안에서 시작해야 함
    assert 4.0 <= best["start"] <= 7.0
    assert 6.0 <= best["dur"] <= 12.0
