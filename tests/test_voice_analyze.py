"""voice_analyze(긴 통화 녹음 → 화자별 ref 추출, UI 플로우 B) 검증 — diarize 는 페이크."""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

from callone.common.schemas import Segment
from callone.serve import voice_analyze as va


def _fake_segments():
    """화자 S0(0~10s, 20~29s) / S1(12~19s) + 겹침 세그먼트 1개(제외돼야 함)."""
    return [
        Segment(start=0.0, end=10.0, local_speaker="S0", snr_db=20.0),
        Segment(start=12.0, end=19.0, local_speaker="S1", snr_db=18.0),
        Segment(start=20.0, end=29.0, local_speaker="S0", snr_db=22.0),
        Segment(start=18.5, end=25.0, local_speaker="S1", snr_db=15.0, overlap=True),
    ]


@pytest.fixture()
def wav_bytes(tmp_path):
    sr = 16000
    t = np.arange(30 * sr) / sr
    y = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    p = tmp_path / "call.wav"
    sf.write(p, y, sr)
    return p.read_bytes()


def _wait_done(jid, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = va.job_status(jid)
        if st and st["stage"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError("분석 job 타임아웃")


def test_analyze_groups_speakers_and_hides_internal(monkeypatch, wav_bytes, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path / "data"))
    from callone.diarize import s2_diarize

    monkeypatch.setattr(s2_diarize, "diarize_one", lambda path, cfg: _fake_segments())
    jid = va.start_job(wav_bytes, suffix=".wav")
    st = _wait_done(jid)
    assert st["stage"] == "done", st.get("error")
    spks = st["speakers"]
    assert [s["id"] for s in spks] == ["S0", "S1"]          # 발화 총량순(S0=19s > S1=7s)
    assert spks[0]["n_segments"] == 2
    for s in spks:
        assert s["sample_wav_b64"]                          # 청취 샘플 포함
        assert not any(k.startswith("_") for k in s)        # 내부 버퍼 미노출


def test_save_pick_writes_preset_with_transcript(monkeypatch, wav_bytes, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path / "data"))
    from callone.diarize import s2_diarize

    monkeypatch.setattr(s2_diarize, "diarize_one", lambda path, cfg: _fake_segments())
    jid = va.start_job(wav_bytes, suffix=".wav")
    assert _wait_done(jid)["stage"] == "done"

    class _Asr:
        def transcribe(self, audio, sr=16000):
            return "안녕하세요 반가워요"

    r = va.save_pick(jid, "S0", "엄마", asr=_Asr())
    assert r["preset_id"] == "엄마" and r["ref_text"] == "안녕하세요 반가워요"
    from callone.serve.voice_presets import list_presets, resolve

    ids = {p["id"] for p in list_presets()}
    assert "엄마" in ids                                    # UI 프리셋에 자동 노출
    wav, text = resolve("엄마")
    assert wav.endswith("엄마.wav") and text == "안녕하세요 반가워요"


def test_save_pick_rejects_unknown(monkeypatch, wav_bytes, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path / "data"))
    from callone.diarize import s2_diarize

    monkeypatch.setattr(s2_diarize, "diarize_one", lambda path, cfg: _fake_segments())
    jid = va.start_job(wav_bytes, suffix=".wav")
    assert _wait_done(jid)["stage"] == "done"
    with pytest.raises(ValueError):
        va.save_pick(jid, "S9", "x")
    with pytest.raises(ValueError):
        va.save_pick("nope", "S0", "x")
