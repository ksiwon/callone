"""test_s0 (§19): 16k mono, 손상 격리, 스키마 라운드트립."""
from callone.common.schemas import CallMeta
from callone.common.audio import estimate_snr_db, load_wav


def test_callmeta_roundtrip():
    m = CallMeta(call_id="c1", src_path="x.m4a", status="ok")
    d = m.model_dump()
    assert CallMeta(**d).call_id == "c1"


def test_wav_is_mono_16k(tmp_wav):
    y, sr = load_wav(tmp_wav, sr=None)
    assert sr == 16000
    assert y.ndim == 1


def test_snr_estimate_positive(tmp_wav):
    y, sr = load_wav(tmp_wav)
    assert estimate_snr_db(y) > 0
