"""pytest 공통 fixture — 합성 데이터로 스테이지 검증 (무거운 모델 없이)."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def tmp_wav(tmp_path):
    """12초 합성 통화 wav (A/B 교번)."""
    sr = 16000
    t = np.linspace(0, 12, sr * 12, endpoint=False)
    y = np.zeros_like(t)
    for i, s in enumerate(np.arange(0, 12, 1.5)):
        freq = 180 if i % 2 == 0 else 140
        mask = (t >= s) & (t < s + 1.2)
        y[mask] += 0.2 * np.sin(2 * np.pi * freq * t[mask])
    p = tmp_path / "call_00000.wav"
    sf.write(str(p), y.astype(np.float32), sr)
    return p


@pytest.fixture
def sample_assignments():
    """전역 화자 연결 더미 (parquet 없이 로직 테스트용)."""
    return [
        {"segment_uid": "c1#0", "call_id": "c1", "start": 0.0, "end": 3.0,
         "global_speaker": "A", "clean": True, "is_overlap": False,
         "snr_db": 20.0, "text": "밥은 묵었나"},
        {"segment_uid": "c1#1", "call_id": "c1", "start": 3.0, "end": 6.0,
         "global_speaker": "B", "clean": True, "is_overlap": False,
         "snr_db": 18.0, "text": "응 먹었어"},
    ]
