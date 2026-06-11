"""S6 스트리밍 TTS (§16) — 문장단위 → 음성 청크.

TTSEngine(S4) 래핑. 첫 청크 지연 목표 ~300ms.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from ..common.io import load_config
from ..tts.infer import TTSEngine


class StreamTTS:
    def __init__(self, speaker: str, cfg: dict | None = None):
        self.engine = TTSEngine(speaker, cfg or load_config("tts_server"))
        self.sr = self.engine.sr

    def speak(self, text: str, chunk_ms: int = 200) -> Iterator[np.ndarray]:
        yield from self.engine.synth_stream(text, chunk_ms=chunk_ms)
