"""S6 스트리밍 TTS (§16) — 문장단위 → 음성 청크.

기본 경로: TTSEngine(S4) 래핑(StreamTTS).
결정서 §2-3 경로: synthesize_streaming() — Qwen3-TTS 감정 instruct + 참조음색 동적 주입.
첫 청크 지연 목표 ~159ms(chunk_size=8).
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

    def synth_stream(self, text: str, chunk_ms: int = 200,
                     emotion: str | None = None) -> Iterator[np.ndarray]:
        # 폴백 엔진은 감정 미지원 → emotion 무시.
        yield from self.engine.synth_stream(text, chunk_ms=chunk_ms)


def synthesize_streaming(text: str, emotion: str, ref_wav_path: str, ref_text: str,
                         speaker: str = "A", chunk_size: int = 8) -> Iterator[np.ndarray]:
    """결정서 §2-3 직접 호출용 — Qwen3-TTS 감정 instruct + 참조 음색 스트리밍.

    ref_wav_path: 화자 참조 WAV(7~10초, 24kHz, 잡음 없음)
    ref_text:     참조 WAV 의 실제 발화 텍스트(전사와 정확히 일치)
    emotion:      LLM 이 판단한 감정 키(happy/sad/angry/neutral/excited)
    """
    from .tts_qwen import QwenTTS

    tts = QwenTTS(speaker, {"ref_wav": ref_wav_path, "ref_text": ref_text,
                            "chunk_size": chunk_size})
    yield from tts.synth_stream(text, emotion=emotion)
