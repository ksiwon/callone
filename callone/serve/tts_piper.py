"""Piper TTS 백엔드 (노트북 온디바이스 폴백) — onnx, torch-free.

과거에 학습해 둔 Piper onnx 를 구동만 한다(학습 경로는 v2 정리 때 폐기 — 목소리는
제로샷 클론이 정본). onnxruntime + piper-phonemize 라 **torch 없이 CPU 실시간**.
문장 단위 스트리밍.

모델 위치: models/tts_piper/{speaker}.onnx (+ {speaker}.onnx.json 설정).
미설치/모델없음 → 예외 → orchestrator 가 KokoroTTS/placeholder 로 폴백.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..common.logging import get_logger

log = get_logger("tts_piper")


def _model_paths(speaker: str) -> tuple[Path, Path]:
    base = Path("models/tts_piper")
    onnx = base / f"{speaker}.onnx"
    cfg = base / f"{speaker}.onnx.json"
    return onnx, cfg


class PiperTTS:
    def __init__(self, speaker: str, cfg: dict | None = None):
        self.speaker = speaker
        self.cfg = cfg or {}
        onnx, jcfg = _model_paths(speaker)
        if not onnx.exists():
            raise FileNotFoundError(
                f"Piper 음성 모델 없음: {onnx} (기존 학습본 onnx 배치 시에만 사용 — 기본은 제로샷)")
        from piper import PiperVoice  # type: ignore  # pip install piper-tts

        self.voice = PiperVoice.load(str(onnx), config_path=str(jcfg) if jcfg.exists() else None)
        self.sr = self._detect_sr()
        log.info("Piper TTS 로드: %s (sr=%d, speaker=%s)", onnx, self.sr, speaker)

    def _detect_sr(self) -> int:
        for attr in ("config", "_config"):
            c = getattr(self.voice, attr, None)
            sr = getattr(c, "sample_rate", None)
            if sr:
                return int(sr)
        return int(self.cfg.get("sample_rate", 22050))

    # ----- 합성 ----------------------------------------------------------
    def _chunks_to_float(self, text: str) -> list[np.ndarray]:
        """piper 버전별 API 차이 흡수 → float32 [-1,1] 청크 리스트."""
        out: list[np.ndarray] = []
        # 신버전: synthesize(text) → AudioChunk(.audio_float_array / .audio_int16_bytes)
        if hasattr(self.voice, "synthesize"):
            try:
                for ch in self.voice.synthesize(text):
                    arr = getattr(ch, "audio_float_array", None)
                    if arr is not None:
                        out.append(np.asarray(arr, dtype=np.float32))
                        continue
                    b = getattr(ch, "audio_int16_bytes", None)
                    if b is None and isinstance(ch, (bytes, bytearray)):
                        b = bytes(ch)
                    if b is not None:
                        out.append(np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0)
                if out:
                    return out
            except TypeError:
                pass  # 구버전 시그니처 → 아래
        # 구버전: synthesize_stream_raw(text) → int16 raw bytes
        if hasattr(self.voice, "synthesize_stream_raw"):
            for b in self.voice.synthesize_stream_raw(text):
                out.append(np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0)
        return out

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.zeros(0, dtype=np.float32), self.sr
        chunks = self._chunks_to_float(text)
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sr
        return np.concatenate(chunks), self.sr

    def synth_stream(self, text: str, chunk_ms: int = 200) -> Iterator[np.ndarray]:
        # piper 가 이미 청크로 주지만, 송출 균일화를 위해 chunk_ms 로 재단.
        y, sr = self.synth(text)
        if len(y) == 0:
            return
        step = max(1, int(sr * chunk_ms / 1000))
        for i in range(0, len(y), step):
            yield y[i:i + step]
