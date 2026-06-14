"""S4 TTS 추론 (§14) — 텍스트 → 음성, 스트리밍 청크.

학습된 per-speaker 모델로 합성. 백엔드 미설치/모델 미존재 시:
간이 사인톤 placeholder 생성(파이프라인/지연 측정용).
실시간(S6)에서 문장단위 스트리밍 호출.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

from ..common.audio import save_wav
from ..common.io import load_config
from ..common.logging import get_logger

log = get_logger("tts_infer")


class TTSEngine:
    """백엔드 추상화. server/phone 공통 인터페이스."""

    def __init__(self, speaker: str, cfg: dict):
        self.speaker = speaker
        self.cfg = cfg
        self.backend = cfg.get("backend", "qwen3-tts")
        # 결정서 §2: Qwen3-TTS 24kHz. config 의 sample_rate 우선.
        self.sr = int(cfg.get("sample_rate", 24000))
        self.model_dir = Path(cfg.get("output_dir", "models/tts_server")) / speaker
        self._engine = None          # 실백엔드(QwenTTS 등)
        self._loaded = self._try_load()

    def _try_load(self) -> bool:
        # 결정서 §2: qwen3-tts 백엔드면 faster_qwen3_tts(QwenTTS) 시도.
        if self.backend == "qwen3-tts":
            try:
                from ..serve.tts_qwen import QwenTTS

                self._engine = QwenTTS(self.speaker)
                self.sr = self._engine.sr
                log.info("TTSEngine: Qwen3-TTS 백엔드 로드(speaker=%s)", self.speaker)
                return True
            except Exception as e:  # noqa: BLE001
                log.warning("Qwen3-TTS 미연결(%s) — placeholder 합성", e)
        elif not self.model_dir.exists():
            log.warning("TTS 모델 없음 %s — placeholder 합성 사용", self.model_dir)
        return False

    def synth(self, text: str, emotion: str | None = None) -> tuple[np.ndarray, int]:
        if self._loaded and self._engine is not None:
            return self._engine.synth(text, emotion=emotion)
        return self._placeholder(text), self.sr

    def synth_stream(self, text: str, chunk_ms: int = 200,
                     emotion: str | None = None) -> Iterator[np.ndarray]:
        """문장단위 → 청크 스트리밍 (S6 지연 예산)."""
        if self._loaded and self._engine is not None:
            yield from self._engine.synth_stream(text, chunk_ms=chunk_ms, emotion=emotion)
            return
        y, sr = self.synth(text)
        step = int(sr * chunk_ms / 1000)
        for i in range(0, len(y), step):
            yield y[i:i + step]

    def _placeholder(self, text: str) -> np.ndarray:
        """글자 수 비례 길이의 톤 — 실제 음성 아님, 배관 검증용."""
        dur = max(0.5, len(text) * 0.08)
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        freq = 180 if self.speaker == "A" else 140
        y = 0.05 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        return y


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 TTS 추론")
    ap.add_argument("--config", default="tts_server")
    ap.add_argument("--speaker", default="A")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    args = ap.parse_args()
    eng = TTSEngine(args.speaker, load_config(args.config))
    y, sr = eng.synth(args.text)
    save_wav(args.out, y, sr)
    log.info("합성 → %s (%.1fs)", args.out, len(y) / sr)


if __name__ == "__main__":
    main()
