"""S6 VAD — 발화끝 검출 (§16).

Silero VAD(선택) 또는 에너지 기반 폴백. end_silence_ms 무음 후 턴 종료.
"""
from __future__ import annotations

import numpy as np

from ..common.logging import get_logger

log = get_logger("vad")


class VAD:
    def __init__(self, cfg: dict | None = None, sr: int = 16000):
        cfg = cfg or {}
        self.sr = sr
        self.end_silence_ms = cfg.get("end_silence_ms", 200)
        # 기본 energy: torch 미사용 → OpenVINO LLM 과 같은 프로세스 충돌(segfault) 회피.
        # Silero(torch) 쓰려면 backend="silero" 명시(단 OV LLM 과 분리 프로세스 권장).
        self.backend = cfg.get("backend", "energy")
        self._model = self._try_silero()
        self._silence_ms = 0.0

    def _try_silero(self):
        if self.backend != "silero":
            return None
        try:
            import torch

            model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
            return model
        except Exception as e:  # noqa: BLE001
            log.warning("Silero VAD 미설치(%s) — 에너지 폴백", e)
            return None

    def is_speech(self, frame: np.ndarray) -> bool:
        if self._model is not None:
            import torch

            with torch.no_grad():
                prob = self._model(torch.from_numpy(frame).float(), self.sr).item()
            return prob > 0.5
        # 에너지 폴백
        return float(np.sqrt(np.mean(frame ** 2))) > 0.01

    def update(self, frame: np.ndarray) -> bool:
        """프레임 입력 → 발화 종료(turn end)면 True."""
        frame_ms = 1000 * len(frame) / self.sr
        if self.is_speech(frame):
            self._silence_ms = 0.0
            return False
        self._silence_ms += frame_ms
        if self._silence_ms >= self.end_silence_ms:
            self._silence_ms = 0.0
            return True
        return False
