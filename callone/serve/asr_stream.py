"""S6 스트리밍 ASR (§16) — 실시간 전화용, 속도 최우선.

기본 모델 = **Whisper large-v3-turbo** (정확도 대비 가장 빠름; 실시간 적합).
방언 적응본(models/asr_dialect)이 있으면 그걸 우선 사용.
실시간이라 beam_size=1, greedy, VAD 필터로 지연 최소화.
디바이스 자동(GPU 없으면 cpu+int8). Gemma 4 오디오 직접 입력(준-E2E)도 허용.

⚠️ §3 검증 의무: turbo 가 현 시점 최선인지 착수 시 웹 재확인(Voxtral/Canary 등).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..common.io import compute_type_for, load_config, resolve_device
from ..common.logging import get_logger

log = get_logger("asr_stream")

# 실시간 기본: turbo (속도). 오프라인 고정확 전사는 asr.yaml 의 large-v3 사용.
REALTIME_DEFAULT = "large-v3-turbo"


@lru_cache(maxsize=2)
def _load(model: str):
    from faster_whisper import WhisperModel  # type: ignore

    dev = resolve_device()
    ct = compute_type_for(dev)
    log.info("스트리밍 ASR 로드: model=%s device=%s compute=%s", model, dev, ct)
    return WhisperModel(model, device=dev, compute_type=ct)


class StreamASR:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        adapted = self.cfg.get("model_dir", "models/asr_dialect")
        configured = self.cfg.get("model", "auto")
        if Path(adapted).exists() and any(Path(adapted).iterdir()):
            self.model_name = adapted          # 방언 적응본 최우선
        elif configured and configured != "auto":
            self.model_name = configured       # 사용자가 고정
        else:
            # 티어별 자동: GPU=turbo, CPU=small
            from ..common.hardware import detect_tier, tier_defaults

            self.model_name = tier_defaults(detect_tier())["asr_realtime"]
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = _load(self.model_name)
        return self._model

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        try:
            model = self._ensure()
            segments, _ = model.transcribe(
                audio, language="ko",
                beam_size=1,                  # greedy → 빠름
                vad_filter=True,
                condition_on_previous_text=False,
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as e:  # noqa: BLE001
            log.warning("스트리밍 ASR 불가(%s) — 빈 문자열", e)
            return ""
