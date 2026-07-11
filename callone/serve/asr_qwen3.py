"""Qwen3-ASR 백엔드 (transformers, 인프로세스) — StreamASR 와 동일 인터페이스(drop-in).

- 한국어 포함 52언어, 0.6B/1.7B(티어 자동: hardware.tier_defaults["asr_qwen_model"]).
- 여기선 **오프라인 transcribe** 만 쓴다. 네이티브 스트리밍은 vLLM 백엔드 전용이라
  serve venv 에 안 끌어온다(무거움) — 발화 중 partial 은 asr_streaming.StreamingTranscriber
  (백엔드 무관 재전사 방식)가 담당. vLLM 전환은 REBUILD_PLAN §5 이후 과제.
- 미설치/로드 실패 시 호출측(_pick_asr)이 faster-whisper 로 폴백.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..common.logging import get_logger

log = get_logger("asr_qwen3")


@lru_cache(maxsize=1)
def _load(model_id: str):
    import torch
    from qwen_asr import Qwen3ASRModel  # type: ignore  # pip install qwen-asr

    kw: dict = {}
    if torch.cuda.is_available():
        kw = {"device_map": "cuda:0", "dtype": torch.bfloat16}
    log.info("Qwen3-ASR 로드: %s %s", model_id, kw or "(cpu)")
    return Qwen3ASRModel.from_pretrained(model_id, **kw)


class Qwen3StreamASR:
    """실시간 통화용 Qwen3-ASR — StreamASR 와 같은 transcribe(audio, sr) 계약."""

    def __init__(self, cfg: dict | None = None):
        import importlib.util

        # fail-fast: 패키지 없는데 lazy 로드로 넘어가면 "조용히 빈 전사만 나오는" 죽은 ASR 가 됨
        # → 생성 시점에 확인해 _pick_asr 가 즉시 whisper 로 폴백하게 한다.
        if importlib.util.find_spec("qwen_asr") is None:
            raise RuntimeError("qwen-asr 미설치 — serve venv 에서 pip install qwen-asr")
        self.cfg = cfg or {}
        model_id = self.cfg.get("qwen_model") or ""
        if not model_id:
            from ..common.hardware import detect_tier, tier_defaults

            model_id = tier_defaults(detect_tier())["asr_qwen_model"]
        if not model_id:
            raise RuntimeError("이 티어엔 Qwen3-ASR 미배정(cpu/mid) — whisper 백엔드 사용")
        self.model_name = model_id
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = _load(self.model_name)
        return self._model

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        try:
            model = self._ensure()
            results = model.transcribe(
                audio=(np.asarray(audio, dtype=np.float32), int(sr)),
                language="Korean",
            )
            return " ".join(r.text for r in results).strip()
        except Exception as e:  # noqa: BLE001
            log.warning("Qwen3-ASR 전사 실패(%s) — 빈 문자열", e)
            return ""
