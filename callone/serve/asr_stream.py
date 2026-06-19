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


def _preload_cuda_libs() -> None:
    """ctranslate2(faster-whisper)가 **torch 번들 cuDNN9/cuBLAS** 를 쓰도록 전역 심볼로 선로드.

    클라우드 이미지의 시스템 cuDNN 과 ct2 요구버전이 어긋나면 GPU 로드가 실패→CPU int8 로
    강등된다(RTX 3090 실측: cuDNN 불일치로 ASR 가 매 턴 CPU 로 떨어져 지연 급증).
    torch[cu124] 는 nvidia-cudnn-cu12 / nvidia-cublas-cu12 를 동반 설치하므로, 그 .so 들을
    RTLD_GLOBAL 로 미리 dlopen 해 두면 ct2 의 dlopen 이 **버전 맞는** 심볼을 잡는다.
    Linux/CUDA 에서만 의미. 실패해도 무해(아래 CPU 폴백 유지)."""
    import ctypes
    import glob
    import os

    try:
        import nvidia  # type: ignore  # torch[cu124] 동반(이 노트북엔 없음, 박스에만)
    except Exception:  # noqa: BLE001
        return
    base = os.path.dirname(nvidia.__file__)
    # cuBLAS 먼저(cuDNN 이 의존), 그담 cuDNN.
    for pat in ("cublas/lib/libcublas*.so*", "cublas/lib/libcublasLt*.so*",
                "cudnn/lib/libcudnn*.so*"):
        for lib in sorted(glob.glob(os.path.join(base, pat))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


@lru_cache(maxsize=2)
def _load(model: str):
    from faster_whisper import WhisperModel  # type: ignore

    dev = resolve_device()
    ct = compute_type_for(dev)
    log.info("스트리밍 ASR 로드: model=%s device=%s compute=%s", model, dev, ct)
    if dev == "cuda":
        _preload_cuda_libs()   # 시스템 cuDNN 불일치로 GPU 로드 실패하는 것 사전 차단
    try:
        return WhisperModel(model, device=dev, compute_type=ct)
    except Exception as e:  # noqa: BLE001
        # 선로드해도 안 되면(드뭄) CPU int8 자동 폴백 — 통화는 살린다(빈 전사 방지).
        if dev == "cuda":
            log.warning("ASR GPU 로드 실패(%s) — CPU int8 폴백(ct2/cuDNN 버전 확인)", e)
            return WhisperModel(model, device="cpu", compute_type="int8")
        raise


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
