"""화자 임베딩 — ECAPA-TDNN(SpeechBrain) / WeSpeaker (§10b).

무거운 모델 미설치 시 MFCC 통계 폴백(파이프라인 검증용).
임베딩은 음색 보존 가드(S1)와 전역 연결(S2c)에서 공유.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..common.logging import get_logger

log = get_logger("embed")

_DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


@lru_cache(maxsize=1)
def _load_speechbrain(model_id: str = _DEFAULT_MODEL):
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

    from ..common.io import resolve_device

    return EncoderClassifier.from_hparams(source=model_id,
                                          run_opts={"device": resolve_device()})


def embed_waveform(y: np.ndarray, sr: int, model_id: str = _DEFAULT_MODEL) -> np.ndarray:
    """파형 → 고정 차원 화자 임베딩 벡터."""
    try:
        import torch

        if sr != 16000:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        enc = _load_speechbrain(model_id)
        with torch.no_grad():
            emb = enc.encode_batch(torch.from_numpy(y).float().unsqueeze(0))
        return emb.squeeze().cpu().numpy()
    except Exception as e:  # noqa: BLE001
        log.warning("SpeechBrain 임베딩 불가(%s) — MFCC 폴백", e)
        return _mfcc_embed(y, sr)


def _mfcc_embed(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def embed_file(path: str, model_id: str = _DEFAULT_MODEL,
               start: float | None = None, end: float | None = None) -> np.ndarray:
    from ..common.audio import load_wav

    y, sr = load_wav(path, sr=16000)
    if start is not None and end is not None:
        y = y[int(start * sr): int(end * sr)]
    return embed_waveform(y, sr, model_id)
