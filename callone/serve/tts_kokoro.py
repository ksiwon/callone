"""Kokoro(+KokoClone) TTS 백엔드 (노트북 온디바이스, 한국어, 경량 82M).

화자별 대표 ECAPA 임베딩(정제 클립 평균)으로 제로샷 음색 클론.
Kokoro 미설치 시 폴백(MeloTTS 또는 placeholder) — 노트북에서 `pip install kokoro` 후 사용.

스트리밍: 문장 단위로 합성(첫 문장 즉시 송출 → 실시간 느낌).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from ..common.io import data_dir
from ..common.logging import get_logger

log = get_logger("tts_kokoro")


@lru_cache(maxsize=4)
def speaker_embedding(speaker: str, topk: int = 50) -> np.ndarray | None:
    """화자 정제 클립 중 SNR 상위 topk 의 ECAPA 임베딩 평균 = 대표 음색 벡터."""
    csv = data_dir() / "datasets" / speaker / "tts" / "metadata.csv"
    if not csv.exists():
        log.warning("TTS셋 없음: %s", csv)
        return None
    rows = []
    for ln in csv.read_text(encoding="utf-8").splitlines():
        p = ln.split("|")
        if len(p) >= 4:
            rows.append((p[0], float(p[3]) if p[3] else 0.0))
    rows.sort(key=lambda r: -r[1])           # SNR 내림차순
    from ..diarize.embeddings import embed_file

    embs = []
    for wav, _snr in rows[:topk]:
        if Path(wav).exists():
            try:
                embs.append(embed_file(wav))
            except Exception:  # noqa: BLE001
                pass
    if not embs:
        return None
    return np.mean(np.stack(embs), axis=0)


class KokoroTTS:
    def __init__(self, speaker: str, cfg: dict | None = None, sr: int = 24000):
        self.speaker = speaker
        self.sr = sr
        self.cfg = cfg or {}
        # 런타임 torch 회피(OV LLM 과 같은 프로세스 → speechbrain 호출 금지).
        # 사전계산된 임베딩(scripts/precompute_voice_emb.py)을 numpy 로 로드.
        self.spk_emb = self._load_precomputed_emb()
        self._engine = self._load()

    def _load_precomputed_emb(self) -> np.ndarray | None:
        p = data_dir() / "speakers" / self.speaker / "voice_emb.npy"
        if p.exists():
            try:
                return np.load(p)
            except Exception:  # noqa: BLE001
                pass
        log.warning("화자 임베딩 없음(%s) — scripts/precompute_voice_emb.py 로 생성 권장", p)
        return None

    def _load(self):
        try:
            from kokoro import KPipeline  # type: ignore

            # 한국어 파이프라인. (lang_code 는 Kokoro 버전별 상이 — 'k'/'ko' 등)
            pipe = KPipeline(lang_code="k")
            log.info("Kokoro 로드 (speaker=%s, 임베딩 %s)",
                     self.speaker, "있음" if self.spk_emb is not None else "없음")
            return pipe
        except Exception as e:  # noqa: BLE001
            log.warning("Kokoro 미설치/로드 실패(%s) — placeholder 폴백. "
                        "노트북서 'pip install kokoro' 후 사용", e)
            return None

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        if self._engine is not None:
            try:
                # KokoClone: 화자 임베딩으로 음색 지정 (API 는 설치본에 맞게 조정)
                audio_chunks = []
                for _gs, _ps, audio in self._engine(text, voice=self.spk_emb):
                    audio_chunks.append(np.asarray(audio, dtype=np.float32))
                if audio_chunks:
                    return np.concatenate(audio_chunks), self.sr
            except Exception as e:  # noqa: BLE001
                log.warning("Kokoro 합성 오류(%s) — placeholder", e)
        return self._placeholder(text), self.sr

    def synth_stream(self, text: str, chunk_ms: int = 200) -> Iterator[np.ndarray]:
        y, sr = self.synth(text)
        step = int(sr * chunk_ms / 1000)
        for i in range(0, len(y), step):
            yield y[i:i + step]

    def _placeholder(self, text: str) -> np.ndarray:
        dur = max(0.4, len(text) * 0.07)
        t = np.linspace(0, dur, int(self.sr * dur), endpoint=False)
        freq = 175 if self.speaker == "A" else 135
        return (0.04 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
