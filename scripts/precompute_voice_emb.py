"""화자 사전계산(오프라인) — 서빙(OpenVINO)이 torch/pandas 없이 쓰도록 준비.

생성물 (data/speakers/{spk}/):
  - voice_emb.npy     : ECAPA 음색 임베딩(TTS 클론용). speechbrain=torch 사용.
  - utterances.json   : 화자 A 실제 발화 목록(RAG 기억용). parquet→json(pandas 사용).

⚠️ torch/pandas 를 쓰므로 **서빙과 별도 프로세스로 1회** 실행. 서빙은 npy/json 만 로드.

사용:
  python scripts/precompute_voice_emb.py --speakers A B
"""
from __future__ import annotations

import argparse

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo 루트 → callone import

from callone.common.io import data_dir, write_json  # noqa: E402
from callone.common.logging import get_logger  # noqa: E402
from callone.serve.tts_kokoro import speaker_embedding  # noqa: E402

log = get_logger("precompute")


def export_utterances(speaker: str):
    """global_assignment.parquet → 화자 깨끗한 발화 목록 utterances.json (RAG용)."""
    ga = data_dir() / "speakers" / "global_assignment.parquet"
    texts = []
    if ga.exists():
        import pandas as pd

        df = pd.read_parquet(ga)
        texts = df[(df["global_speaker"] == speaker) & (df["clean"])]["text"].dropna().tolist()
    texts = [t for t in texts if t and t.strip()]
    out = data_dir() / "speakers" / speaker / "utterances.json"
    write_json(out, texts)
    log.info("화자 %s 발화 %d개 → %s", speaker, len(texts), out)


def main():
    ap = argparse.ArgumentParser(description="화자 ECAPA 음색 임베딩 사전계산")
    ap.add_argument("--speakers", nargs="+", default=["A", "B"])
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    for spk in args.speakers:
        export_utterances(spk)                       # RAG 발화 JSON
        emb = speaker_embedding(spk, args.topk)      # 음색 임베딩
        if emb is None:
            log.warning("화자 %s 음색 임베딩 실패(클립 없음?)", spk)
            continue
        out = data_dir() / "speakers" / spk / "voice_emb.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, emb)
        log.info("화자 %s 음색 임베딩 저장: %s (dim=%d)", spk, out, emb.shape[-1])


if __name__ == "__main__":
    main()
