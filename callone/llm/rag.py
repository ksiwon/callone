"""S5 RAG (§15.2 층3) — 실제 발화 임베딩 → 벡터DB 검색.

EmbeddingGemma 로 화자 발화 임베딩 → FAISS 색인 → 질의 시 유사 발화 검색.
"예전에 ~했잖아" 일관성/사실성. 임베더 미설치 시 키워드 폴백.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common.io import data_dir, read_json
from ..common.logging import get_logger

log = get_logger("rag")

_EMBED_MODEL = "google/embeddinggemma"


class UtteranceRAG:
    def __init__(self, speaker: str, cfg: dict | None = None, use_vectors: bool = False):
        # use_vectors=False(기본): 키워드 검색 — 빠르고 임베더 다운로드 없음(EmbeddingGemma는
        # 게이트+대용량이라 온디바이스 초기화가 느림/행). 정밀 의미검색 필요시 use_vectors=True.
        self.speaker = speaker
        self.cfg = cfg or {}
        self.use_vectors = use_vectors or (cfg or {}).get("use_vectors", False)
        self.texts: list[str] = []
        self.index = None
        self._embedder = None
        self._build()

    def _load_texts(self) -> list[str]:
        # ⚠️ 서빙 프로세스에선 pandas 금지(OpenVINO 와 segfault). 사전 export 된
        #    plain JSON(utterances.json)만 읽는다(json, torch/pandas 무관).
        #    JSON 없으면 precompute(오프라인)에서 parquet→json 생성 후 사용.
        up = data_dir() / "speakers" / self.speaker / "utterances.json"
        if up.exists():
            try:
                return [t for t in read_json(up) if isinstance(t, str) and t.strip()]
            except Exception:  # noqa: BLE001
                pass
        log.warning("발화 JSON 없음(%s) — scripts/precompute_voice_emb.py 로 생성", up)
        return []

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer(
                self.cfg.get("embedder", _EMBED_MODEL))
        return self._embedder

    def _build(self):
        self.texts = [t for t in self._load_texts() if t.strip()]
        if not self.texts:
            return
        if not self.use_vectors:
            log.info("RAG(키워드) %s: %d 발화", self.speaker, len(self.texts))
            return
        try:
            import faiss  # type: ignore

            emb = self._get_embedder().encode(self.texts, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(np.asarray(emb, dtype=np.float32))
            log.info("RAG 벡터색인 %s: %d 발화", self.speaker, len(self.texts))
        except Exception as e:  # noqa: BLE001
            log.warning("벡터 색인 불가(%s) — 키워드 폴백", e)
            self.index = None

    def search(self, query: str, k: int = 3) -> list[str]:
        if not self.texts:
            return []
        if self.index is not None:
            q = self._get_embedder().encode([query], normalize_embeddings=True)
            _, idx = self.index.search(np.asarray(q, dtype=np.float32), k)
            return [self.texts[i] for i in idx[0] if i < len(self.texts)]
        # 키워드 폴백
        toks = set(query.split())
        scored = sorted(self.texts, key=lambda t: -len(toks & set(t.split())))
        return scored[:k]

    def context(self, query: str, k: int = 3) -> str:
        hits = self.search(query, k)
        return "\n".join(f"- {h}" for h in hits)
