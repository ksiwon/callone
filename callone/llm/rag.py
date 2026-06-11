"""S5 기억 회상 (§15.2) — 화자 '사실(기억)' 의미 검색.

Mem0 식 2단계의 2단계(검색). 우선순위:
  1) memories.json (extract_memories.py 가 뽑은 원자적 사실) — 삼천포 없이 정밀 회상
  2) utterances.json (날것 발화) — memories 없을 때 폴백

임베딩은 **fastembed(onnxruntime, torch 불필요)** → torch-free 서빙에 그대로 맞음.
fastembed 미설치/실패 시 키워드 검색 폴백. 임베딩은 디스크 캐시(memories_emb.npy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..common.io import data_dir, read_json
from ..common.logging import get_logger

log = get_logger("rag")

# 다국어(한국어 포함) 경량 임베더. cfg 의 'embedder' 로 교체 가능.
_EMBED_MODEL = "intfloat/multilingual-e5-small"


class UtteranceRAG:
    def __init__(self, speaker: str, cfg: dict | None = None, use_vectors: bool = True):
        self.speaker = speaker
        self.cfg = cfg or {}
        self.use_vectors = use_vectors and not (cfg or {}).get("keyword_only", False)
        self.model_name = (cfg or {}).get("embedder", _EMBED_MODEL)
        self.kind = "memories"          # memories | utterances
        self.texts: list[str] = []
        self._emb: np.ndarray | None = None
        self._embedder = None
        self._build()

    # ----- 소스 로드: memories 우선, 없으면 utterances ---------------------
    def _spk_dir(self) -> Path:
        return data_dir() / "speakers" / self.speaker

    def _load_texts(self) -> list[str]:
        mp = self._spk_dir() / "memories.json"
        if mp.exists():
            try:
                t = [x for x in read_json(mp) if isinstance(x, str) and x.strip()]
                if t:
                    self.kind = "memories"
                    return t
            except Exception:  # noqa: BLE001
                pass
        up = self._spk_dir() / "utterances.json"
        if up.exists():
            try:
                self.kind = "utterances"
                return [x for x in read_json(up) if isinstance(x, str) and x.strip()]
            except Exception:  # noqa: BLE001
                pass
        log.warning("기억/발화 없음(%s) — extract_memories.py 로 생성 권장", self._spk_dir())
        return []

    # ----- 임베딩(fastembed, onnx) ---------------------------------------
    def _is_e5(self) -> bool:
        return "e5" in self.model_name.lower()

    def _get_embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding  # type: ignore  # onnxruntime 기반(torch X)

            self._embedder = TextEmbedding(self.model_name)
        return self._embedder

    def _embed(self, texts: list[str], is_query: bool) -> np.ndarray:
        if self._is_e5():  # e5 계열은 prefix 필요
            pre = "query: " if is_query else "passage: "
            texts = [pre + t for t in texts]
        vecs = np.asarray(list(self._get_embedder().embed(texts)), dtype=np.float32)
        n = np.linalg.norm(vecs, axis=1, keepdims=True)   # 정규화(코사인=내적)
        return vecs / np.clip(n, 1e-9, None)

    def _build(self):
        self.texts = self._load_texts()
        if not self.texts:
            return
        if not self.use_vectors:
            log.info("RAG(키워드) %s: %s %d개", self.speaker, self.kind, len(self.texts))
            return
        cache = self._spk_dir() / "memories_emb.npy"
        try:
            if cache.exists():
                arr = np.load(cache)
                if arr.shape[0] == len(self.texts):
                    self._emb = arr
            if self._emb is None:
                self._emb = self._embed(self.texts, is_query=False)
                try:
                    np.save(cache, self._emb)
                except Exception:  # noqa: BLE001
                    pass
            log.info("RAG(의미검색) %s: %s %d개 (%s)",
                     self.speaker, self.kind, len(self.texts), self.model_name)
        except Exception as e:  # noqa: BLE001
            log.warning("임베더 불가(%s) — 키워드 폴백", e)
            self._emb = None
            self.use_vectors = False

    # ----- 검색 -----------------------------------------------------------
    def search(self, query: str, k: int = 3) -> list[str]:
        if not self.texts:
            return []
        if self._emb is not None:
            try:
                q = self._embed([query], is_query=True)[0]
                sims = self._emb @ q
                idx = np.argsort(-sims)[:k]
                return [self.texts[i] for i in idx]
            except Exception as e:  # noqa: BLE001
                log.warning("의미검색 오류(%s) — 키워드", e)
        toks = set(query.split())
        return sorted(self.texts, key=lambda t: -len(toks & set(t.split())))[:k]

    def context(self, query: str, k: int = 3) -> str:
        hits = self.search(query, k)
        return "\n".join(f"- {h}" for h in hits)
