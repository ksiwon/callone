"""S5 장기 메모리 (§15.2 층3) — 세션 간 기억(Mem0/Zep).

대화 사이 사실/약속/맥락 유지. 라이브러리 미설치 시 JSON 파일 폴백.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.io import data_dir
from ..common.logging import get_logger

log = get_logger("memory")


class LongTermMemory:
    def __init__(self, speaker: str, user_id: str = "default", backend: str = "mem0"):
        self.speaker = speaker
        self.user_id = user_id
        self.backend = backend
        self._mem = self._try_backend()
        self._file = data_dir() / "speakers" / speaker / f"memory_{user_id}.json"

    def _try_backend(self):
        if self.backend == "mem0":
            try:
                from mem0 import Memory  # type: ignore

                return Memory()
            except Exception as e:  # noqa: BLE001
                log.warning("Mem0 미설치(%s) — JSON 폴백", e)
        return None

    def add(self, text: str, role: str = "user") -> None:
        if self._mem is not None:
            self._mem.add(text, user_id=f"{self.speaker}:{self.user_id}",
                          metadata={"role": role})
            return
        items = self._load()
        items.append({"role": role, "text": text})
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def recall(self, query: str, k: int = 5) -> str:
        if self._mem is not None:
            res = self._mem.search(query, user_id=f"{self.speaker}:{self.user_id}", limit=k)
            return "\n".join(f"- {r['memory']}" for r in res.get("results", []))
        items = self._load()
        toks = set(query.split())
        scored = sorted(items, key=lambda it: -len(toks & set(it["text"].split())))
        return "\n".join(f"- {it['text']}" for it in scored[:k])

    def _load(self) -> list[dict]:
        if self._file.exists():
            return json.loads(self._file.read_text(encoding="utf-8"))
        return []
