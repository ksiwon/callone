"""SQLite 메타 DB — calls 테이블 + manifest parquet 동기화.

가벼운 wrapper. 무거운 ORM 안 씀. 재현성 위해 parquet 도 같이 씀.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .io import REPO_ROOT
from .schemas import CallMeta

DB_PATH = REPO_ROOT / "db" / "callone.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  call_id       TEXT PRIMARY KEY,
  src_path      TEXT,
  wav16k_path   TEXT,
  restored_path TEXT,
  duration_sec  REAL,
  orig_sr       INTEGER,
  orig_channels INTEGER,
  codec         TEXT,
  status        TEXT,
  error         TEXT,
  created_at    TEXT
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


def upsert_call(con: sqlite3.Connection, m: CallMeta) -> None:
    d = m.model_dump()
    cols = list(d.keys())
    placeholders = ",".join("?" for _ in cols)
    updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "call_id")
    sql = (
        f"INSERT INTO calls ({','.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(call_id) DO UPDATE SET {updates}"
    )
    con.execute(sql, [d[c] for c in cols])
    con.commit()


def all_calls(con: sqlite3.Connection) -> list[CallMeta]:
    rows = con.execute("SELECT * FROM calls ORDER BY call_id").fetchall()
    return [CallMeta(**dict(r)) for r in rows]


def write_manifest(calls: Iterable[CallMeta], path: str | Path) -> None:
    """manifest.parquet 작성 (pandas/pyarrow)."""
    import pandas as pd

    df = pd.DataFrame([c.model_dump() for c in calls])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
