"""manifest 헬퍼 — 통화 메타 수집 + parquet/DB 동기화 (§7.1)."""
from __future__ import annotations

from pathlib import Path

from ..common.audio import ffprobe
from ..common.schemas import CallMeta


def probe_call(call_id: str, src: str | Path) -> CallMeta:
    info = ffprobe(src)
    return CallMeta(
        call_id=call_id,
        src_path=str(src),
        duration_sec=info.get("duration", 0.0),
        orig_sr=info.get("sr", 0),
        orig_channels=info.get("channels", 0),
        codec=info.get("codec", ""),
        status="pending",
    )


def call_id_from_path(p: Path, idx: int) -> str:
    """파일명 우선, 없으면 인덱스 기반 call_00001."""
    stem = p.stem
    if stem:
        return stem
    return f"call_{idx:05d}"
