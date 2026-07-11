"""전시(call:one) 소멸 카운터 — 개인 데이터 0, 순수 집계.

벽면 연출 "오늘 ___개의 목소리가 태어나고 사라졌습니다"의 숫자. 통화(세션)가 끝나
목소리가 폐기될 때 키오스크가 bump() 를 호출한다. 날짜가 바뀌면 오늘 카운트는 0부터
(total 은 누적). ephemeral 원칙과 무관 — 세었다는 사실만 남고 무엇을 세었는지는 없다.
"""
from __future__ import annotations

import threading
from datetime import date

from ..common.io import data_dir, read_json, write_json

_LOCK = threading.Lock()


def _path():
    return data_dir() / "exhibit_count.json"


def _load(today: str) -> dict:
    p = _path()
    st = {"day": today, "today": 0, "total": 0}
    if p.exists():
        try:
            old = read_json(p)
            st["total"] = int(old.get("total", 0))
            if old.get("day") == today:
                st["today"] = int(old.get("today", 0))
        except Exception:  # noqa: BLE001
            pass
    return st


def current(today: str | None = None) -> dict:
    """오늘/누적 카운트 조회. {day, today, total}"""
    today = today or date.today().isoformat()
    with _LOCK:
        return _load(today)


def bump(today: str | None = None) -> dict:
    """소멸 1회 기록(세션 종료 시). 날짜 바뀌면 today 리셋."""
    today = today or date.today().isoformat()
    with _LOCK:
        st = _load(today)
        st["today"] += 1
        st["total"] += 1
        write_json(_path(), st)
        return st
