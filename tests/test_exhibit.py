"""전시 소멸 카운터 — 오늘/누적, 날짜 넘어가면 today 리셋."""
from __future__ import annotations

from callone.serve.exhibit import bump, current


def test_counter_bump_and_daily_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path))

    assert current("2026-11-21") == {"day": "2026-11-21", "today": 0, "total": 0}
    bump("2026-11-21")
    st = bump("2026-11-21")
    assert st["today"] == 2 and st["total"] == 2

    st = bump("2026-11-22")                      # 다음 날 — today 리셋, total 누적
    assert st["today"] == 1 and st["total"] == 3
    assert current("2026-11-22")["today"] == 1


def test_counter_survives_corrupt_file(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path))
    (tmp_path / "exhibit_count.json").write_text("{broken", encoding="utf-8")
    st = bump("2026-11-21")                      # 깨진 파일 → 0부터, 크래시 없음
    assert st == {"day": "2026-11-21", "today": 1, "total": 1}
