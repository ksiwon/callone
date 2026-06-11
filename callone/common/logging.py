"""구조적 로깅 — rich 있으면 컬러, 없으면 표준 logging 폴백."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _force_utf8_stdio() -> None:
    """Windows cp949 콘솔에서 한글/유니코드 출력 깨짐/크래시 방지."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def get_logger(name: str = "callone") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        _force_utf8_stdio()
        level = os.environ.get("CALLONE_LOG", "INFO").upper()
        try:
            from rich.console import Console  # type: ignore
            from rich.logging import RichHandler  # type: ignore

            # legacy_windows=False → cp949 콘솔 크래시 회피, UTF-8 사용
            console = Console(file=sys.stdout, legacy_windows=False)
            logging.basicConfig(
                level=level,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
            )
        except Exception:
            logging.basicConfig(
                level=level,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        _CONFIGURED = True
    return logging.getLogger(name)
