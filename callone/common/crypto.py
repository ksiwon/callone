"""디스크 암호화 — §20 보안 요건.

통화 원본/전사/학습셋은 ENCRYPTION_KEY 로 대칭 암호화(Fernet).
키 없으면 평문 폴백(개발용) + 경고. 운영 시 키 필수.

사용:
  python -m callone.common.crypto genkey   # 새 키 출력 → .env 에 넣기
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .logging import get_logger

log = get_logger("crypto")


def _get_fernet():
    key = os.environ.get("ENCRYPTION_KEY", "").strip()
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet

        return Fernet(key.encode())
    except Exception as e:  # noqa: BLE001
        log.warning("ENCRYPTION_KEY 유효하지 않음(%s) — 평문 폴백", e)
        return None


def gen_key() -> str:
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def encrypt_bytes(data: bytes) -> bytes:
    f = _get_fernet()
    return f.encrypt(data) if f else data


def decrypt_bytes(data: bytes) -> bytes:
    f = _get_fernet()
    return f.decrypt(data) if f else data


def encrypt_file(path: str | Path) -> None:
    """제자리 암호화 (.enc 접미사). 키 없으면 no-op + 경고."""
    f = _get_fernet()
    if not f:
        log.warning("키 없음 — %s 암호화 건너뜀(개발 모드)", path)
        return
    p = Path(path)
    enc = f.encrypt(p.read_bytes())
    Path(str(p) + ".enc").write_bytes(enc)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "genkey":
        print(gen_key())
    else:
        print("사용법: python -m callone.common.crypto genkey")


if __name__ == "__main__":
    main()
