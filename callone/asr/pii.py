"""PII 마스킹 (§12, §20) — 학습셋 저장 시 강제 적용.

정규식 + (선택) 한국어 NER 로 이름·전화·주소·주민번호·계좌 마스킹.
원본은 암호화 보관, 학습셋은 마스킹본만.
"""
from __future__ import annotations

import re

# 한국 전화/휴대폰
_PHONE = re.compile(r"\b(01[016789]|0\d{1,2})[-\s]?\d{3,4}[-\s]?\d{4}\b")
# 주민등록번호
_RRN = re.compile(r"\b\d{6}[-\s]?[1-4]\d{6}\b")
# 계좌번호 (느슨)
_ACCOUNT = re.compile(r"\b\d{2,6}[-\s]\d{2,6}[-\s]\d{2,7}\b")
# 주소 키워드
_ADDR = re.compile(r"[가-힣]+(시|도)\s?[가-힣]+(구|군|시)\s?[가-힣]+(동|읍|면|로|길)\s?\d*")
# 이메일
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

DEFAULT_TOKENS = {
    "name": "[NAME]", "phone": "[PHONE]", "addr": "[ADDR]",
    "rrn": "[RRN]", "account": "[ACCOUNT]", "email": "[EMAIL]",
}


def mask_text(text: str, tokens: dict | None = None, use_ner: bool = False) -> str:
    t = tokens or DEFAULT_TOKENS
    out = text
    out = _RRN.sub(t["rrn"], out)
    out = _PHONE.sub(t["phone"], out)        # 전화 먼저 (계좌 패턴과 겹침 방지)
    out = _ACCOUNT.sub(t["account"], out)
    out = _EMAIL.sub(t.get("email", "[EMAIL]"), out)
    out = _ADDR.sub(t["addr"], out)
    if use_ner:
        out = _mask_names_ner(out, t["name"])
    return out


def _mask_names_ner(text: str, token: str) -> str:
    """한국어 NER 로 인명 마스킹 (선택, 미설치 시 no-op)."""
    try:
        # 예: transformers 한국어 NER 파이프라인 (로컬 모델)
        from transformers import pipeline  # type: ignore

        ner = _get_ner()
        for ent in ner(text):
            if ent.get("entity_group") in ("PS", "PER", "PERSON"):
                text = text.replace(ent["word"], token)
    except Exception:
        pass
    return text


_NER_CACHE = {}


def _get_ner():
    if "ner" not in _NER_CACHE:
        from transformers import pipeline  # type: ignore

        _NER_CACHE["ner"] = pipeline("token-classification",
                                     model="KPF/KPF-bert-ner",
                                     aggregation_strategy="simple")
    return _NER_CACHE["ner"]


def scan_pii(text: str) -> list[str]:
    """PII 누출 검사용 — 발견된 패턴 종류 반환 (test 에서 사용, §19)."""
    found = []
    if _RRN.search(text):
        found.append("rrn")
    if _PHONE.search(text):
        found.append("phone")
    if _ACCOUNT.search(text):
        found.append("account")
    if _EMAIL.search(text):
        found.append("email")
    return found
