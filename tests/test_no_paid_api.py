"""공통 (§19, §20): 외부 유료 API 호출 0 — 코드 정적 검사로 금지 패턴 차단.

CI/빌드에서 이 테스트 실패 = 유료 API 흔적 발견 = 빌드 실패.
"""
import re
from pathlib import Path

# 금지 도메인/패키지 (§2, §20). 호출/import 금지.
FORBIDDEN = [
    r"api\.openai\.com",
    r"api\.elevenlabs\.io",
    r"api\.assemblyai\.com",
    r"generativelanguage\.googleapis\.com",   # Gemini API
    r"dashscope\.aliyuncs\.com",              # Alibaba 클라우드 보이스 API
    r"import\s+openai\b",
    r"from\s+openai\b",
    r"import\s+elevenlabs\b",
    r"import\s+assemblyai\b",
    r"precision-2",                            # pyannote 유료 모델
]

REPO = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["callone", "configs", "scripts"]


def _iter_files():
    for d in SCAN_DIRS:
        base = REPO / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix in (".py", ".yaml", ".yml", ".sh", ".toml", ".md"):
                yield p


def test_no_forbidden_paid_api():
    hits = []
    pats = [re.compile(f) for f in FORBIDDEN]
    for p in _iter_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        # 주석으로 '금지'를 설명하는 줄은 허용 (이 테스트 파일 자체 + 설명)
        for pat in pats:
            for m in pat.finditer(text):
                line = text[:m.start()].count("\n") + 1
                ctx = text.splitlines()[line - 1] if line - 1 < len(text.splitlines()) else ""
                # 금지/forbidden/호출 금지 설명 줄은 면제
                if any(k in ctx for k in ("금지", "forbidden", "FORBIDDEN", "차단", "호출 금지")):
                    continue
                hits.append(f"{p.relative_to(REPO)}:{line}: {pat.pattern}")
    assert not hits, "유료 API 금지 패턴 발견:\n" + "\n".join(hits)
