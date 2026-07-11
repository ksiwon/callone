"""통화 이력 → 기억 성장 (라이브 버전의 extract_memories).

연구 근거(docs/REBUILD_PLAN·research): 기억은 대화로 재구성·성장한다(CHI26 Remember You)
+ "feeling heard"가 동반자 효과의 핵심(HBS) → 지난 통화 내용을 다음 통화가 기억하면
관계 연속성이 생긴다.

흐름: 유저가 통화 후 [기억시키기] 누름(클라 소유 이력을 명시적으로 서버 기억에 승격 —
ephemeral 원칙 위반 아님, export 와 같은 유저 주도 영속화) → llama-server 로 원자적
사실 추출 → data/speakers/<spk>/memories.json 에 중복 제거 후 append →
UtteranceRAG(use_rag: auto)가 다음 통화부터 회상.

오프라인 대량 버전(전사 파이프라인)은 scripts/extract_memories.py — 프롬프트/파싱 규약 동일.
"""
from __future__ import annotations

import json
import re
import urllib.request

from ..common.io import data_dir, read_json, write_json
from ..common.logging import get_logger

log = get_logger("memory_update")

_SYS = (
    "너는 통화 대화에서 '다음 통화 때 기억하면 좋을 사실'을 뽑는 도구다.\n"
    "규칙: ① 사용자(user)가 밝힌 근황·계획·감정·관계 사실 위주 ② 한 사실 = 짧은 한 문장, "
    "'사용자는 …' 형태 ③ 인사말·맞장구·일반 상식은 제외 ④ JSON 문자열 배열로만 출력.\n"
    '예시 출력: ["사용자는 다음 주에 이사한다", "사용자는 요즘 야근이 잦아 피곤해한다"]'
)


def _chat(base_url: str, history_text: str, timeout: float = 60) -> str:
    payload = {
        "messages": [{"role": "system", "content": _SYS},
                     {"role": "user", "content": f"대화:\n{history_text}\n\n출력:"}],
        "max_tokens": 500, "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def _parse_facts(text: str) -> list[str]:
    """LLM 출력 → 사실 리스트 (extract_memories.py 와 동일 규약: 첫 JSON 배열)."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    return [re.sub(r"\s+", " ", f.strip()) for f in arr
            if isinstance(f, str) and len(f.strip()) >= 6]


def _key(f: str) -> str:
    return re.sub(r"[^가-힣a-z0-9]", "", f.lower())


def remember_from_history(speaker: str, history: list[dict], base_url: str,
                          chat_fn=None) -> dict:
    """이력 → 새 기억 추출·병합. 반환 {added, total}. chat_fn 은 테스트 주입용."""
    lines = [f"{'사용자' if m.get('role') == 'user' else '상대'}: {m.get('content', '').strip()}"
             for m in history if m.get("content", "").strip()]
    if not lines:
        return {"added": 0, "total": 0}
    raw = (chat_fn or _chat)(base_url, "\n".join(lines[-80:]))   # 최근 80발화까지만
    facts = _parse_facts(raw)

    mp = data_dir() / "speakers" / speaker / "memories.json"
    old: list[str] = []
    if mp.exists():
        try:
            old = [x for x in read_json(mp) if isinstance(x, str)]
        except Exception:  # noqa: BLE001
            old = []
    seen = {_key(x) for x in old}
    added = [f for f in facts if _key(f) and _key(f) not in seen]
    if added:
        write_json(mp, old + added)
    log.info("기억 성장: +%d (총 %d) — speaker=%s", len(added), len(old) + len(added), speaker)
    return {"added": len(added), "total": len(old) + len(added)}
