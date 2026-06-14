"""S5 페르소나 프롬프트 조립 (§15.2 층1).

페르소나 카드(system) + RAG 검색 컨텍스트 + 메모리 → 최종 프롬프트 메시지.
추론 시(S6) 매 턴 호출.
"""
from __future__ import annotations

from ..common.io import data_dir, read_json
from ..common.schemas import SpeakerProfile
from ..dataset.persona_card import build_persona_card


def load_persona(speaker: str) -> str:
    pj = data_dir() / "speakers" / speaker / "profile.json"
    if pj.exists():
        return build_persona_card(SpeakerProfile(**read_json(pj)))
    return f"너는 화자 {speaker}이다. 자연스럽게 대화한다."


def build_messages(speaker: str, user_text: str, history: list[dict] | None = None,
                   rag_context: str | None = None, memory: str | None = None,
                   persona_override: str | None = None,
                   situation: str | None = None) -> list[dict]:
    # persona_override: 통화 시 사용자가 직접 지정한 "이 사람은 누구인가"(학습 페르소나 대체).
    system = persona_override.strip() if (persona_override and persona_override.strip()) \
        else load_persona(speaker)
    if situation and situation.strip():
        system += f"\n\n[지금 통화 상황 — 이 맥락에서 대화한다]\n{situation.strip()}"
    if rag_context:
        system += f"\n\n[참고할 실제 발화 기억]\n{rag_context}"
    if memory:
        system += f"\n\n[장기 기억]\n{memory}"
    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user_text})
    return msgs
