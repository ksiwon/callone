"""설문 → 페르소나 (전시 call:one 트랙①·② 공용, LLM 불필요 — 순수 템플릿).

연구 근거(demo/EXHIBIT_PLAN §2): 스탠퍼드 1,052명 생성 에이전트 — 잘 짠 구조화 설문만으로
2시간 인터뷰의 99%(82% vs 83%) 성능. 짧은 설문 답을 캐릭터 카드 + 기억 시드로 변환하면
기존 파이프라인(캐릭터 카드 프롬프트 + UtteranceRAG 회상)이 그대로 "그 사람"을 만든다.

두 모드:
- future_self (트랙① 《나》): 6문항 → "10년 뒤 나" 카드 + 기억 시드 + 부메랑(작별 직전
  되돌려줄 한마디 — farewell 지시문에 이어붙인다).
- loved_one  (트랙② 《그 사람》): 관계 설문 → 전사에서 안 나오는 관계 고유 신호(호칭·입버릇·
  장난)를 카드에 주입 + 함께한 기억을 기억 시드로.

반환 card 키는 serve 의 SessionInit 캐릭터 카드 필드와 1:1
(persona/personality/background/situation/first_message/example_dialogue/user_persona).
memories 는 memory_update 와 같은 규약(짧은 한 문장 문자열 배열) — 세션 시드 or
memories.json append 어느 쪽에도 쓸 수 있다.
"""
from __future__ import annotations

import re

CARD_KEYS = ("persona", "personality", "background", "situation",
             "first_message", "example_dialogue", "user_persona")


def _clean(v) -> str:
    return re.sub(r"\s+", " ", str(v or "").strip())


def _slider(v, low: str, mid: str, high: str) -> str:
    """0~1 슬라이더 → 성격 서술. 잘못된 값은 중간 취급."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return mid
    return low if x < 0.35 else (high if x > 0.65 else mid)


def _dedup(facts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for f in facts:
        f = _clean(f)
        k = re.sub(r"[^가-힣a-z0-9]", "", f.lower())
        if len(f) >= 4 and k and k not in seen:
            seen.add(k)
            out.append(f)
    return out


def _future_self(name: str, a: dict) -> dict:
    worry, future = _clean(a.get("worry")), _clean(a.get("future"))
    person, joy = _clean(a.get("person")), _clean(a.get("joy"))
    message = _clean(a.get("message"))

    tone = _slider(a.get("extraversion"),
                   "말수가 적고 조용조용하다", "차분하다", "말이 많고 활기차다")
    pace = _slider(a.get("spontaneity"),
                   "신중하게 한 번 더 생각하고 말한다", "담백하게 말한다",
                   "즉흥적이고 스스럼없이 말한다")

    card = {
        "persona": f"{name}의 10년 뒤 자신. 그 사이의 일을 다 겪었고, 지금의 {name}을 누구보다 잘 안다.",
        "user_persona": f"10년 전의 나({name})",
        "personality": (f"{tone}. {pace}. 상대는 과거의 나 자신이라 말투가 서로 똑같다 — "
                        "편한 반말, 놀리듯 다정하게."),
        "situation": "10년 뒤의 내가 과거의 나에게 딱 한 번 허락된 전화를 걸었다. 통화는 길지 않다.",
        "first_message": "여보세요, 나야. …진짜 나. 목소리 듣자마자 알겠지?",
    }
    if worry:
        card["background"] = f"과거의 나는 지금 {worry} 때문에 고민하고 있다. 나는 그 일이 어떻게 되는지 안다."

    memories = _dedup([
        worry and f"사용자는 요즘 {worry} 때문에 고민한다",
        future and f"사용자는 10년 뒤 자신이 {future} 모습이길 상상한다",
        person and f"사용자가 가장 아끼는 사람은 {person}이다",
        joy and f"사용자는 최근 {joy} 덕분에 웃었다",
        message and f"사용자가 10년 뒤 자신에게 남긴 한마디: {message}",
    ])

    out = {"card": {k: v for k, v in card.items() if v}, "memories": memories}
    if message:
        # 부메랑: farewell 지시문 뒤에 이어붙여 작별 직전 되돌려주게 한다(전시 정점 연출).
        out["boomerang"] = (f"(작별 인사 직전에, 과거의 내가 남긴 한마디 '{message}' 를 "
                            "언급하며 같은 말을 따뜻하게 되돌려줘라.)")
    return out


def _loved_one(name: str, a: dict) -> dict:
    nick_me = _clean(a.get("nickname_me"))       # 그 사람이 나를 부르던 호칭
    nick_them = _clean(a.get("nickname_them"))   # 내가 그 사람을 부르는 호칭
    catch, teasing = _clean(a.get("catchphrase")), _clean(a.get("teasing"))
    last_topic = _clean(a.get("last_topic"))
    mems = a.get("memories") or []
    if isinstance(mems, str):
        mems = mems.splitlines()
    mems = [_clean(m) for m in mems if len(_clean(m)) >= 2]   # 접두어 붙기 전, 원답 기준 필터

    traits = [f"사용자를 '{nick_me}'(이)라고 부른다" if nick_me else "",
              f"입버릇처럼 자주 하는 말: \"{catch}\"" if catch else "",
              f"사용자를 이렇게 놀리곤 한다: {teasing}" if teasing else ""]
    card = {
        "persona": f"{name}. 사용자와 아주 가까운 사이다.",
        "user_persona": (f"{name}에게 소중한 사람" + (f" — {name}을(를) '{nick_them}'(이)라고 부른다"
                                                  if nick_them else "")),
        "personality": ". ".join(t for t in traits if t) or "평소 통화하던 그대로, 편하고 자연스럽게.",
        "situation": "오랜만에 사용자에게 전화가 걸려 왔다.",
    }
    if last_topic:
        card["background"] = f"마지막으로 나눈 대화 주제는 {last_topic}였다. 자연스럽게 그 뒤가 궁금하다."

    memories = _dedup(
        [f"사용자와의 기억: {m}" for m in mems]
        + [last_topic and f"사용자와 마지막으로 {last_topic} 이야기를 나눴다",
           nick_me and f"{name}은(는) 사용자를 '{nick_me}'(이)라고 부른다"])
    return {"card": {k: v for k, v in card.items() if v}, "memories": memories}


def persona_from_survey(name: str, answers: dict, mode: str = "future_self") -> dict:
    """설문 답 → {card, memories, [boomerang]}.

    name: 통화 상대 이름(트랙①=관람객 본인, 트랙②=그 사람).
    answers: 설문 키-값. future_self: worry/future/person/joy/message/extraversion/spontaneity.
             loved_one: nickname_me/nickname_them/catchphrase/teasing/last_topic/memories.
    """
    name = _clean(name) or "나"
    if mode == "loved_one":
        return _loved_one(name, answers or {})
    if mode != "future_self":
        raise ValueError(f"unknown persona survey mode: {mode}")
    return _future_self(name, answers or {})
