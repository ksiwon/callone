"""설문→페르소나 변환 검증 — 전시 트랙①(10년 뒤 나)·트랙②(그 사람)."""
from __future__ import annotations

import pytest

from callone.llm.persona_survey import CARD_KEYS, persona_from_survey


def test_future_self_full():
    r = persona_from_survey("시원", {
        "worry": "졸업 전시", "future": "좋아하는 일을 하며 사는",
        "person": "엄마", "joy": "친구들과의 여행",
        "message": "너무 걱정하지 마", "extraversion": 0.9, "spontaneity": 0.1,
    })
    card = r["card"]
    assert set(card) <= set(CARD_KEYS)                      # SessionInit 필드와 정합
    assert "시원" in card["persona"] and "10년" in card["persona"]
    assert "말이 많고 활기차다" in card["personality"]       # 슬라이더 high
    assert "신중하게" in card["personality"]                 # 슬라이더 low
    assert "졸업 전시" in card["background"]
    assert any("엄마" in m for m in r["memories"])
    assert any("너무 걱정하지 마" in m for m in r["memories"])
    assert "너무 걱정하지 마" in r["boomerang"]              # 부메랑 = 작별 직전 연출 재료


def test_future_self_sparse_answers():
    r = persona_from_survey("", {"worry": "  ", "extraversion": "not-a-number"})
    card = r["card"]
    assert "나" in card["persona"]                           # 이름 없으면 "나"
    assert "차분하다" in card["personality"]                  # 잘못된 슬라이더 → 중간
    assert "background" not in card                          # 빈 답 → 필드 생략
    assert r["memories"] == []
    assert "boomerang" not in r                              # 한마디 없으면 부메랑 없음


def test_loved_one_card_and_memories():
    r = persona_from_survey("엄마", {
        "nickname_me": "우리 강아지", "nickname_them": "엄마",
        "catchphrase": "밥은 먹었나", "teasing": "잠꾸러기라고 놀림",
        "last_topic": "이사 준비",
        "memories": "같이 부산 여행\n김장하던 날\n",
    }, mode="loved_one")
    card = r["card"]
    assert set(card) <= set(CARD_KEYS)
    assert "우리 강아지" in card["personality"]
    assert "밥은 먹었나" in card["personality"]
    assert "이사 준비" in card["background"]
    assert any("부산 여행" in m for m in r["memories"])
    assert any("김장" in m for m in r["memories"])


def test_memories_dedup_and_short_filter():
    r = persona_from_survey("A", {"memories": ["같이 간 제주도", "같이 간 제주도", "ㅎ"]},
                            mode="loved_one")
    assert sum("제주도" in m for m in r["memories"]) == 1    # 중복 1개만
    assert not any(m.endswith("ㅎ") for m in r["memories"])  # 너무 짧은 답 제외


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        persona_from_survey("A", {}, mode="nope")
