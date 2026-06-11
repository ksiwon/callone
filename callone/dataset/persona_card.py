"""페르소나 카드 생성 (§7.7, §15.2) — profile.json(자동+사용자) → system 프롬프트.

포함: 정체성, 말투, 사투리(지역·세기·대표 어미 — 자동 측정값), 입버릇,
자주 묻고/답하는 주제, 유머, 금기, "모르면 모른다" 원칙.
"""
from __future__ import annotations

from ..common.schemas import SpeakerProfile


def build_persona_card(prof: SpeakerProfile) -> str:
    u = prof.user
    a = prof.auto
    region = prof.effective_region()
    intensity = prof.effective_intensity()
    name = u.name or f"화자 {prof.speaker_id}"

    markers = ", ".join(f"{m.form}({m.std})" if m.std else m.form
                        for m in a.dialect.markers[:8])
    fillers = ", ".join(a.speech.top_fillers) or "없음"
    catch = ", ".join(u.catchphrases) or "없음"
    traits = ", ".join(u.traits) or "정보 없음"
    taboo = ", ".join(u.taboo) or "없음"

    intensity_label = (
        "거의 표준어" if intensity < 0.05 else
        "약한 사투리" if intensity < 0.15 else
        "중간 사투리" if intensity < 0.30 else "강한 사투리"
    )

    lines = [
        f"너는 '{name}'이다. 관계: {u.relation or '미상'}.",
        f"성격/특징: {traits}.",
        f"말투: {u.register}. 평균 문장 길이 약 {a.speech.avg_sentence_len:.0f}어절, "
        f"질문 비율 {a.speech.question_rate:.0%}.",
        f"사투리: {region} 방언, 세기 {intensity:.2f}({intensity_label}). "
        f"대표 어미/어휘: {markers}." if region not in ("standard", "unknown")
        else "사투리: 표준어에 가깝게 말한다.",
        f"입버릇/감탄사: {fillers}.",
        f"자주 쓰는 말: {catch}.",
        f"금기(하지 말 것): {taboo}.",
        "원칙: 모르거나 기억나지 않으면 솔직히 '모른다/기억 안 난다'고 말한다. "
        "지어내지 않는다.",
        "너는 실제 통화하듯 짧고 자연스럽게 반응한다. 그 사람 말투를 그대로 쓴다.",
    ]
    return "\n".join(lines)


def card_from_profile_path(path: str) -> str:
    from ..common.io import read_json

    return build_persona_card(SpeakerProfile(**read_json(path)))
