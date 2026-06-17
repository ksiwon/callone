"""_parse_emotion 회귀 — 감정 태그가 TTS 로 새지 않게(대괄호 누출 방지).

실통화 버그(2026-06-17): LLM 이 목록에 없는 [emotion:surprised] 를 내자 태그가
제거되지 않아 "대괄호가 그대로 읽히고" 음색까지 교란됨. 단어 무관 제거로 수정.
"""
from callone.serve.orchestrator import _parse_emotion


def test_known_emotion_extracted_and_stripped():
    assert _parse_emotion("[emotion:happy] 어, 왔나!") == ("happy", "어, 왔나!")
    assert _parse_emotion("[emotion:surprised] 아! 네가 있네") == ("surprised", "아! 네가 있네")


def test_rich_emotions_passthrough():
    # 확장 팔레트(tender/playful/...)도 그대로 전달 + 태그 제거
    assert _parse_emotion("[emotion:tender] 우리 딸~") == ("tender", "우리 딸~")
    assert _parse_emotion("[emotion:playful] 야 장난이지") == ("playful", "야 장난이지")


def test_unknown_emotion_no_bracket_leak():
    # 모르는 감정([emotion:X])도 태그는 반드시 제거(브래킷 누출 0). 감정값은 TTS 가 neutral 폴백.
    for tag in ("[emotion:joyful] 안녕", "[emotion:기쁨] 안녕"):
        emo, clean = _parse_emotion(tag)
        assert clean == "안녕" and "[" not in clean and "emotion" not in clean


def test_bare_and_paren_tags():
    assert _parse_emotion("[happy] 하이") == ("happy", "하이")
    assert _parse_emotion("(sad) 흠") == ("sad", "흠")


def test_plain_text_untouched():
    assert _parse_emotion("그냥 평문") == ("neutral", "그냥 평문")


def test_json_form():
    emo, clean = _parse_emotion('{"emotion":"sad","reply":"어이구..."}')
    assert emo == "sad" and clean == "어이구..."
