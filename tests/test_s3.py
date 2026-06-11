"""test_s3 (§19): PII 누출 0 + 텍스트 정규화 + 대화셋 역할 정합."""
from callone.asr.pii import mask_text, scan_pii
from callone.dataset.build_tts import normalize_ko
from callone.dataset.persona_card import build_persona_card
from callone.common.schemas import SpeakerProfile


def test_pii_masked():
    txt = "내 번호 010-1234-5678 이고 주민번호 901231-1234567 이야"
    masked = mask_text(txt)
    assert "[PHONE]" in masked
    assert "[RRN]" in masked
    assert scan_pii(masked) == []   # 누출 0


def test_normalize_numbers():
    assert "이" in normalize_ko("2시")  # 숫자→한글 음독


def test_persona_card_contains_dialect():
    p = SpeakerProfile(speaker_id="A")
    p.user.name = "화자 A"
    p.user.relation = "어머니"
    p.auto.dialect.region_est = "gyeongsang"
    p.auto.dialect.intensity_0to1 = 0.6
    card = build_persona_card(p)
    assert "화자 A" in card
    assert "gyeongsang" in card
    assert "모른" in card   # "모르면 모른다" 원칙
