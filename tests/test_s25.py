"""test_s25 (§19): dialect intensity 연속값 산출 + 프로필 라운드트립."""
from callone.profile.s25_dialect import profile_dialect
from callone.common.schemas import SpeakerProfile


CFG = {"dialect": {"regions": ["gyeongsang", "jeolla", "chungcheong", "gangwon"],
                   "markers_dir": "resources/dialect_markers", "min_tokens": 3}}


def test_intensity_continuous_gyeongsang():
    # 경상 마커 다수 → 지역=경상, 세기>0
    text = "밥은 묵었나 이래가 안 된다카이 머라카노 억수로 마 됐다아이가"
    da = profile_dialect(text, CFG)
    assert da.region_est == "gyeongsang"
    assert 0.0 < da.intensity_0to1 <= 1.0


def test_intensity_varies_by_speaker():
    weak = profile_dialect("밥은 먹었니 그래 알겠어 고마워 응 그래", CFG)
    strong = profile_dialect("묵었나 카이 머라카노 억수로 안카나 됐다아이가 와이라노", CFG)
    assert strong.intensity_0to1 >= weak.intensity_0to1


def test_standard_fallback():
    da = profile_dialect("네 알겠습니다 감사합니다 안녕하세요", CFG)
    assert da.region_est in ("standard", "gangwon", "chungcheong")  # 표준 또는 약신호


def test_profile_roundtrip():
    p = SpeakerProfile(speaker_id="A")
    p.user.name = "화자 A"
    p.user.dialect_intensity_override = 0.7
    d = p.model_dump()
    p2 = SpeakerProfile(**d)
    assert p2.effective_intensity() == 0.7
    assert p2.user.name == "화자 A"
