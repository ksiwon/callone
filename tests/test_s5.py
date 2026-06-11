"""test_s5 (§19): 페르소나 프롬프트 조립 + "모름" 캘리브레이션 존재."""
from callone.llm.persona_prompt import build_messages
from callone.llm.prepare_sft import _DONTKNOW


def test_build_messages_structure():
    msgs = build_messages("A", "화자 A 나 왔어", rag_context="- 밥은 묵었나")
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "밥은 묵었나" in msgs[0]["content"]


def test_dontknow_calibration_present():
    # 환각 억제: "모른다" 예시가 SFT 에 주입됨
    assert any("기억" in ex["a"] or "모른" in ex["a"] or "알 수" in ex["a"]
               for ex in _DONTKNOW)
