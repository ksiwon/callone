"""연구 근거 기반 개선 검증 — 작별 메타턴(record=False) + 대화→기억 성장."""
from __future__ import annotations

import numpy as np

from callone.llm.memory_update import remember_from_history
from tests.test_v2_rebuild import _mini_orch


# ----- 작별 인사 = 메타 턴: 이력·user 이벤트에 안 남아야 함 --------------------
def test_meta_turn_hides_instruction_and_history():
    o = _mini_orch()
    events = list(o.stream_turn(np.zeros(1, dtype=np.float32), 16000,
                                user_text="(작별 인사 지시)", record=False))
    kinds = [k for k, _ in events]
    assert "user" not in kinds                       # 지시문이 사용자 발화로 위장 안 됨
    assert "audio" in kinds and "end" in kinds       # 인사는 합성돼 나감
    assert o.history == []                           # 이력 오염 0


def test_normal_turn_still_records():
    o = _mini_orch()
    events = list(o.stream_turn(np.zeros(1, dtype=np.float32), 16000, user_text="안녕"))
    assert ("user", "안녕") in events
    assert len(o.history) == 2                       # user + assistant


# ----- 대화 → 기억 성장(remember_from_history) --------------------------------
def _fake_chat(base_url, text):
    return '["사용자는 다음 주에 이사한다", "사용자는 야근이 잦다", "짧음"]'


def test_remember_appends_and_dedups(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path))
    hist = [{"role": "user", "content": "나 다음주에 이사해"},
            {"role": "assistant", "content": "어이구 힘들겠네"}]
    r1 = remember_from_history("A", hist, "http://x", chat_fn=_fake_chat)
    assert r1["added"] == 2 and r1["total"] == 2     # "짧음"(6자 미만)은 걸러짐
    # 같은 사실 재추출 → 중복 0
    r2 = remember_from_history("A", hist, "http://x", chat_fn=_fake_chat)
    assert r2["added"] == 0 and r2["total"] == 2
    # 저장 파일이 RAG 소스 규약(문자열 배열)과 일치
    from callone.common.io import data_dir, read_json

    mems = read_json(data_dir() / "speakers" / "A" / "memories.json")
    assert mems == ["사용자는 다음 주에 이사한다", "사용자는 야근이 잦다"]


def test_remember_empty_history_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path))
    assert remember_from_history("A", [], "http://x", chat_fn=_fake_chat) == {"added": 0, "total": 0}
