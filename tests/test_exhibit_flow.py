"""전시(call:one) 엔드투엔드 시뮬레이션 — 실제 모델 없이 API 레벨 전 구간.

패이크 Orchestrator/ASR/LLM 으로 키오스크 관람객 여정 전체를 재현:
  설문→페르소나 → WS 통화(session_init→턴→farewell+부메랑) → 소멸 카운터,
그리고 트랙②(폰 업로드→job 이어받기→화자선택→기억 구축), 이벤트 버스(GPIO),
업로드 페이지, AI 인터뷰어까지. GPU 박스에선 같은 경로가 실모델로 돈다.
"""
from __future__ import annotations

import base64
import io
import json
import time

import numpy as np
import pytest
import soundfile as sf

from callone.common.schemas import Segment
from callone.serve import voice_analyze as va

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ───────────────────────── 패이크(모델 대역) ─────────────────────────
class FakeASR:
    """StreamingTranscriber 계약: transcribe(audio, sr) -> str"""

    def __init__(self, text="여보세요"):
        self.text = text

    def transcribe(self, audio, sr=16000):
        return self.text


_REGISTRY: dict = {}


class FakeOrch:
    """Orchestrator 대역 — stream_turn 이벤트 규약(user/text/audio)과 record 게이트 재현."""

    def __init__(self, speaker_id):
        self.speaker_id = speaker_id
        self.history: list[dict] = []
        self.asr = FakeASR()
        self.init_kwargs = None
        self.meta_texts: list[str] = []      # record=False 로 들어온 지시문(작별 등)
        _REGISTRY[speaker_id] = self

    def init_session(self, **kw):
        self.history = list(kw.get("history") or [])
        self.init_kwargs = kw

    def interrupt(self):
        pass

    def stream_turn(self, audio, sr=16000, user_text=None, record=True):
        ut = (user_text or "").strip() or "(무음)"
        if record:
            yield ("user", ut)
        else:
            self.meta_texts.append(ut)
        reply = "그래, 나야. 잘 지냈어?"
        yield ("text", reply)
        yield ("audio", np.zeros(240, dtype=np.float32))
        if record:
            self.history += [{"role": "user", "content": ut},
                             {"role": "assistant", "content": reply}]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path))
    import callone.serve.orchestrator as om

    monkeypatch.setattr(om, "Orchestrator", FakeOrch)
    _REGISTRY.clear()
    from fastapi.testclient import TestClient

    from callone.serve.app import create_app

    return TestClient(create_app())


def _tiny_wav_b64(sec=1.0, sr=16000) -> str:
    t = np.arange(int(sec * sr)) / sr
    y = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _drain_turn(ws):
    """audio_end 까지 수신 — (텍스트 메시지 목록, 오디오 청크 수)."""
    texts, audio = [], 0
    while True:
        m = ws.receive()
        if m.get("text"):
            j = json.loads(m["text"])
            if j.get("type") == "audio_end":
                return texts, audio
            texts.append(j)
        elif m.get("bytes"):
            audio += 1


# ───────────────────── 트랙① 키오스크 여정(핵심 시뮬레이션) ─────────────────────
def test_kiosk_full_journey(client):
    # 1) 설문 → 페르소나(카드+부메랑)
    p = client.post("/api/exhibit/persona", json={
        "name": "시원",
        "answers": {"worry": "졸업 전시", "future": "좋아하는 일을 하는", "person": "엄마",
                    "joy": "친구와의 여행", "message": "너무 걱정하지 마",
                    "extraversion": 0.8, "spontaneity": 0.3},
    }).json()
    assert "10년" in p["card"]["persona"] and "너무 걱정하지 마" in p["boomerang"]

    # 2) 통화: init(현장 녹음 wav + 카드) → 한 턴 → farewell(부메랑) → 종료
    with client.websocket_connect("/ws/call/kiosk") as ws:
        ws.send_text(json.dumps({"type": "session_init", "ref_text": "여보세요, 나야.",
                                 "ref_audio_b64": _tiny_wav_b64(), **p["card"]}))
        assert ws.receive_json()["type"] == "session_ready"

        ws.send_bytes(np.zeros(4096, dtype=np.float32).tobytes())   # 마이크 한 청크
        ws.send_text(json.dumps({"type": "end_turn"}))
        texts, audio = _drain_turn(ws)
        kinds = [t["type"] for t in texts]
        assert "user" in kinds and "reply" in kinds and audio >= 1  # 전사·응답·음성 전부 흐름

        ws.send_text(json.dumps({"type": "farewell", "extra": p["boomerang"]}))
        texts2, audio2 = _drain_turn(ws)
        kinds2 = [t["type"] for t in texts2]
        assert "user" not in kinds2 and "reply" in kinds2 and audio2 >= 1
        ws.send_text(json.dumps({"type": "stop"}))

    orch = _REGISTRY["kiosk"]
    assert orch.init_kwargs["ref_audio"] is not None                # 현장 녹음이 ref 로 들어감
    assert len(orch.history) == 2                                   # 일반 턴만 이력에
    assert any("너무 걱정하지 마" in t for t in orch.meta_texts)     # 부메랑이 작별 지시에 주입

    # 3) 소멸 카운터
    assert client.post("/api/exhibit/dissolve").json()["today"] == 1
    assert client.post("/api/exhibit/dissolve").json()["today"] == 2
    assert client.get("/api/exhibit/count").json()["total"] == 2


# ───────────────── 트랙② 폰 업로드 → 화자선택 → 기억 구축 ─────────────────
def _fake_segments():
    return [
        Segment(start=0.0, end=10.0, local_speaker="S0", snr_db=20.0),
        Segment(start=12.0, end=19.0, local_speaker="S1", snr_db=18.0),
        Segment(start=20.0, end=29.0, local_speaker="S0", snr_db=22.0),
        Segment(start=18.5, end=25.0, local_speaker="S1", snr_db=15.0, overlap=True),
    ]


def _wav_bytes_30s():
    sr = 16000
    t = np.arange(30 * sr) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def _wait_done(jid, timeout=60.0):   # 첫 회는 잡 스레드의 whisperx/torch 임포트가 느릴 수 있음
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = va.job_status(jid)
        if st and st["stage"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError("분석 job 타임아웃")


def test_upload_to_memories_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("CALLONE_DATA_DIR", str(tmp_path))
    from callone.diarize import s2_diarize

    monkeypatch.setattr(s2_diarize, "diarize_one", lambda path, cfg: _fake_segments())

    jid = va.start_job(_wav_bytes_30s(), suffix=".wav")             # = 폰이 /upload 로 보낸 것
    assert any(j["job_id"] == jid for j in va.list_jobs())          # 데스크가 이어받기
    assert _wait_done(jid)["stage"] == "done"

    seen = {}

    class _Asr:                                                     # 창마다 다른 전사
        n = 0

        def transcribe(self, audio, sr=16000):
            _Asr.n += 1
            return f"{_Asr.n}번째 이야기 조각"

    def _chat(base_url, text):
        seen["text"] = text
        return '["사용자는 다음 달에 이사한다"]'

    r = va.remember_pick(jid, "S0", "엄마", asr=_Asr(), base_url="http://x", chat_fn=_chat)
    assert r["windows"] == 3 and r["added"] == 1                    # 겹침 세그먼트 제외 3창
    # 화자 매핑: 선택 화자(그 사람)=상대, 나머지=사용자
    lines = seen["text"].splitlines()
    assert lines[0].startswith("상대:") and lines[1].startswith("사용자:")

    from callone.common.io import data_dir, read_json

    mems = read_json(data_dir() / "speakers" / "엄마" / "memories.json")
    assert mems == ["사용자는 다음 달에 이사한다"]                    # RAG 규약 그대로


# ───────────────── 이벤트 버스(GPIO 브리지 ↔ 키오스크) ─────────────────
def test_event_bus_broadcast(client):
    with client.websocket_connect("/ws/exhibit/events") as ws:
        r = client.post("/api/exhibit/event", json={"event": "hook_up"})
        assert r.json()["delivered"] == 1
        assert ws.receive_json() == {"type": "event", "event": "hook_up"}
        client.post("/api/exhibit/event", json={"event": "ring_start"})
        assert ws.receive_json()["event"] == "ring_start"
    assert client.post("/api/exhibit/event", json={"event": "bogus"}).status_code == 400


# ───────────────── 업로드 페이지·인터뷰어 ─────────────────
def test_upload_page_served(client):
    r = client.get("/upload")
    assert r.status_code == 200
    assert "이 파일은 이 방을 떠나지 않습니다" in r.text
    assert "api/voice/analyze" in r.text                            # 기존 분석 API 에 붙음


def test_interviewer_card(client):
    r = client.get("/api/exhibit/interviewer", params={"name": "협력자"}).json()
    assert len(r["questions"]) >= 15                                # AVP 변형 질문지
    assert "협력자" in r["card"]["user_persona"]
    assert "인생 이야기" in r["card"]["first_message"]               # 1번 질문으로 시작
    assert "1." in r["card"]["situation"]                           # 질문지가 카드에 내장
