"""v2 재구축 단위 검증 (docs/REBUILD_PLAN.md §5) — GPU/서버 없이 도는 범위만.

- StreamingTranscriber: 발화 중 partial + finalize 재사용(ASR 지연 0 경로)
- stream_turn(user_text=...): 선전사 주입 시 ASR 건너뜀
- 백엔드 선택: qwen3 미설치 → whisper 폴백 / TTS 체인은 서버 없이도 안전 폴백
"""
from __future__ import annotations

import time

import numpy as np

from callone.serve.asr_streaming import StreamingTranscriber
from callone.serve.orchestrator import Orchestrator, _pick_asr


class _SpyASR:
    """transcribe(audio, sr)->str 계약 스파이 — 호출 수와 마지막 샘플 수 기록."""

    def __init__(self):
        self.calls = 0
        self.last_n = 0

    def transcribe(self, audio, sr=16000):
        self.calls += 1
        self.last_n = len(audio)
        return f"전사{self.last_n}"


def _wait(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


# ----- StreamingTranscriber ---------------------------------------------------
def test_partial_then_finalize_reuses_last_transcription():
    """발화 중 partial 이 만들어지고, 새 오디오 없으면 finalize 가 재전사 없이 반환(지연 0)."""
    asr = _SpyASR()
    got: list[str] = []
    st = StreamingTranscriber(asr, sr=16000, interval_ms=50, on_partial=got.append)
    st.feed(np.zeros(16000, dtype=np.float32))          # 1s (min 0.4s 초과)
    assert _wait(lambda: asr.calls >= 1), "partial 재전사가 안 돎"
    assert _wait(lambda: len(got) >= 1)
    calls_before = asr.calls
    # 워커가 이미 전부 소화 → finalize 는 추가 전사 없이 마지막 partial 반환
    assert _wait(lambda: asr.last_n == 16000)
    text = st.finalize()
    assert text == got[-1]
    assert asr.calls == calls_before


def test_finalize_transcribes_tail():
    """마지막 partial 이후 꼬리 오디오가 있으면 finalize 가 1회만 더 전사."""
    asr = _SpyASR()
    st = StreamingTranscriber(asr, sr=16000, interval_ms=10_000)   # 워커가 못 돌게 큰 주기
    st.feed(np.zeros(16000, dtype=np.float32))
    text = st.finalize()                                # 워커 미소화분 → 여기서 전사
    assert text == "전사16000"
    assert asr.calls == 1


def test_close_discards_buffer():
    asr = _SpyASR()
    st = StreamingTranscriber(asr, sr=16000, interval_ms=10_000)
    st.feed(np.ones(8000, dtype=np.float32))
    st.close()
    assert st.finalize() == ""                          # 버퍼 폐기됨(ephemeral)


# ----- stream_turn user_text 주입(선전사 → ASR 스킵) ---------------------------
class _FakeLLM:
    def chat_stream(self, user_text, history):
        yield "응 알았어."


class _FakeTTS:
    sr = 24000
    ref_wav = ""

    def synth_stream(self, text, emotion=None):
        yield np.zeros(240, dtype=np.float32)


def _mini_orch():
    o = Orchestrator.__new__(Orchestrator)
    o.asr = _SpyASR()
    o.llm = _FakeLLM()
    o.tts = _FakeTTS()
    o.avatar = None
    o.history = []
    o.synth_mode = "full"
    o.sentence_pause_ms = 0
    o.log_content = False
    import threading
    o._interrupt = threading.Event()
    return o


def test_stream_turn_skips_asr_with_pretranscribed_text():
    o = _mini_orch()
    events = list(o.stream_turn(np.zeros(1600, dtype=np.float32), 16000, user_text="안녕"))
    kinds = [k for k, _ in events]
    assert o.asr.calls == 0                             # 핵심: ASR 안 돌았다
    assert ("user", "안녕") in events
    assert "audio" in kinds and "end" in kinds
    timing = dict(events)["timing"]
    assert timing["asr_ms"] < 50                        # 선전사 → asr 단계 ≈ 0


def test_stream_turn_falls_back_to_asr_without_text():
    o = _mini_orch()
    list(o.stream_turn(np.zeros(1600, dtype=np.float32), 16000))
    assert o.asr.calls == 1                             # 기존 경로 유지(폴백)


# ----- 백엔드 선택 폴백 --------------------------------------------------------
def test_pick_asr_falls_back_to_whisper_when_qwen_missing():
    """qwen-asr 미설치 환경(이 박스) → StreamASR 폴백. 통화는 항상 산다."""
    from callone.serve.asr_stream import StreamASR

    asr = _pick_asr({"asr": {"backend": "qwen3", "qwen_model": "Qwen/Qwen3-ASR-0.6B"}})
    assert isinstance(asr, StreamASR)


def test_pick_tts_auto_never_raises_without_servers():
    """auto 체인: qwen3tts(:8093)·cosy(:8092) 다운이어도 예외 없이 폴백 백엔드 반환."""
    from callone.serve.orchestrator import _pick_tts

    tts = _pick_tts("t", {"tts": {"backend": "auto"}})
    assert tts is not None and hasattr(tts, "synth_stream")
