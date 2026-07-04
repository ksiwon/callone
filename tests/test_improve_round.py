"""test_improve_round — 2026-06 개선 라운드 회귀(모델/GPU/서버 없이).

GPU 첫 세션이 버그를 처음 만나는 곳이 되지 않게, 무-GPU로 검증 가능한 로직을 잠근다:
  1) 작업2 샘플러: _payload 에 DRY/min_p, repeat_last_n=64(settled). config 로 조정됨.
  2) 작업3 EXAONE 프롬프트: /no_think 같은 타 모델 토큰 미주입.
  3) 작업1 TTS 프레이밍: [4B LE 길이][f32 PCM] 서버 인코딩 ↔ 클라 파서 라운드트립 + 부분수신.
  4) 작업4-A 아바타 워밍업: 더미 1s 오디오로 frames_for 1회 호출(없으면 무동작).
"""
import io
import struct

import numpy as np

from callone.serve.llama_llm import LlamaPersonaLLM
from callone.serve.tts_cosyvoice import CosyVoiceTTS, _read_exactly
from callone.serve.orchestrator import Orchestrator, Turn


# ----- 작업2: 샘플러(DRY/min_p) ----------------------------------------------
def test_payload_has_dry_and_min_p_defaults():
    llm = LlamaPersonaLLM("test", use_rag=False, probe=False)
    p = llm._payload("안녕", [], stream=False)
    assert p["min_p"] == 0.05
    assert p["dry_multiplier"] == 0.8
    assert p["dry_base"] == 1.75
    assert p["dry_allowed_length"] == 2
    assert p["dry_penalty_last_n"] == -1
    # settled 값 유지(넓히면 조사 반복까지 눌러 비문 위험)
    assert p["repeat_last_n"] == 64
    assert p["repeat_penalty"] == 1.05


def test_payload_sampler_config_driven():
    llm = LlamaPersonaLLM("test", use_rag=False, probe=False,
                          rag_cfg={"min_p": 0.07, "dry_multiplier": 1.2,
                                   "dry_base": 2.0, "dry_allowed_length": 3})
    p = llm._payload("x", None, stream=False)
    assert p["min_p"] == 0.07
    assert p["dry_multiplier"] == 1.2
    assert p["dry_base"] == 2.0
    assert p["dry_allowed_length"] == 3


def test_prompt_conflict_rule_consolidated():
    """충돌 3개('짧게'+'질문말라'+'반복말라')를 1개 긍정 규칙으로 통합했는지."""
    sysmsg = LlamaPersonaLLM("test", use_rag=False, probe=False)._system("안녕")
    assert "질문은 꼭 필요할 때만" in sysmsg            # 통합된 긍정 규칙
    assert "매 문장을 질문으로 끝내지 마라" not in sysmsg  # 옛 충돌 규칙 제거됨


# ----- 작업3: EXAONE 프롬프트 -----------------------------------------------
def test_exaone_prompt_has_no_foreign_nothink_token():
    sysmsg = LlamaPersonaLLM("test", use_rag=False, probe=False)._system("안녕")
    assert "/no_think" not in sysmsg                   # EXAONE 기본: 리터럴 미주입


# ----- 작업1: TTS 스트리밍 프레이밍 [4B LE len][f32 PCM] ---------------------
def test_cosyvoice_init_reads_reference_asr_from_cfg(monkeypatch):
    monkeypatch.setattr(CosyVoiceTTS, "_probe", lambda self: None)
    tts = CosyVoiceTTS("test", {"ref_transcribe_model": "small"})
    assert tts.ref_asr_model == "small"
def _encode(a: np.ndarray) -> bytes:
    b = a.astype(np.float32).tobytes()
    return struct.pack("<I", len(b)) + b


def test_stream_framing_roundtrip():
    a0 = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    a1 = np.array([0.5, 0.6], dtype=np.float32)
    resp = io.BytesIO(_encode(a0) + _encode(a1))
    out = []
    while True:
        head = _read_exactly(resp, 4)
        if not head:
            break
        (n,) = struct.unpack("<I", head)
        body = _read_exactly(resp, n)
        assert len(body) == n
        out.append(np.frombuffer(body, dtype=np.float32))
    assert len(out) == 2
    assert np.allclose(out[0], a0) and np.allclose(out[1], a1)


def test_read_exactly_handles_partial_socket_reads():
    """소켓이 조금씩 줄 때도 정확히 n바이트 모으고, EOF 면 모인 만큼."""
    class _Drip:
        def __init__(self, data): self.data, self.i = data, 0
        def read(self, n):
            if self.i >= len(self.data):
                return b""
            chunk = self.data[self.i:self.i + 1]   # 1바이트씩 흘림
            self.i += 1
            return chunk

    d = _Drip(b"hello")
    assert _read_exactly(d, 5) == b"hello"
    assert _read_exactly(d, 5) == b""              # EOF → 빈 바이트(루프 종료 신호)


# ----- 작업4-A: 아바타 세션 워밍업 ------------------------------------------
class _SpyAvatar:
    fps = 25

    def __init__(self):
        self.calls = []

    def frames_for(self, audio, sr):
        self.calls.append((len(audio), sr))
        yield b"frame"                              # 1프레임(폐기됨)


class _Tts:
    sr = 24000


def test_warmup_avatar_runs_dummy_second():
    spy = _SpyAvatar()
    fake = Orchestrator.__new__(Orchestrator)      # __init__ 우회(무거운 모델 로드 회피)
    fake.avatar = spy
    fake.tts = _Tts()
    fake._warmup_avatar()
    assert len(spy.calls) == 1
    n_samples, sr = spy.calls[0]
    assert sr == 24000 and n_samples == 24000       # 더미 1s @ 24k


def test_warmup_avatar_noop_when_none():
    fake = Orchestrator.__new__(Orchestrator)
    fake.avatar = None
    fake.tts = _Tts()
    fake._warmup_avatar()                           # 예외 없이 즉시 반환


# ----- 섹시/ASMR 모드: 세션 레퍼런스 선택(우선순위1) --------------------------
class _RefTts:
    sr = 24000

    def __init__(self, has_nsfw=True):
        self.calls: list = []
        self._has_nsfw = has_nsfw
        self.ref_wav = ""

    def use_nsfw_reference(self):
        self.calls.append(("nsfw",))
        if self._has_nsfw:
            self.ref_wav = "memory"
            return True
        return False

    def set_reference(self, a, sr, t=None):
        self.calls.append(("caller", len(a), sr))
        self.ref_wav = "memory"


def _fake_orch(tts):
    o = Orchestrator.__new__(Orchestrator)
    o.tts = tts
    return o


def test_nsfw_uses_preset_reference_over_caller():
    """nsfw=True + 프리셋 지원 → 고정 섹시 레퍼런스 사용, 프론트 음성 무시."""
    tts = _RefTts(has_nsfw=True)
    ok = _fake_orch(tts)._apply_session_reference(
        True, np.zeros(16000, dtype=np.float32), 16000, None)
    assert ok is True
    assert tts.calls == [("nsfw",)]                 # caller set_reference 호출 안 됨


def test_nsfw_falls_back_to_caller_when_no_preset():
    """nsfw=True 지만 프리셋 미설정(use_nsfw_reference False) → 프론트 음성으로 폴백."""
    tts = _RefTts(has_nsfw=False)
    ok = _fake_orch(tts)._apply_session_reference(
        True, np.zeros(8000, dtype=np.float32), 16000, None)
    assert ok is True
    assert tts.calls == [("nsfw",), ("caller", 8000, 16000)]


def test_normal_mode_never_touches_nsfw_reference():
    """nsfw=False → use_nsfw_reference 안 부르고 프론트 음성만."""
    tts = _RefTts(has_nsfw=True)
    ok = _fake_orch(tts)._apply_session_reference(
        False, np.zeros(4000, dtype=np.float32), 24000, "ref")
    assert ok is True
    assert tts.calls == [("caller", 4000, 24000)]


def test_no_reference_returns_false():
    """일반 모드 + 음성 없음 → ref 미설정(False)."""
    tts = _RefTts(has_nsfw=True)
    ok = _fake_orch(tts)._apply_session_reference(False, None, 24000, None)
    assert ok is False
    assert tts.calls == []


# ----- 턴 로직 단일화: handle_utterance = stream_turn 소비 어댑터 --------------
def test_handle_utterance_adapts_stream_turn_events():
    """handle_utterance 가 stream_turn 이벤트를 Turn 으로 정확히 취합하는지(중복 제거 회귀)."""
    fake = Orchestrator.__new__(Orchestrator)
    events = [("user", "안녕"), ("emotion", "happy"), ("text", "응 안녕"),
              ("latency", 123.0), ("audio", np.ones(3, dtype=np.float32)),
              ("audio", np.zeros(2, dtype=np.float32)),
              ("timing", {"asr_ms": 10, "first_audio_ms": 123.0}), ("end", "응 안녕")]
    fake.stream_turn = lambda audio, sr=16000: iter(events)   # type: ignore[method-assign]
    turn = fake.handle_utterance(np.zeros(4, dtype=np.float32), 16000)
    assert turn.user_text == "안녕"
    assert turn.reply_text == "응 안녕"
    assert turn.first_audio_latency_ms == 123.0
    assert len(turn.audio_chunks) == 2
    assert turn.timings == {"asr_ms": 10, "first_audio_ms": 123.0}
    assert turn.interrupted is False


def test_handle_utterance_marks_interrupted():
    fake = Orchestrator.__new__(Orchestrator)
    fake.stream_turn = lambda audio, sr=16000: iter(   # type: ignore[method-assign]
        [("user", "x"), ("interrupted", None)])
    turn = fake.handle_utterance(np.zeros(2, dtype=np.float32), 16000)
    assert turn.interrupted is True
    assert turn.reply_text == ""
