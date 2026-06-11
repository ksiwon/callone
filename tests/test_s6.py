"""test_s6 (§19): 오케스트레이터 턴 처리 + 첫 음성 지연 측정 경로 + VAD."""
import numpy as np

from callone.serve.vad import VAD
from callone.serve.llm_server import _split_sentences


def test_vad_energy_turn_end():
    vad = VAD({"backend": "energy", "end_silence_ms": 100}, sr=16000)
    speech = 0.5 * np.sin(np.linspace(0, 50, 1600)).astype(np.float32)
    silence = np.zeros(1600, np.float32)
    assert vad.update(speech) is False
    # 무음 두 프레임(각 100ms) → 턴 종료
    vad.update(silence)
    assert vad.update(silence) is True


def test_sentence_split():
    parts = _split_sentences("안녕? 잘 지냈나~ 밥은 묵었나.")
    assert len(parts) == 3


def test_orchestrator_latency_field():
    from callone.serve.orchestrator import Turn

    t = Turn(user_text="안녕")
    assert hasattr(t, "first_audio_latency_ms")
