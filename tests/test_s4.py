"""test_s4: 현재 경량 TTS 폴백 인터페이스."""
from callone.serve.tts_kokoro import KokoroTTS

def test_tts_placeholder_synth():
    eng = KokoroTTS("A", sr=16000)
    y, sr = eng.synth("밥은 묵었나")
    assert sr == 16000
    assert len(y) > 0

def test_tts_stream_chunks():
    eng = KokoroTTS("A", sr=16000)
    chunks = list(eng.synth_stream("안녕 잘 지냈나", chunk_ms=200))
    assert chunks
    assert sum(len(c) for c in chunks) > 0
