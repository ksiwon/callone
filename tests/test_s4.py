"""test_s4 (§19): TTS 엔진 인터페이스 + 스트리밍 청크 (placeholder)."""
from callone.tts.infer import TTSEngine


def test_tts_synth_placeholder():
    eng = TTSEngine("A", {"sample_rate": 16000, "backend": "qwen3-tts",
                          "output_dir": "models/tts_server"})
    y, sr = eng.synth("밥은 묵었나")
    assert sr == 16000
    assert len(y) > 0


def test_tts_stream_chunks():
    eng = TTSEngine("A", {"sample_rate": 16000, "output_dir": "models/tts_server"})
    chunks = list(eng.synth_stream("안녕 잘 지냈나", chunk_ms=200))
    assert len(chunks) >= 1
    assert sum(len(c) for c in chunks) > 0
