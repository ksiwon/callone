"""노트북 마이크 실시간 통화 테스트 (브라우저 없이).

마이크 → ASR(faster-whisper 한국어) → LLM(화자 A, llama-server) → TTS → 스피커.
실제 callone Orchestrator 그대로 사용. TTS 음성 모델(models/tts_piper/A.onnx) 있으면
화자 A 목소리, 없으면 placeholder beep(루프·지연 확인용).

전제: llama-server 가 8080 에 떠 있어야 함(화자 A LLM). 첫 실행 시 ASR 모델 자동 다운(~1.5GB).

사용:
  pip install sounddevice
  python scripts/call_mic.py --speaker A
  → [Enter] 누르고 말하기 → 끝나면 [Enter] 다시 → 화자 A 응답
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000


def record_until_enter() -> np.ndarray:
    import sounddevice as sd

    frames: list[np.ndarray] = []
    stop = threading.Event()

    def cb(indata, n, t, status):  # noqa: ANN001
        if not stop.is_set():
            frames.append(indata.copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=cb):
        input()              # Enter 누르면 녹음 종료
        stop.set()
    if not frames:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(frames).flatten().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="마이크 실시간 통화 테스트")
    ap.add_argument("--speaker", default="A")
    args = ap.parse_args()

    import sounddevice as sd

    from callone.serve.orchestrator import Orchestrator

    print(f"☎️  callone 통화 — 화자 {args.speaker} 로딩 중...")
    orch = Orchestrator(args.speaker)
    out_sr = getattr(orch.tts, "sr", 24000)
    voice = "화자 A 목소리" if type(orch.tts).__name__ == "PiperTTS" else "placeholder(beep)"
    print(f"   LLM={type(orch.llm).__name__}  TTS={type(orch.tts).__name__}({voice})  ASR 준비됨")
    print("   ──────────────────────────────────────────")

    try:
        while True:
            input("\n🎤 [Enter] 누르고 말하세요 (다 말하면 [Enter] 다시) ...")
            print("   녹음 중... (끝나면 Enter)")
            audio = record_until_enter()
            secs = len(audio) / SR
            if secs < 0.3:
                print("   (너무 짧음, 다시)")
                continue
            print(f"   ⏳ 처리 중 ({secs:.1f}초 녹음)...")

            chunks: list[np.ndarray] = []
            for kind, val in orch.stream_turn(audio, sr=SR):
                if kind == "user":
                    print(f"   🗣️  나: {val}")
                elif kind == "text":
                    print(f"   👩 화자 A: {val}")
                elif kind == "latency":
                    print(f"   ⚡ 첫 음성 {val:.0f}ms")
                elif kind == "audio":
                    chunks.append(val)
                elif kind == "interrupted":
                    print("   (중단됨)")
            if chunks:
                y = np.concatenate(chunks)
                sd.play(y, out_sr)
                sd.wait()
    except KeyboardInterrupt:
        print("\n☎️  통화 종료")


if __name__ == "__main__":
    main()
