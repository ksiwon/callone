"""통화 지연 단계별 실측 — RunPod(3090) 박스에서 진짜 병목을 찾는다.

여기 노트북엔 GPU 모델이 없어 측정 불가 → 이 스크립트를 **박스에서** 돌려
ASR / LLM / TTS 각 단계가 첫 음성까지 몇 ms 먹는지 숫자로 본다(목소리엔 영향 0).

전제(docs/FRESH_SETUP.md · scripts/run_all.sh): llama-server(:8090) + cosyvoice-server(:8092) + 화자 ref 준비.

사용:
  # 텍스트 입력(ASR 건너뜀 — LLM+TTS 만 측정). 가장 간단.
  callone-bench --speaker A --turns 5 --text "여보세요, 밥은 먹었어?"

  # 실제 마이크 발화(WAV) 입력 → ASR 포함 전체 관통.
  callone-bench --speaker A --turns 5 --audio data/speakers/A/ref_24k.wav

출력: 턴마다 단계별 ms + 중앙값(median). 첫 턴은 워밍업 직후라 보통 가장 안정적.
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Iterator

import numpy as np

from ..common.io import load_config
from ..common.logging import get_logger

log = get_logger("bench")


def _load_audio_16k(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype("float32")
    if sr != 16000:
        # 간단 선형 리샘플(측정용 — 품질 무관).
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    return audio


def _bench_text(orch, text: str) -> dict:
    """ASR 건너뛰고 LLM+TTS 단계만 측정(handle_utterance 의 LLM/TTS 구간 미러)."""
    from .orchestrator import _parse_emotion, _tts_stream_emo

    t0 = time.time()
    emotion = "neutral"
    parts: list[str] = []
    t_llm_first = 0.0
    for sentence in orch.llm.chat_stream(text, []):
        if not t_llm_first:
            t_llm_first = time.time()
        emotion, clean = _parse_emotion(sentence, emotion)
        if clean:
            parts.append(clean)
    t_llm_done = time.time()
    reply = " ".join(parts)

    t_tts_first = 0.0
    n_chunks = 0
    segments = [reply] if orch.synth_mode == "full" else parts
    for seg in segments:
        for _chunk in _tts_stream_emo(orch.tts, seg, emotion):
            if not t_tts_first:
                t_tts_first = time.time()
            n_chunks += 1
    t_done = time.time()
    return {
        "reply": reply,
        "llm_first_ms": (t_llm_first - t0) * 1000 if t_llm_first else 0.0,
        "llm_total_ms": (t_llm_done - t0) * 1000,
        "tts_first_ms": (t_tts_first - t_llm_done) * 1000 if t_tts_first else 0.0,
        "first_audio_ms": (t_tts_first - t0) * 1000 if t_tts_first else 0.0,
        "synth_total_ms": (t_done - t0) * 1000,
        "chunks": n_chunks,
    }


def _fmt(d: dict) -> str:
    keys = [k for k in ("asr_ms", "llm_first_ms", "llm_total_ms",
                        "tts_first_ms", "first_audio_ms", "synth_total_ms") if k in d]
    return "  ".join(f"{k}={d[k]:.0f}" for k in keys)


def _median(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key)]
    return statistics.median(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="callone 통화 지연 단계별 실측(RunPod)")
    ap.add_argument("--speaker", default="A")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--text", default="여보세요, 잘 지냈어? 밥은 먹었고?",
                    help="텍스트 입력(ASR 건너뜀). --audio 주면 무시")
    ap.add_argument("--audio", default=None, help="WAV 경로 → ASR 포함 전체 관통")
    ap.add_argument("--no-warmup", action="store_true", help="워밍업 끄고 콜드 지연도 본다")
    args = ap.parse_args()

    from .orchestrator import Orchestrator

    cfg = load_config("serve")
    if args.no_warmup:
        cfg["warmup"] = False

    print(f"== callone 지연 벤치 == speaker={args.speaker} turns={args.turns} "
          f"mode={'audio' if args.audio else 'text'} warmup={not args.no_warmup}")
    t_build = time.time()
    orch = Orchestrator(args.speaker, cfg)   # warmup 여기서 실행(켜져 있으면)
    print(f"[setup] Orchestrator 생성+워밍업 {(time.time() - t_build) * 1000:.0f}ms  "
          f"(synth_mode={orch.synth_mode})")

    rows: list[dict] = []
    audio = _load_audio_16k(args.audio) if args.audio else None
    for i in range(args.turns):
        orch.reset_history()   # 턴마다 동일 조건(이력 누적 영향 제거)
        if audio is not None:
            turn = orch.handle_utterance(audio, 16000)
            row = dict(turn.timings)
            reply = turn.reply_text
        else:
            row = _bench_text(orch, args.text)
            reply = row.pop("reply", "")
        rows.append(row)
        print(f"[turn {i + 1}] {_fmt(row)}   → '{reply[:40]}'")

    print("\n== 중앙값(median) ==")
    for k in ("asr_ms", "llm_first_ms", "llm_total_ms", "tts_first_ms",
              "first_audio_ms", "synth_total_ms"):
        m = _median(rows, k)
        if m:
            print(f"  {k:16s} {m:7.0f} ms")
    print("\n해석: first_audio_ms 가 '말 끝나고 첫 음성까지'. 이게 큰 단계가 병목."
          " 목소리 안 바꾸고 깎을 수 있는 건 asr/llm/워밍업 쪽.")


if __name__ == "__main__":
    main()
