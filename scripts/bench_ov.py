"""OpenVINO GenAI LLM 속도 벤치 — Lunar Lake(Arc iGPU/NPU/CPU)에서 실시간 가능한지 측정.

방법(견고): TTFT = 1토큰 생성 시간, throughput = 64토큰 강제생성(ignore_eos) / 시간.
첫 음성 지연 추정 = STT 0.6s + TTFT + (첫문장 ~12토큰)/tput + TTS 0.3s.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import openvino_genai as ov_genai

PROMPT = "화자 A 나 왔어. 오늘 좀 늦었지?"


def _cfg(n: int):
    c = ov_genai.GenerationConfig()
    c.max_new_tokens = n
    try:
        c.ignore_eos = True       # 정확히 n토큰 강제(짧게 끝나는 것 방지)
    except Exception:
        pass
    return c


def bench(model_dir: str, device: str):
    try:
        pipe = ov_genai.LLMPipeline(model_dir, device)
    except Exception as e:  # noqa: BLE001
        print(f"  [{device:3}] 로드 실패: {str(e).splitlines()[0][:90]}")
        return
    try:
        pipe.generate("안녕", _cfg(8))                       # 워밍업
        t = time.perf_counter(); pipe.generate(PROMPT, _cfg(1)); ttft = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); pipe.generate(PROMPT, _cfg(64)); wall = time.perf_counter() - t
        tput = 64 / wall
        first_audio = 0.6 + ttft / 1000 + 12 / max(tput, 1) + 0.3
        verdict = "실시간 OK" if first_audio <= 1.6 else ("아슬" if first_audio <= 2.2 else "느림")
        print(f"  [{device:3}] {tput:5.1f} tok/s | TTFT {ttft:5.0f}ms | "
              f"추정 첫음성 ~{first_audio:.1f}s  → {verdict}")
    except Exception as e:  # noqa: BLE001
        print(f"  [{device:3}] 생성 실패: {str(e).splitlines()[0][:90]}")


def main():
    models = sys.argv[1:] or ["models_ov/qwen3-1.7b-int4", "models_ov/qwen3-4b-int4"]
    for md in models:
        if not Path(md).exists():
            print(f"\n{md}: 없음")
            continue
        print(f"\n=== {Path(md).name} ===")
        for dev in ("GPU", "CPU"):     # NPU 는 LLM 정적셰이프 설정 필요 → 별도
            bench(md, dev)


if __name__ == "__main__":
    main()
