"""더미 데이터 생성 (§6, 부록 B-6) — 실제 통화 없이 파이프라인 검증.

두 가상 화자(A=180Hz, B=140Hz)의 합성 톤 통화 wav 생성.
ffmpeg 있으면 m4a 도 만들어 S0 부터 전 과정 테스트 가능.

사용:
  python scripts/make_dummy_data.py --n 6
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def make_call(path: Path, dur: float = 12.0, sr: int = 16000) -> None:
    """A/B 가 번갈아 말하는 합성 통화(모노믹스)."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = np.zeros_like(t)
    seg = 1.5
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    for i, s in enumerate(np.arange(0, dur, seg)):
        freq = 180 if i % 2 == 0 else 140       # A / B
        mask = (t >= s) & (t < s + seg * 0.8)   # 0.8 발화 + 0.2 무음
        y[mask] += 0.2 * np.sin(2 * np.pi * freq * t[mask])
        y[mask] += 0.02 * rng.standard_normal(mask.sum())  # 잡음
    sf.write(str(path), y.astype(np.float32), sr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--force", action="store_true", help="실데이터 있어도 강행")
    args = ap.parse_args()

    raw = Path(args.out)
    raw.mkdir(parents=True, exist_ok=True)

    # ⚠️ footgun 가드: data/raw 에 진짜 녹음(dummy_ 아님)이 있으면 섞지 않도록 중단
    real = [p for p in raw.glob("*.m4a") if not p.name.startswith("dummy_")]
    if real and not args.force:
        print(f"중단: {raw} 에 실제 녹음 {len(real)}개가 있음. 더미와 섞이면 안 됨.")
        print("  → 다른 폴더로:  python scripts/make_dummy_data.py --out data/raw_dummy")
        print("  → 그래도 강행:  --force")
        return

    have_ff = __import__("shutil").which("ffmpeg") is not None
    for i in range(args.n):
        wav = raw / f"dummy_call_{i:05d}.wav"
        make_call(wav)
        if have_ff:
            m4a = raw / f"dummy_call_{i:05d}.m4a"
            subprocess.run(["ffmpeg", "-y", "-i", str(wav), "-c:a", "aac", str(m4a)],
                           capture_output=True)
            wav.unlink()
            print(f"생성: {m4a}")
        else:
            print(f"생성: {wav} (ffmpeg 없음 — m4a 변환 생략)")
    print(f"더미 통화 {args.n}개 → {raw} (파일명 dummy_ 접두)")


if __name__ == "__main__":
    main()
