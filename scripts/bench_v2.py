#!/usr/bin/env python3
"""bench_v2 — v2 TTS 교체 게이트: 음색 안정성(턴 간 편차) + 지연 실측 A/B.

왜?  과거 Qwen3 계열 TTS 는 "턴마다 음색 튐" 으로 기각됨(REBUILD_PLAN §1). 신형
     Qwen3-TTS-12Hz(:8093)를 기본 백엔드로 승격하려면 이 게이트를 먼저 통과해야 한다:
       ① 같은 ref 로 N턴 합성 → 턴 간 스피커 유사도(자기유사) 행렬 — cosy 와 비교
       ② /synth_stream 첫패킷 지연 + 통짜 RTF — GPU 실측(공개 벤치는 A100급)
       ③ 저장 wav 블라인드 청취(사람 최종 판정)

사용(박스에서, 두 TTS 서버 켠 뒤):
    python scripts/bench_v2.py --ref data/refs/sample.wav --ref-text "안녕하세요 반가워요" \
        [--turns 10] [--out out/bench_v2]
    # 특정 백엔드만: --backends qwen3   /   --backends cosy

torch 불필요(urllib+numpy+soundfile). 유사도는 resemblyzer 있으면 자동(없으면 청취 안내만).
판정 기준(권장): qwen3 자기유사 평균 ≥ cosy 자기유사 평균 - 0.02  AND  첫패킷 < 300ms.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import time
import urllib.request

import numpy as np
import soundfile as sf

BACKENDS = {
    "qwen3": os.environ.get("QWEN_TTS_URL", "http://127.0.0.1:8093"),
    "cosy":  os.environ.get("COSYVOICE_URL", "http://127.0.0.1:8092"),
}
TEXTS = [   # 턴마다 다른 문장(실통화 조건) — 같은 문장 반복은 --same-text
    "안녕, 오늘 하루 어땠어?",
    "나는 방금 산책하고 왔어. 날씨가 진짜 좋더라.",
    "저녁은 뭐 먹을지 정했어? 같이 고민해 줄까?",
    "요즘 즐겨 듣는 노래 있으면 하나만 추천해 줘.",
    "주말에 시간 되면 오랜만에 얼굴 한번 보자.",
    "어제 말한 그 일은 잘 해결됐어?",
    "너무 무리하지 말고 틈틈이 쉬어 가면서 해.",
    "갑자기 비 온다더라, 우산 꼭 챙겨.",
    "사진 보니까 머리 스타일 바꿨네? 잘 어울려.",
    "이제 슬슬 자야겠다. 내일 또 통화하자.",
]


def _b64_ref(path: str) -> tuple[str, int]:
    y, sr = sf.read(path, dtype="float32")
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    return base64.b64encode(np.asarray(y, dtype=np.float32).tobytes()).decode(), int(sr)


def _payload(text: str, ref_b64: str, ref_sr: int, ref_text: str) -> bytes:
    return json.dumps({"text": text, "ref_audio_b64": ref_b64, "ref_sr": ref_sr,
                       "prompt_text": ref_text, "language": "Korean"}).encode()


def synth_full(url: str, text: str, ref_b64: str, ref_sr: int, ref_text: str,
               timeout: float) -> tuple[np.ndarray, int, float]:
    """POST /synth → (audio, sr, 총 소요 s)."""
    req = urllib.request.Request(f"{url}/synth", data=_payload(text, ref_b64, ref_sr, ref_text),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        sr = int(r.headers.get("X-Sample-Rate", "24000"))
    return np.frombuffer(raw, dtype=np.float32), sr, time.time() - t0


def synth_stream_first(url: str, text: str, ref_b64: str, ref_sr: int, ref_text: str,
                       timeout: float) -> float:
    """POST /synth_stream → 첫 프레임([len][pcm]) 도착까지 s."""
    req = urllib.request.Request(f"{url}/synth_stream", data=_payload(text, ref_b64, ref_sr, ref_text),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        head = r.read(4)                       # 길이 프리픽스 4바이트 = 첫 청크 시작
        if len(head) == 4:
            n = struct.unpack("<I", head)[0]
            r.read(min(n, 4096))               # 첫 청크 일부 소비(도착 확정)
    return time.time() - t0


def _embed_all(wavs: list[str]):
    """resemblyzer 스피커 임베딩(선택 의존성). 없으면 None."""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore
    except ImportError:
        return None
    enc = VoiceEncoder()
    return [enc.embed_utterance(preprocess_wav(w)) for w in wavs]


def self_similarity(wavs: list[str]) -> float | None:
    """턴 간 자기유사(모든 쌍 코사인 평균). 1.0=완전 동일 음색. None=resemblyzer 없음."""
    embs = _embed_all(wavs)
    if embs is None or len(embs) < 2:
        return None
    sims = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            for i, a in enumerate(embs) for b in embs[i + 1:]]
    return float(np.mean(sims))


def main() -> None:
    ap = argparse.ArgumentParser(description="v2 TTS 게이트: 음색 안정성 + 지연 A/B")
    ap.add_argument("--ref", required=True, help="레퍼런스 wav (mono)")
    ap.add_argument("--ref-text", default="", help="레퍼런스 전사(유사도↑)")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--same-text", action="store_true", help="매 턴 같은 문장(순수 편차 측정)")
    ap.add_argument("--backends", default="qwen3,cosy")
    ap.add_argument("--out", default="out/bench_v2")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    ref_b64, ref_sr = _b64_ref(args.ref)
    os.makedirs(args.out, exist_ok=True)
    report: dict[str, dict] = {}

    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        url = BACKENDS.get(name)
        if not url:
            print(f"✗ 알 수 없는 백엔드: {name}"); continue
        try:
            urllib.request.urlopen(f"{url}/health", timeout=5)
        except Exception as e:  # noqa: BLE001
            print(f"✗ {name}({url}) 다운({e}) — 스킵"); continue

        print(f"\n== {name} ({url}) — {args.turns}턴 ==")
        wavs, rtfs = [], []
        for i in range(args.turns):
            text = TEXTS[0] if args.same_text else TEXTS[i % len(TEXTS)]
            audio, sr, dt = synth_full(url, text, ref_b64, ref_sr, args.ref_text, args.timeout)
            path = os.path.join(args.out, f"{name}_{i:02d}.wav")
            sf.write(path, audio, sr)
            wavs.append(path)
            dur = len(audio) / max(1, sr)
            rtfs.append(dt / max(0.01, dur))
            print(f"  턴{i:02d}: {dt*1000:6.0f}ms  {dur:4.1f}s  RTF {rtfs[-1]:.3f}  '{text[:18]}…'")
        # 스트리밍 첫패킷(3회 중앙값)
        fp = sorted(synth_stream_first(url, TEXTS[0], ref_b64, ref_sr, args.ref_text, args.timeout)
                    for _ in range(3))[1]
        sim = self_similarity(wavs)
        report[name] = {"rtf_mean": float(np.mean(rtfs)), "first_packet_ms": fp * 1000,
                        "self_sim": sim}
        print(f"  ▶ RTF 평균 {np.mean(rtfs):.3f} · 스트림 첫패킷 {fp*1000:.0f}ms"
              + (f" · 자기유사 {sim:.4f}" if sim is not None else " · (resemblyzer 없음 — 청취 판정)"))

    print("\n===== 게이트 판정 =====")
    q, c = report.get("qwen3"), report.get("cosy")
    if q and c and q["self_sim"] is not None and c["self_sim"] is not None:
        ok_sim = q["self_sim"] >= c["self_sim"] - 0.02
        ok_lat = q["first_packet_ms"] < 300
        print(f"음색 안정성: qwen3 {q['self_sim']:.4f} vs cosy {c['self_sim']:.4f} → {'✅' if ok_sim else '❌'}")
        print(f"첫패킷: qwen3 {q['first_packet_ms']:.0f}ms (<300) → {'✅' if ok_lat else '❌'}")
        print("둘 다 ✅ + 블라인드 청취 통과 시 serve.yaml tts.backend: auto 로 승격."
              if (ok_sim and ok_lat) else "❌ 있음 → cosyvoice3 유지(승격 금지).")
    else:
        print("정량 비교 불충분 — out/ 의 wav 를 블라인드 청취로 판정하라"
              "(pip install resemblyzer 하면 자기유사 자동).")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
