#!/usr/bin/env python3
"""tag_ab_test — CosyVoice3 체크포인트가 인라인 태그를 **실제로 파싱**하는지 A/B 판정.

왜?  orchestrator 는 지금 [breath]/[moan] 같은 대괄호를 TTS 도달 전에 다 지운다(발화 누출 방지).
     태그를 화이트리스트로 살리려면, 이 서버 모델(Fun-CosyVoice3-0.5B)이 그 토큰을 **소리로**
     해석하는지 먼저 확인해야 한다. 논문/데모가 지원해도 이 0.5B 체크포인트가 될지는 별개.
     안 되면 태그는 그냥 글자로 읽혀 오히려 나빠진다 → 살리기 전 필수 검증.

동작:  cosyvoice_server(:8092) /synth 에 레퍼런스 클립 + (플레인 vs 태그) 문장쌍을 보내
       out_dir 에 wav 로 저장. 사람이 들어서 판정:
         - 태그 케이스에서 **숨소리/웃음/한숨이 들리고** '브레스/래프터' 같은 글자를 안 읽으면 → 파싱됨(살려도 됨)
         - 태그 글자를 그대로 읽거나 플레인과 똑같으면 → 미파싱(살리지 말 것)

사용:
    python scripts/tag_ab_test.py --ref data/refs/sample.wav \
        --ref-text "안녕하세요 반가워요" [--url http://127.0.0.1:8092] [--out out/tag_ab]

torch 불필요(urllib+numpy+soundfile). serve venv 에서 그대로 실행.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request

import numpy as np
import soundfile as sf

# 플레인 ↔ 태그 대응쌍(같은 문장, 태그만 추가). 모델이 태그를 소리로 바꾸는지 대조.
CASES: dict[str, str] = {
    "01_plain_greet":     "안녕하세요. 오랜만이에요.",
    "02_breath_greet":    "안녕하세요. [breath] 오랜만이에요.",
    "03_plain_laugh":     "그래요? 정말 웃기네요.",
    "04_laughter":        "그래요? [laughter] 정말 웃기네요.",
    "05_sigh":            "하아... [sigh] 보고 싶었어요.",
    "06_moan":            "가까이 와요. [moan] 더 가까이.",
    "07_whisper_tag":     "[whisper] 아무한테도 말하지 마요.",
    "08_literal_control": "다음 단어를 읽어보세요 breath.",   # 통제군: 'breath' 를 글자로 읽는지 기준
}


def _load_ref_b64(path: str) -> tuple[str, int]:
    y, sr = sf.read(path, dtype="float32")
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    return base64.b64encode(y.tobytes()).decode(), int(sr)


def _synth(url: str, text: str, ref_b64: str, ref_sr: int,
           ref_text: str, language: str, timeout: float) -> tuple[np.ndarray, int]:
    payload = json.dumps({
        "text": text, "ref_audio_b64": ref_b64, "ref_sr": ref_sr,
        "prompt_text": ref_text, "language": language,
    }).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/synth", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        out_sr = int(r.headers.get("X-Sample-Rate", "24000"))
    return np.frombuffer(raw, dtype=np.float32), out_sr


def main() -> None:
    ap = argparse.ArgumentParser(description="CosyVoice3 인라인 태그 파싱 A/B 검증")
    ap.add_argument("--ref", required=True, help="레퍼런스 wav (mono 권장)")
    ap.add_argument("--ref-text", default="", help="레퍼런스 전사(ICL 유사도↑, 권장)")
    ap.add_argument("--url", default=os.environ.get("COSYVOICE_URL", "http://127.0.0.1:8092"))
    ap.add_argument("--language", default="Korean")
    ap.add_argument("--out", default="out/tag_ab")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    if not os.path.exists(args.ref):
        raise SystemExit(f"레퍼런스 없음: {args.ref}")
    os.makedirs(args.out, exist_ok=True)
    ref_b64, ref_sr = _load_ref_b64(args.ref)
    print(f"[tag_ab] ref={args.ref} ({ref_sr}Hz) → {args.url}  out={args.out}")

    for name, text in CASES.items():
        try:
            audio, sr = _synth(args.url, text, ref_b64, ref_sr, args.ref_text,
                               args.language, args.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {name}: {e}")
            continue
        path = os.path.join(args.out, f"{name}.wav")
        sf.write(path, audio, sr)
        print(f"  ✓ {name:20s} {len(audio)/max(1,sr):4.1f}s  '{text}'  → {path}")

    print("\n판정법: 02/04/05/06/07 을 01/03 과 비교해 들어라.")
    print("  숨소리·웃음·한숨이 소리로 들리고 08 은 'breath' 를 글자로 읽으면 → 태그 파싱됨(화이트리스트 OK).")
    print("  태그 글자를 읽거나 플레인과 동일하면 → 미파싱(orchestrator stripping 유지).")


if __name__ == "__main__":
    main()
