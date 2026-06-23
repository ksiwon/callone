"""CosyVoice2 제로샷 한국어 음성복제 테스트 (Piper 외계어 대체 검증).

화자 A의 깨끗한 레퍼런스 클립 + 한국어 텍스트 → A 목소리로 한국어 합성.
학습 불필요(제로샷). 한국어가 제대로 나오는지 + 속도가 견딜 만한지 확인용.

전제: CosyVoice 설치된 conda 환경에서, CosyVoice 저장소 폴더 안에서 실행.
  (cosyvoice_test.py 를 CosyVoice 저장소로 복사하거나, --repo 로 경로 지정)

사용(CosyVoice 저장소 안에서):
  python cosyvoice_test.py --ref /abs/cosyvoice_ref/A_ref1.wav --ref-text "오늘 맛있더라..." \
      --text "내 왔다 아이가 밥은 묵었나"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pretrained_models/Fun-CosyVoice3-0.5B")
    ap.add_argument("--ref", required=True, help="레퍼런스 wav (화자 A 깨끗한 클립)")
    ap.add_argument("--ref-text", required=True, help="레퍼런스 wav 의 발화 텍스트")
    ap.add_argument("--text", default="내 왔다 아이가 밥은 묵었나. 오늘 저녁은 뭐 먹을라꼬?",
                    help="합성할 한국어 텍스트")
    ap.add_argument("--out", default="cosy_out.wav")
    args = ap.parse_args()

    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    print(f"[1/3] 모델 로드: {args.model} (CPU면 좀 걸림)")
    t0 = time.time()
    cosy = CosyVoice2(args.model, load_jit=False, load_trt=False, fp16=False)
    print(f"      로드 {time.time()-t0:.1f}s")

    print(f"[2/3] 레퍼런스: {args.ref}")
    prompt = load_wav(args.ref, 16000)

    print(f"[3/3] 합성: {args.text!r}")
    t1 = time.time()
    audios = []
    for out in cosy.inference_zero_shot(args.text, args.ref_text, prompt, stream=False):
        audios.append(out["tts_speech"])
    import torch
    wav = torch.cat(audios, dim=1)
    torchaudio.save(args.out, wav, cosy.sample_rate)
    dur = wav.shape[1] / cosy.sample_rate
    gen = time.time() - t1
    print(f"\n완료 → {args.out}")
    print(f"  음성 길이 {dur:.1f}s, 생성 {gen:.1f}s, 실시간배율 {gen/dur:.1f}x "
          f"({'실시간 OK' if gen < dur else '실시간 빠듯' if gen < dur*2 else '느림'})")
    print("  → 들어보고: 한국어 제대로 나오나? A 목소리 닮았나? 속도 견딜 만한가?")


if __name__ == "__main__":
    main()
