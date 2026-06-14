"""OpenVoice V2 + MeloTTS 한국어 음성복제 품질 검증 (Piper 외계어 / CosyVoice 느림 대체).

구조: MeloTTS(KR 네이티브) 가 한국어 합성 → OpenVoice 톤컬러 변환으로 화자 A 음색 입힘.
제로샷(학습 불필요, A 레퍼런스 클립만). feed-forward 라 빠름. 먼저 torch(CPU)로 품질 확인,
좋으면 OpenVINO 로 Arc 가속.

전제: OpenVoice 저장소 + MeloTTS 설치 + checkpoints_v2 다운로드 (scripts/setup_openvoice.md).
실행은 OpenVoice 저장소 폴더 안에서(또는 PYTHONPATH 에 추가).

사용:
  python openvoice_clone_test.py --ref C:/.../cosyvoice_ref/A_ref1.wav \
      --text "내 왔다 아이가 밥은 묵었나" --ckpt checkpoints_v2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="화자 A 레퍼런스 wav")
    ap.add_argument("--text", default="내 왔다 아이가 밥은 묵었나. 오늘 저녁은 뭐 묵을라꼬?")
    ap.add_argument("--ckpt", default="checkpoints_v2", help="OpenVoice V2 checkpoints 폴더")
    ap.add_argument("--out", default="A_clone.wav")
    ap.add_argument("--device", default="cpu", help="cpu (torch 검증). OpenVINO는 별도")
    args = ap.parse_args()

    import torch
    from melo.api import TTS
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    ck = Path(args.ckpt)

    # 1) MeloTTS 한국어 베이스 합성 (네이티브 한국어 → 외국인 억양 없음)
    print("[1/3] MeloTTS 한국어 베이스 합성")
    t0 = time.time()
    tts = TTS(language="KR", device=args.device)
    spk_id = list(tts.hps.data.spk2id.values())[0]
    tts.tts_to_file(args.text, spk_id, "base_kr.wav", speed=1.0)
    t_base = time.time() - t0

    # 2) OpenVoice 톤컬러 변환기 로드
    print("[2/3] OpenVoice 톤컬러 변환기 로드 + A 음색 추출")
    conv = ToneColorConverter(str(ck / "converter" / "config.json"), device=args.device)
    conv.load_ckpt(str(ck / "converter" / "checkpoint.pth"))
    src_se = torch.load(str(ck / "base_speakers" / "ses" / "kr.pth"), map_location=args.device)
    tgt_se, _ = se_extractor.get_se(args.ref, conv, vad=True)   # A 레퍼런스에서 음색 추출

    # 3) 변환: 베이스 음성 → A 음색
    print("[3/3] 음색 변환 → A 목소리")
    t1 = time.time()
    conv.convert(audio_src_path="base_kr.wav", src_se=src_se, tgt_se=tgt_se,
                 output_path=args.out, message="@callone")
    t_conv = time.time() - t1

    import soundfile as sf
    dur = sf.info(args.out).duration
    total = t_base + t_conv
    print(f"\n완료 → {args.out}")
    print(f"  음성 {dur:.1f}s | 베이스 {t_base:.1f}s + 변환 {t_conv:.1f}s = {total:.1f}s "
          f"| 실시간배율 {total/dur:.1f}x")
    print("  → 들어보고: ① 진짜 한국어인가(외계어/외국인억양 아닌가) ② A 음색 닮았나 ③ 속도")
    print("  (이건 torch-CPU 측정. OpenVINO+Arc 면 더 빠름)")


if __name__ == "__main__":
    main()
