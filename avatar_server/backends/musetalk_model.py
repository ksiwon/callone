"""MuseTalk 어댑터 — TMElyralab/MuseTalk (입/하관 위주, 가볍고 실시간 검증됨).

레퍼런스(PunithVT/ai-avatar-system, 2026-03)가 MuseTalk persistent worker 로 실시간 토킹헤드를
**실제 성공**시킨 검증된 경로. 머리·표정 큰 움직임은 Ditto, 입 중심 경량·안정은 MuseTalk.
**GPU 단계서 채운다.** ~9GB, 256², ~30fps(24GB GPU). mmcv/mmpose/diffusers 필요(별 venv).

통합 지점(GPU 박스):
  1) MuseTalk repo + 체크포인트(`bash scripts/setup_musetalk.sh` 류) 준비.
  2) __init__: UNet/VAE/whisper(audio feat) 모델 1회 로드(persistent worker).
  3) start(): 사진 → 얼굴 검출·latent 사전계산(통화당 1회).
  4) frames(): 오디오 → whisper feature → latent inpainting(1-step, 확산 아님) → 프레임 → JPEG yield.
"""
from __future__ import annotations

import numpy as np

from ..app import AvatarModel


class MuseTalkModel(AvatarModel):
    name = "musetalk"

    def __init__(self):
        raise RuntimeError(
            "MuseTalkModel 미구현(GPU 단계). MuseTalk repo+checkpoints 준비 후 __init__/start/frames "
            "를 채워라(persistent worker 로 모델 1회 로드). 지금은 static 폴백.")

    def start(self, image_bytes: bytes, fps: int, resolution: int) -> None:
        raise NotImplementedError

    def frames(self, audio: np.ndarray, sr: int):
        raise NotImplementedError
