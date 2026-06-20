"""Ditto 토킹헤드 어댑터 — antgroup/ditto-talkinghead (ACM MM 2025).

증명사진 1장 + 오디오 → 입·표정·고개 움직임. **GPU 단계서 채운다**(이 노트북엔 모델 없음).
웹검증(2026-06-20): CLI `inference.py`(배치) + repo 에 `stream_pipeline_online.py`(스트리밍) 존재.
모델 HF `digital-avatar/ditto-talkinghead`(TensorRT Ampere+ / PyTorch). A100=TensorRT 권장.

통합 지점(GPU 박스에서):
  1) Ditto repo clone + `git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints`
  2) 아래 __init__ 에서 StreamSDK(또는 stream_pipeline_online 의 진입점) 로드 — **프로세스 1회**(persistent).
     예) from stream_pipeline_online import StreamSDK
         self.sdk = StreamSDK(cfg_pkl, data_root)
  3) start(): 사진 등록·source 셋업(Ditto setup) — 통화당 1회(identity 사전추출).
  4) frames(): 오디오청크 → SDK 에 밀어넣고 나오는 프레임(BGR/RGB ndarray)을 JPEG 로 인코딩해 yield.
     ⚠️ Ditto online 파이프라인의 정확한 청크 입력/프레임 출력 시그니처는 repo 코드로 확인해 맞출 것.
"""
from __future__ import annotations

import io

import numpy as np

from ..app import AvatarModel


def _jpeg(frame_rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame_rgb.astype("uint8")).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class DittoModel(AvatarModel):
    name = "ditto"

    def __init__(self):
        # TODO(GPU): Ditto SDK 를 여기서 1회 로드(persistent). 체크포인트 경로는 env 로.
        #   import os; from stream_pipeline_online import StreamSDK
        #   self.sdk = StreamSDK(os.environ["DITTO_CFG_PKL"], os.environ["DITTO_DATA_ROOT"])
        raise RuntimeError(
            "DittoModel 미구현(GPU 단계). Ditto repo+checkpoints 준비 후 이 어댑터의 "
            "__init__/start/frames 를 stream_pipeline_online API 로 채워라. 지금은 static 폴백.")

    def start(self, image_bytes: bytes, fps: int, resolution: int) -> None:
        # TODO(GPU): 사진 → Ditto source 셋업(얼굴 검출·정렬은 Ditto 내부). 통화당 1회.
        raise NotImplementedError

    def frames(self, audio: np.ndarray, sr: int):
        # TODO(GPU): audio 청크 → SDK → 프레임 ndarray 들 → _jpeg 로 yield.
        raise NotImplementedError
