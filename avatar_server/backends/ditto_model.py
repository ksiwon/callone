"""Ditto 토킹헤드 어댑터 — antgroup/ditto-talkinghead (ACM MM 2025). PyTorch 백엔드.

증명사진 1장 + 오디오 → 입·표정·고개. 실 API(웹검증 2026-06-20, repo 소스):
  from stream_pipeline_online import StreamSDK          # 온라인(스트리밍) 모드
  SDK = StreamSDK(cfg_pkl, data_root)
  SDK.setup(source_path, output_path, online_mode=True) # 사진 등록 + 파이프라인 셋업
  SDK.run_chunk(audio_chunk_16k, chunksize=(3,5,2))     # 오디오 청크 → 모션 → 프레임(워커스레드)
  SDK.close()
  # 프레임은 워커가 self.writer(res_frame_rgb, fmt="rgb") 로 파일에 씀
  #   → 우리는 self.writer 를 큐 싱크로 바꿔치기해 **프레임을 WS 로 빼낸다**(파일 안 씀).

환경변수(setup_avatar_gpu.sh 가 안내):
  DITTO_REPO      = ditto-talkinghead repo 경로(sys.path 추가)
  DITTO_DATA_ROOT = checkpoints/ditto_pytorch/models   (PyTorch 백엔드)
  DITTO_CFG_PKL   = checkpoints/.../v0.4_hubert_cfg_pytorch.pkl

⚠️ GPU 박스에서 검증할 지점(주석 [V]): ① setup_Nd 필요여부 ② run_chunk 입력 청크 크기/리샘플
   ③ writer 싱크 attribute 명(self.writer) ④ 프레임 색공간(rgb ndarray). repo 코드로 맞춰라.
"""
from __future__ import annotations

import io
import os
import queue
import sys
import tempfile
import threading

import numpy as np

from ..app import AvatarModel


class _FrameSink:
    """StreamSDK.writer 대체 — 워커가 self.writer(frame, fmt="rgb") 호출하면 큐에 담는다(파일 안 씀)."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()

    def __call__(self, frame, fmt="rgb"):   # [V] 워커 호출 시그니처 = self.writer(res_frame_rgb, fmt="rgb")
        self.q.put(frame)

    def close(self):                         # VideoWriterByImageIO 호환(no-op)
        pass


def _jpeg(frame_rgb: np.ndarray, resolution: int) -> bytes:
    from PIL import Image

    im = Image.fromarray(np.asarray(frame_rgb).astype("uint8"))
    if im.size != (resolution, resolution):
        im = im.resize((resolution, resolution))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class DittoModel(AvatarModel):
    name = "ditto"

    def __init__(self):
        repo = os.environ.get("DITTO_REPO")
        data_root = os.environ.get("DITTO_DATA_ROOT")
        cfg_pkl = os.environ.get("DITTO_CFG_PKL")
        if not (repo and data_root and cfg_pkl):
            raise RuntimeError(
                "Ditto 환경변수 필요: DITTO_REPO, DITTO_DATA_ROOT(.../ditto_pytorch/models), "
                "DITTO_CFG_PKL(.../v0.4_hubert_cfg_pytorch.pkl). setup_avatar_gpu.sh 참고.")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        # persistent: 프로세스 시작 시 1회 로드(콜드 제거). 실패하면 app._pick_model 이 static 폴백.
        from stream_pipeline_online import StreamSDK  # type: ignore

        self._SDK_cls = StreamSDK
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self.resolution = 256
        self.sink: _FrameSink | None = None
        self._lock = threading.Lock()

    def start(self, image_bytes: bytes, fps: int, resolution: int) -> None:
        self.resolution = int(resolution)
        # 사진을 임시파일로(Ditto setup 은 source_path 받음). 얼굴 검출·정렬은 Ditto 내부.
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.write(image_bytes)
        f.close()
        dummy_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        # [V] online_mode=True 로 스트리밍 셋업. setup_Nd 가 필요하면 여기서 호출(총 프레임 수 미정→online 처리).
        self.sdk.setup(f.name, dummy_out, online_mode=True)
        # 프레임 싱크로 writer 바꿔치기 → 파일 대신 우리 큐로.
        self.sink = _FrameSink()
        self.sdk.writer = self.sink            # [V] attribute 명 self.writer 확인

    def frames(self, audio: np.ndarray, sr: int):
        """오디오 청크(임의 sr) → 16k 리샘플 → run_chunk → 워커가 생성한 프레임 drain → JPEG yield."""
        if self.sink is None:
            return
        a = np.asarray(audio, dtype=np.float32)
        if sr != 16000:                         # [V] Ditto 는 16kHz(librosa load sr=16000)
            try:
                import librosa

                a = librosa.resample(a, orig_sr=sr, target_sr=16000)
            except Exception:                   # noqa: BLE001  librosa 없으면 선형 리샘플(품질 무관)
                n = int(len(a) * 16000 / sr)
                a = np.interp(np.linspace(0, len(a), n, endpoint=False),
                              np.arange(len(a)), a).astype("float32")
        with self._lock:
            self.sdk.run_chunk(a, chunksize=(3, 5, 2))   # [V] 입력 청크 크기 정합은 repo 확인
        # 워커 스레드가 비동기로 프레임 생성 → 현재까지 나온 것 drain(타임아웃으로 idle 감지).
        while True:
            try:
                fr = self.sink.q.get(timeout=0.2)
            except queue.Empty:
                break
            if fr is None:
                break
            yield _jpeg(fr, self.resolution)

    def stop(self) -> None:
        try:
            self.sdk.close()
        except Exception:  # noqa: BLE001
            pass
