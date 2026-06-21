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
    # 원본 비율 유지(정사각 강제 리사이즈는 찌그러짐). 긴 변만 maxside 로 제한(대역폭 절충).
    maxside = max(resolution * 2, 256)              # resolution=256 → 긴 변 512
    w, h = im.size
    if max(w, h) > maxside:
        s = maxside / float(max(w, h))
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
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
        self._cfg_pkl = cfg_pkl
        self._data_root = data_root
        self.sdk = StreamSDK(cfg_pkl, data_root)   # persistent: 모델 1회 로드(턴마다 재로드 안 함)
        self.resolution = 256
        self._fps = 25
        self._image_bytes: bytes | None = None     # start()가 보관 → frames()가 발화마다 setup 에 사용
        self._lock = threading.Lock()

    def start(self, image_bytes: bytes, fps: int, resolution: int) -> None:
        # ★멀티턴 누적 차단(2026-06-21 실측)★: setup 1회 + 턴마다 setup_Nd 재무장만 반복하면 SDK 내부
        #   모션/워커 버퍼가 쌓여 추론이 thrashing(40→3 it/s) → 프레임 지연/클라 타임아웃 → 영상 끊김.
        #   repo inference.py 라이프사이클은 **발화 1개 = setup→run→close**. 그래서 여기선 사진만 보관하고
        #   실제 source setup 은 frames()에서 발화마다 수행한다. persistent SDK(모델)는 __init__ 로드 유지
        #   → 턴당 추가비용은 source 전처리(~1s)뿐(모델 재로드 아님).
        self.resolution = int(resolution)
        self._fps = int(fps)
        self._image_bytes = image_bytes

    def _tmpbase(self) -> str:
        return "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()

    def _feed(self, sdk, a16: np.ndarray) -> int:
        """16k 오디오를 온라인 청크 루프로 SDK 에 먹인다(repo 정합). 반환=예상 프레임수(num_f)."""
        import math
        chunksize = (3, 5, 2)
        num_f = max(1, math.ceil(len(a16) / 16000 * 25))
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        sdk.setup_Nd(N_d=num_f)                            # 발화당 프레임수
        a = np.pad(a16, (chunksize[0] * 640, 0))           # 앞 zero 패딩(repo 정합)
        for i in range(0, len(a), chunksize[1] * 640):
            chunk = a[i:i + split_len]
            if len(chunk) < split_len:
                chunk = np.pad(chunk, (0, split_len - len(chunk)))
            sdk.run_chunk(chunk, chunksize)
        return num_f

    def frames(self, audio: np.ndarray, sr: int):
        """**한 발화 전체 오디오**(임의 sr) → 16k → 발화 단위 setup→run→close → 프레임 JPEG yield.

        멀티턴 누적 방지를 위해 발화마다 **새 setup→close 사이클**(repo inference.py 라이프사이클).
        지점: setup(사진 등록) → writer 를 _FrameSink 로 가로채기 → setup_Nd+run_chunk → num_f drain
              → close(워커/내부버퍼 정리). 다음 발화는 또 깨끗한 setup 부터.
        """
        if self._image_bytes is None:
            return
        a = np.asarray(audio, dtype=np.float32).flatten()
        if sr != 16000:                         # Ditto 는 16kHz
            try:
                import librosa

                a = librosa.resample(a, orig_sr=sr, target_sr=16000)
            except Exception:                   # noqa: BLE001  librosa 없으면 선형 리샘플
                n = int(len(a) * 16000 / sr)
                a = np.interp(np.linspace(0, len(a), n, endpoint=False),
                              np.arange(len(a)), a).astype("float32")
        if len(a) == 0:
            return
        # 프라이버시: 사진을 tmpfs(/dev/shm, RAM)에 잠깐 쓰고 setup 직후 즉시 삭제(디스크 영속 0).
        base = self._tmpbase()
        fd, src_path = tempfile.mkstemp(suffix=".png", prefix="ditto_src_", dir=base)
        with os.fdopen(fd, "wb") as f:
            f.write(self._image_bytes)
        dummy_out = os.path.join(base, f"ditto_out_{os.getpid()}.mp4")  # 안 쓰임(writer 가로채기)
        sink = _FrameSink()
        num_f = 0
        with self._lock:                        # 발화 단위 setup+run (턴은 순차 처리라 경합 없음)
            try:
                try:
                    self.sdk.setup(src_path, dummy_out, online_mode=True)
                except Exception:               # noqa: BLE001  close() 후 재setup 불가 상태면
                    self.sdk = self._SDK_cls(self._cfg_pkl, self._data_root)  # SDK 1회 재생성(모델 재로드 수초)
                    self.sdk.setup(src_path, dummy_out, online_mode=True)
                self.sdk.writer = sink          # [V] attr 명 self.writer — 파일 대신 우리 큐로
                num_f = self._feed(self.sdk, a)
            finally:
                for p in (src_path, dummy_out):  # 사진·더미 즉시 삭제
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        # 워커가 비동기로 num_f 프레임 생성 → 모일 때까지 drain. 첫 프레임은 콜드(TRT 첫추론 ~수십초)라
        # 아주 길게(90s), 이후는 짧게(2s). (클라 WS 타임아웃도 이보다 커야 함 — avatar.py)
        got = 0
        while got < num_f:
            try:
                fr = sink.q.get(timeout=(90.0 if got == 0 else 2.0))
            except queue.Empty:
                break
            if fr is None:
                break
            got += 1
            yield _jpeg(fr, self.resolution)
        # 발화 종료: close()로 워커/내부버퍼 정리 → 다음 발화 누적 0(멀티턴 thrashing 차단).
        with self._lock:
            try:
                self.sdk.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._image_bytes = None
        with self._lock:
            try:
                self.sdk.close()
            except Exception:  # noqa: BLE001
                pass
