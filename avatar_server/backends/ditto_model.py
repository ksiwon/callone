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
        self._cfg_pkl = cfg_pkl                 # 세션마다 SDK 재생성용(누적 방지) — 아래 start 참고
        self._data_root = data_root
        self.sdk = StreamSDK(cfg_pkl, data_root)
        self._dirty = False                     # SDK 가 한 번이라도 setup/run 됐나(세션 누적 → OOM 방지)
        self.resolution = 256
        self.sink: _FrameSink | None = None
        self._lock = threading.Lock()

    def start(self, image_bytes: bytes, fps: int, resolution: int) -> None:
        self.resolution = int(resolution)
        # ★세션 정리(2026-06-21 실측 OOM 사망 대응)★: 이전 통화에서 쓴 SDK 를 그대로 재 setup 하면
        #   워커스레드·내부버퍼·프레임큐가 누적돼 다음 세션에서 추론 thrashing → 프로세스 OOM 사망.
        #   새 통화마다 close()로 워커 정리 후 SDK 재생성(깨끗한 상태). 모델 재로드 수초지만 통화당 1회라 수용.
        if self._dirty:
            try:
                self.sdk.close()
            except Exception:  # noqa: BLE001
                pass
            self.sdk = self._SDK_cls(self._cfg_pkl, self._data_root)
            self._dirty = False
        # 프라이버시: 사진을 **tmpfs(/dev/shm, RAM)** 에 잠깐 쓰고 setup 직후 즉시 삭제(디스크 영속 0).
        #   Ditto setup 은 source_path 를 받아 메모리로 로드 → 그 뒤 파일 불필요.
        base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
        fd, src_path = tempfile.mkstemp(suffix=".png", prefix="ditto_src_", dir=base)
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)
        dummy_out = os.path.join(base, f"ditto_out_{os.getpid()}.mp4")  # 실제 안 쓰임(writer 가로채기로 파일출력 우회)
        try:
            self.sdk.setup(src_path, dummy_out, online_mode=True)
        finally:
            for p in (src_path, dummy_out):    # 사진·더미 출력 즉시 삭제
                try:
                    os.remove(p)
                except OSError:
                    pass
        # 프레임 싱크로 writer 바꿔치기 → 파일 대신 우리 큐로.
        self.sink = _FrameSink()
        self.sdk.writer = self.sink            # [V] attribute 명 self.writer 확인
        self._dirty = True                     # 이 SDK 는 이제 사용됨 → 다음 통화 start 에서 재생성
        # 예열 안 함: 첫 턴은 콜드(TRT 첫 추론 ~수십초)지만 사용자 수용. 예열을 넣으면 드레인 경쟁/
        # 클라 타임아웃과 엉켜 0프레임 났음(제거가 맞음). 첫 턴만 느리고 이후 RTF<1.

    def _drain_queue(self) -> None:
        """프레임 큐의 잔여를 비운다. 이전 턴에서 SDK 가 num_f 초과로 흘린 프레임이 남으면 다음 턴
        프레임과 섞여 desync + 큐 무한증가(RAM 누수)가 되므로 턴 경계에서 비운다(non-blocking)."""
        if self.sink is None:
            return
        try:
            while True:
                self.sink.q.get_nowait()
        except queue.Empty:
            pass

    def _feed(self, a16: np.ndarray) -> int:
        """16k 오디오를 온라인 청크 루프로 SDK 에 먹인다. 반환=예상 프레임수(num_f)."""
        import math
        chunksize = (3, 5, 2)
        num_f = max(1, math.ceil(len(a16) / 16000 * 25))
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        with self._lock:
            self.sdk.setup_Nd(N_d=num_f)                       # 발화당 프레임수 재무장
            a = np.pad(a16, (chunksize[0] * 640, 0))           # 앞 zero 패딩(repo 정합)
            for i in range(0, len(a), chunksize[1] * 640):
                chunk = a[i:i + split_len]
                if len(chunk) < split_len:
                    chunk = np.pad(chunk, (0, split_len - len(chunk)))
                self.sdk.run_chunk(chunk, chunksize)
        return num_f

    def frames(self, audio: np.ndarray, sr: int):
        """**한 발화 전체 오디오**(임의 sr) → 16k → setup_Nd(프레임수) → 온라인 청크 루프 run_chunk
        → 워커가 비동기 생성한 프레임을 num_f 개 모일 때까지 drain → JPEG yield.

        repo inference.py 온라인 흐름 정합(2026-06-21 검증):
          num_f = ceil(len/16000*25); setup_Nd(N_d=num_f);
          앞 zero패딩 chunksize[0]*640; split_len=sum(chunksize)*0.04*16000+80;
          for i in range(0,len,chunksize[1]*640): run_chunk(audio[i:i+split_len], chunksize)
        [V] GPU 검증: ① close() 없이 발화마다 setup_Nd 재무장 가능한지(안 되면 발화당 재 setup)
                      ② 마지막 프레임 flush 타이밍(close 가 flush 하는데 멀티턴이라 close 안 함).
        """
        if self.sink is None:
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
        self._drain_queue()                                   # 이전 턴 잔여 프레임 제거(섞임/누수 방지)
        num_f = self._feed(a)                                  # setup_Nd + 청크 루프
        # 워커가 비동기로 num_f 프레임 생성 → 모일 때까지 drain. 첫 프레임은 콜드(TRT 첫추론 ~수십초)라
        # 아주 길게(90s), 이후는 짧게(2s). (클라 WS 타임아웃도 이보다 커야 함 — avatar.py)
        got = 0
        while got < num_f:
            try:
                fr = self.sink.q.get(timeout=(90.0 if got == 0 else 2.0))
            except queue.Empty:
                break
            if fr is None:
                break
            got += 1
            yield _jpeg(fr, self.resolution)
        self._drain_queue()                                   # 이번 턴에 SDK 가 더 흘린 잔여 비움(누적 방지)

    def stop(self) -> None:
        try:
            self.sdk.close()
        except Exception:  # noqa: BLE001
            pass
