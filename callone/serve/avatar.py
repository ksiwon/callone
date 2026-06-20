"""S7+ 토킹헤드(사진→움직이는 얼굴) 추상화 — docs/avatar_talking_head_design.md.

증명사진 1장 + 통화 TTS 음성 → 입·표정·고개가 움직이는 영상 프레임(JPEG).
음성 스택과 동일 철학: **무거운 모델 미설치/저사양이면 정지사진으로 안전 폴백**.
영상은 음성 통화의 *부가 레이어*지 의존 대상이 아니다.

백엔드 선택(_pick_avatar, _pick_tts 와 동형 폴백 체인):
  1) DittoAvatar   — 별도 프로세스 avatar-server(Ditto, GPU)에 HTTP/WS 로 호출.
  2) (후속) MuseTalk
  3) StaticImageAvatar — 사진 한 장을 그대로 프레임으로(움직임 없음). 모델/서버 없을 때.

인터페이스(모든 백엔드 공통):
  - start_call(image_path)              통화당 1회: 사진 등록(identity 사전추출은 서버 몫)
  - frames_for(audio, sr) -> Iterator[bytes]   오디오 청크 → JPEG 프레임들
  - interrupt()                         barge-in 시 현재 프레임 생성 중단
  - stop()                              세션 종료/리소스 해제
"""
from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from ..common.logging import get_logger

log = get_logger("avatar")


@runtime_checkable
class AvatarBackend(Protocol):
    fps: int

    def start_call(self, image_path: str) -> None: ...
    def frames_for(self, audio: np.ndarray, sr: int) -> Iterator[bytes]: ...
    def interrupt(self) -> None: ...
    def stop(self) -> None: ...


def _encode_jpeg(image_path: str, resolution: int = 256) -> bytes:
    """사진 → 정사각 리사이즈 JPEG 바이트. Pillow 없으면 원본 바이트 그대로(이미 jpg/png)."""
    try:
        from PIL import Image  # type: ignore

        im = Image.open(image_path).convert("RGB")
        # 증명사진 가정(정면·얼굴 중앙) → 가운데 정사각 크롭 후 resolution 리사이즈.
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((resolution, resolution))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:  # noqa: BLE001
        log.warning("Pillow 인코딩 불가(%s) — 원본 바이트 사용", e)
        return Path(image_path).read_bytes()


class StaticImageAvatar:
    """폴백: 사진 한 장을 움직임 없이 프레임으로 반복. 모델/서버 없어도 영상통화 화면은 뜬다.

    오디오 길이에 맞춰 fps 만큼 같은 프레임을 emit(립싱크 없음). avatar-server 가 뜨면
    DittoAvatar 가 대신 잡는다(아래 _pick_avatar)."""

    def __init__(self, cfg: dict | None = None):
        acfg = cfg or {}
        self.fps = int(acfg.get("fps", 25))
        self.resolution = int(acfg.get("resolution", 256))
        self._frame: bytes | None = None
        self._interrupt = threading.Event()

    def start_call(self, image_path: str) -> None:
        if not image_path or not Path(image_path).exists():
            raise RuntimeError(f"avatar 사진 없음: {image_path!r}")
        self._frame = _encode_jpeg(image_path, self.resolution)
        log.info("StaticImageAvatar: %s (%dB, %dpx, %dfps)",
                 image_path, len(self._frame), self.resolution, self.fps)

    def frames_for(self, audio: np.ndarray, sr: int) -> Iterator[bytes]:
        if self._frame is None:
            return
        self._interrupt.clear()
        dur = len(audio) / float(sr or 1)
        n = max(1, int(dur * self.fps))
        for _ in range(n):
            if self._interrupt.is_set():
                break
            yield self._frame

    def interrupt(self) -> None:
        self._interrupt.set()

    def stop(self) -> None:
        self._frame = None


class DittoAvatar:
    """Ditto 토킹헤드 — 별도 프로세스 avatar-server(GPU)의 HTTP/WS 클라이언트.

    서빙 venv 엔 Ditto 의존성 0(transformers/diffusers 충돌 회피) → llama-server 와 동일하게
    **HTTP/WS 로만 호출**. avatar-server 미가동이면 probe 실패 → orchestrator 가 StaticImage 폴백.

    avatar-server 인터페이스(설계서 §2-2):
      POST /session/start  {image_b64, fps, resolution} -> {session_id}
      WS   /session/{id}/stream  ← PCM f32 오디오 업, → JPEG 프레임 다운
      POST /session/{id}/stop
      GET  /health
    ⚠️ WS 스트리밍 본체는 avatar-server 구현 후 연결(GPU 단계). 지금은 probe+세션 골격만.
    """

    def __init__(self, cfg: dict | None = None, probe: bool = True):
        acfg = cfg or {}
        self.base_url = str(acfg.get("base_url", "http://127.0.0.1:8091")).rstrip("/")
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.fps = int(acfg.get("fps", 25))
        self.resolution = int(acfg.get("resolution", 256))
        self.sr = int(acfg.get("sr", 24000))   # TTS 출력 sr(오디오 청크)
        self.timeout = float(acfg.get("timeout", 30.0))
        self.session_id: str | None = None
        self._ws = None
        self._interrupt = threading.Event()
        if probe:
            self._probe()
        log.info("DittoAvatar: %s (%dfps, %dpx)", self.base_url, self.fps, self.resolution)

    def _probe(self) -> None:
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=3) as r:
                if r.status >= 400:
                    raise RuntimeError(f"health {r.status}")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"avatar-server 응답 없음({self.base_url}). 별도 프로세스로 띄워야 함"
                f"(scripts/setup_avatar_gpu.sh, GPU). 미가동 시 StaticImage 폴백: {e}") from e

    def start_call(self, image_path: str) -> None:
        import base64
        import json
        import urllib.request

        if not image_path or not Path(image_path).exists():
            raise RuntimeError(f"avatar 사진 없음: {image_path!r}")
        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        body = json.dumps({"image_b64": img_b64, "fps": self.fps,
                           "resolution": self.resolution, "sr": self.sr}).encode()
        req = urllib.request.Request(f"{self.base_url}/session/start", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            self.session_id = json.loads(r.read().decode()).get("session_id")
        log.info("DittoAvatar 세션 시작: %s (사진 %s)", self.session_id, image_path)

    def _ensure_ws(self):
        """세션 WS 를 lazy 연결(청크마다 재사용). websocket-client 필요."""
        if self._ws is not None:
            return self._ws
        import websocket  # type: ignore  # pip install websocket-client

        if not self.session_id:
            raise RuntimeError("start_call 먼저 — session 없음")
        self._ws = websocket.create_connection(
            f"{self.ws_url}/session/{self.session_id}/stream", timeout=self.timeout)
        return self._ws

    def frames_for(self, audio: np.ndarray, sr: int) -> Iterator[bytes]:
        """오디오 청크 → avatar-server WS 로 보내고 JPEG 프레임 받아 yield.
        프로토콜: 청크(f32 bytes) 업 → 서버가 프레임들(binary) 후 {"type":"chunk_done"}(text)."""
        self._interrupt.clear()
        ws = self._ensure_ws()
        ws.send_binary(np.asarray(audio, dtype=np.float32).tobytes())
        while True:
            if self._interrupt.is_set():
                try:
                    ws.send(json.dumps({"type": "interrupt"}))
                except Exception:  # noqa: BLE001
                    pass
                break
            op = ws.recv()
            if isinstance(op, (bytes, bytearray)):
                if op:
                    yield bytes(op)
            else:  # text = 제어
                try:
                    if json.loads(op).get("type") == "chunk_done":
                        break
                except Exception:  # noqa: BLE001
                    break

    def interrupt(self) -> None:
        self._interrupt.set()

    def stop(self) -> None:
        if self._ws is not None:
            try:
                self._ws.send(json.dumps({"type": "stop"}))
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None
        if not self.session_id:
            return
        import urllib.request

        try:
            req = urllib.request.Request(
                f"{self.base_url}/session/{self.session_id}/stop", method="POST")
            urllib.request.urlopen(req, timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self.session_id = None


def _pick_avatar(serve_cfg: dict | None = None) -> AvatarBackend:
    """_pick_tts 와 동형 폴백 체인. avatar.enabled=false 면 항상 StaticImage(이미지 없으면도 안전)."""
    acfg = (serve_cfg or {}).get("avatar", {}) or {}
    backend = acfg.get("backend", "auto")

    if backend in ("ditto", "auto"):
        try:
            return DittoAvatar(acfg)
        except Exception as e:  # noqa: BLE001
            log.warning("DittoAvatar 불가(%s) — StaticImage 폴백", e)
            if backend == "ditto":
                return StaticImageAvatar(acfg)
    # (후속) MuseTalk 자리
    return StaticImageAvatar(acfg)
