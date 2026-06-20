"""avatar-server — 토킹헤드 별도 프로세스(설계서 §2-2). callone 서빙과 분리.

왜 별 프로세스: Ditto/MuseTalk 은 자체 diffusers/mmcv/torch 스택 → callone 서빙 venv
(transformers==4.57.3 하드핀)과 충돌 → llama-server 와 동일하게 **별 venv·별 프로세스**로
띄우고 HTTP/WS 로만 호출(서빙 파이썬엔 토킹헤드 의존성 0).

엔드포인트:
  GET  /health                          → {"status":"ok", "backend": ...}
  POST /session/start  {image_b64,fps,resolution,sr} → {session_id}
  WS   /session/{id}/stream             ← 오디오청크(f32 LE bytes) 업, → JPEG 프레임(bytes) 다운
       per 청크: 서버가 프레임들(binary) 후 {"type":"chunk_done"}(text). 클라가 control(text) 보내면 중단.
  POST /session/{id}/stop

백엔드(_pick_model, callone _pick_avatar 와 동형):
  Ditto/MuseTalk(GPU, 자리만) → StaticModel(정지사진, CPU — 모델 없이 전체 파이프라인 검증).

설계 교훈 반영(PunithVT/ai-avatar-system, 2026-03):
  - 모델은 **프로세스 시작 시 1회 로드(persistent)** → 첫 요청 콜드 제거.
  - **오디오 duration 이 프레임 수 결정**(fps 가 아니라). 사진 전처리는 모델 내부(원본 그대로 넘김).

기동: AVATAR_BACKEND=static python -m avatar_server   (GPU: ditto|musetalk, 별 venv)
"""
from __future__ import annotations

import base64
import io
import json
import os
import uuid
from pathlib import Path

import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
except Exception:  # noqa: BLE001
    FastAPI = None  # type: ignore


# ===================================================================== 백엔드 추상화
class AvatarModel:
    """프레임 생성기 인터페이스. start(image)=identity 사전추출(통화당 1회), frames(audio,sr)=프레임들."""

    name = "base"

    def start(self, image_bytes: bytes, fps: int, resolution: int) -> None: ...
    def frames(self, audio: np.ndarray, sr: int):  # -> Iterator[bytes(jpeg)]
        raise NotImplementedError


class StaticModel(AvatarModel):
    """정지사진 백엔드(CPU) — 모델 없이 전체 WS 파이프라인 검증용. 움직임 없음.

    Ditto/MuseTalk 가 들어오면 _pick_model 이 대신 잡는다. 오디오 duration × fps 만큼 같은 프레임 emit."""

    name = "static"

    def __init__(self):
        self.fps = 25
        self.sr = 24000
        self._frame: bytes | None = None

    def start(self, image_bytes: bytes, fps: int, resolution: int) -> None:
        self.fps = int(fps)
        self._frame = _to_jpeg(image_bytes, resolution)

    def frames(self, audio: np.ndarray, sr: int):
        if self._frame is None:
            return
        n = max(1, int(len(audio) / float(sr or 1) * self.fps))
        for _ in range(n):
            yield self._frame


def _to_jpeg(image_bytes: bytes, resolution: int) -> bytes:
    """원본 사진 → 정사각 리사이즈 JPEG(정지 백엔드 표시용). Pillow 없으면 원본 그대로."""
    try:
        from PIL import Image  # type: ignore

        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize(
            (resolution, resolution))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return image_bytes


def _pick_model() -> AvatarModel:
    """AVATAR_BACKEND=ditto|musetalk|static. GPU 백엔드는 import 실패 시 static 폴백."""
    backend = os.environ.get("AVATAR_BACKEND", "static").lower()
    if backend == "ditto":
        try:
            from .backends.ditto_model import DittoModel  # type: ignore

            return DittoModel()
        except Exception as e:  # noqa: BLE001
            print(f"[avatar] Ditto 로드 실패({e}) — static 폴백")
    elif backend == "musetalk":
        try:
            from .backends.musetalk_model import MuseTalkModel  # type: ignore

            return MuseTalkModel()
        except Exception as e:  # noqa: BLE001
            print(f"[avatar] MuseTalk 로드 실패({e}) — static 폴백")
    return StaticModel()


# ===================================================================== 앱
def create_app():
    if FastAPI is None:
        raise RuntimeError("fastapi 미설치 — pip install fastapi uvicorn")
    app = FastAPI(title="callone-avatar", version="0.1.0")
    model = _pick_model()          # persistent: 프로세스 시작 시 1회 로드(콜드 제거)
    sessions: dict[str, dict] = {}
    print(f"[avatar] backend={model.name}")

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": model.name}

    @app.post("/session/start")
    async def session_start(payload: dict):
        try:
            img = base64.b64decode(payload["image_b64"])
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "image_b64 필요"}, status_code=400)
        fps = int(payload.get("fps", 25))
        res = int(payload.get("resolution", 256))
        sid = uuid.uuid4().hex[:12]
        # 사진 등록 + identity 사전추출(모델 내부). 통화당 1회.
        model.start(img, fps, res)
        sessions[sid] = {"fps": fps, "resolution": res,
                         "sr": int(payload.get("sr", 24000))}
        return {"session_id": sid, "fps": fps, "resolution": res}

    @app.post("/session/{sid}/stop")
    def session_stop(sid: str):
        sessions.pop(sid, None)
        return {"status": "stopped"}

    @app.websocket("/session/{sid}/stream")
    async def stream(ws: WebSocket, sid: str):
        await ws.accept()
        sess = sessions.get(sid)
        if sess is None:
            await ws.close(code=4404)
            return
        sr = sess["sr"]
        try:
            while True:
                msg = await ws.receive()
                # 클라 연결 끊김 → disconnect 메시지 받으면 즉시 종료(이후 receive() 호출 시 RuntimeError 방지).
                if msg.get("type") == "websocket.disconnect":
                    break
                if "text" in msg and msg["text"]:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") in ("stop", "interrupt", "end"):
                        break
                elif "bytes" in msg and msg["bytes"] is not None:
                    audio = np.frombuffer(msg["bytes"], dtype=np.float32)
                    for fr in model.frames(audio, sr):
                        await ws.send_bytes(fr)
                    await ws.send_text(json.dumps({"type": "chunk_done"}))
        except WebSocketDisconnect:
            pass

    return app


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="callone avatar-server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
