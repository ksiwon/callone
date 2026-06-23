"""test_avatar_server — avatar-server(static) ↔ DittoAvatar 클라이언트 전체 WS 파이프라인(GPU 없이).

Ditto를 붙이기 전에 **HTTP/WS 배관 자체**가 도는지 검증한다(설계서 §3 통합).
실제 uvicorn 서버를 스레드로 띄우고, callone 쪽 DittoAvatar 로 사진 등록→오디오청크→프레임 수신.
"""
import io
import socket
import threading
import time
import urllib.request

import numpy as np
import pytest

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")
pytest.importorskip("websocket")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _dummy_jpeg() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (128, 128), (120, 120, 120)).save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return b"\xff\xd8\xff\xe0dummy"


@pytest.fixture
def server(tmp_path):
    from avatar_server.app import create_app

    port = _free_port()
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    # /health 뜰 때까지 대기
    ok = False
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:  # noqa: BLE001
            time.sleep(0.1)
    assert ok, "avatar-server 안 뜸"
    yield port
    srv.should_exit = True
    th.join(timeout=5)


def test_e2e_static_pipeline(server, tmp_path):
    from callone.serve.avatar import DittoAvatar

    img = tmp_path / "portrait.jpg"
    img.write_bytes(_dummy_jpeg())

    av = DittoAvatar({"base_url": f"http://127.0.0.1:{server}", "fps": 10, "sr": 24000})
    av.start_call(str(img))
    assert av.session_id

    audio = np.zeros(24000, dtype=np.float32)          # 1초 @24k
    frames = list(av.frames_for(audio, 24000))
    av.stop()

    assert len(frames) == 10                           # 1초 * 10fps
    assert all(isinstance(f, bytes) and len(f) > 0 for f in frames)
