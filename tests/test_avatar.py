"""test_avatar — 토킹헤드 폴백 경로(모델/GPU 없이). StaticImage 가 사진을 프레임으로 emit + 폴백 체인."""
import numpy as np
import pytest

from callone.serve.avatar import DittoAvatar, StaticImageAvatar, _pick_avatar


def test_pick_avatar_static_backend():
    av = _pick_avatar({"avatar": {"backend": "static", "fps": 25}})
    assert isinstance(av, StaticImageAvatar)


def test_pick_avatar_auto_falls_back_when_no_server():
    # avatar-server(8091) 없음 → DittoAvatar probe 실패 → StaticImage 폴백
    av = _pick_avatar({"avatar": {"backend": "auto", "base_url": "http://127.0.0.1:8099"}})
    assert isinstance(av, StaticImageAvatar)


def test_static_emits_frames_for_duration(tmp_path):
    img = tmp_path / "portrait.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0dummyjpeg")   # 더미(Pillow 깨지면 원본 바이트 폴백)
    av = StaticImageAvatar({"fps": 10, "resolution": 64})
    av.start_call(str(img))
    frames = list(av.frames_for(np.zeros(16000, dtype=np.float32), 16000))   # 1초 @16k
    assert len(frames) == 10                          # 1초 * 10fps
    assert all(isinstance(f, (bytes, bytearray)) and len(f) > 0 for f in frames)


def test_static_no_image_raises():
    with pytest.raises(RuntimeError):
        StaticImageAvatar({}).start_call("")


def test_ditto_probe_fails_without_server():
    with pytest.raises(RuntimeError):
        DittoAvatar({"base_url": "http://127.0.0.1:8099"}, probe=True)
