"""오디오 IO + ffmpeg/ffprobe wrapper + 신호 측정 헬퍼.

무거운 torch 없이도 동작(librosa/soundfile). ffmpeg 는 시스템 의존.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .logging import get_logger

log = get_logger("audio")


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def ffprobe(path: str | Path) -> dict:
    """sr/channels/codec/duration 추출. ffprobe 없으면 빈 dict."""
    if shutil.which("ffprobe") is None:
        return {}
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        info = json.loads(out)
    except Exception as e:  # noqa: BLE001
        log.warning("ffprobe 실패 %s: %s", path, e)
        return {}
    astream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = info.get("format", {})
    return {
        "sr": int(astream.get("sample_rate", 0) or 0),
        "channels": int(astream.get("channels", 0) or 0),
        "codec": astream.get("codec_name", ""),
        "duration": float(fmt.get("duration", 0) or 0),
    }


def to_wav16k(src: str | Path, dst: str | Path) -> bool:
    """m4a → 16k mono wav. 라우드니스 정규화 + highpass (§8).

    ffmpeg -i src -ac 1 -ar 16000 -af "loudnorm=I=-23:LRA=7,highpass=f=40" dst
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if not have_ffmpeg():
        log.error("ffmpeg 없음 — 변환 불가. 시스템에 ffmpeg 설치 필요.")
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-af", "loudnorm=I=-23:LRA=7,highpass=f=40",
        str(dst),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg 변환 실패 %s: %s", src, e.stderr[-400:] if e.stderr else e)
        return False


def load_wav(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """모노 float32 로드."""
    import librosa

    y, _sr = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), _sr


def save_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def estimate_snr_db(y: np.ndarray, frame: int = 2048) -> float:
    """간이 SNR 추정: 상위 프레임 에너지 vs 하위(잡음 바닥) 비율(dB).

    정밀하진 않지만 세그먼트 품질 필터/리포트용으로 충분.
    """
    if y.size == 0:
        return 0.0
    n = (y.size // frame) * frame
    if n == 0:
        return 0.0
    frames = y[:n].reshape(-1, frame)
    energy = np.mean(frames ** 2, axis=1) + 1e-12
    e_sorted = np.sort(energy)
    noise = np.mean(e_sorted[: max(1, len(e_sorted) // 10)])
    signal = np.mean(e_sorted[-max(1, len(e_sorted) // 2):])
    return float(10.0 * np.log10(signal / (noise + 1e-12)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
