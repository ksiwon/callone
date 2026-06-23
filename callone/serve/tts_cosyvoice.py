"""CosyVoice3 제로샷 클론 TTS — **별 프로세스(cosyvoice_server :8092)** HTTP 클라이언트.

왜 별 프로세스?  CosyVoice3 는 전용 conda env(python3.10) + torch2.6 + Matcha-TTS PYTHONPATH +
9.75GB 모델이라 callone .venv-serve 에 합치면 의존성 충돌. → llama-server/avatar-server 와
동일하게 **독립 서비스로 띄우고 HTTP 로 호출**(serve venv 엔 torch/cosyvoice 전혀 없음).

왜 CosyVoice3?  실측(사용자): Qwen3-TTS 제로샷은 턴마다 음색이 튈 때가 있는데 CosyVoice3-0.5B
는 안정적(원본 충실). 우선순위①(목소리 유사도) → 이 백엔드 채택.

인터페이스는 QwenTTS 와 동일(orchestrator 가 그대로 호출):
  .sr / set_reference(audio,sr,ref_text) / synth_stream(text,emotion=) / synth / cleanup_reference

프라이버시: 레퍼런스는 **인메모리(ndarray→b64)** 로만 들고 매 합성 시 서버로 전송(서버도 디스크에
영속 안 함). 통화 끝나면 cleanup_reference 로 폐기. 디스크 파일/로그 본문 0.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import urllib.error
import urllib.request
from typing import Iterator

import numpy as np

from ..common.io import compute_type_for, load_config, resolve_device
from ..common.logging import get_logger

log = get_logger("tts_cosyvoice")


def _read_exactly(resp, n: int) -> bytes:
    """HTTP 스트림 본문에서 정확히 n바이트 모아 읽음(소켓 부분수신 대비). EOF 면 모인 만큼 반환."""
    buf = b""
    while len(buf) < n:
        chunk = resp.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


class CosyVoiceTTS:
    def __init__(self, speaker: str, cfg: dict | None = None):
        self.speaker = speaker
        scfg = cfg or {}
        self.sr = 24000                       # 서버가 24k 로 리샘플해 반환(프론트 재생 24k 와 일치)
        self.base_url = str(scfg.get("cosyvoice_url")
                            or os.environ.get("COSYVOICE_URL", "http://127.0.0.1:8092")).rstrip("/")
        self.language = str(scfg.get("language", "Korean"))
        # stream=True → 서버 /synth_stream(네이티브 bistream, 첫음성 ~150ms). false → /synth(통짜, 기준선).
        #   음색은 동일 연속 생성이나 우선순위①(목소리)이라 저장 wav A/B 통과 전엔 false 권장.
        self.stream = bool(scfg.get("stream", True))
        self.timeout = float(scfg.get("cosyvoice_timeout", 60.0))
        self.ref_asr_model = str(scfg.get("ref_transcribe_model", "large-v3-turbo"))
        # 인메모리 레퍼런스(디스크 0). ref_wav 는 orchestrator 워밍업 게이트용 truthy 마커.
        self._ref_b64: str | None = None
        self._ref_sr: int = 16000
        self.ref_text: str = ""
        self.ref_wav: str = ""                # set_reference 후 "memory" 로 채워짐(존재 신호)
        self._cur_text: str = ""              # synth_stream 진입 시 세팅(_payload 공용)
        self._probe()
        log.info("CosyVoice3 TTS: %s (speaker=%s, lang=%s)", self.base_url, speaker, self.language)

    def _probe(self):
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                if r.status >= 400:
                    raise RuntimeError(f"health {r.status}")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"cosyvoice_server 응답 없음({self.base_url}). 먼저 띄워라: "
                f"bash scripts/setup_cosyvoice_gpu.sh (최초) / conda run -n cosyvoice "
                f"python cosyvoice_server/app.py: {e}") from e

    # ----- 통화 세션 ref(인메모리, 프라이버시) -----------------------------
    def set_reference(self, audio: np.ndarray, sr: int, ref_text: str | None = None) -> None:
        """프론트 음성으로 ref 교체 — RAM 에만. CosyVoice 는 16k ref 선호 → 16k 리샘플 후 b64."""
        a = np.asarray(audio, dtype=np.float32).flatten()
        a16 = a if sr == 16000 else self._resample(a, sr, 16000)
        self._ref_b64 = base64.b64encode(a16.astype(np.float32).tobytes()).decode()
        self._ref_sr = 16000
        self.ref_text = (ref_text.strip() if ref_text and ref_text.strip()
                         else self._transcribe(a16, 16000))
        self.ref_wav = "memory"
        log.info("세션 ref 설정(인메모리, %.1fs, ref_text %d자)", len(a) / max(1, sr), len(self.ref_text))

    def cleanup_reference(self) -> None:
        self._ref_b64 = None
        self.ref_text = ""
        self.ref_wav = ""

    @staticmethod
    def _resample(a: np.ndarray, src: int, dst: int) -> np.ndarray:
        if src == dst or len(a) == 0:
            return a
        n = int(round(len(a) * dst / src))
        return np.interp(np.linspace(0, len(a), n, endpoint=False),
                         np.arange(len(a)), a).astype(np.float32)

    def _transcribe(self, audio: np.ndarray, sr: int) -> str:
        """ref_text 미지정 시 인메모리 전사(ICL 유사도에 중요). 실패 시 빈 문자열."""
        try:
            from faster_whisper import WhisperModel

            m = WhisperModel(self.ref_asr_model, device=resolve_device(),
                             compute_type=compute_type_for())
            segs, _ = m.transcribe(audio, language="ko")
            t = "".join(s.text for s in segs).strip()
            del m
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            return t
        except Exception as e:  # noqa: BLE001
            log.warning("ref 전사 실패(%s) — 빈 ref_text(품질 저하 가능)", e)
            return ""

    def _payload(self) -> bytes:
        return json.dumps({
            "text": self._cur_text,
            "ref_audio_b64": self._ref_b64,
            "ref_sr": self._ref_sr,
            "prompt_text": self.ref_text,
            "language": self.language,
        }).encode()

    # ----- 합성: stream=True 면 /synth_stream(나오는 대로), 아니면 /synth(통짜) -----
    def synth_stream(self, text: str, chunk_ms: int = 200,
                     emotion: str | None = None) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        if not self._ref_b64:
            log.warning("CosyVoice3: 레퍼런스 없음 — 합성 생략")
            return
        self._cur_text = text.strip()
        if self.stream:
            yield from self._synth_stream_live()
        else:
            yield from self._synth_whole(chunk_ms)

    def _synth_stream_live(self) -> Iterator[np.ndarray]:
        """서버 /synth_stream 의 [4바이트 LE 길이][f32 PCM] 프레임을 나오는 대로 yield(첫음성 ~150ms)."""
        req = urllib.request.Request(
            f"{self.base_url}/synth_stream", data=self._payload(),
            headers={"Content-Type": "application/json"})
        r = None
        try:
            r = urllib.request.urlopen(req, timeout=self.timeout)
            out_sr = int(r.headers.get("X-Sample-Rate", self.sr))
            while True:
                head = _read_exactly(r, 4)
                if not head:
                    break
                (n,) = struct.unpack("<I", head)
                body = _read_exactly(r, n)
                if len(body) < n:
                    break
                audio = np.frombuffer(body, dtype=np.float32)
                if out_sr != self.sr:
                    audio = self._resample(audio, out_sr, self.sr)
                yield audio
        except urllib.error.HTTPError as e:
            log.warning("CosyVoice3 stream HTTP %s: %s", e.code, e.read().decode("utf-8", "ignore")[:200])
        except Exception as e:  # noqa: BLE001
            log.warning("CosyVoice3 stream 오류(%s)", e)
        finally:
            # barge-in 으로 제너레이터가 중간에 버려지면(GeneratorExit) 여기로 와 소켓 확실히 닫음(누수 방지).
            if r is not None:
                try:
                    r.close()
                except Exception:  # noqa: BLE001
                    pass

    def _synth_whole(self, chunk_ms: int = 200) -> Iterator[np.ndarray]:
        """통짜 합성(/synth, stream=False) — A/B 기준선·폴백. 받은 전체를 chunk_ms 단위로 흘림."""
        req = urllib.request.Request(
            f"{self.base_url}/synth", data=self._payload(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                out_sr = int(r.headers.get("X-Sample-Rate", self.sr))
        except urllib.error.HTTPError as e:
            log.warning("CosyVoice3 HTTP %s: %s", e.code, e.read().decode("utf-8", "ignore")[:200])
            return
        except Exception as e:  # noqa: BLE001
            log.warning("CosyVoice3 합성 오류(%s)", e)
            return
        audio = np.frombuffer(raw, dtype=np.float32)
        if out_sr != self.sr:
            audio = self._resample(audio, out_sr, self.sr)
        step = max(1, int(self.sr * chunk_ms / 1000))
        for i in range(0, len(audio), step):
            yield audio[i:i + step]

    def synth(self, text: str, emotion: str | None = None) -> tuple[np.ndarray, int]:
        chunks = list(self.synth_stream(text, emotion=emotion))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sr
        return np.concatenate(chunks), self.sr
