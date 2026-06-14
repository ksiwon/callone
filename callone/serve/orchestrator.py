"""S6 실시간 대화 오케스트레이터 (노트북 온디바이스) — 전화 느낌 우선.

마이크 → VAD(말끝) → ASR → LLM(스트리밍) → 문장단위 TTS → 스피커.
실시간 핵심:
  - 문장 단위 스트리밍: LLM 첫 문장 나오자마자 TTS→스피커 (첫 음성 빨리)
  - barge-in: 클론이 말하는 중 사용자가 말하면 즉시 중단(interrupt)
  - 짧은 응답: OV LLM system 프롬프트 + max_new_tokens 로 전화처럼 1~2문장

LLM: OVPersonaLLM(OpenVINO, Arc GPU) 우선 → 없으면 PersonaLLM 폴백.
TTS: KokoroTTS(화자 A 음색) → 없으면 placeholder.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ..common.io import load_config
from ..common.logging import get_logger
from .asr_stream import StreamASR
from .vad import VAD

log = get_logger("orchestrator")


@dataclass
class Turn:
    user_text: str
    reply_text: str = ""
    first_audio_latency_ms: float = 0.0
    audio_chunks: list = field(default_factory=list)
    interrupted: bool = False


def _pick_llm(speaker: str, serve_cfg: dict):
    """LLM 백엔드 선택 (우선순위):
      1) llama-server(HTTP) — Qwen3.5-4B+LoRA (qwen3_5 는 OV 변환 불가 → llama.cpp).
      2) OVPersonaLLM(OpenVINO/Arc) — base Qwen3-4B int4 (qwen3.5 미지원 시).
      3) PersonaLLM — 순수 폴백.
    serve.yaml 의 llm.backend('llama'|'ov'|'auto'), llm.base_url 로 제어.
    """
    llm_cfg = (serve_cfg or {}).get("llm", {})
    backend = llm_cfg.get("backend", "auto")
    base_url = llm_cfg.get("base_url", "http://127.0.0.1:8080")
    # LoRA 가 화자 A 말투·습관을 이미 내재화 → 일상 대화엔 RAG OFF 가 더 자연스럽다
    # (RAG 키워드 발화 주입이 삼천포 유발). 온도 0.5 로 산만함 억제. 둘 다 serve.yaml 로 조정.
    use_rag = bool(llm_cfg.get("use_rag", False))
    temperature = float(llm_cfg.get("temperature", 0.5))
    max_new = int(llm_cfg.get("max_new_tokens", 80))

    # 1) llama-server (Qwen3.5 + LoRA) — 서버가 떠 있을 때만(probe 실패 시 다음으로)
    if backend in ("llama", "auto"):
        try:
            from .llama_llm import LlamaPersonaLLM

            return LlamaPersonaLLM(speaker, base_url=base_url, use_rag=use_rag,
                                   max_new_tokens=max_new, temperature=temperature,
                                   rag_cfg=llm_cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("llama-server LLM 불가(%s) — 다음 백엔드", e)
            if backend == "llama":
                from .llm_server import PersonaLLM
                return PersonaLLM(speaker)

    # 2) OpenVINO (base Qwen3-4B) — 배포 변환본 → 테스트용 base OV
    if backend in ("ov", "auto"):
        for md in (f"models/llm_ov/{speaker}", "models_ov/qwen3-4b-int4"):
            if Path(md).exists():
                try:
                    from .ov_llm import OVPersonaLLM

                    dev = load_config("llm_phone").get("device", "GPU")
                    return OVPersonaLLM(speaker, md, device=dev,
                                        max_new_tokens=160, temperature=0.7)
                except Exception as e:  # noqa: BLE001
                    log.warning("OV LLM 로드 실패(%s) — 폴백", e)
                    break

    # 3) 폴백
    from .llm_server import PersonaLLM

    return PersonaLLM(speaker)


def _pick_tts(speaker: str, serve_cfg: dict):
    tts_cfg = (serve_cfg or {}).get("tts", {})
    backend = tts_cfg.get("backend", "qwen3-tts")
    # 0) Qwen3-TTS(서버 GPU, 감정 instruct 통합) — 결정서 §2. backend=qwen3-tts 일 때만.
    if backend == "qwen3-tts":
        try:
            from .tts_qwen import QwenTTS

            return QwenTTS(speaker, tts_cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("Qwen3-TTS 불가(%s) — Piper 시도", e)
    # 1) Piper(화자 A 음색 학습본, onnx torch-free)
    try:
        from .tts_piper import PiperTTS

        return PiperTTS(speaker, tts_cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("Piper TTS 불가(%s) — Kokoro 시도", e)
    # 2) Kokoro(제로샷)
    try:
        from .tts_kokoro import KokoroTTS

        return KokoroTTS(speaker, tts_cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("Kokoro TTS 실패(%s) — placeholder", e)
        from .tts_stream import StreamTTS

        return StreamTTS(speaker)


_EMOTIONS = ("happy", "sad", "angry", "neutral", "excited")


def _parse_emotion(text: str, default: str = "neutral") -> tuple[str, str]:
    """LLM 출력에서 감정 키 추출 (§4-1). 반환 (emotion, 정제텍스트).

    지원 형태:
      - JSON: {"emotion":"sad","reply":"어이구..."}
      - 태그: [emotion:happy] 안녕  /  (sad) 안녕
      - 없으면 (default, 원문)
    """
    s = text.strip()
    if s.startswith("{") and '"reply"' in s:
        try:
            obj = json.loads(s)
            emo = str(obj.get("emotion", default)).lower()
            return (emo if emo in _EMOTIONS else default), str(obj.get("reply", "")).strip()
        except json.JSONDecodeError:
            pass
    m = re.match(r"^\s*[\[(]\s*(?:emotion\s*[:=]\s*)?(happy|sad|angry|neutral|excited)\s*[\])]\s*",
                 s, re.IGNORECASE)
    if m:
        return m.group(1).lower(), s[m.end():].strip()
    return default, s


def _tts_stream_emo(tts, text: str, emotion: str):
    """TTS 백엔드가 emotion 인자를 지원하면 넘기고, 아니면 무시(폴백 호환)."""
    try:
        yield from tts.synth_stream(text, emotion=emotion)
    except TypeError:
        yield from tts.synth_stream(text)


class Orchestrator:
    def __init__(self, speaker: str, serve_cfg: dict | None = None):
        cfg = serve_cfg or load_config("serve")
        self.speaker = speaker
        self.vad = VAD(cfg.get("vad", {}))
        self.asr = StreamASR(cfg.get("asr", {}))
        self.llm = _pick_llm(speaker, cfg)
        self.tts = _pick_tts(speaker, cfg)
        self.history: list[dict] = []
        self._interrupt = threading.Event()

    def interrupt(self):
        """barge-in: 진행 중인 응답 중단 (app 이 마이크에서 사용자 발화 감지 시 호출)."""
        self._interrupt.set()

    def set_context(self, persona: str | None = None, situation: str | None = None):
        """통화 페르소나/상황 주입 — 매 통화 시작 시 호출.

        persona:   이 사람이 누구인지(말투/성격/관계). 비면 학습된 페르소나 유지.
        situation: 지금 어떤 상황에서 통화 중인지(배경지식/맥락).
        LLM 백엔드가 set_context 를 지원하면 그대로 전달.
        """
        fn = getattr(self.llm, "set_context", None)
        if callable(fn):
            fn(persona=persona, situation=situation)
        else:
            log.warning("LLM 백엔드가 set_context 미지원 — 페르소나/상황 무시")

    def reset_history(self):
        """새 통화 시작 — 대화 이력 초기화."""
        self.history = []

    def handle_utterance(self, audio: np.ndarray, sr: int = 16000) -> Turn:
        """완결된 발화 → 응답 텍스트 + 음성 청크 (문장 스트리밍, barge-in 존중)."""
        self._interrupt.clear()
        t0 = time.time()
        user_text = self.asr.transcribe(audio, sr)
        turn = Turn(user_text=user_text)
        if not user_text.strip():
            return turn

        first = False
        emotion = "neutral"
        parts = []
        for sentence in self.llm.chat_stream(user_text, self.history):
            if self._interrupt.is_set():
                turn.interrupted = True
                break
            emotion, clean = _parse_emotion(sentence, emotion)   # §4-1 감정 추출
            if not clean:
                continue
            parts.append(clean)
            for chunk in _tts_stream_emo(self.tts, clean, emotion):
                if self._interrupt.is_set():
                    turn.interrupted = True
                    break
                if not first:
                    turn.first_audio_latency_ms = (time.time() - t0) * 1000
                    first = True
                turn.audio_chunks.append(chunk)
            if turn.interrupted:
                break

        turn.reply_text = " ".join(parts)
        if turn.reply_text:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": turn.reply_text})
        log.info("턴: '%s' → '%s' (첫음성 %.0fms%s)", user_text[:30],
                 turn.reply_text[:40], turn.first_audio_latency_ms,
                 ", 중단됨" if turn.interrupted else "")
        return turn

    def stream_audio_chunks(self, audio: np.ndarray, sr: int = 16000) -> Iterator[np.ndarray]:
        yield from self.handle_utterance(audio, sr).audio_chunks

    def stream_turn(self, audio: np.ndarray, sr: int = 16000) -> Iterator[tuple]:
        """이벤트 제너레이터 — app(WS)이 실시간 송출 + barge-in 감지에 사용.
        yield 이벤트: ("user", text) / ("text", sentence) / ("latency", ms) /
                      ("audio", np.ndarray) / ("end", reply) / ("interrupted", None)
        """
        self._interrupt.clear()
        t0 = time.time()
        user_text = self.asr.transcribe(audio, sr)
        yield ("user", user_text)
        if not user_text.strip():
            yield ("end", "")
            return
        first = False
        emotion = "neutral"
        parts: list[str] = []
        for sentence in self.llm.chat_stream(user_text, self.history):
            if self._interrupt.is_set():
                yield ("interrupted", None); break
            emotion, clean = _parse_emotion(sentence, emotion)   # §4-1 감정 추출
            if not clean:
                continue
            parts.append(clean)
            yield ("emotion", emotion)
            yield ("text", clean)
            for chunk in _tts_stream_emo(self.tts, clean, emotion):
                if self._interrupt.is_set():
                    yield ("interrupted", None); break
                if not first:
                    yield ("latency", (time.time() - t0) * 1000); first = True
                yield ("audio", chunk)
            else:
                continue
            break
        reply = " ".join(parts)
        if reply:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
        yield ("end", reply)
