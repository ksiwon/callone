"""Qwen3-TTS 1.7B 백엔드 (서버 GPU) — callone_stack_decision.md §2.

faster_qwen3_tts.QwenTTS 래핑. 단일 모델에서 zero-shot 화자복제 + 자연어 instruct
감정 제어를 **동시에** 받는다(§2-3). 화자 LoRA(models/tts_server/{spk}) 있으면 적용.

인터페이스(다른 TTS 백엔드와 동일):
  - .sr                     출력 샘플레이트(24000)
  - synth(text) -> (np, sr)
  - synth_stream(text, chunk_ms=200, emotion=None) -> Iterator[np]

미설치(faster_qwen3_tts 없음)/참조없음 → 예외 → orchestrator 가 piper/kokoro/placeholder 폴백.
⚠️ 로컬 가중치만. Alibaba 클라우드 보이스 API 금지(§2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..common.io import load_config
from ..common.logging import get_logger

log = get_logger("tts_qwen")

# §2-3 EMOTION_MAP — LLM 이 판단한 emotion 키 → 자연어 instruct
EMOTION_MAP = {
    "happy":   "Speak with a bright, cheerful, and warm tone.",
    "sad":     "Speak with a low, slow, comforting, and sad tone.",
    "angry":   "Speak with an annoyed, sharp, and frustrated voice.",
    "neutral": "Speak in a natural, relaxed, conversational tone.",
    "excited": "Speak with high energy and an excited, upbeat tone.",
}


class QwenTTS:
    def __init__(self, speaker: str, cfg: dict | None = None):
        self.speaker = speaker
        scfg = cfg or {}
        tcfg = load_config("tts_server")
        self.sr = int(tcfg.get("sample_rate", 24000))
        self.model_id = tcfg.get("model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        self.chunk_size = int(scfg.get("chunk_size", tcfg.get("inference", {}).get("chunk_size", 8)))
        self.lora_scale = float(scfg.get("lora_scale",
                                         tcfg.get("inference", {}).get("lora_scale", 0.3)))
        self.emotion_map = tcfg.get("emotion_instruct", EMOTION_MAP)
        # 참조 음색(zero-shot): serve.tts.ref_wav/ref_text 또는 화자 ref 파일
        self.ref_wav = scfg.get("ref_wav") or self._default_ref()
        self.ref_text = scfg.get("ref_text", "")
        # 화자 LoRA 어댑터(학습본)
        self.adapter_dir = Path(tcfg.get("output_dir", "models/tts_server")) / speaker
        self._tts = self._load()

    def _default_ref(self) -> str:
        for cand in (Path("data/speakers") / self.speaker / "ref_24k.wav",
                     Path("data/speakers") / self.speaker / "ref_16k.wav"):
            if cand.exists():
                return str(cand)
        return ""

    def _load(self):
        try:
            from faster_qwen3_tts import QwenTTS as _Q  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"faster_qwen3_tts 미설치({e}). pip install faster-qwen3-tts "
                f"(GPU 노드). 미설치 시 orchestrator 가 piper/kokoro 폴백.") from e
        kwargs = {"model_name": self.model_id}
        if self.adapter_dir.exists():
            kwargs["adapter_dir"] = str(self.adapter_dir)
            kwargs["lora_scale"] = self.lora_scale
            log.info("Qwen3-TTS LoRA 적용: %s (scale=%.2f)", self.adapter_dir, self.lora_scale)
        tts = _Q(**kwargs)
        log.info("Qwen3-TTS 로드: %s (sr=%d, speaker=%s, ref=%s)",
                 self.model_id, self.sr, self.speaker, self.ref_wav or "(없음)")
        return tts

    def _instruct(self, emotion: str | None) -> str:
        return self.emotion_map.get(emotion or "neutral", self.emotion_map.get("neutral", ""))

    # ----- 스트리밍 합성(감정 instruct 동적 주입) -------------------------
    def synth_stream(self, text: str, chunk_ms: int = 200,
                     emotion: str | None = None) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        instruct = self._instruct(emotion)
        try:
            for chunk in self._tts.generate_voice_clone_streaming(
                    text=text, ref_audio_path=self.ref_wav, ref_text=self.ref_text,
                    instruct_text=instruct, chunk_size=self.chunk_size):
                yield np.asarray(chunk, dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            log.warning("Qwen3-TTS 스트림 오류(%s)", e)

    def synth(self, text: str, emotion: str | None = None) -> tuple[np.ndarray, int]:
        chunks = list(self.synth_stream(text, emotion=emotion))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sr
        return np.concatenate(chunks), self.sr
