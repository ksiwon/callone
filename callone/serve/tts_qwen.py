"""Qwen3-TTS 1.7B 백엔드 (서버 GPU) — callone_stack_decision.md §2.

faster_qwen3_tts.**FasterQwen3TTS** 래핑(실 API 검증 2026-06-17, andimarafioti/faster-qwen3-tts).
단일 모델에서 zero-shot 화자복제(ref_audio) + 자연어 instruct 감정 제어를 **동시에** 받는다
(generate_voice_clone_streaming 가 instruct 인자를 직접 지원 — §2-3).

⚠️ 실 API 제약(웹검증 반영):
  - 클래스는 FasterQwen3TTS, 생성은 `FasterQwen3TTS.from_pretrained(model_name, device, dtype)`.
  - **LoRA 어댑터 미지원**(adapter_dir/lora_scale 인자 없음). 화자 충실도는 zero-shot 참조음성
    품질로 결정 → ref_wav(7~10초, 24kHz, 깨끗) 필수. LoRA 충실도가 필요하면 Piper 학습본 사용.
  - generate_voice_clone_streaming(text, language, ref_audio, ref_text, chunk_size, instruct)
    → yield (np.ndarray, sr, timing_dict). language 필수(한국어="Korean").
  - torch+CUDA 필요(CUDA graph). 서버(4090)엔 OpenVINO 없으니 torch OK. 노트북(Arc)은 Piper.

인터페이스(다른 TTS 백엔드와 동일):
  - .sr                     출력 샘플레이트(24000)
  - synth(text, emotion=) -> (np, sr)
  - synth_stream(text, chunk_ms=, emotion=) -> Iterator[np]

미설치(faster_qwen3_tts 없음)/참조없음 → 예외 → orchestrator 가 piper/kokoro/placeholder 폴백.
⚠️ 로컬 가중치만. Alibaba 클라우드 보이스 API 금지(§2).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np

from ..common.io import compute_type_for, load_config, resolve_device
from ..common.logging import get_logger

log = get_logger("tts_qwen")

# §2-3 EMOTION_MAP — LLM 이 판단한 emotion 키 → 자연어 instruct(generate 의 instruct 인자로 주입)
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
        # 모델 경로: 환경변수(CALLONE_TTS_MODEL=로컬 다운본 경로) 우선 → config(HF id).
        # 로컬에 받아둔 가중치를 재다운로드 없이 쓰려면 env 로(머신별 경로라 repo 엔 HF id 유지).
        self.model_id = os.environ.get("CALLONE_TTS_MODEL") or tcfg.get(
            "model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        # language: 한국어 발화 → "Korean"(전체 영문명). "Auto" 도 가능.
        self.language = str(scfg.get("language", tcfg.get("language", "Korean")))
        # 실측 튜닝(2026-06-17): chunk_size 25 + temperature 0.5 = 스트리밍 톤/안정 최적
        # (cs=8 은 끊김, temp=0.9 는 흔들림). 첫청크 ~0.67s(워밍업 후).
        ic = tcfg.get("inference", {})
        self.chunk_size = int(scfg.get("chunk_size", ic.get("chunk_size", 25)))
        self.temperature = float(scfg.get("temperature", ic.get("temperature", 0.5)))
        self.device = str(scfg.get("device", tcfg.get("device", "cuda")))
        self.dtype_name = str(tcfg.get("dtype", "bfloat16"))
        self.ref_asr_model = str(tcfg.get("ref_transcribe_model", "small"))
        self.emotion_map = tcfg.get("emotion_instruct", EMOTION_MAP)
        # 참조 음색(zero-shot 클론의 핵심): serve.tts.ref_wav 또는 화자 ref 파일
        self.ref_wav = scfg.get("ref_wav") or self._default_ref()
        if not self.ref_wav:
            raise RuntimeError(
                f"Qwen3-TTS zero-shot 클론엔 참조 WAV 필수(화자={speaker}). "
                f"data/speakers/{speaker}/ref_24k.wav 두거나 serve.yaml tts.ref_wav 지정. "
                f"미지정 시 orchestrator 가 Piper 폴백.")
        # ref_text(ICL 모드 필수): config → ref_text.txt(권장) → 자동전사 폴백.
        self.ref_text = scfg.get("ref_text") or self._resolve_ref_text()
        # ⚠️ faster_qwen3_tts 는 LoRA 미지원 — 학습 어댑터가 있어도 서빙엔 못 붙음(Piper 사용 권장).
        adapter = Path(tcfg.get("output_dir", "models/tts_server")) / speaker
        if adapter.exists():
            log.warning("화자 LoRA(%s) 존재하나 faster_qwen3_tts 는 어댑터 미지원 → zero-shot 참조로만 합성. "
                        "LoRA 충실도가 필요하면 TTS 백엔드를 Piper 로.", adapter)
        self._tts = self._load()

    def _default_ref(self) -> str:
        for cand in (Path("data/speakers") / self.speaker / "ref_24k.wav",
                     Path("data/speakers") / self.speaker / "ref_16k.wav"):
            if cand.exists():
                return str(cand)
        return ""

    def _resolve_ref_text(self) -> str:
        """ICL 모드 필수 ref_text 확보(품질 큰 영향).
        우선순위: ref_24k.wav 옆 ref_text.txt(정확·모델로드 없음, 권장) → faster-whisper 자동전사 폴백.
        """
        side = Path(self.ref_wav).with_name("ref_text.txt")
        if side.exists():
            t = side.read_text(encoding="utf-8").strip()
            if t:
                log.info("ref_text: %s 사용", side)
                return t
        # 폴백: 참조 클립 자동 전사. 정확도 위해선 ref_text.txt 두기를 권장(작은 모델은 오인식 가능).
        try:
            from faster_whisper import WhisperModel

            log.warning("ref_text 미지정 → faster-whisper(%s) 자동 전사. 품질 위해 %s 권장.",
                        self.ref_asr_model, side)
            m = WhisperModel(self.ref_asr_model, device=resolve_device(),
                             compute_type=compute_type_for())
            segs, _ = m.transcribe(self.ref_wav, language="ko")
            t = "".join(s.text for s in segs).strip()
            del m
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            log.info("ref_text 자동전사: %s", t[:40])
            return t
        except Exception as e:  # noqa: BLE001
            log.warning("ref_text 자동전사 실패(%s) — 빈 ref_text(품질 저하 가능)", e)
            return ""

    def _load(self):
        try:
            import torch  # CUDA graph 백엔드라 torch 필수
            from faster_qwen3_tts import FasterQwen3TTS  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"faster_qwen3_tts/torch 미설치({e}). pip install faster-qwen3-tts "
                f"(torch>=2.5.1 + NVIDIA CUDA). 미설치 시 orchestrator 가 piper/kokoro 폴백.") from e
        dtype = getattr(torch, self.dtype_name, torch.bfloat16)
        tts = FasterQwen3TTS.from_pretrained(self.model_id, device=self.device, dtype=dtype)
        log.info("Qwen3-TTS 로드: %s (device=%s, dtype=%s, sr=%d, speaker=%s, ref=%s, lang=%s)",
                 self.model_id, self.device, self.dtype_name, self.sr, self.speaker,
                 self.ref_wav, self.language)
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
            for audio_chunk, sr, _timing in self._tts.generate_voice_clone_streaming(
                    text=text, language=self.language,
                    ref_audio=self.ref_wav, ref_text=self.ref_text,
                    chunk_size=self.chunk_size, instruct=instruct,
                    temperature=self.temperature):
                self.sr = int(sr)  # 모델이 알려주는 실제 sr 로 갱신
                yield np.asarray(audio_chunk, dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            log.warning("Qwen3-TTS 스트림 오류(%s)", e)

    def synth(self, text: str, emotion: str | None = None) -> tuple[np.ndarray, int]:
        chunks = list(self.synth_stream(text, emotion=emotion))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sr
        return np.concatenate(chunks), self.sr
