"""S6 LLM 추론 서버 (§16).

티어 자동 선택(common/hardware):
  - server_gpu(H100) → llama.cpp/vLLM(EXAONE-3.5-7.8B + LoRA)  [llm_server.yaml]
  - laptop_cpu → OpenVINO/llama.cpp(EXAONE-3.5-2.4B + LoRA)  [llm_laptop.yaml]
페르소나 프롬프트 + RAG + 메모리 결합. 미설치 시 규칙 기반 폴백 응답(지연 측정용).
"""
from __future__ import annotations

from typing import Iterator

from ..common.hardware import detect_tier, tier_defaults
from ..common.io import load_config
from ..common.logging import get_logger
from ..llm.persona_prompt import build_messages
from ..llm.rag import UtteranceRAG

log = get_logger("llm_server")


class PersonaLLM:
    def __init__(self, speaker: str, cfg: dict | None = None):
        self.speaker = speaker
        if cfg is None:
            # 하드웨어 티어에 맞는 LLM config 자동 선택
            tier = detect_tier()
            llm_cfg_name = tier_defaults(tier)["llm_config"]   # llm_server | llm_laptop
            log.info("LLM 티어=%s → config=%s", tier, llm_cfg_name)
            cfg = load_config(llm_cfg_name)
        self.cfg = cfg
        self.gen = self.cfg.get("generation", {})
        self._persona_override: str | None = None   # 통화 시 주입(이 사람은 누구)
        self._situation: str | None = None          # 통화 시 주입(지금 상황)
        self._llm = self._try_load()
        self._rag = None
        try:
            self._rag = UtteranceRAG(speaker, load_config("serve").get("rag", {}))
        except Exception:  # noqa: BLE001
            pass

    def _try_load(self):
        try:
            from vllm import LLM, SamplingParams  # type: ignore  # noqa: F401

            base = self.cfg.get("base_model")
            log.info("vLLM 로드 base=%s (+LoRA %s)", base, self.speaker)
            # 실제 vLLM LLM(...) 인스턴스 + LoRA 어댑터 로드 지점
            return None  # repo/모델 준비 시 연결
        except Exception as e:  # noqa: BLE001
            log.warning("vLLM 미설치/모델없음(%s) — 폴백 응답", e)
            return None

    def set_context(self, persona: str | None = None, situation: str | None = None):
        """통화 페르소나/상황 주입(orchestrator.set_context 가 호출)."""
        self._persona_override = persona
        self._situation = situation

    def _rag_ctx(self, user_text: str) -> str | None:
        if self._rag:
            try:
                return self._rag.context(user_text, k=3)
            except Exception:  # noqa: BLE001
                return None
        return None

    def chat(self, user_text: str, history: list[dict] | None = None) -> str:
        msgs = build_messages(self.speaker, user_text, history,
                              rag_context=self._rag_ctx(user_text),
                              persona_override=self._persona_override,
                              situation=self._situation)
        if self._llm is not None:
            raise NotImplementedError("vLLM generate 연결 지점")
        return self._fallback(user_text)

    def chat_stream(self, user_text: str, history: list[dict] | None = None) -> Iterator[str]:
        """문장단위 토큰 스트리밍 (TTS 청크 트리거)."""
        text = self.chat(user_text, history)
        for sent in _split_sentences(text):
            yield sent

    def _fallback(self, user_text: str) -> str:
        """모델 미준비 시 페르소나 톤 흉내 폴백 (실제 클론 아님)."""
        ctx = self._rag_ctx(user_text)
        if ctx:
            first = ctx.splitlines()[0].lstrip("- ")
            return first or "응, 그래."
        return "응, 그래. (모델 준비 전 폴백 응답)"


def _split_sentences(text: str):
    import re

    parts = re.split(r"(?<=[.?!~])\s+", text.strip())
    return [p for p in parts if p]
