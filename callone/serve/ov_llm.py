"""OpenVINO GenAI LLM 백엔드 (노트북 온디바이스) — Qwen3.5-4B int4, Arc iGPU.

실측(갤럭시북5 Pro): 4B int4 GPU 29.9 tok/s, 첫음성 ~1.5초 = 실시간.
페르소나 카드(system) + RAG(화자 A 실제 발화) 주입 + 문장단위 스트리밍(TTS 트리거).

모델 경로: OpenVINO IR 디렉토리 (scripts/merge_to_ov.py 로 LoRA 병합+변환한 것,
또는 테스트용 base OV 모델 models_ov/qwen3-4b-int4).
"""
from __future__ import annotations

import queue
import re
import threading
from pathlib import Path
from typing import Iterator

from ..common.logging import get_logger
from ..llm.persona_prompt import load_persona

log = get_logger("ov_llm")

_SENT_END = re.compile(r"(?<=[.?!~…])\s|(?<=[다요네까나])\s")  # 한국어 문장 경계 근사


class OVPersonaLLM:
    """OpenVINO 로 도는 페르소나 LLM. chat()/chat_stream()."""

    def __init__(self, speaker: str, model_dir: str, device: str = "GPU",
                 max_new_tokens: int = 160, temperature: float = 0.7,
                 use_rag: bool = True):
        # RAG 는 키워드(utterances.json, pandas 미사용) → OV 와 segfault 없음.
        # 화자 A 실제 발화 검색("예전에 ~했잖아")으로 기억 반영.
        import openvino_genai as ov_genai  # 지연 import

        self.speaker = speaker
        self.device = device
        self._genai = ov_genai
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"OV 모델 없음: {model_dir} (merge_to_ov.py 로 생성)")
        log.info("OpenVINO LLM 로드: %s (%s)", model_dir, device)
        self.pipe = ov_genai.LLMPipeline(model_dir, device)
        # 정식 채팅 템플릿(enable_thinking=False)용 HF 토크나이저
        self._tok = None
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("HF 토크나이저 없음(%s) — 수동 ChatML 폴백", e)
        self.cfg = ov_genai.GenerationConfig()
        self.cfg.max_new_tokens = max_new_tokens
        self.cfg.temperature = temperature
        self.cfg.do_sample = temperature > 0
        self.persona = load_persona(speaker)
        self._rag = None
        if use_rag:
            try:
                from ..llm.rag import UtteranceRAG

                self._rag = UtteranceRAG(speaker)
            except Exception as e:  # noqa: BLE001
                log.warning("RAG 비활성(%s)", e)

    # ----- 프롬프트(ChatML, Qwen 형식) ------------------------------------
    def _system(self, user_text: str) -> str:
        sys = self.persona
        if self._rag:
            try:
                ctx = self._rag.context(user_text, k=3)
                if ctx:
                    sys += f"\n\n[참고할 실제 발화]\n{ctx}"
            except Exception:  # noqa: BLE001
                pass
        # 실시간 전화 느낌: 짧게
        sys += "\n전화 통화처럼 1~2문장으로 짧고 자연스럽게 답한다."
        return sys

    def _build_prompt(self, user_text: str, history: list[dict] | None) -> str:
        msgs = [{"role": "system", "content": self._system(user_text)}]
        msgs += list(history or [])[-8:]
        msgs.append({"role": "user", "content": user_text})
        if self._tok is not None:
            try:
                return self._tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:  # enable_thinking 미지원 토크나이저
                return self._tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
        # 폴백: 수동 ChatML (thinking off)
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in msgs]
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "\n".join(parts)

    # ----- 생성 -----------------------------------------------------------
    @staticmethod
    def _strip_think(text: str) -> str:
        # thinking 블록 제거. 닫힌 </think> 있으면 그 뒤(실제 답)만, 닫힌 쌍은 통째 제거.
        if "</think>" in text:
            text = text.split("</think>")[-1]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.replace("<think>", "").strip()

    def chat(self, user_text: str, history: list[dict] | None = None) -> str:
        prompt = self._build_prompt(user_text, history)
        out = self.pipe.generate(prompt, self.cfg)
        return self._strip_think(str(out))

    def chat_stream(self, user_text: str, history: list[dict] | None = None) -> Iterator[str]:
        """문장 단위 스트리밍 — 생성 스레드 + 큐. 첫 문장 즉시 TTS 가능."""
        prompt = self._build_prompt(user_text, history)
        q: queue.Queue = queue.Queue()
        DONE = object()

        def _streamer(subword: str):
            q.put(subword)
            return False  # 계속(중단 안 함)

        def _run():
            try:
                self.pipe.generate(prompt, self.cfg, _streamer)
            except Exception as e:  # noqa: BLE001
                log.warning("OV 생성 오류: %s", e)
            finally:
                q.put(DONE)

        threading.Thread(target=_run, daemon=True).start()

        buf = ""
        while True:
            item = q.get()
            if item is DONE:
                break
            buf += item
            # 문장 경계에서 잘라 yield
            while True:
                m = _SENT_END.search(buf)
                if not m:
                    break
                sent, buf = buf[:m.start() + 1].strip(), buf[m.end():]
                if sent:
                    yield sent
        if buf.strip():
            yield buf.strip()
