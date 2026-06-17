"""llama.cpp(llama-server) LLM 백엔드 — Qwen3.5-4B + LoRA, 노트북 온디바이스.

OpenVINO 는 qwen3_5 아키텍처(Gated Delta Net+MoE+MTP) 변환을 아직 못 한다
(optimum-intel #1628). 반면 llama.cpp 는 Qwen3.5 를 지원하고 LoRA 도 적용 가능.
그래서 **llama-server 를 별도 프로세스로 띄우고 HTTP(OpenAI 호환)로 호출**한다.

장점:
  - 서빙 파이썬엔 torch/OpenVINO 가 전혀 없음 → segfault 원천 차단(HTTP 클라이언트뿐).
  - 페르소나 카드(system) + RAG(화자 A 실제 발화) + 문장단위 스트리밍 그대로 재사용.
  - GPU 백엔드(Vulkan coopmat-off / SYCL / IPEX) 든 CPU 든 **같은 HTTP 코드**.

서버 띄우는 법(노트북, 별도 터미널) — scripts/run_llama_server.md 참고:
  llama-server -m qwen3.5-4b-Q4_K_M.gguf --lora mom-lora-f16.gguf \
      -ngl 99 --host 127.0.0.1 --port 8080 -c 4096
(Arc Vulkan TDR 회피: 환경변수 GGML_VK_DISABLE_COOPMAT=1)

인터페이스는 OVPersonaLLM 과 동일: chat() / chat_stream().
"""
from __future__ import annotations

import json
import re
from typing import Iterator

from ..common.logging import get_logger
from ..llm.persona_prompt import load_persona

log = get_logger("llama_llm")

_SENT_END = re.compile(r"(?<=[.?!~…])\s|(?<=[다요네까나])\s")  # 한국어 문장 경계 근사


class LlamaPersonaLLM:
    """llama-server(HTTP, OpenAI 호환)에 붙는 페르소나 LLM."""

    def __init__(self, speaker: str, base_url: str = "http://127.0.0.1:8080",
                 max_new_tokens: int = 80, temperature: float = 0.7,
                 use_rag: bool = True, timeout: float = 60.0,
                 probe: bool = True, rag_cfg: dict | None = None,
                 max_history: int = 24):
        self.speaker = speaker
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/v1/chat/completions"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # 통화 맥락 누적: 최근 N개 메시지(=N/2턴)를 LLM 에 전달. 시스템 프롬프트가
        # 고정(RAG off)이면 llama.cpp prefix 캐시가 먹어 길어도 추가비용 적음.
        self.max_history = int((rag_cfg or {}).get("max_history", max_history))
        self.timeout = timeout
        self.persona = load_persona(speaker)
        self._persona_override: str | None = None   # 통화 시 주입(이 사람은 누구)
        self._situation: str | None = None          # 통화 시 주입(지금 상황)
        self._emotion_labels = False                 # True 면 응답 앞에 [emotion:..] 출력 지시
        self._rag = None
        if use_rag:
            try:
                from ..llm.rag import UtteranceRAG

                self._rag = UtteranceRAG(speaker, cfg=rag_cfg or {})
            except Exception as e:  # noqa: BLE001
                log.warning("RAG 비활성(%s)", e)
        if probe:
            self._probe()  # 서버 없으면 즉시 예외 → orchestrator 가 폴백
        log.info("llama-server LLM: %s (speaker=%s)", self.base_url, speaker)

    # ----- 서버 헬스체크 --------------------------------------------------
    def _probe(self):
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=3) as r:
                if r.status >= 400:
                    raise RuntimeError(f"health {r.status}")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"llama-server 응답 없음({self.base_url}). "
                f"먼저 llama-server 를 띄워라(scripts/run_llama_server.md): {e}") from e

    # ----- 컨텍스트 주입(통화 페르소나/상황) ------------------------------
    def set_context(self, persona: str | None = None, situation: str | None = None):
        self._persona_override = persona
        self._situation = situation

    def set_emotion_labeling(self, on: bool = True):
        """감정 라벨 출력 on/off. orchestrator 가 TTS 감정 활성 시 켠다.
        켜지면 응답 맨 앞 [emotion:..] → _parse_emotion 이 추출·제거 → TTS instruct 동적 주입.
        """
        self._emotion_labels = bool(on)

    # ----- 메시지 조립 ----------------------------------------------------
    def _system(self, user_text: str) -> str:
        # persona_override 있으면 학습 페르소나 대체(통화 시 "이 사람은 누구").
        sys = (self._persona_override.strip()
               if (self._persona_override and self._persona_override.strip())
               else self.persona)
        if self._situation and self._situation.strip():
            sys += ("\n\n[지금 통화 상황 — 이 맥락에서 대화한다]\n"
                    + self._situation.strip())
        if self._rag:
            try:
                ctx = self._rag.context(user_text, k=3)
                if ctx:   # 게이트(rag): 관련 기억 있을 때만 채워짐 → 평소엔 빈 값이라 주입 안 됨
                    sys += ("\n\n[참고할 기억 — 지금 질문과 자연스럽게 이어질 때만 활용하고, "
                            "관련 없으면 무시하고 평소처럼 대화한다]\n" + ctx)
            except Exception:  # noqa: BLE001
                pass
        sys += "\n전화 통화처럼 1~2문장으로 짧고 자연스럽게 답한다."
        if self._emotion_labels:
            # 응답 맨 앞에 감정 태그 1개. _parse_emotion 이 추출·제거 → TTS 톤 동적 변화.
            sys += ("\n응답 맨 앞에 지금 감정을 [emotion:happy|sad|angry|neutral|excited] "
                    "중 하나로 딱 붙여라(은은하게). 예: '[emotion:happy] 어, 왔나!'")
        return sys

    def _messages(self, user_text: str, history: list[dict] | None) -> list[dict]:
        msgs = [{"role": "system", "content": self._system(user_text)}]
        msgs += list(history or [])[-self.max_history:]   # 통화 맥락 누적(최근 max_history 개)
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def _payload(self, user_text: str, history: list[dict] | None, stream: bool) -> dict:
        return {
            "messages": self._messages(user_text, history),
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": stream,
            # 반복 루프 방지("뭐. 뭐. 뭐." 등). LoRA 가 화자 A filler 물고 늘어지는 것 억제.
            "repeat_penalty": 1.3,
            "repeat_last_n": 64,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            # Qwen3.5 thinking 끄기(짧고 빠른 전화 응답). 서버가 무시해도 _strip_think 가 처리.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    @staticmethod
    def _strip_think(text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.replace("<think>", "").strip()

    # ----- 생성(비스트리밍) ----------------------------------------------
    def chat(self, user_text: str, history: list[dict] | None = None) -> str:
        import urllib.request

        data = json.dumps(self._payload(user_text, history, False)).encode()
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            obj = json.loads(r.read().decode())
        text = obj["choices"][0]["message"]["content"]
        return self._strip_think(text)

    # ----- 생성(SSE 스트리밍, 문장 단위 yield) ---------------------------
    def chat_stream(self, user_text: str, history: list[dict] | None = None) -> Iterator[str]:
        import urllib.request

        data = json.dumps(self._payload(user_text, history, True)).encode()
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"})
        buf = ""
        in_think = False
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            log.warning("llama-server 스트림 오류(%s)", e)
            return
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
            except Exception:  # noqa: BLE001
                continue
            piece = delta.get("content") or ""
            if not piece:
                continue
            # thinking 블록 토큰 스트림 제거
            if "<think>" in piece:
                in_think = True
                piece = piece.split("<think>")[0]
            if in_think:
                if "</think>" in piece:
                    in_think = False
                    piece = piece.split("</think>")[-1]
                else:
                    continue
            buf += piece
            while True:
                m = _SENT_END.search(buf)
                if not m:
                    break
                sent, buf = buf[:m.start() + 1].strip(), buf[m.end():]
                if sent:
                    yield sent
        if buf.strip():
            yield buf.strip()
