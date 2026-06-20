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
        sys += (
            "\n\n[말하는 방식 — 반드시 지켜라]"
            "\n- 너는 AI 비서가 아니라 전화기 너머의 진짜 사람이다. 끝까지 그 사람으로 연기하고, 절대 AI/모델이라 밝히지 마라."
            "\n- 실제 통화처럼 1~2문장으로 짧게. 글이 아니라 입에서 나오는 구어체로 말해라."
            "\n- 줄임말과 구어체를 자연스럽게 써라. 단 '야!/아!/어!' 같은 감탄사로 문장을 **시작하는 건 가끔만** — 매번 시작하지 말고, 바로 본론으로 자연스럽게 들어가라."
            "\n- 도움/설명/정리하는 비서 말투 절대 금지(\"도와드릴게요\", \"~할 수 있습니다\", 목록·번호 금지)."
            "\n- **입으로 말할 수 있는 것만** 출력해라. 이모지·이모티콘 절대 금지. 괄호()로 행동·상황·해설·자기설명"
            " 넣지 마라(예: '(웃으며)', '(이모지는 상상이에요)' 같은 거 절대 금지). 오직 실제 입에서 나오는 말만."
            "\n- 상대 말에 진짜 사람처럼 감정으로 반응해라(기쁘면 들뜨고, 걱정되면 떨리고, 서운하면 토라지게). 맞장구치며 공감해라."
            "\n- **대화를 앞으로 진전시켜라**: 위 대화 내용을 기억하고, 상대가 답하면 그 답에 구체적으로 반응한 뒤"
            " 자연스럽게 **새로운 화제나 다음 이야기로 넘어가라**. 이미 했던 질문·했던 말을 다시 하지 마라(같은 질문 반복 금지)."
            "\n- 매 문장을 질문으로 끝내지 마라. 질문은 가끔만, 보통은 네 얘기·감상·반응으로 대화를 이어가라."
        )
        if self._emotion_labels:
            # 응답 맨 앞에 감정 태그 1개. _parse_emotion 이 추출·제거 → TTS 톤 동적 변화.
            sys += (
                "\n- 응답 맨 앞에 지금 감정을 [emotion:WORD] 로 딱 하나, 영어 한 단어로 붙여라(다른 형식·여러개 금지)."
                " 고를 감정: happy, sad, angry, excited, surprised, tender, playful, worried, shy, tired,"
                " disappointed, proud, neutral. 매번 neutral 만 쓰지 말고 맥락에 맞춰 다채롭게."
                " 예: '[emotion:tender] 아이고 우리 딸~ 밥은 묵었나?'"
            )
        # Qwen3.5 thinking 강제 OFF(모델레벨 소프트 스위치) — 시스템 프롬프트 붙으면 thinking 이
        # 다시 켜져 content 가 비는(reasoning_content 로만 가는) 실측 버그 차단. --jinja 와 별개로 확실.
        sys += "\n\n/no_think"
        return sys

    def _messages(self, user_text: str, history: list[dict] | None) -> list[dict]:
        msgs = [{"role": "system", "content": self._system(user_text)}]
        # history 새니타이즈: 클라가 import 한 이력에 None/빈/비문자열 content 나 알 수 없는
        # role 이 섞이면 (특히 이전 실패 턴이 저장한 빈 응답) chat 템플릿 렌더가 터져 llama 가
        # 빈손/500 → 통화서 'LLM 0자'. user/assistant + 비어있지 않은 문자열만 통과시킨다.
        for m in list(history or [])[-self.max_history:]:
            role = (m or {}).get("role")
            content = (m or {}).get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                msgs.append({"role": role, "content": content})
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
            "repeat_last_n": 256,        # 64→256: 여러 턴 전 표현까지 반복 억제(같은 질문 되풀이 방지)
            "frequency_penalty": 0.5,
            "presence_penalty": 0.7,     # 0.3→0.7: 새 화제·새 표현으로 밀어 주제고착/반복 완화
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
        import urllib.error
        import urllib.request

        data = json.dumps(self._payload(user_text, history, False)).encode()
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                obj = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            # llama 가 템플릿 렌더 실패 등으로 4xx/5xx 면 본문(원인)을 남긴다.
            body = e.read().decode("utf-8", "ignore")[:300]
            log.warning("llama-server HTTP %s: %s", e.code, body)
            return ""
        text = obj["choices"][0]["message"].get("content") or ""
        return self._strip_think(text)

    # ----- 생성(SSE 스트리밍, 문장 단위 yield) ---------------------------
    def chat_stream(self, user_text: str, history: list[dict] | None = None) -> Iterator[str]:
        import urllib.error
        import urllib.request

        data = json.dumps(self._payload(user_text, history, True)).encode()
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"})
        buf = ""
        in_think = False
        yielded = False
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
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
                        yielded = True
                        yield sent
            if buf.strip():
                yielded = True
                yield buf.strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            log.warning("llama-server 스트림 HTTP %s: %s", e.code, body)
        except Exception as e:  # noqa: BLE001
            log.warning("llama-server 스트림 오류(%s)", e)
        # 스트리밍이 한 글자도 못 냈으면(HTTP 에러·thinking 누출 등 모든 경우) **반드시**
        # 비스트리밍 chat() 로 폴백 — 이 경로는 system+history 포함해도 작동 검증됨.
        if not yielded:
            # 스트리밍이 빈 응답(thinking 누출 등) → 비스트리밍 chat() 폴백(content 보장).
            log.info("스트리밍 빈 응답 → 비스트리밍 폴백")
            try:
                txt = self.chat(user_text, history)
                if txt.strip():
                    yield txt.strip()
            except Exception as e:  # noqa: BLE001
                log.warning("LLM 비스트리밍 폴백 실패(%s)", e)
