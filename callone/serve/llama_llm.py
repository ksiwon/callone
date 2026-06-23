"""llama.cpp(llama-server) LLM 백엔드 — EXAONE GGUF(서버) / 경량 GGUF(노트북) + LoRA, HTTP.

llama.cpp 는 GGUF 로 모델을 바로 서빙하고 LoRA 도 적용 가능 → **llama-server 를 별도
프로세스로 띄우고 HTTP(OpenAI 호환)로 호출**한다. 기본 서빙은 EXAONE-3.5-7.8B GGUF
(bootstrap_gpu.sh 가 다운·기동). 노트북은 OpenVINO(ov_llm) 경로가 별도.

장점:
  - 서빙 파이썬엔 torch/OpenVINO 가 전혀 없음 → segfault 원천 차단(HTTP 클라이언트뿐).
  - 페르소나 카드(system) + RAG(화자 A 실제 발화) + 문장단위 스트리밍 그대로 재사용.
  - GPU 백엔드(Vulkan coopmat-off / SYCL / IPEX) 든 CPU 든 **같은 HTTP 코드**.

서버 띄우는 법(별도 터미널) — scripts/run_llama_server.md / bootstrap_gpu.sh 참고:
  llama-server -m EXAONE-3.5-7.8B-...-Q6_K.gguf [--lora spk-lora-f16.gguf] \
      -ngl 99 --host 127.0.0.1 --port 8090 -c 4096 --jinja
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
        # 샘플러(반복 억제) — DRY 가 주력: n-gram 반복만 정확히 끊고 조사·어미 문법은 안 건드림
        #   (repeat_penalty 강하게의 부작용 '그치마니/하렴 야호' 비문 회피). min_p 로 저확률 토큰
        #   잘라 자연스러움 강건화(top_k/top_p 조합 대체). 전부 serve.yaml llm 블록에서 조정.
        _c = rag_cfg or {}
        self.min_p = float(_c.get("min_p", 0.05))
        self.dry_multiplier = float(_c.get("dry_multiplier", 0.8))
        self.dry_base = float(_c.get("dry_base", 1.75))
        self.dry_allowed_length = int(_c.get("dry_allowed_length", 2))
        self.timeout = timeout
        self.persona = load_persona(speaker)
        self._persona_override: str | None = None   # 이름·관계(이 사람은 누구)
        self._situation: str | None = None          # 지금 상황(scenario)
        self._personality: str | None = None        # 성격·말투
        self._background: str | None = None          # 배경
        self._first_message: str | None = None       # 첫 마디(말투 예시)
        self._example_dialogue: str | None = None    # 예시 대화(말투 충실도)
        self._user_persona: str | None = None        # 상대(나)는 누구 = 관계
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

    # ----- 컨텍스트 주입(캐릭터 카드 — 상황극 페르소나) -------------------
    def set_context(self, persona: str | None = None, situation: str | None = None, *,
                    personality: str | None = None, background: str | None = None,
                    first_message: str | None = None, example_dialogue: str | None = None,
                    user_persona: str | None = None):
        self._persona_override = persona
        self._situation = situation
        self._personality = personality
        self._background = background
        self._first_message = first_message
        self._example_dialogue = example_dialogue
        self._user_persona = user_persona

    def set_emotion_labeling(self, on: bool = True):
        """감정 라벨 출력 on/off. orchestrator 가 TTS 감정 활성 시 켠다.
        켜지면 응답 맨 앞 [emotion:..] → _parse_emotion 이 추출·제거 → TTS instruct 동적 주입.
        """
        self._emotion_labels = bool(on)

    # ----- 메시지 조립 ----------------------------------------------------
    def _system(self, user_text: str) -> str:
        # ── 캐릭터 카드(상황극) ── 정체/성격/배경 → 관계 → 상황 → 말투예시 순으로 조립.
        def _s(v):
            return v.strip() if (v and v.strip()) else ""
        who = _s(self._persona_override) or self.persona
        block = ["[너의 정체 — 전화기 너머의 진짜 사람. 이 사람으로 끝까지 연기하고 절대 AI/모델이라 밝히지 마라]",
                 who]
        if _s(self._personality):
            block.append("성격·말투: " + _s(self._personality))
        if _s(self._background):
            block.append("배경: " + _s(self._background))
        sys = "\n".join(block)
        if _s(self._user_persona):
            sys += ("\n\n[상대 — 지금 너와 통화하는 사람]\n상대는 너에게 '" + _s(self._user_persona)
                    + "'. 이 관계에 맞는 호칭·태도로 대하라.")
        if _s(self._situation):
            sys += "\n\n[지금 상황 — 이 맥락 속에서 대화한다]\n" + _s(self._situation)
        ex = [e for e in (_s(self._first_message), _s(self._example_dialogue)) if e]
        if ex:
            sys += ("\n\n[너의 말투 예시 — 이런 톤·길이로 말한다(말투만 따라하고 내용은 상황에 맞춰)]\n"
                    + "\n".join(ex))
        if self._rag:
            try:
                ctx = self._rag.context(user_text, k=3)
                if ctx:   # 게이트(rag): 관련 기억 있을 때만 채워짐 → 평소엔 빈 값이라 주입 안 됨
                    sys += ("\n\n[참고할 기억 — 지금 질문과 자연스럽게 이어질 때만 활용하고, "
                            "관련 없으면 무시하고 평소처럼 대화한다]\n" + ctx)
            except Exception:  # noqa: BLE001
                pass
        # 규칙은 '부정 나열'이 아니라 '긍정·통합'으로(7.8B 급은 부정 과적재 시 지시 충돌→퇴행:
        #   '짧게'+'질문말라'+'반복말라'가 서로 부딪쳐 되묻기/반복 패턴으로 붕괴). 충돌 3개를 1개 긍정으로.
        sys += (
            "\n\n[말하는 방식]"
            "\n- 너는 전화기 너머의 진짜 사람이다. 끝까지 그 사람으로 연기하고, AI/모델이라 밝히지 마라."
            "\n- 한두 문장으로 말해라. '응'·'그래' 같은 단답으로 끝내지 말고, 네 생각·느낌·반응을 한 마디라도 붙여 자연스럽게(단 길게 늘어놓진 마라)."
            "\n- 질문은 꼭 필요할 때만(서너 번에 한 번꼴) 하고, 매 턴 되묻지 마라. 네 얘기·감상으로 대화를 이어가라."
            "\n- 앞 대화를 기억해 상대 말에 구체적으로 반응하고 다음 이야기로 자연스럽게 이어가라. 이미 한 말·질문은 반복하지 마라."
            "\n- 상대 말에 진짜 사람처럼 감정으로 반응해라(기쁘면 들뜨고, 걱정되면 떨리고, 서운하면 토라지게)."
            "\n- 정해진 말투(반말/존댓말)를 처음부터 끝까지 일관되게. 지정 없으면 반말로 통일."
            "\n- 순 한국어(한글)로만 말해라. 한자·외국어·이모지·괄호 속 행동/상황/혼잣말은 한 글자도 쓰지 마라."
            "\n- 비서 말투 금지(\"도와드릴게요\", \"~할 수 있습니다\", 목록·번호)."
        )
        if self._emotion_labels:
            # 응답 맨 앞에 감정 태그 1개. _parse_emotion 이 추출·제거 → TTS 톤 동적 변화.
            sys += (
                "\n- 응답 맨 앞에 지금 감정을 [emotion:WORD] 로 딱 하나, 영어 한 단어로 붙여라(다른 형식·여러개 금지)."
                " 고를 감정: happy, sad, angry, excited, surprised, tender, playful, worried, shy, tired,"
                " disappointed, proud, neutral. 매번 neutral 만 쓰지 말고 맥락에 맞춰 다채롭게."
                " 예: '[emotion:tender] 아이고 우리 딸~ 밥은 묵었나?'"
            )
        return sys

    def _post_history(self) -> str:
        """이력 '뒤'에 다시 박는 핵심 규칙 — 연구·캐릭터카드 스펙상 시스템(앞)보다 훨씬 강하게 지켜진다.
        길어진 대화에서 말투/길이 붕괴 방지. 캐릭터 자체는 위 시스템에, 여기엔 '행동 규칙'만 짧게."""
        who = (self._persona_override or "").strip()
        rule = ("[지금 답하기 직전 — 반드시 지켜라] "
                + (f"너는 '{who}'다. " if who else "")
                + "위 설정·말투를 유지한다. 반말/존댓말 섞지 말고 일관되게, 1~2문장으로 아주 짧게, "
                "순 한국어로, 입으로 소리내는 말만(이모지·괄호설명·한자 금지). 이미 한 질문은 반복 금지.")
        return rule

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
        # post-history 규칙(이력 뒤·현재 발화 앞 system) — 약화되기 쉬운 말투/길이 규칙을 강하게 재주입.
        msgs.append({"role": "system", "content": self._post_history()})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def _payload(self, user_text: str, history: list[dict] | None, stream: bool) -> dict:
        return {
            "messages": self._messages(user_text, history),
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "min_p": self.min_p,                  # 저확률 토큰 컷(top_k/top_p 대체, 자연스러움 강건)
            "stream": stream,
            # 반복 억제 = DRY 주력. n-gram 반복만 정확히 끊어 "뭐. 뭐. 뭐."/같은 질문 되풀이 차단,
            # 조사·어미 문법은 안 건드림(repeat_penalty 강하게의 비문 부작용 회피). dry_penalty_last_n=-1=전체.
            "dry_multiplier": self.dry_multiplier,
            "dry_base": self.dry_base,
            "dry_allowed_length": self.dry_allowed_length,
            "dry_penalty_last_n": -1,
            # 보조 페널티는 약하게(DRY 가 주력이라 완화). 과하면 한국어 조사·어미 깨짐(실측 비문).
            # repeat_last_n 은 settled 값 64 유지(넓히면 1.05 라도 조사 반복까지 눌러 위험) — 장거리
            # 반복은 dry_penalty_last_n=-1(전체)이 담당.
            "repeat_penalty": 1.05,
            "repeat_last_n": 64,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.3,
            # EXAONE reasoning 비활성. _strip_think는 구형 서버 응답의 안전망.
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
