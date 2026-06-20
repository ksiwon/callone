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
    # 단계별 측정(병목 진단용) — asr/llm_first/llm_total/tts_first/first_audio (ms).
    # RunPod 박스에서 실측해 진짜 병목을 찾는다(목소리 영향 0).
    timings: dict = field(default_factory=dict)


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


_EMOTIONS = ("happy", "sad", "angry", "neutral", "excited", "surprised",
             "tender", "playful", "worried", "shy", "tired", "disappointed", "proud")


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
    # 맨 앞 감정 태그 제거 — 오타·변형까지 단어 무관하게(브래킷이 TTS 로 새어 읽히는 것 방지).
    #   실측: 모델이 [emtion:happy] 처럼 **오타** 내면 'emotion' 정확매칭 정규식이 못 걸러 누출됨.
    #   → ① 맨 앞 대괄호 [..] 는 (오타 포함) 통째 제거 + 안에서 알려진 감정 추출(있으면).
    emotion = default
    m = re.match(r"^\s*\[[^\]]*\]\s*", s)
    if m:
        low = m.group(0).lower()
        emotion = next((e for e in _EMOTIONS if e in low), default)
        s = s[m.end():]
    else:
        #   ② 베어 (happy)/(sad) 괄호 — 알려진 감정만(괄호 속 일반 텍스트 오제거 방지).
        m = re.match(r"^\s*\(\s*(" + "|".join(_EMOTIONS) + r")\s*\)\s*", s, re.IGNORECASE)
        if m:
            emotion = m.group(1).lower()
            s = s[m.end():]
    #   ③ 안전망: 본문 어디든 남은 대괄호 그룹 전부 제거(구어체 발화엔 대괄호가 없어야 함).
    s = re.sub(r"\[[^\]]*\]", "", s).strip()
    return emotion, s


def _tts_stream_emo(tts, text: str, emotion: str):
    """TTS 백엔드가 emotion 인자를 지원하면 넘기고, 아니면 무시(폴백 호환)."""
    try:
        yield from tts.synth_stream(text, emotion=emotion)
    except TypeError:
        yield from tts.synth_stream(text)


def _silence(sr: int, ms: int) -> np.ndarray:
    """문장 사이 텀(무음) — 급박하게 안 들리게."""
    return np.zeros(max(0, int(sr * ms / 1000)), dtype=np.float32)


class Orchestrator:
    def __init__(self, speaker: str, serve_cfg: dict | None = None):
        cfg = serve_cfg or load_config("serve")
        self._cfg = cfg
        self.speaker = speaker
        self.vad = VAD(cfg.get("vad", {}))
        self.asr = StreamASR(cfg.get("asr", {}))
        self.llm = _pick_llm(speaker, cfg)
        self.tts = _pick_tts(speaker, cfg)
        # TTS 감정 활성 시 LLM 이 [emotion:..] 라벨을 내도록(문맥→톤 변화). 백엔드가 지원할 때만.
        tcfg = cfg.get("tts", {}) or {}
        if bool(tcfg.get("emotion", True)):
            fn = getattr(self.llm, "set_emotion_labeling", None)
            if callable(fn):
                fn(True)
        # 합성 단위: full = 응답 통째 1회(운율 일관 + 구두점 자연 텀, 톤 안 튐) /
        #           sentence = 문장별(첫음성 최저지연, WS 스트리밍용). 사이엔 무음 텀.
        self.synth_mode = str(tcfg.get("synth_mode", "full"))
        self.sentence_pause_ms = int(tcfg.get("sentence_pause_ms", 180))
        self.history: list[dict] = []
        self._interrupt = threading.Event()
        # 프라이버시: 기본 false → 로그에 사용자 발화/응답 본문을 절대 안 남긴다(길이·지연만).
        #   디버그로 잠깐 볼 땐 serve.yaml log_content: true. 개인데이터 로그 누출 원천차단.
        self.log_content = bool(cfg.get("log_content", False))
        # 토킹헤드(선택, 기본 off): 사진→움직이는 얼굴. 없거나 실패해도 음성은 그대로(부가 레이어).
        self.avatar = self._init_avatar(cfg)
        # 워밍업: 부팅 시 ASR/LLM/TTS 더미 1회 → CUDA graph 빌드 + ref 인코딩 + 슬롯 예열.
        # 첫 통화 턴의 콜드스타트(수 초) 제거. 목소리/출력엔 영향 0(폐기됨). 초기 셋업만 길어짐.
        if bool(cfg.get("warmup", True)):
            self.warmup()

    def warmup(self) -> dict:
        """첫 턴 콜드스타트 제거 — 각 모델을 더미 입력으로 1회 예열(출력 폐기).

        - ASR: 짧은 무음 전사 → faster-whisper 모델 로드 + 첫 추론 그래프.
        - LLM: 짧은 프롬프트 → llama-server 슬롯/프리픽스 캐시 예열(이력에 안 남김).
        - TTS: 짧은 문장 합성을 끝까지 소비 → CUDA graph 빌드 + ref_audio 인코딩 캐시.
        결과 ms 를 로깅. 실패해도 통화는 폴백으로 계속(예외 안 던짐)."""
        import numpy as _np

        t = {}
        t0 = time.time()
        try:
            self.asr.transcribe(_np.zeros(16000, dtype=_np.float32), 16000)
        except Exception as e:  # noqa: BLE001
            log.warning("ASR 워밍업 스킵(%s)", e)
        t["asr_ms"] = (time.time() - t0) * 1000

        t1 = time.time()
        try:
            for _ in self.llm.chat_stream("안녕", []):
                pass   # 이력(self.history)엔 안 넣음 — 통화 맥락 오염 방지
        except Exception as e:  # noqa: BLE001
            log.warning("LLM 워밍업 스킵(%s)", e)
        t["llm_ms"] = (time.time() - t1) * 1000

        t2 = time.time()
        # ephemeral: ref 가 아직 없으면(통화 시작 시 프론트가 줌) TTS 워밍업 스킵 — 빈 ref 합성
        # 오류 로그 안 내려고. ref 있으면(디스크/세션) 끝까지 소비해 CUDA graph + ref 인코딩 예열.
        if getattr(self.tts, "ref_wav", None):
            try:
                for _ in _tts_stream_emo(self.tts, "네, 안녕하세요.", "neutral"):
                    pass
            except Exception as e:  # noqa: BLE001
                log.warning("TTS 워밍업 스킵(%s)", e)
        t["tts_ms"] = (time.time() - t2) * 1000
        t["total_ms"] = (time.time() - t0) * 1000
        log.info("워밍업 완료: ASR %.0f + LLM %.0f + TTS %.0f = %.0fms",
                 t["asr_ms"], t["llm_ms"], t["tts_ms"], t["total_ms"])
        return t

    def _default_portrait(self, speaker: str) -> str:
        """화자 폴더에서 증명사진 자동탐색(portrait.jpg/jpeg/png)."""
        for ext in ("jpg", "jpeg", "png"):
            p = Path("data/speakers") / speaker / f"portrait.{ext}"
            if p.exists():
                return str(p)
        return ""

    def _init_avatar(self, cfg: dict):
        """토킹헤드 백엔드 준비(선택). enabled=false 면 None. **사진 없으면(ephemeral) 통화 시작 시
        프론트 사진으로 init_session 에서 설정** — 부팅 땐 음성전용(이건 정상, 경고 아님)."""
        acfg = cfg.get("avatar", {}) or {}
        if not bool(acfg.get("enabled", False)):
            return None
        img = acfg.get("image") or self._default_portrait(self.speaker)
        if not img:
            log.info("avatar: 통화 시작 시 프론트 사진으로 설정(지금은 음성전용 — 정상)")
            return None
        try:
            from .avatar import _pick_avatar

            av = _pick_avatar(cfg)
            av.start_call(img)
            log.info("토킹헤드 활성: %s", type(av).__name__)
            return av
        except Exception as e:  # noqa: BLE001
            log.warning("토킹헤드 비활성(%s) — 음성전용", e)
            return None

    def interrupt(self):
        """barge-in: 진행 중인 응답 중단 (app 이 마이크에서 사용자 발화 감지 시 호출)."""
        self._interrupt.set()
        if self.avatar is not None:
            try:
                self.avatar.interrupt()
            except Exception:  # noqa: BLE001
                pass

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

    # ----- 프라이버시 세션(인메모리, 디스크·로그 영속 0) ------------------
    def init_session(self, *, ref_audio=None, ref_sr: int = 24000, ref_text=None,
                     portrait=None, persona=None, situation=None, history=None):
        """통화 시작 — 프론트가 보낸 개인데이터로 세션 구성(전부 **인메모리**).

        ref_audio: 화자 음성 ndarray(목소리 복제) / portrait: 사진 bytes(얼굴) /
        history: 이전 대화(클라가 보관·복원). 디스크 파일 안 만들고 로그에 본문 안 남긴다.
        통화 끝나면 cleanup_session() 으로 즉시 폐기."""
        self.reset_history()
        if history:
            self.history = list(history)
        if (persona and persona.strip()) or (situation and situation.strip()):
            self.set_context(persona=persona or None, situation=situation or None)
        # 목소리(인메모리 ref) — TTS 백엔드가 지원하면.
        if ref_audio is not None:
            fn = getattr(self.tts, "set_reference", None)
            if callable(fn):
                try:
                    fn(ref_audio, ref_sr, ref_text)
                except Exception as e:  # noqa: BLE001
                    log.warning("세션 ref 설정 실패(%s)", e)
        # 얼굴(사진 bytes) — avatar 켜져 있을 때만 세션 아바타 구성.
        if portrait is not None and bool((self._cfg.get("avatar") or {}).get("enabled", False)):
            try:
                from .avatar import _pick_avatar

                av = _pick_avatar(self._cfg)
                av.start_call(portrait)            # bytes 직접(디스크 안 거침)
                if self.avatar is not None:
                    try:
                        self.avatar.stop()
                    except Exception:  # noqa: BLE001
                        pass
                self.avatar = av
            except Exception as e:  # noqa: BLE001
                log.warning("세션 아바타 설정 실패(%s) — 영상 생략", e)
                self.avatar = None
        log.info("세션 시작(인메모리): ref=%s portrait=%s history=%d턴",
                 ref_audio is not None, portrait is not None, len(self.history) // 2)

    def export_history(self) -> list[dict]:
        """현재 대화 이력 반환 — 클라가 내보내기/저장용(서버엔 안 남김)."""
        return list(self.history)

    def cleanup_session(self):
        """통화 종료 — 인메모리 개인데이터 **즉시 폐기**(흔적 0): ref tmpfs 삭제·이력·아바타 해제."""
        fn = getattr(self.tts, "cleanup_reference", None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        if self.avatar is not None:
            try:
                self.avatar.stop()
            except Exception:  # noqa: BLE001
                pass
            self.avatar = None
        self.reset_history()
        log.info("세션 종료 — 인메모리 개인데이터 폐기 완료")

    def handle_utterance(self, audio: np.ndarray, sr: int = 16000) -> Turn:
        """완결된 발화 → 응답 텍스트 + 음성 청크 (문장 스트리밍, barge-in 존중)."""
        self._interrupt.clear()
        t0 = time.time()
        user_text = self.asr.transcribe(audio, sr)
        t_asr = time.time()
        turn = Turn(user_text=user_text)
        if not user_text.strip():
            turn.timings = {"asr_ms": (t_asr - t0) * 1000}
            return turn

        # 1) LLM 응답 전체 수집(감정은 첫 감정 유지). 전화 응답은 짧아 수집 비용 작음.
        emotion = "neutral"
        parts: list[str] = []
        t_llm_first = 0.0
        for sentence in self.llm.chat_stream(user_text, self.history):
            if not t_llm_first:
                t_llm_first = time.time()
            if self._interrupt.is_set():
                turn.interrupted = True
                break
            emotion, clean = _parse_emotion(sentence, emotion)   # §4-1 감정 추출
            if clean:
                parts.append(clean)
        t_llm_done = time.time()
        turn.reply_text = " ".join(parts)

        # 2) 합성: full = 응답 통째 1회(운율 일관·톤 안 튐, 구두점 자연 텀) /
        #          sentence = 문장별 + 사이 무음(저지연). barge-in 은 청크마다 존중.
        t_tts_first = 0.0
        if parts and not turn.interrupted:
            sr_out = getattr(self.tts, "sr", 24000)
            segments = [turn.reply_text] if self.synth_mode == "full" else parts
            first = False
            for i, seg in enumerate(segments):
                if i > 0 and self.sentence_pause_ms > 0:
                    turn.audio_chunks.append(_silence(sr_out, self.sentence_pause_ms))
                for chunk in _tts_stream_emo(self.tts, seg, emotion):
                    if self._interrupt.is_set():
                        turn.interrupted = True
                        break
                    if not first:
                        t_tts_first = time.time()
                        turn.first_audio_latency_ms = (t_tts_first - t0) * 1000
                        first = True
                    turn.audio_chunks.append(chunk)
                if turn.interrupted:
                    break

        # 단계별 시계: 어디서 시간 먹는지(ASR vs LLM vs TTS) 병목 진단(목소리 영향 0).
        turn.timings = {
            "asr_ms": (t_asr - t0) * 1000,
            "llm_first_ms": (t_llm_first - t_asr) * 1000 if t_llm_first else 0.0,
            "llm_total_ms": (t_llm_done - t_asr) * 1000,
            "tts_first_ms": (t_tts_first - t_llm_done) * 1000 if t_tts_first else 0.0,
            "first_audio_ms": turn.first_audio_latency_ms,
        }

        if turn.reply_text:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": turn.reply_text})
        tm = turn.timings
        if self.log_content:   # 디버그 전용(기본 off) — 평소엔 본문 로그 0
            log.info("턴: '%s' → '%s' (첫음성 %.0fms = ASR %.0f + LLM첫 %.0f + TTS첫 %.0f%s)",
                     user_text[:30], turn.reply_text[:40], turn.first_audio_latency_ms,
                     tm.get("asr_ms", 0), tm.get("llm_first_ms", 0), tm.get("tts_first_ms", 0),
                     ", 중단됨" if turn.interrupted else "")
        else:                  # 프라이버시: 본문 없이 길이·지연만
            log.info("턴 완료(usr %d자→rep %d자, 첫음성 %.0fms = ASR %.0f + LLM첫 %.0f + TTS첫 %.0f%s)",
                     len(user_text), len(turn.reply_text), turn.first_audio_latency_ms,
                     tm.get("asr_ms", 0), tm.get("llm_first_ms", 0), tm.get("tts_first_ms", 0),
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
        t_asr = time.time()
        yield ("user", user_text)
        if not user_text.strip():
            yield ("end", "")
            return
        # 1) LLM 응답 전체 수집 — full 합성으로 studio 와 **같은 음색**(우선순위 1: 목소리 유사도).
        #    문장별 독립 합성은 운율 불연속(톤 튐)으로 음색이 덜 닮음 → 통째 1회가 충실도 최선.
        emotion = "neutral"
        parts: list[str] = []
        t_llm_first = 0.0
        for sentence in self.llm.chat_stream(user_text, self.history):
            if not t_llm_first:
                t_llm_first = time.time()
            if self._interrupt.is_set():
                yield ("interrupted", None); break
            emotion, clean = _parse_emotion(sentence, emotion)   # §4-1 감정 추출
            if clean:
                parts.append(clean)
        t_llm_done = time.time()
        reply = " ".join(parts)
        if reply and not self._interrupt.is_set():
            yield ("emotion", emotion)
            yield ("text", reply)

        # 2) 합성: full=응답 통째 1회(음색 일관=studio 동일) / sentence=문장별+무음. 청크마다 audio(+frame),
        #    barge-in 은 청크마다 존중. 토킹헤드는 같은 오디오로 프레임 생성(오디오가 master clock).
        first = False
        t_tts_first = 0.0
        sr_out = getattr(self.tts, "sr", 24000)
        segments = [reply] if self.synth_mode == "full" else parts
        for i, seg in enumerate(segments):
            if self._interrupt.is_set():
                yield ("interrupted", None); break
            if i > 0 and self.sentence_pause_ms > 0:             # 세그먼트 사이 텀(무음)
                yield ("audio", _silence(sr_out, self.sentence_pause_ms))
            for chunk in _tts_stream_emo(self.tts, seg, emotion):
                if self._interrupt.is_set():
                    yield ("interrupted", None); break
                if not first:
                    t_tts_first = time.time()
                    yield ("latency", (t_tts_first - t0) * 1000); first = True
                yield ("audio", chunk)
                # 토킹헤드(선택): 같은 오디오로 얼굴 프레임 → ("frame", jpeg). 없으면 스킵.
                if self.avatar is not None:
                    try:
                        for fr in self.avatar.frames_for(chunk, sr_out):
                            yield ("frame", fr)
                    except Exception as e:  # noqa: BLE001
                        log.warning("avatar 프레임 생성 오류(%s) — 영상 생략", e)
                        self.avatar = None
            else:
                continue
            break

        if reply:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
        yield ("timing", {
            "asr_ms": (t_asr - t0) * 1000,
            "llm_first_ms": (t_llm_first - t_asr) * 1000 if t_llm_first else 0.0,
            "llm_total_ms": (t_llm_done - t_asr) * 1000,
            "tts_first_ms": (t_tts_first - t_llm_done) * 1000 if t_tts_first else 0.0,
            "first_audio_ms": (t_tts_first - t0) * 1000 if t_tts_first else 0.0,
        })
        yield ("end", reply)
