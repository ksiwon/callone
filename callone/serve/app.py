"""S7 서빙 API (§16, §17) — FastAPI + WebSocket.

엔드포인트:
  GET  /api/speakers                  화자 목록(ContactList)
  GET  /api/speakers/{id}/profile     프로필 조회 (라벨링 편집기)
  PUT  /api/speakers/{id}/profile     프로필 저장
  GET  /api/speakers/{id}/samples     대표 발화(재생용)
  WS   /ws/call/{speaker_id}          실시간 통화: 오디오 청크 업/다운

클라이언트(ui/)가 통화 시작 시 speaker_id 지정 → 오디오 업스트림,
음성 청크 다운스트림. 제어 메시지(start/stop/mute).

사용:
  callone-serve            # uvicorn 기동
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..common.io import data_dir, load_config, read_json, write_json
from ..common.logging import get_logger
from ..common.schemas import SpeakerProfile

log = get_logger("app")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except Exception:  # noqa: BLE001
    FastAPI = None  # type: ignore


def _speakers_dir() -> Path:
    return data_dir() / "speakers"


def list_speakers() -> list[dict]:
    out = []
    sd = _speakers_dir()
    if not sd.exists():
        return out
    for p in sorted(sd.iterdir()):
        pj = p / "profile.json"
        if pj.exists():
            prof = SpeakerProfile(**read_json(pj))
            out.append({
                "speaker_id": prof.speaker_id,
                "name": prof.user.name or f"화자 {prof.speaker_id}",
                "relation": prof.user.relation,
                "region": prof.effective_region(),
            })
    return out


def _parse_session_init(ctrl: dict) -> dict:
    """session_init 메시지 → orch.init_session kwargs. 전부 인메모리(디스크 파일 안 만듦)."""
    import base64
    import io

    kw: dict = {"persona": ctrl.get("persona"), "situation": ctrl.get("situation"),
                "ref_text": ctrl.get("ref_text"), "history": ctrl.get("history"),
                "nsfw": bool(ctrl.get("nsfw", False)),
                "preset_id": ctrl.get("preset_id") or None,
                # 캐릭터 카드 추가 필드(전부 선택)
                "personality": ctrl.get("personality"), "background": ctrl.get("background"),
                "first_message": ctrl.get("first_message"),
                "example_dialogue": ctrl.get("example_dialogue"),
                "user_persona": ctrl.get("user_persona")}
    if ctrl.get("ref_audio_b64"):
        import soundfile as sf

        raw = base64.b64decode(ctrl["ref_audio_b64"])
        a, asr = sf.read(io.BytesIO(raw), dtype="float32")
        if getattr(a, "ndim", 1) > 1:
            a = a.mean(axis=1)
        kw["ref_audio"] = a
        kw["ref_sr"] = int(asr)
    if ctrl.get("portrait_b64"):
        kw["portrait"] = base64.b64decode(ctrl["portrait_b64"])
    return kw


def create_app():
    if FastAPI is None:
        raise RuntimeError("fastapi 미설치 — pip install fastapi uvicorn")
    app = FastAPI(title="callone", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    _orchestrators: dict[str, Any] = {}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "note": "callone 로컬 서버 — 외부 API 0"}

    @app.get("/api/speakers")
    def speakers():
        return list_speakers()

    @app.get("/api/voice/presets")
    def voice_presets():
        """준비된 프리셋 목소리 목록(data/voice_presets/*.wav). '내 목소리 업로드' 대안으로 UI 가 표시.
        클립 자체는 서버 로컬(gitignore) — 목록만 노출(id·label)."""
        from .voice_presets import list_presets
        return list_presets()

    @app.get("/api/speakers/{sid}/profile")
    def get_profile(sid: str):
        pj = _speakers_dir() / sid / "profile.json"
        if not pj.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return read_json(pj)

    @app.put("/api/speakers/{sid}/profile")
    async def put_profile(sid: str, payload: dict):
        prof = SpeakerProfile(**payload)
        write_json(_speakers_dir() / sid / "profile.json", prof)
        return {"status": "saved", "speaker_id": sid}

    @app.get("/api/speakers/{sid}/samples")
    def samples(sid: str):
        sp = _speakers_dir() / sid / "sample_utterances.json"
        return read_json(sp) if sp.exists() else []

    _preview: dict = {}   # 프로세스 캐시: CosyVoiceTTS 핸들(재프로브 비용 절감)
    _preview_lock = threading.Lock()  # 동시 미리듣기 요청 직렬화(ref overwrite 방지)

    @app.post("/api/voice/preview")
    async def voice_preview(payload: dict):
        """목소리 미리듣기 — 업로드한 참조로 짧은 문장을 복제 합성(통화 전 유사도 확인).

        body: {ref_audio_b64, ref_text?, text?}
        resp: {ref_text, sr, audio_b64(float32 PCM)}
        프라이버시: 참조는 인메모리만(디스크/로그 0). 합성 직후 cleanup_reference 로 폐기.
        """
        import asyncio
        import base64
        import io

        b64 = payload.get("ref_audio_b64")
        if not b64:
            return JSONResponse({"error": "ref_audio_b64 필요(음성 먼저 업로드)"}, status_code=400)
        text = (payload.get("text")
                or "안녕하세요. 제 목소리가 이렇게 복제됐어요. 오랜만이에요, 잘 지냈죠?").strip()
        import soundfile as sf

        raw = base64.b64decode(b64)
        a, asr = sf.read(io.BytesIO(raw), dtype="float32")
        if getattr(a, "ndim", 1) > 1:
            a = a.mean(axis=1)

        def _work():
            from .tts_cosyvoice import CosyVoiceTTS
            with _preview_lock:   # 동시 요청이 같은 tts 객체의 ref 를 덮어쓰는 것 방지
                tts = _preview.get("tts")
                if tts is None:
                    tts = CosyVoiceTTS("preview", load_config("serve").get("tts", {}))
                    _preview["tts"] = tts
                try:
                    tts.set_reference(a, int(asr), payload.get("ref_text") or None)
                    audio, sr = tts.synth(text)
                    return tts.ref_text, audio, sr
                finally:
                    tts.cleanup_reference()   # 인메모리 개인데이터 즉시 폐기

        loop = asyncio.get_running_loop()
        try:
            ref_text, audio, sr = await loop.run_in_executor(None, _work)
        except Exception as e:  # noqa: BLE001
            return JSONResponse(
                {"error": f"미리듣기 실패(cosyvoice-server 확인): {e}"}, status_code=503)
        if audio is None or len(audio) == 0:
            return JSONResponse(
                {"error": "합성 결과 없음 — cosyvoice-server(:8092) 상태 확인"}, status_code=503)
        return {"ref_text": ref_text, "sr": int(sr),
                "audio_b64": base64.b64encode(audio.astype(np.float32).tobytes()).decode()}

    @app.websocket("/ws/call/{speaker_id}")
    async def call(ws: WebSocket, speaker_id: str):
        await ws.accept()
        import asyncio

        from .orchestrator import Orchestrator

        loop = asyncio.get_running_loop()
        if speaker_id not in _orchestrators:
            # 생성+워밍업(CUDA graph/ref 인코딩)은 수 초 → 이벤트 루프 막지 않게 executor 에서.
            _orchestrators[speaker_id] = await loop.run_in_executor(
                None, Orchestrator, speaker_id)
        orch = _orchestrators[speaker_id]
        buf: list = []
        gen_task = None     # 현재 응답 생성·송출 태스크(한 번에 하나)

        async def _run_turn(audio):
            """응답 생성(스레드) → 큐 → WS 송출. 이 코루틴은 **수신 안 함**(단일 수신자 규칙)."""
            q: asyncio.Queue = asyncio.Queue()

            def _producer():
                for ev in orch.stream_turn(audio, sr=16000):
                    loop.call_soon_threadsafe(q.put_nowait, ev)
                loop.call_soon_threadsafe(q.put_nowait, ("_done", None))

            gen = loop.run_in_executor(None, _producer)
            try:
                while True:
                    kind, val = await q.get()
                    if kind == "_done":
                        break
                    if kind == "audio":
                        await ws.send_bytes(val.astype(np.float32).tobytes())
                    elif kind == "frame":
                        import base64
                        await ws.send_text(json.dumps(
                            {"type": "frame", "jpeg_b64": base64.b64encode(val).decode()}))
                    elif kind == "text":
                        await ws.send_text(json.dumps({"type": "reply", "text": val}, ensure_ascii=False))
                    elif kind == "user":   # 사용자 전사 → 클라 대화이력(export)
                        await ws.send_text(json.dumps({"type": "user", "text": val}, ensure_ascii=False))
                    elif kind == "latency":
                        await ws.send_text(json.dumps({"type": "latency_ms", "value": val}))
                    elif kind == "interrupted":
                        await ws.send_text(json.dumps({"type": "interrupted"}))
            finally:
                await gen
                try:
                    await ws.send_text(json.dumps({"type": "audio_end"}))
                except Exception:  # noqa: BLE001
                    pass

        async def _finish_gen():
            """진행 중 응답이 있으면 중단하고 끝까지 정리(동시 송출 2개 방지 — 직렬화)."""
            nonlocal gen_task
            if gen_task and not gen_task.done():
                orch.interrupt()
                try:
                    await gen_task
                except Exception:  # noqa: BLE001
                    pass
            gen_task = None

        try:
            while True:                                  # 단일 수신자 — 여기서만 ws.receive()
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":   # 끊김 → 종료(재 receive 시 RuntimeError 방지)
                    break
                if "text" in msg and msg["text"]:
                    ctrl = json.loads(msg["text"])
                    t = ctrl.get("type")
                    if t == "session_init":
                        kw = _parse_session_init(ctrl)
                        await loop.run_in_executor(None, lambda: orch.init_session(**kw))
                        await ws.send_text(json.dumps({"type": "session_ready"}))
                    elif t == "end_turn":
                        await _finish_gen()              # 이전 응답 정리 후 새 턴(직렬화)
                        audio = np.concatenate(buf) if buf else np.zeros(1, np.float32)
                        buf = []
                        gen_task = asyncio.ensure_future(_run_turn(audio))
                    elif t in ("interrupt", "stop"):
                        orch.interrupt()
                        if t == "stop":
                            break
                elif "bytes" in msg and msg["bytes"]:
                    if gen_task and not gen_task.done():
                        # 응답 생성·재생 중 들어오는 오디오 = 마이크가 계속 흘리는 에코/무음.
                        # 이걸 barge-in(orch.interrupt())으로 보면 **턴 시작 직후 자기 응답을
                        # 즉시 끊어** 0자가 된다(실측 근본원인). 버튼 UX(응답 전송)라 자동 barge-in
                        # 불필요 → 생성 중 바이트는 버퍼링도 인터럽트도 안 하고 버린다.
                        # 진짜 끊기는 명시적 'interrupt'/'stop' 제어 메시지로만.
                        continue
                    buf.append(np.frombuffer(msg["bytes"], dtype=np.float32))
        except WebSocketDisconnect:
            log.info("통화 종료 speaker=%s", speaker_id)
        finally:
            if gen_task and not gen_task.done():
                gen_task.cancel()
            # 프라이버시: 연결 끊기면 인메모리 개인데이터(ref tmpfs·이력·아바타) 즉시 폐기.
            try:
                await loop.run_in_executor(None, orch.cleanup_session)
            except Exception:  # noqa: BLE001
                pass

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="callone 서빙 API")
    ap.add_argument("--config", default="serve")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    import uvicorn

    uvicorn.run(create_app(), host=args.host or cfg.get("host", "0.0.0.0"),
                port=args.port or cfg.get("port", 8000))


if __name__ == "__main__":
    main()
