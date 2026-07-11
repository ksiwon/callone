"""Qwen3-TTS-12Hz 제로샷 클론 TTS 서버 — callone 전용(별 venv `.venv-qwentts` 에서 실행).

cosyvoice_server 와 **동일 API 계약**(주소만 :8093) — serve 쪽은 tts_qwen3.Qwen3TTS 가
같은 HTTP 클라이언트로 붙는다. 도입 근거/게이트는 docs/REBUILD_PLAN.md §1:
  ⚠️ 과거 Qwen3 계열 TTS 는 "턴마다 음색 튐" 으로 기각된 이력이 있다(구모델).
     이 신형(12Hz, 2026-01)은 scripts/bench_v2.py 음색 안정성 게이트 통과 전엔
     serve.yaml 기본 백엔드로 올리지 말 것(cosyvoice3 유지, backend=auto/qwen3tts 로만 시험).

실행(최초 셋업: scripts/setup_qwen_tts_gpu.sh):
  source .venv-qwentts/bin/activate
  QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base python qwen_tts_server/app.py   # :8093

엔드포인트(cosyvoice_server 와 동일):
  GET  /health         → {"status":"ok", "sr":24000, "model":...}
  POST /synth          {text, ref_audio_b64(float32 mono), ref_sr, prompt_text, language}
                       → 본문=float32 PCM bytes, 헤더 X-Sample-Rate
  POST /synth_stream   → [4바이트 LE 길이][f32 PCM] 프레임 스트림

레퍼런스는 매 요청 인메모리 → tmpfs 임시 wav → 즉시 삭제(디스크 영속 0, cosy 와 동일).
"""
from __future__ import annotations

import base64
import os
import struct
import tempfile
import threading

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel  # type: ignore  # pip install qwen-tts

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
PORT = int(os.environ.get("PORT", "8093"))
OUT_SR = 24000   # callone/프론트 재생 sr 통일(cosy 와 동일)
# >=0 시드 고정 → 같은 ref/텍스트에서 매 합성 동일 음색(턴 편차 억제 — 과거 기각 사유 재발 방지 1차 수단)
SEED = int(os.environ.get("QWEN_TTS_SEED", "1234"))
LANG_DEFAULT = os.environ.get("QWEN_TTS_LANG", "Korean")

print(f"[qwen-tts] 모델 로드: {MODEL_ID}")
_kw: dict = {"device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
             "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
try:
    _model = Qwen3TTSModel.from_pretrained(MODEL_ID, attn_implementation="flash_attention_2", **_kw)
except Exception as e:  # noqa: BLE001  # flash-attn 미설치/미지원 GPU(3090 등 Ampere 는 fa2 OK, 구형만 해당)
    print(f"[qwen-tts] flash_attention_2 불가({e}) → 기본 attention 재시도")
    _model = Qwen3TTSModel.from_pretrained(MODEL_ID, **_kw)
print("[qwen-tts] 로드 완료")

# 동시 요청 직렬화(GPU 단일 + seed 상태 공유 — cosyvoice_server 와 동일 이유)
_infer_lock = threading.Lock()

app = FastAPI(title="callone qwen3-tts")


class SynthReq(BaseModel):
    text: str
    ref_audio_b64: str
    ref_sr: int = 16000
    prompt_text: str = ""
    language: str = LANG_DEFAULT


def _resample(a: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(a) == 0:
        return a
    n = int(round(len(a) * dst / src))
    return np.interp(np.linspace(0, len(a), n, endpoint=False),
                     np.arange(len(a)), a).astype(np.float32)


def _ref_to_tmpfs(req: SynthReq) -> str:
    """레퍼런스 b64 → 16k mono wav (tmpfs). 호출측이 finally 로 삭제."""
    ref = np.frombuffer(base64.b64decode(req.ref_audio_b64), dtype=np.float32)
    ref16 = ref if req.ref_sr == 16000 else _resample(ref, req.ref_sr, 16000)
    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    fd, ref_path = tempfile.mkstemp(suffix=".wav", prefix="qwen_ref_", dir=base)
    os.close(fd)
    sf.write(ref_path, ref16.astype(np.float32), 16000)
    return ref_path


def _seed():
    if SEED >= 0:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)


def _clone_full(text: str, ref_path: str, ref_text: str, language: str) -> np.ndarray:
    """통짜 합성 → OUT_SR float32 mono."""
    wavs, sr = _model.generate_voice_clone(
        text=text, language=language, ref_audio=ref_path,
        ref_text=(ref_text or None))
    a = np.asarray(wavs[0] if isinstance(wavs, (list, tuple)) else wavs,
                   dtype=np.float32).squeeze()
    return np.clip(_resample(a, int(sr), OUT_SR), -1.0, 1.0).astype(np.float32)


@app.get("/health")
def health():
    return {"status": "ok", "sr": OUT_SR, "model": MODEL_ID}


@app.post("/synth")
def synth(req: SynthReq):
    if not req.text.strip():
        return JSONResponse({"error": "empty text"}, status_code=400)
    ref_path = _ref_to_tmpfs(req)
    try:
        with _infer_lock:
            _seed()
            audio = _clone_full(req.text.strip(), ref_path, req.prompt_text, req.language)
    finally:
        try:
            os.remove(ref_path)
        except OSError:
            pass
    return Response(content=audio.tobytes(), media_type="application/octet-stream",
                    headers={"X-Sample-Rate": str(OUT_SR)})


@app.post("/synth_stream")
def synth_stream(req: SynthReq):
    """스트리밍 — 네이티브 스트림 API 가 있으면 그걸(12Hz 완전 causal, 첫패킷 ~100ms),
    없으면 통짜 합성 후 청크로 흘리는 폴백(계약은 동일, 첫음성 이득만 없음).
    프레임 형식은 cosyvoice_server 와 동일: [4바이트 LE 길이][f32 PCM]."""
    if not req.text.strip():
        return JSONResponse({"error": "empty text"}, status_code=400)
    ref_path = _ref_to_tmpfs(req)

    def _frames(a: np.ndarray):
        b = a.astype(np.float32).tobytes()
        return struct.pack("<I", len(b)) + b

    def gen():
        try:
            with _infer_lock:
                _seed()
                stream_fn = getattr(_model, "generate_voice_clone_stream", None)
                if callable(stream_fn):                     # 네이티브 스트리밍
                    for chunk, sr in stream_fn(text=req.text.strip(), language=req.language,
                                               ref_audio=ref_path,
                                               ref_text=(req.prompt_text or None)):
                        a = np.asarray(chunk, dtype=np.float32).squeeze()
                        a = np.clip(_resample(a, int(sr), OUT_SR), -1.0, 1.0).astype(np.float32)
                        yield _frames(a)
                    return
                audio = _clone_full(req.text.strip(), ref_path, req.prompt_text, req.language)
            step = OUT_SR // 4                              # 폴백: 250ms 청크
            for i in range(0, len(audio), step):
                yield _frames(audio[i:i + step])
        finally:
            try:
                os.remove(ref_path)
            except OSError:
                pass

    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": str(OUT_SR)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
