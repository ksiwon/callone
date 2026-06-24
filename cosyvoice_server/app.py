"""CosyVoice3 제로샷 클론 TTS 서버 — callone 전용(별 conda env `cosyvoice` 에서 실행).

callone serve(.venv-serve)는 torch/cosyvoice 가 없으므로 이 서버를 독립 프로세스로 띄우고
HTTP 로 호출한다(llama-server/avatar-server 와 동일 패턴). 음색 안정성 때문에 Qwen3-TTS 대신
선택(실측: CosyVoice3-0.5B 는 턴마다 음색 안 튐).

실행(최초 셋업 후):
  cd ~/CosyVoice                      # setup_cosyvoice_gpu.sh 가 만든 위치
  conda activate cosyvoice
  export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
  COSYVOICE_MODEL_DIR=pretrained_models/Fun-CosyVoice3-0.5B \
    python /path/to/callone/cosyvoice_server/app.py    # :8092

엔드포인트:
  GET  /health         → {"status":"ok"}
  POST /synth          {text, ref_audio_b64(float32 mono), ref_sr, prompt_text, language}
                       → 본문=float32 PCM bytes, 헤더 X-Sample-Rate=24000

레퍼런스는 매 요청 인메모리로 받아 tmpfs 임시 wav 로만 쓰고 즉시 삭제(디스크 영속 0).
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

from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore

MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
PORT = int(os.environ.get("PORT", "8092"))
OUT_SR = 24000   # callone/프론트 재생 sr 통일
SEED = int(os.environ.get("COSYVOICE_SEED", "1234"))   # >=0 고정 → 턴마다 음색 일관(Qwen 백엔드와 동일 의도)

print(f"[cosyvoice] 모델 로드: {MODEL_DIR}")
_cosy = AutoModel(model_dir=MODEL_DIR)
SR = _cosy.sample_rate
# cosyvoice3.yaml 존재 여부로 판정(type().__name__ 은 AutoModel 래퍼라 믿을 수 없음).
NEEDS_SYS = os.path.exists(os.path.join(MODEL_DIR, "cosyvoice3.yaml"))
SYS_PROMPT = "You are a helpful assistant.<|endofprompt|>"
print(f"[cosyvoice] 로드 완료 (sr={SR}, CosyVoice3={NEEDS_SYS})")

# uvicorn 이 sync 핸들러를 threadpool 에서 실행 → 동시 요청이 _cosy / torch.manual_seed 공유.
# GPU 단일 디바이스라 어차피 직렬이지만 seed 상태가 섞이는 것 방지.
_infer_lock = threading.Lock()

app = FastAPI(title="callone cosyvoice3")


class SynthReq(BaseModel):
    text: str
    ref_audio_b64: str
    ref_sr: int = 16000
    prompt_text: str = ""
    language: str = "Korean"


def _resample(a: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(a) == 0:
        return a
    n = int(round(len(a) * dst / src))
    return np.interp(np.linspace(0, len(a), n, endpoint=False),
                     np.arange(len(a)), a).astype(np.float32)


@app.get("/health")
def health():
    return {"status": "ok", "sr": SR, "cosyvoice3": NEEDS_SYS}


@app.post("/synth")
def synth(req: SynthReq):
    if not req.text.strip():
        return JSONResponse({"error": "empty text"}, status_code=400)
    # 레퍼런스(float32 mono) → 16k wav(tmpfs) 임시저장(CosyVoice 는 경로 인자).
    ref = np.frombuffer(base64.b64decode(req.ref_audio_b64), dtype=np.float32)
    ref16 = ref if req.ref_sr == 16000 else _resample(ref, req.ref_sr, 16000)
    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    fd, ref_path = tempfile.mkstemp(suffix=".wav", prefix="cosy_ref_", dir=base)
    os.close(fd)
    sf.write(ref_path, ref16.astype(np.float32), 16000)
    try:
        with _infer_lock:
            # 시드 고정 → 같은 ref/텍스트면 매 합성 동일 음색(턴 편차 제거). CosyVoice 내부 샘플링도 통일.
            if SEED >= 0:
                torch.manual_seed(SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(SEED)
            pt = (req.prompt_text or "").strip()
            if NEEDS_SYS:
                gen = _cosy.inference_zero_shot(req.text.strip(), SYS_PROMPT + pt, ref_path, stream=False)
            elif pt:
                gen = _cosy.inference_zero_shot(req.text.strip(), pt, ref_path, stream=False)
            else:
                gen = _cosy.inference_cross_lingual(req.text.strip(), ref_path, stream=False)
            chunks = [j["tts_speech"] for j in gen]
            audio = torch.concat(chunks, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    finally:
        try:
            os.remove(ref_path)
        except OSError:
            pass
    audio = np.clip(_resample(audio, SR, OUT_SR), -1.0, 1.0).astype(np.float32)
    return Response(content=audio.tobytes(), media_type="application/octet-stream",
                    headers={"X-Sample-Rate": str(OUT_SR)})


def _gen_iter(text: str, ref_path: str, prompt_text: str, stream: bool):
    """CosyVoice inference 제너레이터 선택(/synth 와 /synth_stream 공용). stream=True 면
    네이티브 bistream(단일 생성 내부 토큰→오디오 흘림 = 운율/음색 연속, 첫 패키지 ~150ms)."""
    pt = (prompt_text or "").strip()
    if NEEDS_SYS:
        return _cosy.inference_zero_shot(text, SYS_PROMPT + pt, ref_path, stream=stream)
    if pt:
        return _cosy.inference_zero_shot(text, pt, ref_path, stream=stream)
    return _cosy.inference_cross_lingual(text, ref_path, stream=stream)


@app.post("/synth_stream")
def synth_stream(req: SynthReq):
    """스트리밍 합성 — CosyVoice stream=True 청크를 나오는 대로 [4바이트 LE 길이][f32 PCM] 프레임으로 흘림.
    첫음성 지연 대폭↓(통짜 합성 대기 제거). 음색은 통짜와 동일 연속 생성(작업1, A/B 검증 후 라이브)."""
    if not req.text.strip():
        return JSONResponse({"error": "empty text"}, status_code=400)
    ref = np.frombuffer(base64.b64decode(req.ref_audio_b64), dtype=np.float32)
    ref16 = ref if req.ref_sr == 16000 else _resample(ref, req.ref_sr, 16000)
    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    fd, ref_path = tempfile.mkstemp(suffix=".wav", prefix="cosy_ref_", dir=base)
    os.close(fd)
    sf.write(ref_path, ref16.astype(np.float32), 16000)

    def gen():
        try:
            with _infer_lock:
                # 시드 고정 → 턴 편차 0(통짜 경로와 동일). lock 안에서 seed+추론을 묶어 thread 간 seed 경쟁 방지.
                if SEED >= 0:
                    torch.manual_seed(SEED)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(SEED)
                for j in _gen_iter(req.text.strip(), ref_path, req.prompt_text, stream=True):
                    a = j["tts_speech"].squeeze(0).detach().cpu().numpy().astype(np.float32)
                    a = np.clip(_resample(a, SR, OUT_SR), -1.0, 1.0).astype(np.float32)
                    b = a.tobytes()
                    yield struct.pack("<I", len(b)) + b
        finally:
            try:
                os.remove(ref_path)
            except OSError:
                pass

    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": str(OUT_SR)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
