"""백엔드 래퍼 — 선택 시점 lazy import. 미설치면 안내 문자열 폴백.

UI(app.py)는 여기 함수만 호출한다. 무거운 모델은 전역 캐시(_CACHE)로 1회 로드.
실패는 예외 대신 (결과, 상태메시지) 튜플로 흡수 → Gradio 가 친절히 표시.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile

from . import env as _env
from . import router as _router

log = logging.getLogger("studio.backends")
_CACHE: dict = {}   # 무거운 모델 핸들 캐시 (키: 백엔드+설정)

PROBE_TEXT = ("안녕하세요. 반갑습니다. 이것은 제 목소리를 그대로 복제해 "
              "비교해 보기 위해 준비한 예시 문장입니다.")


# ===================================================================== 공용 오디오
def _have_ffmpeg() -> bool:
    import shutil as _sh

    return _sh.which("ffmpeg") is not None


def to_16k_wav(in_path: str) -> str:
    """업로드 음성(wav/m4a/mp3, 스테레오 포함) → 16kHz 모노 wav 임시경로.

    ffmpeg 있으면 pydub(만능). 없으면 wav 한정으로 soundfile+선형리샘플 폴백
    (Chrome 마이크는 opus/webm 라 ffmpeg 필요 — 그 경우만 명확히 에러).
    """
    out = tempfile.mktemp(suffix=".wav")
    if _have_ffmpeg():
        from pydub import AudioSegment

        seg = AudioSegment.from_file(in_path).set_channels(1).set_frame_rate(16000)
        seg.export(out, format="wav")
        return out
    # ffmpeg 없음 → soundfile 로 직접(wav/flac 등 PCM 만 가능)
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(in_path, dtype="float32")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "ffmpeg 미설치 + 비PCM 입력(브라우저 마이크는 opus/webm). "
            "ffmpeg 설치 필요: choco install ffmpeg (또는 PATH 등록). "
            f"원본 오류: {e}") from e
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if sr != 16000:                      # 간이 선형 리샘플(ffmpeg 없을 때만)
        n = int(round(len(data) * 16000 / sr))
        if n > 0:
            data = np.interp(np.linspace(0, len(data) - 1, n),
                             np.arange(len(data)), data).astype("float32")
    sf.write(out, data, 16000)
    return out


# ===================================================================== 전사 (ASR)
def transcribe(audio_path: str, env_choice: str, mode: str = "zeroshot") -> tuple[str, str]:
    """faster-whisper 로 한국어 전사. 풀클론 모드면 방언적응본 우선(있을 때)."""
    if not audio_path:
        return "", "⚠️ 음성 파일을 먼저 올려라."
    plan = _router.make_plan("transcribe", mode, env_choice)
    if not plan.available:
        return "", f"⚠️ 전사 백엔드 미설치.\n{plan.recipe}"
    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # noqa: BLE001
        return "", f"⚠️ faster-whisper 로드 실패: {e}\n{plan.recipe}"
    key = ("whisper", plan.model, plan.compute)
    note = ""
    try:
        if key not in _CACHE:
            device = "cuda" if plan.compute == "float16" else "cpu"
            try:
                _CACHE[key] = WhisperModel(plan.model, device=device,
                                           compute_type=plan.compute)
            except Exception as ge:  # noqa: BLE001
                # ct2>=4.5 cuDNN9/CUDA12.3 불일치 등 GPU 로드 실패 → CPU int8 폴백
                if device == "cuda":
                    log.warning("Whisper GPU 로드 실패(%s) — CPU int8 폴백", ge)
                    note = " (GPU 실패 → CPU int8 폴백; ct2/cuDNN 버전 확인)"
                    key = ("whisper", plan.model, "int8")
                    _CACHE[key] = WhisperModel(plan.model, device="cpu",
                                               compute_type="int8")
                else:
                    raise
        wav16 = to_16k_wav(audio_path)
        segs, _ = _CACHE[key].transcribe(wav16, language="ko")
        text = "".join(s.text for s in segs).strip()
        return (text or "(전사 비었음 — 직접 입력)"), f"✅ 전사 완료 · {plan.summary}{note}"
    except Exception as e:  # noqa: BLE001
        return "", f"⚠️ 전사 실패: {e}"


# ===================================================================== 제로샷 TTS (CosyVoice3)
def _load_cosyvoice(model_dir: str):
    key = ("cosyvoice", model_dir)
    if key in _CACHE:
        return _CACHE[key]
    import sys

    # voice_clone 앱은 third_party/Matcha-TTS 를 path 에 요구
    vc_root = _env.VOICECLONE_ROOT
    for cand in (vc_root / "GPU_A100", vc_root / "CPU_laptop_WSL2", vc_root):
        mt = cand / "third_party" / "Matcha-TTS"
        if mt.exists() and str(mt) not in sys.path:
            sys.path.append(str(mt))
    from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore

    try:
        m = AutoModel(model_dir=model_dir, load_jit=False, load_trt=False, fp16=_env.has_cuda())
    except TypeError:
        m = AutoModel(model_dir=model_dir)
    _CACHE[key] = m
    return m


def _cosy_synth(model, text: str, prompt_text: str, ref_wav_path: str) -> str:
    import numpy as np
    import soundfile as sf
    import torch

    needs_sys = "CosyVoice3" in type(model).__name__
    sys_prompt = "You are a helpful assistant.<|endofprompt|>"
    pt = (prompt_text or "").strip()
    if pt.startswith("("):
        pt = ""
    if needs_sys:
        gen = model.inference_zero_shot(text.strip(), sys_prompt + pt, ref_wav_path, stream=False)
    elif pt:
        gen = model.inference_zero_shot(text.strip(), pt, ref_wav_path, stream=False)
    else:
        gen = model.inference_cross_lingual(text.strip(), ref_wav_path, stream=False)
    audio = torch.concat([j["tts_speech"] for j in gen], dim=1)
    arr = np.clip(audio.squeeze(0).detach().cpu().numpy().astype("float32"), -1.0, 1.0)
    out = tempfile.mktemp(suffix=".wav")
    sf.write(out, arr, model.sample_rate)
    return out


# ----- 음성 프로필 저장소 (voice_clone/voices, 둘 공유) ----------------------
def list_voices() -> list[str]:
    d = _env.VOICES_DIR
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if (p / "ref_16k.wav").is_file())


def make_candidates(audio_path: str, prompt_text: str, env_choice: str):
    """제로샷: 원본/디노이즈 후보 합성 → (후보경로들, cand_map, 상태)."""
    plan = _router.make_plan("tts", "zeroshot", env_choice)
    if not audio_path:
        return [], {}, "⚠️ 음성 파일을 먼저 올려라."
    if not plan.available:
        return [], {}, f"⚠️ 제로샷 TTS 미설치.\n{plan.recipe}"
    try:
        model = _load_cosyvoice(plan.model)
    except Exception as e:  # noqa: BLE001
        return [], {}, f"⚠️ CosyVoice 로드 실패: {e}\n{plan.recipe}"
    import soundfile as sf

    ref16 = to_16k_wav(audio_path)
    variants = [("원본", ref16)]
    try:
        import noisereduce as nr

        data, sr = sf.read(ref16)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        p = tempfile.mktemp(suffix=".wav")
        sf.write(p, nr.reduce_noise(y=data, sr=sr), sr)
        variants.append(("디노이즈", p))
    except Exception:  # noqa: BLE001
        pass
    outs, cand_map = [], {}
    for label, vpath in variants:
        try:
            outs.append((label, _cosy_synth(model, PROBE_TEXT, prompt_text, vpath)))
            cand_map[label] = vpath
        except Exception as e:  # noqa: BLE001
            print(f"[make_candidates] '{label}' 실패: {e}")
    if not outs:
        return [], {}, "⚠️ 후보 생성 실패 — 콘솔 로그 확인."
    return outs, cand_map, f"✅ 후보 {len(outs)}종 — 가장 본인 같은 걸 골라 저장."


def save_voice(name: str, label: str, ref_path: str | None,
               prompt_text: str) -> tuple[list[str], str]:
    if not name or not name.strip():
        return list_voices(), "⚠️ 저장할 이름을 입력해라."
    if not ref_path:
        return list_voices(), "⚠️ 먼저 후보를 만들고 하나를 골라라."
    name = name.strip()
    vdir = _env.VOICES_DIR / name
    vdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ref_path, vdir / "ref_16k.wav")
    (vdir / "meta.json").write_text(
        json.dumps({"prompt_text": (prompt_text or "").strip(), "variant": label},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return list_voices(), f"💾 저장 완료: '{name}' ({label})"


def tts_zeroshot(voice_name: str, text: str, env_choice: str) -> tuple[str | None, str]:
    """저장된 제로샷 프로필 + 텍스트 → 합성 wav."""
    if not voice_name:
        return None, "⚠️ 저장된 목소리를 선택해라."
    if not text or not text.strip():
        return None, "⚠️ 읽을 텍스트를 입력해라."
    plan = _router.make_plan("tts", "zeroshot", env_choice)
    if not plan.available:
        return None, f"⚠️ 제로샷 TTS 미설치.\n{plan.recipe}"
    vdir = _env.VOICES_DIR / voice_name
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    try:
        model = _load_cosyvoice(plan.model)
        out = _cosy_synth(model, text, meta.get("prompt_text", ""), str(vdir / "ref_16k.wav"))
        return out, f"✅ 합성 완료 · {voice_name} · {plan.summary}"
    except Exception as e:  # noqa: BLE001
        return None, f"⚠️ 합성 실패: {e}"


# ===================================================================== 풀클론 TTS (callone)
def tts_fullclone(speaker: str, text: str, env_choice: str) -> tuple[str | None, str]:
    """callone per-speaker 학습본으로 합성. 모델 없으면 placeholder + 레시피."""
    plan = _router.make_plan("tts", "fullclone", env_choice)
    if not text or not text.strip():
        return None, "⚠️ 읽을 텍스트를 입력해라."
    _env.ensure_paths()
    try:
        from callone.common.io import load_config
        from callone.tts.infer import TTSEngine
        from callone.common.audio import save_wav
    except Exception as e:  # noqa: BLE001
        return None, f"⚠️ callone 로드 실패: {e}\n{plan.recipe}"
    try:
        eng = TTSEngine(speaker or "A", load_config("tts_server"))
        y, sr = eng.synth(text)
        out = tempfile.mktemp(suffix=".wav")
        save_wav(out, y, sr)
        tag = "✅ 학습본 합성" if plan.available else "⚠️ 학습본 없음 → placeholder(배관확인용)"
        msg = f"{tag} · {plan.summary}"
        if not plan.available:
            msg += f"\n{plan.recipe}"
        return out, msg
    except Exception as e:  # noqa: BLE001
        return None, f"⚠️ 합성 실패: {e}\n{plan.recipe}"


# ===================================================================== 통화 (Orchestrator)
def call_turn(audio_path: str, speaker: str, mode: str, env_choice: str,
              persona: str = "", situation: str = ""):
    """녹음 1턴 → (응답오디오 wav, 응답텍스트, 상태). 턴제 간이 통화.

    persona:   이 사람이 누구인지(말투/성격/관계) — 학습 페르소나 대체.
    situation: 지금 어떤 상황에서 통화 중인지(배경지식/맥락).
    """
    if not audio_path:
        return None, "", "⚠️ 마이크로 한 마디 말하고 멈춰라."
    plan = _router.make_plan("call", mode, env_choice)
    _env.ensure_paths()
    try:
        import numpy as np
        import soundfile as sf
        from callone.serve.orchestrator import Orchestrator
        from callone.common.audio import save_wav
    except Exception as e:  # noqa: BLE001
        return None, "", f"⚠️ callone 통화 로드 실패: {e}\n{plan.recipe}"
    key = ("orch", speaker or "A")
    try:
        if key not in _CACHE:
            _CACHE[key] = Orchestrator(speaker or "A")
        orch = _CACHE[key]
        # 통화 페르소나/상황 주입(매 턴 반영 — 입력 바뀌면 즉시 적용)
        if (persona or "").strip() or (situation or "").strip():
            orch.set_context(persona=persona or None, situation=situation or None)
        wav16 = to_16k_wav(audio_path)
        audio, sr = sf.read(wav16)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        turn = orch.handle_utterance(audio.astype("float32"), sr)
        if not turn.audio_chunks:
            return None, turn.reply_text, (
                f"(응답 음성 없음) 你말: '{turn.user_text}' → '{turn.reply_text}'")
        out = tempfile.mktemp(suffix=".wav")
        save_wav(out, np.concatenate(turn.audio_chunks), orch.tts.sr if hasattr(orch.tts, "sr") else sr)
        status = (f"✅ 첫음성 {turn.first_audio_latency_ms:.0f}ms · {plan.summary}\n"
                  f"나: {turn.user_text}")
        return out, turn.reply_text, status
    except Exception as e:  # noqa: BLE001
        return None, "", f"⚠️ 통화 턴 실패: {e}\n{plan.recipe}"


def full_serve_hint() -> str:
    """풀 실시간(WebRTC) 안내 — 앱 내장 턴제 대신 callone-serve 띄우기."""
    return ("풀 실시간 통화(WebRTC, barge-in)는 callone-serve 로:\n"
            "  cd callone && callone-serve        # FastAPI :8000\n"
            "  cd callone/ui && npm install && npm run dev   # :5173\n"
            "앱 내장 통화는 녹음→응답 턴제 간이판이다.")
