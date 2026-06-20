#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# 한국어 음성 클로닝 스튜디오 (수정판) · Fun-CosyVoice3-0.5B · WSL2/CPU
# -----------------------------------------------------------------------------
# 핵심 수정:
#  1) AutoModel 로 로드 → CosyVoice3 코드 경로가 자동 선택됨 (외계어 원인 제거)
#  2) 참조 음성은 "파일 경로"를 그대로 넘김 (공식 example.py 방식, load_wav 텐서 문제 회피)
#  3) 출력 저장은 soundfile 사용 (torchaudio.save / torchcodec 우회)
#  소스 파일(llm.py, model.py, file_utils.py, cosyvoice.py, *.yaml)은 절대 건드리지 않음.
# =============================================================================
import os
import sys
import json
import shutil
import tempfile

sys.path.append("third_party/Matcha-TTS")

import torch
import numpy as np
import soundfile as sf
import gradio as gr
from pydub import AudioSegment

# === 핵심: AutoModel 로 로드 (CosyVoice2/CosyVoice 직접 호출 금지) ===
from cosyvoice.cli.cosyvoice import AutoModel

# --- CPU 스레드 튜닝 ---
# Lunar Lake(258V) = P코어 4 + E코어 4(하이퍼스레딩 없음). os.cpu_count()=8 을 다 쓰면
# 느린 E코어가 P코어 발목을 잡음 → 기본 P코어 수(4)만 사용. 환경변수 TTS_THREADS 로 조절.
_threads = int(os.environ.get("TTS_THREADS", "4"))
try:
    torch.set_num_threads(_threads)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# --- 디바이스 선택: 기본 cpu / COSYVOICE_DEVICE=xpu 면 Intel Arc 140V iGPU 시도 ---
WANT = os.environ.get("COSYVOICE_DEVICE", "cpu").strip().lower()
DEVICE = "cpu"
USE_FP16 = False
if WANT == "xpu":
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            DEVICE = "xpu"
            # 주의: CosyVoice 는 fp16=True 면 torch.cuda.amp.autocast(CUDA 전용)를 켬.
            # XPU 엔 CUDA autocast 가 안 먹어서 half 가중치↔float 입력 dtype 충돌/멈춤 발생.
            # → 기본 fp32(안전). 정 빠르게 원하면 TTS_FP16=1 로 실험(권장 안 함).
            USE_FP16 = (os.environ.get("TTS_FP16", "0") == "1")
            try:
                print(f"[device] Intel XPU 감지: {torch.xpu.get_device_name(0)}")
            except Exception:
                print("[device] Intel XPU 감지")
        else:
            print("[device] XPU 요청했으나 torch.xpu 사용불가 → CPU 폴백")
    except Exception as e:
        print(f"[device] XPU 확인 실패 ({e}) → CPU 폴백")
print(f"[device] 사용: {DEVICE}  (threads={_threads}, fp16={USE_FP16})")

MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
VOICES_DIR = os.path.abspath("voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# 후보 비교용 문장 — 너무 짧으면 커널 에러가 나므로 넉넉하게.
PROBE_TEXT = "안녕하세요. 반갑습니다. 이것은 제 목소리를 그대로 복제해 비교해 보기 위해 준비한 예시 문장입니다."

print(f"[load] AutoModel 로딩({DEVICE}): {MODEL_DIR}  (처음 한 번은 느립니다)")
try:
    cosyvoice = AutoModel(model_dir=MODEL_DIR, load_jit=False, load_trt=False, fp16=USE_FP16)
except TypeError:
    # 일부 버전은 kwargs 시그니처가 다름 → 최소 인자로 폴백
    cosyvoice = AutoModel(model_dir=MODEL_DIR)
SR = cosyvoice.sample_rate
print(f"[load] 완료. 출력 SR={SR}")

# --- XPU 면 모델 서브모듈을 iGPU 로 best-effort 이동 (실패해도 CPU 로 계속) ---
# 주: CosyVoice 추론 내부에 literal .cuda() 가 남아있으면 합성 시 터질 수 있음.
#     그때는 run.sh 의 COSYVOICE_DEVICE=xpu 줄을 주석처리 → 순수 CPU 로 복귀.
if DEVICE == "xpu":
    try:
        m = getattr(cosyvoice, "model", None)
        if m is not None:
            try:
                m.device = torch.device("xpu")
            except Exception:
                pass
            for attr in ("llm", "flow", "hift"):
                sub = getattr(m, attr, None)
                if sub is not None and hasattr(sub, "to"):
                    try:
                        sub.to("xpu")
                    except Exception as e:
                        print(f"[xpu] {attr} 이동 실패: {e}")
            print("[xpu] 서브모듈 iGPU 이동 시도 완료")
    except Exception as e:
        print(f"[xpu] 이동 전체 실패 → CPU 로 진행: {e}")
        DEVICE = "cpu"
SYS_PROMPT = "You are a helpful assistant.<|endofprompt|>"
NEEDS_SYS = "CosyVoice3" in type(cosyvoice).__name__
print(f"[v3 system prompt needed] {NEEDS_SYS}")

_whisper = None


# ------------------------------------------------------------------ 유틸
def to_16k_wav(in_path: str) -> str:
    """업로드 음성(WAV/m4a/mp3, 스테레오 포함) → 16kHz 모노 wav 경로."""
    seg = AudioSegment.from_file(in_path).set_channels(1).set_frame_rate(16000)
    out = tempfile.mktemp(suffix=".wav")
    seg.export(out, format="wav")
    return out


def transcribe(ref_audio):
    global _whisper
    if not ref_audio:
        return ""
    try:
        from faster_whisper import WhisperModel
        if _whisper is None:
            # 짧은 참조 클립(5~10초) 전사용 → large-v3-turbo 가 large-v3 대비 CPU 에서 수배 빠름.
            # 정확도 더 원하면 WHISPER_MODEL=large-v3 로 바꾸기. (전사는 ASR 이라 항상 CPU)
            wmodel = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
            print(f"[asr] faster-whisper({wmodel}, CPU/int8) 로딩...")
            try:
                _whisper = WhisperModel(wmodel, device="cpu", compute_type="int8")
            except Exception as e:
                print(f"[asr] {wmodel} 로드 실패({e}) → large-v3 폴백")
                _whisper = WhisperModel("large-v3", device="cpu", compute_type="int8")
        wav16 = to_16k_wav(ref_audio)
        segs, _ = _whisper.transcribe(wav16, language="ko")
        t = "".join(s.text for s in segs).strip()
        return t if t else "(전사 비었음 — 직접 입력)"
    except Exception as e:
        return f"(자동 전사 실패: {e} — 직접 입력)"


def _save_wav(tts_speech, sr) -> str:
    """CosyVoice 출력 텐서([1,T])를 soundfile로 안전 저장."""
    arr = tts_speech.squeeze(0).detach().cpu().numpy().astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    p = tempfile.mktemp(suffix=".wav")
    sf.write(p, arr, sr)
    return p


def build_variants(ref16k_path):
    """원본 / 디노이즈 후보 (둘 다 16kHz 모노 wav 파일 경로)."""
    variants = [("원본", ref16k_path)]
    try:
        import noisereduce as nr
        data, sr = sf.read(ref16k_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        reduced = nr.reduce_noise(y=data, sr=sr)
        p = tempfile.mktemp(suffix=".wav")
        sf.write(p, reduced, sr)
        variants.append(("디노이즈", p))
    except Exception as e:
        print(f"[warn] 디노이즈 생략 ({e})")
    return variants


def _synth(text, prompt_text, ref_wav_path):
    pt = (prompt_text or "").strip()
    if pt.startswith("("):
        pt = ""
    if NEEDS_SYS:  # CosyVoice3 는 시스템프롬프트+<|endofprompt|> 필수
        gen = cosyvoice.inference_zero_shot(text.strip(), SYS_PROMPT + pt, ref_wav_path, stream=False)
    elif pt:
        gen = cosyvoice.inference_zero_shot(text.strip(), pt, ref_wav_path, stream=False)
    else:
        gen = cosyvoice.inference_cross_lingual(text.strip(), ref_wav_path, stream=False)
    chunks = [j["tts_speech"] for j in gen]
    audio = torch.concat(chunks, dim=1)
    return _save_wav(audio, SR)


# ------------------------------------------------------------------ [만들기]
def make_candidates(ref_audio, prompt_text):
    if not ref_audio:
        return (gr.update(), gr.update(), gr.update(),
                gr.update(choices=[], value=None), {}, "⚠️ 먼저 음성 파일을 올려주세요.")
    ref16 = to_16k_wav(ref_audio)
    variants = build_variants(ref16)

    slots = [gr.update(visible=False, value=None) for _ in range(3)]
    cand_map = {}
    labels = []
    for i, (label, vpath) in enumerate(variants[:3]):
        try:
            probe_out = _synth(PROBE_TEXT, prompt_text, vpath)
            slots[i] = gr.update(visible=True, value=probe_out, label=f"후보: {label}")
            cand_map[label] = vpath
            labels.append(label)
        except Exception as e:
            print(f"[error] 후보 '{label}' 생성 실패: {e}")

    if not labels:
        return slots[0], slots[1], slots[2], gr.update(choices=[], value=None), {}, "⚠️ 후보 생성 실패 — 콘솔 로그를 확인해 주세요."
    radio = gr.update(choices=labels, value=labels[0])
    return slots[0], slots[1], slots[2], radio, cand_map, f"✅ 후보 {len(labels)}종 생성 완료 — 가장 본인 같은 걸 고르세요."


def save_voice(name, chosen_label, cand_map, prompt_text):
    if not name or not name.strip():
        return gr.update(), "⚠️ 저장할 이름을 입력하세요."
    if not cand_map or not chosen_label or chosen_label not in cand_map:
        return gr.update(), "⚠️ 먼저 후보를 생성하고 하나를 선택하세요."
    name = name.strip()
    vdir = os.path.join(VOICES_DIR, name)
    os.makedirs(vdir, exist_ok=True)
    # 선택된 16k 모노 wav를 그대로 보관
    shutil.copyfile(cand_map[chosen_label], os.path.join(vdir, "ref_16k.wav"))
    meta = {"prompt_text": (prompt_text or "").strip(), "variant": chosen_label}
    with open(os.path.join(vdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return gr.update(choices=list_voices(), value=name), f"💾 저장 완료: '{name}' ({chosen_label})"


# ------------------------------------------------------------------ [사용하기]
def list_voices():
    if not os.path.isdir(VOICES_DIR):
        return []
    return sorted(d for d in os.listdir(VOICES_DIR)
                  if os.path.isfile(os.path.join(VOICES_DIR, d, "ref_16k.wav")))


def tts_from_voice(voice_name, text):
    if not voice_name:
        return None, "⚠️ 저장된 목소리를 선택하세요."
    if not text or not text.strip():
        return None, "⚠️ 읽을 텍스트를 입력하세요."
    vdir = os.path.join(VOICES_DIR, voice_name)
    with open(os.path.join(vdir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    out = _synth(text, meta.get("prompt_text", ""), os.path.join(vdir, "ref_16k.wav"))
    return out, f"✅ 완료 — 목소리: {voice_name}  ·  오디오 ⤓로 다운로드"


# ------------------------------------------------------------------ UI
with gr.Blocks(title="한국어 음성 클로닝 스튜디오") as demo:
    gr.Markdown("# 🎙️ 한국어 음성 클로닝 스튜디오 (Fun-CosyVoice3-0.5B · CPU)")

    with gr.Tab("① 만들기"):
        gr.Markdown("음성 업로드(**WAV 권장**) → 자동 전사 → 후보 생성 → 가장 똑같은 것 선택 → 이름 붙여 저장")
        with gr.Row():
            with gr.Column():
                ref_audio = gr.Audio(label="단일 화자 음성 (WAV 권장 · 한국어 5~10초)",
                                     type="filepath", sources=["upload", "microphone"])
                prompt_text = gr.Textbox(label="참조에서 말한 내용 (정확할수록 유사도↑)", lines=2)
                tr_btn = gr.Button("🔎 자동 전사")
                gen_btn = gr.Button("🧪 후보 생성", variant="primary")
            with gr.Column():
                cand1 = gr.Audio(label="후보 1", type="filepath", visible=False)
                cand2 = gr.Audio(label="후보 2", type="filepath", visible=False)
                cand3 = gr.Audio(label="후보 3", type="filepath", visible=False)
                choice = gr.Radio(label="가장 본인 같은 후보 선택", choices=[])
                save_name = gr.Textbox(label="저장할 목소리 이름 (예: my_voice)")
                save_btn = gr.Button("💾 이 목소리로 저장", variant="primary")
                make_status = gr.Markdown("")
        cand_state = gr.State({})

        tr_btn.click(transcribe, inputs=ref_audio, outputs=prompt_text)
        gen_btn.click(make_candidates, inputs=[ref_audio, prompt_text],
                      outputs=[cand1, cand2, cand3, choice, cand_state, make_status])

    with gr.Tab("② 사용하기"):
        gr.Markdown("저장된 목소리 선택 → 텍스트 입력 → 합성 → 듣기/다운로드")
        with gr.Row():
            with gr.Column():
                voice_dd = gr.Dropdown(label="저장된 목소리", choices=list_voices())
                refresh_btn = gr.Button("🔄 목록 새로고침")
                use_text = gr.Textbox(label="읽게 할 한국어 텍스트", lines=5,
                                      placeholder="예) 오늘 일정은 오후 세 시 회의입니다.")
                use_btn = gr.Button("▶️ 합성", variant="primary")
            with gr.Column():
                use_audio = gr.Audio(label="결과 (재생/다운로드)", type="filepath")
                use_status = gr.Markdown("")

        refresh_btn.click(lambda: gr.update(choices=list_voices()), outputs=voice_dd)
        use_btn.click(tts_from_voice, inputs=[voice_dd, use_text],
                      outputs=[use_audio, use_status])

    save_btn.click(save_voice, inputs=[save_name, choice, cand_state, prompt_text],
                   outputs=[voice_dd, make_status])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "50000"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
