"""callone studio — 통합 Gradio 앱.

진입 시 헤더에서 [환경][목적][데이터모드] 선택 → 아래 패널이 그 칸에 맞게 바뀐다.
callone(풀클론·통화) + voice_clone(제로샷 TTS)을 한 진입점으로 합친다.

실행:  python studio.py   (또는 python -m studio.app)
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # 외부 텔레메트리 차단

import gradio as gr

from . import backends as B
from . import env as _env
from . import router as _router

ENV_CHOICES = [("자동 감지", "auto"), ("GPU 강제", "gpu"), ("CPU 강제", "cpu")]
PURPOSE_CHOICES = [("전사 (음성→텍스트)", "transcribe"),
                   ("TTS 출력 (텍스트→음성)", "tts"),
                   ("실시간 통화", "call")]
MODE_CHOICES = [("제로샷 (5~10초)", "zeroshot"), ("풀클론 (대량 녹음)", "fullclone")]


def _plan_badge(purpose: str, mode: str, env_choice: str) -> str:
    p = _router.make_plan(purpose, mode, env_choice)
    dot = "🟢 바로 실행 가능" if p.available else "🟡 미설치 — 레시피 폴백"
    md = (f"### {p.label}\n"
          f"- {_env.describe_env(env_choice)}\n"
          f"- 백엔드: `{p.backend}` · 모델: `{p.model}`"
          f"{' · ' + p.compute if p.compute else ''}\n"
          f"- {p.summary}\n"
          f"- 상태: **{dot}**")
    if not p.available and p.recipe:
        md += f"\n\n> 실행법:\n> ```\n> {p.recipe}\n> ```"
    return md


def _toggle(purpose: str, mode: str):
    """목적/모드 → 패널 가시성."""
    return (gr.update(visible=purpose == "transcribe"),
            gr.update(visible=purpose == "tts" and mode == "zeroshot"),
            gr.update(visible=purpose == "tts" and mode == "fullclone"),
            gr.update(visible=purpose == "call"))


def build() -> gr.Blocks:
    with gr.Blocks(title="callone studio — 통합 음성 클론") as demo:
        gr.Markdown("# 🎙️ callone studio — 전사 · 음성복제 · 실시간 통화 통합")

        # ---------------- 헤더: 3축 선택 ----------------
        with gr.Row():
            env_dd = gr.Dropdown(ENV_CHOICES, value="auto", label="① 환경", scale=1)
            purpose_dd = gr.Dropdown(PURPOSE_CHOICES, value="transcribe", label="② 목적", scale=1)
            mode_dd = gr.Dropdown(MODE_CHOICES, value="zeroshot", label="③ 데이터 모드", scale=1)
        plan_md = gr.Markdown(_plan_badge("transcribe", "zeroshot", "auto"))

        # ---------------- 패널: 전사 ----------------
        with gr.Group(visible=True) as g_tr:
            gr.Markdown("## 📝 전사 — 음성 올리면 한국어 텍스트로")
            with gr.Row():
                with gr.Column():
                    tr_audio = gr.Audio(label="음성 파일", type="filepath",
                                        sources=["upload", "microphone"])
                    tr_btn = gr.Button("🔎 전사", variant="primary")
                with gr.Column():
                    tr_out = gr.Textbox(label="전사 결과", lines=6)
                    tr_status = gr.Markdown("")
            tr_btn.click(B.transcribe, [tr_audio, env_dd, mode_dd], [tr_out, tr_status])

        # ---------------- 패널: 제로샷 TTS ----------------
        with gr.Group(visible=False) as g_tts_zero:
            gr.Markdown("## 🗣️ 제로샷 TTS — 5~10초 참조로 목소리 복제")
            with gr.Tab("① 만들기"):
                with gr.Row():
                    with gr.Column():
                        z_ref = gr.Audio(label="참조 음성 (WAV 권장 · 5~10초)",
                                         type="filepath", sources=["upload", "microphone"])
                        z_prompt = gr.Textbox(label="참조에서 말한 내용 (정확할수록 유사도↑)", lines=2)
                        z_tr = gr.Button("🔎 자동 전사")
                        z_gen = gr.Button("🧪 후보 생성", variant="primary")
                    with gr.Column():
                        z_c1 = gr.Audio(label="후보 1", type="filepath", visible=False)
                        z_c2 = gr.Audio(label="후보 2", type="filepath", visible=False)
                        z_choice = gr.Radio(label="가장 본인 같은 후보", choices=[])
                        z_name = gr.Textbox(label="저장할 목소리 이름")
                        z_save = gr.Button("💾 저장", variant="primary")
                        z_status = gr.Markdown("")
                z_candmap = gr.State({})
            with gr.Tab("② 사용하기"):
                with gr.Row():
                    with gr.Column():
                        z_voice = gr.Dropdown(label="저장된 목소리", choices=B.list_voices())
                        z_refresh = gr.Button("🔄 새로고침")
                        z_text = gr.Textbox(label="읽게 할 한국어 텍스트", lines=4)
                        z_speak = gr.Button("▶️ 합성", variant="primary")
                    with gr.Column():
                        z_out = gr.Audio(label="결과", type="filepath")
                        z_use_status = gr.Markdown("")

            z_tr.click(lambda a, e: B.transcribe(a, e, "zeroshot")[0], [z_ref, env_dd], z_prompt)

            def _gen(ref, prompt, e):
                outs, cmap, status = B.make_candidates(ref, prompt, e)
                slots = [gr.update(visible=False, value=None) for _ in range(2)]
                labels = []
                for i, (lab, path) in enumerate(outs[:2]):
                    slots[i] = gr.update(visible=True, value=path, label=f"후보: {lab}")
                    labels.append(lab)
                radio = gr.update(choices=labels, value=labels[0] if labels else None)
                return slots[0], slots[1], radio, cmap, status

            z_gen.click(_gen, [z_ref, z_prompt, env_dd],
                        [z_c1, z_c2, z_choice, z_candmap, z_status])

            def _save(name, label, cmap, prompt):
                ref = (cmap or {}).get(label)
                voices, status = B.save_voice(name, label or "", ref, prompt)
                return gr.update(choices=voices, value=name), status

            z_save.click(_save, [z_name, z_choice, z_candmap, z_prompt],
                         [z_voice, z_status])
            z_refresh.click(lambda: gr.update(choices=B.list_voices()), None, z_voice)
            z_speak.click(B.tts_zeroshot, [z_voice, z_text, env_dd], [z_out, z_use_status])

        # ---------------- 패널: 풀클론 TTS ----------------
        with gr.Group(visible=False) as g_tts_full:
            gr.Markdown("## 🧬 풀클론 TTS — 대량 녹음으로 학습한 화자 모델")
            with gr.Row():
                with gr.Column():
                    f_spk = gr.Textbox(label="화자 ID (예: A)", value="A")
                    f_text = gr.Textbox(label="읽게 할 한국어 텍스트", lines=4)
                    f_btn = gr.Button("▶️ 합성", variant="primary")
                with gr.Column():
                    f_out = gr.Audio(label="결과", type="filepath")
                    f_status = gr.Markdown("학습본 없으면 placeholder(배관 확인용)로 합성된다.")
            f_btn.click(B.tts_fullclone, [f_spk, f_text, env_dd], [f_out, f_status])

        # ---------------- 패널: 실시간 통화 ----------------
        with gr.Group(visible=False) as g_call:
            gr.Markdown("## ☎️ 실시간 통화 (앱 내장 = 턴제 간이판)")
            with gr.Accordion("👤 페르소나 · 상황 설정 (이 사람은 누구? 지금 무슨 상황?)", open=True):
                c_persona = gr.Textbox(
                    label="페르소나 — 이 사람이 누구인지(말투·성격·관계)",
                    lines=3,
                    placeholder="예) 너는 부산 사는 60대 우리 엄마. 무뚝뚝하지만 정 많고, 경상도 사투리로 짧게 말한다.")
                c_situation = gr.Textbox(
                    label="상황 — 지금 어떤 상황에서 통화 중인지(배경지식·맥락)",
                    lines=3,
                    placeholder="예) 내가 오랜만에 전화했다. 엄마는 김장 중이고, 다음 주 내 생일을 챙기려 한다.")
            with gr.Row():
                with gr.Column():
                    c_spk = gr.Textbox(label="상대 화자 ID", value="A")
                    c_audio = gr.Audio(label="한 마디 말하고 멈춤", type="filepath",
                                       sources=["microphone", "upload"])
                    c_btn = gr.Button("📞 보내기", variant="primary")
                with gr.Column():
                    c_out = gr.Audio(label="상대 응답 음성", type="filepath")
                    c_reply = gr.Textbox(label="상대 응답(텍스트)", lines=2)
                    c_status = gr.Markdown("")
            gr.Markdown(B.full_serve_hint())
            c_btn.click(B.call_turn,
                        [c_audio, c_spk, mode_dd, env_dd, c_persona, c_situation],
                        [c_out, c_reply, c_status])

        # ---------------- 헤더 변경 → 배지 + 패널 토글 ----------------
        def _on_change(purpose, mode, e):
            return (_plan_badge(purpose, mode, e), *_toggle(purpose, mode))

        outs = [plan_md, g_tr, g_tts_zero, g_tts_full, g_call]
        for dd in (env_dd, purpose_dd, mode_dd):
            dd.change(_on_change, [purpose_dd, mode_dd, env_dd], outs)

    return demo


def main() -> None:
    demo = build()
    port = int(os.environ.get("PORT", "50000"))
    # GRADIO_SHARE=1 이면 gradio 공개 링크(*.gradio.live) 생성 — RunPod 등 원격에서
    # 포트 노출/재시작 없이 브라우저+마이크 접속(HTTPS 라 마이크 권한 OK).
    share = os.environ.get("GRADIO_SHARE", "0").lower() in ("1", "true", "yes")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    main()
