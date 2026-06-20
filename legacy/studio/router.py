"""라우팅 매트릭스 — (목적 × 데이터모드 × 환경) → 실행 Plan.

목적(purpose):   transcribe(전사) | tts(음성합성) | call(실시간통화)
데이터모드(mode): zeroshot(5~10초 제로샷) | fullclone(대량녹음 LoRA 풀클론)
환경(env):       gpu | cpu  (티어 server_gpu | laptop_cpu)

각 칸은 백엔드/모델/가용성/실행레시피를 담은 Plan 으로 해석된다.
가용성은 import 가능 여부 + 모델 파일 존재로 가볍게 판정(실모델 로드 X).
미설치 칸은 available=False + recipe(어떻게 돌리는지) 로 폴백한다.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from . import env as _env

PURPOSES = {"transcribe": "전사", "tts": "TTS 출력", "call": "실시간 통화"}
MODES = {"zeroshot": "제로샷 (5~10초)", "fullclone": "풀클론 (대량 녹음)"}


@dataclass
class Plan:
    purpose: str
    mode: str
    tier: str                       # server_gpu | laptop_cpu
    backend: str                    # 실제 백엔드 식별자
    model: str                      # 모델 id/경로
    compute: str = ""               # fp16 | int8 등
    available: bool = False         # 지금 환경서 바로 실행 가능?
    summary: str = ""               # UI 한 줄 설명
    recipe: str = ""                # 미설치 시 실행 안내
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{PURPOSES.get(self.purpose, self.purpose)} · {MODES.get(self.mode, self.mode)}"


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _gpu(tier: str) -> bool:
    return tier == "server_gpu"


# --------------------------------------------------------------------- 전사
def _plan_transcribe(mode: str, tier: str) -> Plan:
    compute = "float16" if _gpu(tier) else "int8"
    model = "large-v3" if _gpu(tier) else "small"
    available = _have("faster_whisper")
    dialect = _env.CALLONE_ROOT / "models" / "asr_dialect"
    summary = f"faster-whisper {model} ({compute})"
    recipe = "pip install faster-whisper  # 전사 백엔드"
    if mode == "fullclone":
        # 풀클론: 방언 적응본 있으면 우선
        if dialect.exists():
            model = str(dialect)
            summary = f"방언적응 Whisper ({compute}) — {dialect.name}"
        else:
            summary += "  (방언적응본 없음 → 기본 모델)"
            recipe = ("방언 적응 학습: callone-asr-train (data/ 준비 후, GPU 노드). "
                      "없으면 기본 large-v3 로 전사.")
    return Plan("transcribe", mode, tier, "faster_whisper", model, compute,
                available, summary, recipe)


# --------------------------------------------------------------------- TTS
def _plan_tts(mode: str, tier: str) -> Plan:
    if mode == "zeroshot":
        # 제로샷: voice_clone CosyVoice3(참조 5~10초). GPU 면 Qwen3-TTS-Instruct 도 가능.
        cosy_dir = _env.cosyvoice_model_dir()
        have_cosy = _have("cosyvoice") and cosy_dir is not None
        backend = "cosyvoice3"
        model = str(cosy_dir) if cosy_dir else "Fun-CosyVoice3-0.5B"
        summary = "제로샷 CosyVoice3 (참조 WAV 5~10초 → 즉시 합성)"
        recipe = ("voice_clone/CPU_laptop_WSL2(또는 GPU_A100)/setup.sh 로 CosyVoice 설치 후 "
                  "COSYVOICE_MODEL_DIR 지정. (제로샷, 학습 불필요)")
        return Plan("tts", mode, tier, backend, model, "fp16" if _gpu(tier) else "fp32",
                    have_cosy, summary, recipe, extra={"needs_ref": True})
    # fullclone: callone per-speaker TTS(학습본). GPU=Qwen3-TTS LoRA, CPU=Piper onnx.
    backend = "qwen3_tts" if _gpu(tier) else "piper"
    spk_models = _env.CALLONE_ROOT / "models" / "tts_server"
    have_model = spk_models.exists() and any(spk_models.iterdir()) if spk_models.exists() else False
    summary = (f"풀클론 {'Qwen3-TTS-1.7B LoRA(화자학습)' if _gpu(tier) else 'Piper(화자학습, onnx)'}")
    recipe = ("화자 TTS셋 빌드 후 학습: callone-build-tts → callone-tts-train "
              f"(GPU 노드). 산출물 models/tts_server/<화자>/. 백엔드={backend}.")
    available = have_model and (_have("callone"))
    return Plan("tts", mode, tier, backend, str(spk_models), "fp16" if _gpu(tier) else "int8",
                available, summary, recipe, extra={"needs_speaker": True})


# --------------------------------------------------------------------- 통화
def _plan_call(mode: str, tier: str) -> Plan:
    # 통화는 callone Orchestrator(ASR→LLM→TTS) 재사용. 모드/환경에 따라 내부 구성만 바뀜.
    have_callone = _have("callone")
    if mode == "zeroshot":
        summary = ("제로샷 통화: 참조음색 + 페르소나-프롬프트 LLM(학습X). "
                   "녹음 적을 때 즉석 대화.")
        recipe = ("Orchestrator 가 placeholder/제로샷 TTS + PersonaLLM 폴백으로 동작. "
                  "더 좋게: GPU 면 llama-server 띄우고 serve.yaml llm.backend=llama.")
    else:
        summary = ("풀클론 통화: 화자 LoRA TTS + LLM 페르소나 SFT. "
                   "callone-serve(FastAPI+React) 풀 실시간 권장.")
        recipe = ("학습 산출물(models/tts_server, models/llm_*) 준비 후 "
                  "callone-serve 로 풀 WebRTC. 앱 내장은 턴제 간이판.")
    backend = "orchestrator"
    model = ("llama-server Qwen3.5 + Qwen3-TTS" if _gpu(tier)
             else "PersonaLLM + Piper/placeholder")
    return Plan("call", mode, tier, backend, model, "", have_callone, summary, recipe,
                extra={"needs_speaker": True, "turn_based": True})


_DISPATCH = {"transcribe": _plan_transcribe, "tts": _plan_tts, "call": _plan_call}


def make_plan(purpose: str, mode: str, env_choice: str) -> Plan:
    """헤더 3선택 → 실행 Plan. 환경은 CALLONE_TIER 로 강제 반영 후 해석."""
    tier = _env.apply_env(env_choice)
    fn = _DISPATCH.get(purpose)
    if fn is None:
        return Plan(purpose, mode, tier, "none", "", available=False,
                    summary=f"알 수 없는 목적: {purpose}")
    return fn(mode, tier)
