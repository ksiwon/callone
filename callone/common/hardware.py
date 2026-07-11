"""하드웨어 티어 자동 감지 — 기기에 맞는 모델 자동 선택 (v2: VRAM 4단).

티어(자동: VRAM 기준, CALLONE_TIER 로 강제):
  - "ultra" : VRAM ≥ 70GB (H100/A100-80) → LLM 32B + Qwen3-TTS 1.7B + Qwen3-ASR 1.7B
  - "high"  : 20 ≤ VRAM < 70 (3090/3090Ti/4090) → LLM 7.8B~14B + Qwen3-TTS/ASR 0.6B
  - "mid"   : 10 ≤ VRAM < 20 → LLM 7.8B Q4 + whisper turbo, 아바타 static 권장
  - "cpu"   : GPU 없음(노트북 Arc iGPU 등) → EXAONE-2.4B OV + 경량 TTS + whisper small int8

레거시 별칭(하위호환): "server_gpu"(→VRAM 기준 ultra/high/mid), "laptop_cpu"(→cpu).

LLM 엔진 주의(docs/REBUILD_PLAN.md §1):
  - HyperCLOVA X SEED Think 는 llama.cpp 미지원(vLLM 전용) → 기본값은 EXAONE GGUF.
  - 실제 GGUF 선택은 scripts/bootstrap_gpu.sh (VRAM + LLM_PRESET env).
"""
from __future__ import annotations

import os
from functools import lru_cache

from .logging import get_logger

log = get_logger("hardware")

GPU_TIERS = ("ultra", "high", "mid")
_LEGACY = {"server_gpu": None, "laptop_cpu": "cpu"}   # server_gpu 는 VRAM 으로 재판정


@lru_cache(maxsize=1)
def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def total_ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / 1e9
    except Exception:
        # psutil 없으면 os 로 추정(Linux), 실패 시 0
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except Exception:
            return 0.0


@lru_cache(maxsize=1)
def gpu_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return 0.0


def _gpu_tier_by_vram(vram: float) -> str:
    if vram >= 70:
        return "ultra"
    if vram >= 20:
        return "high"
    if vram >= 10:
        return "mid"
    return "cpu"          # 10GB 미만 GPU 는 통화 4모델 공존 불가 → 경량 경로


def detect_tier(override: str | None = None) -> str:
    """실행 티어 결정. CALLONE_TIER 환경변수 또는 인자로 강제 가능.
    레거시 값(server_gpu/laptop_cpu)도 받아서 새 4단으로 해석한다."""
    forced = override or os.environ.get("CALLONE_TIER")
    if forced in ("ultra", "high", "mid", "cpu"):
        return forced
    if forced in _LEGACY:
        mapped = _LEGACY[forced]
        return mapped if mapped else (_gpu_tier_by_vram(gpu_vram_gb()) if has_cuda() else "high")
        # ↑ server_gpu 강제인데 CUDA 프로브 불가(설치 전 등)면 high 가정(24GB 박스가 다수)
    if has_cuda():
        return _gpu_tier_by_vram(gpu_vram_gb())
    return "cpu"


def is_gpu_tier(tier: str | None = None) -> bool:
    return (tier or detect_tier()) in GPU_TIERS


def tier_defaults(tier: str | None = None) -> dict:
    """티어별 권장 모델/설정 — config 가 'auto' 일 때 채택.

    키:
      llm_config      : configs/<이름>.yaml (LoRA 학습·폴백 LLM 참조용)
      asr_realtime/asr_compute : faster-whisper 모델/정밀도(폴백 겸 whisper 백엔드)
      asr_backend     : qwen3(스트리밍 partial) | whisper
      asr_qwen_model  : Qwen3-ASR HF id
      tts_qwen_model  : Qwen3-TTS HF id (qwen-tts-server 가 사용; 빈 값=티어에서 미지원)
      avatar_resolution : Ditto 해상도
    """
    tier = tier or detect_tier()
    table = {
        "ultra": {
            "llm_config": "llm_server",
            "asr_realtime": "large-v3-turbo", "asr_compute": "float16",
            "asr_backend": "qwen3", "asr_qwen_model": "Qwen/Qwen3-ASR-1.7B",
            "tts_qwen_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "avatar_resolution": 512,
        },
        "high": {
            "llm_config": "llm_server",
            "asr_realtime": "large-v3-turbo", "asr_compute": "float16",
            "asr_backend": "qwen3", "asr_qwen_model": "Qwen/Qwen3-ASR-0.6B",
            "tts_qwen_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "avatar_resolution": 256,
        },
        "mid": {
            "llm_config": "llm_server",
            "asr_realtime": "large-v3-turbo", "asr_compute": "int8_float16",
            "asr_backend": "whisper", "asr_qwen_model": "",
            "tts_qwen_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "avatar_resolution": 256,
        },
        "cpu": {
            "llm_config": "llm_laptop",
            "asr_realtime": "small", "asr_compute": "int8",
            "asr_backend": "whisper", "asr_qwen_model": "",
            "tts_qwen_model": "",
            "avatar_resolution": 256,
        },
    }
    # 레거시 이름 호환(외부 스크립트가 tier_defaults("server_gpu") 로 부를 수 있음)
    if tier in _LEGACY:
        tier = detect_tier(tier)
    return table[tier]


def describe() -> str:
    t = detect_tier()
    return (f"tier={t} cuda={has_cuda()} vram={gpu_vram_gb():.0f}GB "
            f"ram={total_ram_gb():.0f}GB → {tier_defaults(t)}")
