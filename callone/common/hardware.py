"""하드웨어 티어 자동 감지 — 기기에 맞는 모델 자동 선택 (§3, §17.2).

세 가지 실행 환경을 코드가 스스로 구분한다:
  - "server_gpu" : NVIDIA GPU(H100 등) → 고품질 모델 (EXAONE-3.5-7.8B, CosyVoice3, 적응 Whisper)
  - "laptop_cpu" : GPU 없는 노트북(예: 32GB RAM) → 경량 (Qwen3.5-4B, 소형 TTS, turbo/small int8)
  - "phone"      : 온디바이스(Android/iOS) → 초경량 (Qwen3.5-4B/2B, per-speaker 소형 TTS)
                   ※ phone 은 파이썬 서버가 아니라 LiteRT/MediaPipe/MLX 런타임이 담당.
                     여기선 명시적 설정(mode=phone)으로만 선택.

근거:
  - EXAONE-3.5-7.8B(LG 한국어 특화) = 한국어 품질 최우선(실사용 판정), 24GB GPU 가뿐.
  - Qwen3.5-4B = 5GB 급, CPU/Arc iGPU 에서 실시간 → 노트북/폰에 적합.
  → 따라서 GPU 없으면 자동으로 경량(Qwen3.5-4B) 사용.
"""
from __future__ import annotations

import os
from functools import lru_cache

from .logging import get_logger

log = get_logger("hardware")


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


def detect_tier(override: str | None = None) -> str:
    """실행 티어 결정. CALLONE_TIER 환경변수 또는 인자로 강제 가능."""
    forced = override or os.environ.get("CALLONE_TIER")
    if forced in ("server_gpu", "laptop_cpu", "phone"):
        return forced
    if has_cuda():
        return "server_gpu"
    return "laptop_cpu"


def tier_defaults(tier: str | None = None) -> dict:
    """티어별 권장 모델/설정 — config 가 'auto' 일 때 채택."""
    tier = tier or detect_tier()
    table = {
        "server_gpu": {
            "llm_config": "llm_server",          # EXAONE-3.5-7.8B
            "tts_config": "tts_server",          # 풀튜닝 화자 TTS(제로샷 통화는 CosyVoice3)
            "asr_realtime": "large-v3-turbo",    # 또는 Voxtral
            "asr_compute": "float16",
        },
        "laptop_cpu": {
            "llm_config": "llm_phone",           # Qwen3.5-4B (CPU/Arc, 경량)
            "tts_config": "tts_phone",           # per-speaker 소형(Piper/MeloTTS/Kokoro)
            "asr_realtime": "small",             # CPU 실시간엔 small/turbo-int8
            "asr_compute": "int8",
        },
        "phone": {
            "llm_config": "llm_phone",           # Qwen3.5-4B/2B (온디바이스)
            "tts_config": "tts_phone",
            "asr_realtime": "small",             # CPU/온디바이스 실시간
            "asr_compute": "int8",
        },
    }
    return table[tier]


def describe() -> str:
    t = detect_tier()
    return (f"tier={t} cuda={has_cuda()} vram={gpu_vram_gb():.0f}GB "
            f"ram={total_ram_gb():.0f}GB → {tier_defaults(t)}")
