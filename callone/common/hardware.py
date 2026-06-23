"""하드웨어 티어 자동 감지 — 기기에 맞는 모델 자동 선택 (§3, §17.2).

두 가지 실행 환경을 코드가 스스로 구분한다(폰 온디바이스 경로는 폐기):
  - "server_gpu" : NVIDIA GPU(H100/A100/4090) → 고품질 (EXAONE-3.5-7.8B, CosyVoice3, 적응 Whisper)
  - "laptop_cpu" : GPU 없는 노트북(Arc iGPU 예: 갤럭시북5 Pro) → 경량 (EXAONE-3.5-2.4B, 소형 TTS, small int8)

근거:
  - EXAONE-3.5-7.8B(LG 한국어 특화) = 한국어 품질 최우선(실사용 판정), 24GB GPU 가뿐.
  - EXAONE-3.5-2.4B = 같은 계열 경량(int4 ≈ 1.5GB), Arc iGPU 실시간 → 노트북에 적합.
  → 따라서 GPU 없으면 자동으로 경량(EXAONE-2.4B) 사용.
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
    if forced in ("server_gpu", "laptop_cpu"):
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
            "asr_realtime": "large-v3-turbo",    # 또는 Voxtral
            "asr_compute": "float16",
        },
        "laptop_cpu": {
            "llm_config": "llm_laptop",          # EXAONE-3.5-2.4B (Arc iGPU, 경량)
            "asr_realtime": "small",             # CPU/Arc 실시간엔 small/turbo-int8
            "asr_compute": "int8",
        },
    }
    return table[tier]


def describe() -> str:
    t = detect_tier()
    return (f"tier={t} cuda={has_cuda()} vram={gpu_vram_gb():.0f}GB "
            f"ram={total_ram_gb():.0f}GB → {tier_defaults(t)}")
