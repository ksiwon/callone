"""기기 티어 자동 선택 검증 (§3, §17.2).

2026-06 웹 검증 결과를 코드에 반영:
  - GPU 서버 → Gemma 4 12B
  - 노트북(무GPU) → Gemma 4 E4B (12B는 CPU에서 너무 느림)
  - 폰 → E4B/E2B
"""
from callone.common.hardware import tier_defaults, detect_tier


def test_tiers_map_to_right_llm():
    assert tier_defaults("server_gpu")["llm_config"] == "llm_server"   # 12B
    assert tier_defaults("laptop_cpu")["llm_config"] == "llm_phone"    # E4B
    assert tier_defaults("phone")["llm_config"] == "llm_phone"


def test_gpu_uses_turbo_cpu_uses_small():
    assert tier_defaults("server_gpu")["asr_realtime"] == "large-v3-turbo"
    assert tier_defaults("laptop_cpu")["asr_realtime"] == "small"
    assert tier_defaults("server_gpu")["asr_compute"] == "float16"
    assert tier_defaults("laptop_cpu")["asr_compute"] == "int8"


def test_detect_tier_override(monkeypatch):
    monkeypatch.setenv("CALLONE_TIER", "server_gpu")
    detect_tier.cache_clear() if hasattr(detect_tier, "cache_clear") else None
    assert detect_tier("server_gpu") == "server_gpu"
