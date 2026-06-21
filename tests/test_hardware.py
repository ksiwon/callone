"""기기 티어 자동 선택 검증 (§3, §17.2).

티어 → config 이름 매핑 검증(모델은 config 가 보유). 폰 경로는 폐기:
  - GPU 서버 → llm_server (EXAONE-3.5-7.8B)
  - 노트북(무GPU/Arc) → llm_laptop (EXAONE-3.5-2.4B, 서버 7.8B는 노트북에서 너무 느림)
"""
from callone.common.hardware import tier_defaults, detect_tier


def test_tiers_map_to_right_llm():
    assert tier_defaults("server_gpu")["llm_config"] == "llm_server"   # EXAONE-3.5-7.8B
    assert tier_defaults("laptop_cpu")["llm_config"] == "llm_laptop"   # EXAONE-3.5-2.4B


def test_gpu_uses_turbo_cpu_uses_small():
    assert tier_defaults("server_gpu")["asr_realtime"] == "large-v3-turbo"
    assert tier_defaults("laptop_cpu")["asr_realtime"] == "small"
    assert tier_defaults("server_gpu")["asr_compute"] == "float16"
    assert tier_defaults("laptop_cpu")["asr_compute"] == "int8"


def test_detect_tier_override(monkeypatch):
    monkeypatch.setenv("CALLONE_TIER", "server_gpu")
    detect_tier.cache_clear() if hasattr(detect_tier, "cache_clear") else None
    assert detect_tier("server_gpu") == "server_gpu"
