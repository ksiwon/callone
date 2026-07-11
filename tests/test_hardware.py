"""기기 티어 자동 선택 검증 (v2: VRAM 4단 — docs/REBUILD_PLAN.md §3).

티어 → config/모델 매핑 검증. 레거시 별칭(server_gpu/laptop_cpu)은 새 4단으로 해석된다.
"""
from callone.common.hardware import detect_tier, is_gpu_tier, tier_defaults


def test_tiers_map_to_right_llm():
    assert tier_defaults("ultra")["llm_config"] == "llm_server"   # EXAONE-4.0-32B Q6_K
    assert tier_defaults("high")["llm_config"] == "llm_server"    # EXAONE-3.5-7.8B Q6_K
    assert tier_defaults("cpu")["llm_config"] == "llm_laptop"     # EXAONE-3.5-2.4B


def test_gpu_uses_turbo_cpu_uses_small():
    assert tier_defaults("high")["asr_realtime"] == "large-v3-turbo"
    assert tier_defaults("cpu")["asr_realtime"] == "small"
    assert tier_defaults("high")["asr_compute"] == "float16"
    assert tier_defaults("cpu")["asr_compute"] == "int8"


def test_qwen_stack_per_tier():
    """ultra=1.7B, high=0.6B, cpu=Qwen 스택 없음(경량 폴백)."""
    assert "1.7B" in tier_defaults("ultra")["tts_qwen_model"]
    assert "0.6B" in tier_defaults("high")["tts_qwen_model"]
    assert tier_defaults("ultra")["asr_backend"] == "qwen3"
    assert tier_defaults("mid")["asr_backend"] == "whisper"
    assert tier_defaults("cpu")["tts_qwen_model"] == ""


def test_legacy_aliases_resolve():
    """구 이름 강제 시에도 새 4단 중 하나로 해석 + tier_defaults 동작."""
    assert detect_tier("laptop_cpu") == "cpu"
    assert detect_tier("server_gpu") in ("ultra", "high", "mid")
    assert tier_defaults("laptop_cpu")["llm_config"] == "llm_laptop"
    assert tier_defaults("server_gpu")["llm_config"] == "llm_server"


def test_detect_tier_override():
    assert detect_tier("ultra") == "ultra"
    assert detect_tier("cpu") == "cpu"
    assert is_gpu_tier("high") and not is_gpu_tier("cpu")
