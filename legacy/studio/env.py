"""환경 해석 — GPU/CPU 선택을 callone 티어로 매핑 + 두 프로젝트 경로 배선.

studio 는 callone(pip 패키지)과 voice_clone(CosyVoice 제로샷 앱)을 한 진입점으로
합친다. 두 프로젝트는 무거운 의존성이 서로 충돌하므로 **여기선 경로만 잡고**,
실제 백엔드 import 는 backends.py 에서 선택 시점에 lazy 로 한다.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

# 단일 루트 구조: callone/ 안에 callone(패키지)·studio·voice_clone 공존.
# 루트 탐색: CALLONE_HOME 환경변수 → CWD → __file__ 상위 순으로 callone+voice_clone 둘 다
# 있는 폴더를 찾는다(진입점 `callone-studio` 든 `python -m studio` 든, 어디서 띄워도 동작).
def _find_root() -> Path:
    cands: list[Path] = []
    home = os.environ.get("CALLONE_HOME")
    if home:
        cands.append(Path(home))
    cands.append(Path.cwd())
    cands.extend(Path(__file__).resolve().parents)
    for c in cands:
        if (c / "callone").is_dir() and (c / "voice_clone").is_dir():
            return c
    return Path(__file__).resolve().parents[1]       # 폴백: studio 의 부모(=레포 루트)


REPO_ROOT = _find_root()                             # callone repo 루트
CALLONE_ROOT = REPO_ROOT                            # `import callone` 용 sys.path 진입점 + models/ 기준
VOICECLONE_ROOT = REPO_ROOT / "voice_clone"         # CosyVoice 제로샷 앱 + voices/
VOICES_DIR = VOICECLONE_ROOT / "voices"             # 공유 음성 프로필 저장소(둘 공용)

# 환경 선택지 → callone hardware 티어
ENV_TO_TIER = {
    "gpu": "server_gpu",
    "cpu": "laptop_cpu",
}
TIER_TO_ENV = {v: k for k, v in ENV_TO_TIER.items()}


def ensure_paths() -> None:
    """callone 패키지를 import 가능하게 sys.path 에 루트 추가."""
    p = str(CALLONE_ROOT)
    if CALLONE_ROOT.exists() and p not in sys.path:
        sys.path.insert(0, p)


@lru_cache(maxsize=1)
def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_tier(env_choice: str) -> str:
    """env_choice('auto'|'gpu'|'cpu') → callone 티어 문자열.

    'auto' 면 callone.detect_tier 로 하드웨어 감지(없으면 cuda 유무로 폴백).
    """
    choice = (env_choice or "auto").lower()
    if choice in ENV_TO_TIER:
        return ENV_TO_TIER[choice]
    # auto
    try:
        ensure_paths()
        from callone.common.hardware import detect_tier

        return detect_tier()
    except Exception:
        return "server_gpu" if has_cuda() else "laptop_cpu"


def apply_env(env_choice: str) -> str:
    """선택을 프로세스 환경에 반영(CALLONE_TIER 강제) 후 해석된 티어 반환.

    callone 내부 로더들이 CALLONE_TIER 를 읽으므로, 사용자가 GPU/CPU 를
    강제 선택하면 그대로 일관 적용된다. 'auto' 면 강제 해제.
    """
    tier = resolve_tier(env_choice)
    if (env_choice or "auto").lower() == "auto":
        os.environ.pop("CALLONE_TIER", None)
    else:
        os.environ["CALLONE_TIER"] = tier
    return tier


def cosyvoice_model_dir() -> Path | None:
    """voice_clone CosyVoice 가중치 디렉터리 탐색(제로샷 CPU/GPU TTS용).

    우선순위: 환경변수 COSYVOICE_MODEL_DIR → voice_clone/*/pretrained_models/*.
    없으면 None(→ backends 가 설치 레시피 폴백).
    """
    env = os.environ.get("COSYVOICE_MODEL_DIR")
    if env and Path(env).exists():
        return Path(env)
    for sub in ("GPU_A100", "CPU_laptop_WSL2", "."):
        base = VOICECLONE_ROOT / sub / "pretrained_models"
        if base.exists():
            for d in sorted(base.glob("*CosyVoice*")):
                if d.is_dir():
                    return d
    return None


def describe_env(env_choice: str) -> str:
    """헤더에 표시할 환경 한 줄 요약."""
    tier = resolve_tier(env_choice)
    cuda = has_cuda()
    forced = "강제" if (env_choice or "auto").lower() != "auto" else "자동감지"
    label = {"server_gpu": "GPU(server_gpu)", "laptop_cpu": "CPU(laptop_cpu)",
             "phone": "phone"}.get(tier, tier)
    return f"환경: {label} · CUDA={'O' if cuda else 'X'} · ({forced})"
