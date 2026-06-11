"""설정 로딩 + 경로 관리 + 디스크 IO 헬퍼.

설정 병합: configs/default.yaml <- 스테이지 yaml <- CLI override.
모든 경로는 CALLONE_DATA_DIR 기준으로 해석.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()  # .env 로드 (있으면)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def data_dir() -> Path:
    d = Path(os.environ.get("CALLONE_DATA_DIR", REPO_ROOT / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def device() -> str:
    """`.env` 의 DEVICE 원시값 (cuda|cpu|mps). 기본 cuda."""
    return os.environ.get("DEVICE", "cuda")


_RESOLVED_DEVICE: str | None = None


def resolve_device() -> str:
    """실제 사용 가능한 디바이스 반환 — 로컬/서버 자동 적응.

    DEVICE=cuda 라도 torch.cuda 가 없으면 자동으로 cpu 로 강등(경고 1회).
    torch 미설치면 cpu. 결과 캐시.
    """
    global _RESOLVED_DEVICE
    if _RESOLVED_DEVICE is not None:
        return _RESOLVED_DEVICE
    want = device()
    resolved = "cpu"
    try:
        import torch

        if want == "cuda" and torch.cuda.is_available():
            resolved = "cuda"
        elif want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            resolved = "mps"
        elif want == "cuda":
            from .logging import get_logger

            get_logger("io").warning("DEVICE=cuda 지만 GPU 없음 → cpu 로 자동 전환(로컬 모드)")
    except Exception:
        pass
    _RESOLVED_DEVICE = resolved
    return resolved


def compute_type_for(dev: str | None = None) -> str:
    """디바이스별 권장 연산 정밀도 (faster-whisper 등). GPU=float16, CPU=int8."""
    dev = dev or resolve_device()
    return "float16" if dev == "cuda" else "int8"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(stage: str | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """default.yaml + 스테이지 yaml + override 병합.

    stage: "s0_ingest" 처럼 configs/{stage}.yaml 을 가리킴.
    """
    cfg: dict[str, Any] = {}
    default_path = CONFIG_DIR / "default.yaml"
    if default_path.exists():
        cfg = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    if stage:
        sp = CONFIG_DIR / f"{stage}.yaml"
        if sp.exists():
            cfg = _deep_merge(cfg, yaml.safe_load(sp.read_text(encoding="utf-8")) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
