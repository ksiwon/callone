"""S4 서버 TTS 파인튜닝 (§14).

Qwen3-TTS 로컬 파인튜닝(한국어·스트리밍) 또는 VoxCPM2(48k LoRA).
⚠️ Qwen3-TTS 로컬 가중치만 — Alibaba 클라우드 보이스 API 호출 금지(§2).
사투리는 음성 학습으로 자동 반영(전사가 사투리 보존이면 됨).

입력: data/datasets/{spk}/tts/metadata.csv. 출력: models/tts_server/{spk}.

백엔드 repo 무거움 → 미설치 시 학습 레시피/명령 안내 출력(폴백).

사용:
  callone-tts-train --speakers A
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..common.io import data_dir, load_config
from ..common.logging import get_logger

log = get_logger("tts_server")


def _check_dataset(spk: str) -> Path | None:
    csv = data_dir() / "datasets" / spk / "tts" / "metadata.csv"
    if not csv.exists() or csv.stat().st_size == 0:
        log.error("TTS셋 없음/빈 파일: %s — callone-build-tts 먼저", csv)
        return None
    return csv


def train_qwen3(spk: str, csv: Path, cfg: dict) -> None:
    out = Path(cfg.get("output_dir", "models/tts_server")) / spk
    out.mkdir(parents=True, exist_ok=True)
    try:
        # Qwen3-TTS 로컬 파인튜닝 진입점 (repo 설치 시)
        import qwen_tts  # type: ignore  # noqa: F401
        log.info("Qwen3-TTS 파인튜닝 시작 spk=%s → %s", spk, out)
        # repo 의 finetune API 호출 (버전별 상이 — README 참조)
        raise NotImplementedError("Qwen3-TTS repo finetune API 연결 지점")
    except Exception as e:  # noqa: BLE001
        log.warning("Qwen3-TTS 직접 학습 미연결(%s).", e)
        _print_recipe(spk, csv, cfg, backend="qwen3-tts")


def train_voxcpm(spk: str, csv: Path, cfg: dict) -> None:
    _print_recipe(spk, csv, cfg, backend="voxcpm2")


def _print_recipe(spk: str, csv: Path, cfg: dict, backend: str) -> None:
    out = Path(cfg.get("output_dir", "models/tts_server")) / spk
    ft = cfg.get("finetune", {})
    log.info(
        "[%s 학습 레시피 spk=%s] (callone_stack_decision §2-4)\n"
        "  데이터: %s — ⚠️ 반드시 %sHz 리샘플(미리샘플 시 학습 크래시)\n"
        "  방법: LoRA r=%s alpha=%s, epochs=%s, lr=%s (⚠️ 2e-5 금지, 2e-6 고정), bsz=%s\n"
        "  사전: annotation 태그 제거 / 각 클립 끝 1초 묵음 / commit 680d4e9+ / PR#178 확인\n"
        "  출력: %s  · 추론 lora_scale=%s (0.2/0.3/0.35/0.5 스윕)\n"
        "  → A100 에서 QwenLM/Qwen3-TTS sft_12hz.py. model_id=로컬 가중치. 클라우드 API 금지(§2).",
        backend, spk, csv, cfg.get("sample_rate"),
        ft.get("lora_r"), ft.get("lora_alpha"), ft.get("epochs"), ft.get("lr"),
        ft.get("batch_size"), out, cfg.get("inference", {}).get("lora_scale"),
    )


def run(cfg: dict, speakers: list[str]) -> None:
    backend = cfg.get("backend", "qwen3-tts")
    for spk in speakers:
        csv = _check_dataset(spk)
        if not csv:
            continue
        if backend == "qwen3-tts":
            train_qwen3(spk, csv, cfg)
        else:
            train_voxcpm(spk, csv, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 서버 TTS 파인튜닝")
    ap.add_argument("--config", default="tts_server")
    ap.add_argument("--speakers", nargs="+", default=["A"])
    args = ap.parse_args()
    run(load_config(args.config), args.speakers)


if __name__ == "__main__":
    main()
