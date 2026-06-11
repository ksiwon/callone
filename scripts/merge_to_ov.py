"""LoRA 어댑터 → base 병합 → OpenVINO int4 변환 (노트북 배포용).

학습 산출물 models/llm_phone/{spk}(LoRA) 를 Qwen3.5-4B 에 병합하고
OpenVINO IR(int4) 로 내보낸다. 결과를 노트북으로 복사해 OVPersonaLLM 으로 구동.

필요: transformers, peft, optimum-intel (학습 venv 에 있음; optimum-intel 추가 설치 가능)
  pip install "optimum-intel[openvino]"

사용:
  python scripts/merge_to_ov.py --speaker A
  python scripts/merge_to_ov.py --speaker A --base Qwen/Qwen3.5-4B --bits int4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo 루트 → callone import

from callone.common.io import load_config  # noqa: E402
from callone.common.logging import get_logger  # noqa: E402

log = get_logger("merge_to_ov")


def merge(speaker: str, base: str, lora_dir: Path, merged_dir: Path):
    import torch  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("base 로드: %s", base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
    log.info("LoRA 병합: %s", lora_dir)
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(base).save_pretrained(str(merged_dir))
    log.info("병합 저장: %s", merged_dir)


def to_openvino(merged_dir: Path, ov_dir: Path, bits: str):
    # optimum-cli 로 OV 변환(int4 가중치). 실패 시 명령 안내.
    ov_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["optimum-cli", "export", "openvino", "--model", str(merged_dir),
           "--weight-format", bits, "--task", "text-generation-with-past", str(ov_dir)]
    log.info("OV 변환: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        log.info("OV 변환 완료: %s", ov_dir)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.error("optimum-cli 실패(%s). 설치: pip install \"optimum-intel[openvino]\"", e)
        log.error("수동: %s", " ".join(cmd))


def main():
    ap = argparse.ArgumentParser(description="LoRA 병합 + OpenVINO 변환")
    ap.add_argument("--speaker", default="A")
    ap.add_argument("--config", default="llm_phone")
    ap.add_argument("--base", default=None)
    ap.add_argument("--bits", default="int4", help="int4 | int8 | fp16")
    ap.add_argument("--keep-merged", action="store_true", help="병합본(fp) 보존")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = args.base or cfg.get("base_model", "Qwen/Qwen3.5-4B")
    root = Path(cfg.get("output_dir", "models/llm_phone")) / args.speaker
    # 어댑터 위치: 루트(adapter_config.json) 또는 checkpoint-*/ 하위
    lora_dir = None
    if (root / "adapter_config.json").exists():
        lora_dir = root
    else:
        cps = sorted(root.glob("checkpoint-*"), key=lambda p: p.name)
        for cp in reversed(cps):                 # 최신 체크포인트 우선
            if (cp / "adapter_config.json").exists():
                lora_dir = cp
                break
    if lora_dir is None:
        log.error("LoRA adapter 없음: %s (또는 checkpoint-*/) — 먼저 callone-llm-train", root)
        sys.exit(1)
    log.info("LoRA: %s", lora_dir)

    merged_dir = Path("models/llm_merged") / args.speaker
    ov_dir = Path("models/llm_ov") / args.speaker

    merge(args.speaker, base, lora_dir, merged_dir)
    to_openvino(merged_dir, ov_dir, args.bits)
    if not args.keep_merged:
        shutil.rmtree(merged_dir, ignore_errors=True)
    log.info("완료 → 노트북으로 복사: %s  (OVPersonaLLM model_dir 로 사용)", ov_dir)


if __name__ == "__main__":
    main()
