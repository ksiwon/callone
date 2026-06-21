"""LoRA → Qwen3.5-4B 병합 → GGUF(Q4_K_M) 변환 (llama.cpp 배포용).

OpenVINO 가 qwen3_5 변환을 못 하므로(아키텍처 GDN+MoE+MTP, optimum-intel #1628),
llama.cpp 로 간다. 이 스크립트는 '병합본 단일 GGUF' 경로(품질 최고).
런타임 LoRA(9GB 다운로드 회피) 경로는 scripts/run_llama_server.md 참고.

필요(오프라인 1회, torch OK — 서빙과 별도):
  pip install "transformers>=5.10" peft torch
  git clone https://github.com/ggml-org/llama.cpp   # convert/quantize 도구

사용:
  python scripts/make_gguf.py --speaker A --llama-cpp C:/tools/llama.cpp
  python scripts/make_gguf.py --speaker A --base Qwen/Qwen3.5-4B --quant Q4_K_M
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

log = get_logger("make_gguf")


def find_lora(root: Path) -> Path | None:
    if (root / "adapter_config.json").exists():
        return root
    for cp in reversed(sorted(root.glob("checkpoint-*"), key=lambda p: p.name)):
        if (cp / "adapter_config.json").exists():
            return cp
    return None


def merge(base: str, lora_dir: Path, merged_dir: Path):
    import torch  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("base 로드: %s", base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
    log.info("LoRA 병합: %s", lora_dir)
    model = PeftModel.from_pretrained(model, str(lora_dir)).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(base).save_pretrained(str(merged_dir))
    # MTP(multi-token prediction) 레이어 가중치는 텍스트 CausalLM 추출 시 빠지는데
    # config 의 mtp_num_hidden_layers 가 남아있으면 GGUF 변환이 blk.N(=32) 을 더 기대해
    # 'missing tensor blk.32...' 로 죽는다 → 0 으로 꺼서 block_count 를 실제 레이어수에 맞춘다.
    import json as _json
    cfg_p = merged_dir / "config.json"
    try:
        cc = _json.load(open(cfg_p))
        if cc.get("mtp_num_hidden_layers"):
            cc["mtp_num_hidden_layers"] = 0
            _json.dump(cc, open(cfg_p, "w"), indent=2)
            log.info("config mtp_num_hidden_layers→0 (MTP 미사용)")
    except Exception as e:  # noqa: BLE001
        log.warning("config mtp 패치 실패(%s)", e)
    log.info("병합 저장: %s", merged_dir)


def to_gguf(merged_dir: Path, llama_cpp: Path, out_dir: Path, speaker: str, quant: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / f"qwen3.5-4b-{speaker}-f16.gguf"
    conv = llama_cpp / "convert_hf_to_gguf.py"
    if not conv.exists():
        log.error("convert_hf_to_gguf.py 없음: %s — git clone llama.cpp 후 --llama-cpp 지정", conv)
        sys.exit(1)
    log.info("GGUF(f16) 변환: %s", f16)
    subprocess.run([sys.executable, str(conv), str(merged_dir),
                    "--outfile", str(f16), "--outtype", "f16"], check=True)
    if quant.lower() in ("f16", "none"):
        log.info("완료(f16): %s", f16)
        return f16
    # llama-quantize 바이너리 탐색(빌드 위치 다양)
    qbin = None
    for cand in ("llama-quantize", "llama-quantize.exe",
                 "build/bin/llama-quantize", "build/bin/llama-quantize.exe"):
        p = llama_cpp / cand
        if p.exists():
            qbin = p
            break
    qout = out_dir / f"qwen3.5-4b-{speaker}-{quant}.gguf"
    if qbin is None:
        log.warning("llama-quantize 못 찾음 — f16 유지(%s). 수동 양자화: llama-quantize %s %s %s",
                    f16, f16, qout, quant)
        return f16
    log.info("양자화 %s: %s", quant, qout)
    subprocess.run([str(qbin), str(f16), str(qout), quant], check=True)
    f16.unlink(missing_ok=True)  # f16 큰 중간물 제거
    log.info("완료: %s", qout)
    return qout


def main():
    ap = argparse.ArgumentParser(description="LoRA 병합 → GGUF 변환")
    ap.add_argument("--speaker", default="A")
    ap.add_argument("--config", default="llm_server",
                    help="llm_server(EXAONE-7.8B 서버) | llm_phone(Qwen3.5-4B 노트북)")
    ap.add_argument("--base", default=None, help="기본: config 의 base_model")
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp 저장소 경로(convert/quantize)")
    ap.add_argument("--out", default="models_gguf")
    ap.add_argument("--quant", default="Q4_K_M", help="Q4_K_M | Q5_K_M | Q8_0 | f16")
    ap.add_argument("--keep-merged", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = args.base or cfg.get("base_model", "Qwen/Qwen3.5-4B")
    root = Path(cfg.get("output_dir", "models/llm_phone")) / args.speaker
    lora_dir = find_lora(root)
    if lora_dir is None:
        log.error("LoRA adapter 없음: %s (또는 checkpoint-*/)", root)
        sys.exit(1)
    log.info("LoRA: %s | base: %s", lora_dir, base)

    merged_dir = Path("models/llm_merged") / args.speaker
    merge(base, lora_dir, merged_dir)
    out = to_gguf(merged_dir, Path(args.llama_cpp), Path(args.out), args.speaker, args.quant)
    if not args.keep_merged:
        shutil.rmtree(merged_dir, ignore_errors=True)
    log.info("배포 GGUF: %s  → llama-server -m 로 구동(run_llama_server.md)", out)


if __name__ == "__main__":
    main()
