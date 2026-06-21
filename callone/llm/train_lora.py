"""S5 화자 페르소나 LoRA/QLoRA 파인튜닝 (선택적·고급, mode B).

같은 SFT 데이터로 서버용·노트북용 LoRA 각각 → models/llm_server/{spk}, models/llm_phone/{spk}.
베이스/템플릿은 configs/llm_server.yaml·llm_phone.yaml 참조(EXAONE 기본). HF+QLoRA 또는 Unsloth.

무거운 의존성(transformers/peft/bitsandbytes) — H100. 미설치 시 레시피 안내.

사용:
  callone-llm-train --config llm_server --speakers A
  callone-llm-train --config llm_phone  --speakers A
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.io import data_dir, load_config
from ..common.logging import get_logger

log = get_logger("llm_train")


def load_sft(spk: str) -> list[dict]:
    p = data_dir() / "datasets" / spk / "dialogue" / "sft.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def train(cfg: dict, spk: str) -> None:
    data = load_sft(spk)
    if not data:
        log.error("SFT 데이터 없음 — callone-llm-sft 먼저")
        return
    log.info("SFT %d 샘플 로드 spk=%s", len(data), spk)

    try:
        import torch  # noqa: F401
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except Exception as e:  # noqa: BLE001
        log.error("학습 의존성 미설치(%s). pip install -e \".[heavy]\" + trl.", e)
        _print_recipe(cfg, spk)
        return

    base = cfg.get("base_model")
    out = Path(cfg.get("output_dir", "models/llm_server")) / spk
    tcfg = cfg.get("train", {})
    lcfg = cfg.get("lora", {})

    # bf16 LoRA 권장(load_in_4bit=false). EXAONE-3.5-7.8B 는 dense 라 24GB 에 bf16 LoRA 가뿐.
    # VRAM 빠듯할 때만 4bit(QLoRA).
    use_4bit = bool(cfg.get("load_in_4bit", cfg.get("method") == "qlora"))
    tok = AutoTokenizer.from_pretrained(base)
    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype="bfloat16")
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                                                     device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16",
                                                     device_map="auto")
    model.config.use_cache = False          # gradient checkpointing 과 호환
    if use_4bit:
        try:                                # 4bit 학습 준비(grad checkpointing 등)
            from peft import prepare_model_for_kbit_training  # type: ignore

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True)
        except Exception:  # noqa: BLE001
            pass

    def _reshape(messages):
        """채팅 템플릿이 받는 user/assistant 교대 형태로 정규화.
        - name(관계)은 user content 에 녹임
        - 같은 역할 연속은 병합(엄격한 user/assistant 교대 보장)
        - 선두 assistant(앞에 user 없음) 제거, 끝의 user 제거(assistant 로 끝나게)
        - user 가 하나도 없으면 None(학습에서 제외)
        반환: messages 또는 None
        """
        sys_msgs = [{"role": "system", "content": m.get("content", "")}
                    for m in messages if m["role"] == "system"]
        convo = []
        for m in messages:
            if m["role"] == "system":
                continue
            c = m.get("content", "") or ""
            if m.get("name") and m["role"] == "user":
                c = f"[{m['name']}] {c}"
            if convo and convo[-1]["role"] == m["role"]:
                convo[-1]["content"] = (convo[-1]["content"] + " " + c).strip()
            else:
                convo.append({"role": m["role"], "content": c})
        while convo and convo[0]["role"] == "assistant":
            convo.pop(0)
        while convo and convo[-1]["role"] == "user":
            convo.pop()
        if not any(m["role"] == "user" for m in convo) or not convo:
            return None
        return sys_msgs + convo

    recs = []
    for ex in data:
        r = _reshape(ex["messages"])
        if r:
            recs.append({"messages": r})
    log.info("SFT 정규화: %d → %d 샘플(템플릿 적합)", len(data), len(recs))
    if not recs:
        log.error("학습 가능한 샘플 0개 — 대화셋 점검 필요")
        return
    data = recs

    # 대화용: thinking 비활성으로 학습(지연 억제, §1-3). 템플릿이 미지원이면 fmt 에서 무시.
    enable_thinking = bool(cfg.get("enable_thinking", False))

    def fmt(ex):
        try:
            text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                           enable_thinking=enable_thinking)
        except TypeError:   # 템플릿이 enable_thinking 미지원(EXAONE/Gemma 등) → 무시
            text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return {"text": text}

    ds = Dataset.from_list(data).map(fmt)
    peft_cfg = LoraConfig(r=lcfg.get("r", 16), lora_alpha=lcfg.get("alpha", 32),
                          lora_dropout=lcfg.get("dropout", 0.05),
                          target_modules=lcfg.get("target", "all-linear"))
    # trl 버전마다 인자명이 다름 → 시그니처 검사해서 맞춰 넣기
    import inspect

    seq_len = tcfg.get("max_seq_len", 2048)
    sft_kwargs = dict(output_dir=str(out), num_train_epochs=tcfg.get("epochs", 3),
                      per_device_train_batch_size=tcfg.get("bsz", 1),
                      gradient_accumulation_steps=tcfg.get("grad_accum", 16),
                      learning_rate=tcfg.get("lr", 1e-4),
                      gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={"use_reentrant": False},
                      bf16=True, logging_steps=5, report_to=[])
    cfg_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in cfg_params:
        sft_kwargs["max_length"] = seq_len
    elif "max_seq_length" in cfg_params:
        sft_kwargs["max_seq_length"] = seq_len
    args = SFTConfig(**sft_kwargs)

    tr_kwargs = dict(model=model, args=args, train_dataset=ds, peft_config=peft_cfg)
    tr_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in tr_params:
        tr_kwargs["processing_class"] = tok
    elif "tokenizer" in tr_params:
        tr_kwargs["tokenizer"] = tok
    SFTTrainer(**tr_kwargs).train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log.info("LLM LoRA 완료 spk=%s → %s", spk, out)


def _print_recipe(cfg: dict, spk: str) -> None:
    log.info(
        "[화자 LoRA 레시피 spk=%s] (base=EXAONE 기본, callone_stack_decision §1)\n"
        "  base=%s method=%s\n"
        "  lora=%s train=%s\n"
        "  데이터: data/datasets/%s/dialogue/sft.jsonl\n"
        "  → A100 에서 trl LoRA + apply_chat_template(enable_thinking=False).\n"
        "  → 학습 후 GGUF q4_k_m 변환(scripts/make_gguf.py) → llama-server.\n"
        "  ※ 학습 없이 빠른 통화는 bootstrap_gpu.sh 의 EXAONE GGUF 직접 서빙(제로샷).",
        spk, cfg.get("base_model"), cfg.get("method"),
        cfg.get("lora"), cfg.get("train"), spk,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="S5 화자 페르소나 LoRA 파인튜닝(EXAONE 기본)")
    ap.add_argument("--config", default="llm_server", help="llm_server | llm_phone")
    ap.add_argument("--speakers", nargs="+", default=["A"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    for spk in args.speakers:
        train(cfg, spk)


if __name__ == "__main__":
    main()
