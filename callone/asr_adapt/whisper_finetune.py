"""S13 Whisper 방언 적응 파인튜닝 (§13).

교정 CSV → Whisper large-v3 LoRA/파인튜닝(transformers).
그 특정 사투리 + 두 화자 + 전화 음질에 동시 적응.
보강(선택): AIHub 방언 + 저음질 통화 데이터.

출력: models/asr_dialect/. 수용기준: held-out WER 개선.

무거운 의존성(transformers/peft) 필요 — H100 학습 노드. 미설치 시 안내만.

사용:
  callone-asr-train --csv data/datasets/asr_correction/to_correct.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..common.io import load_config
from ..common.logging import get_logger

log = get_logger("asr_train")


def load_correction_csv(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            txt = (r.get("corrected_text") or "").strip()
            if txt and Path(r["wav_clip"]).exists():
                rows.append({"audio": r["wav_clip"], "text": txt})
    return rows


def train(cfg: dict, csv_path: str) -> None:
    rows = load_correction_csv(csv_path)
    if not rows:
        log.error("교정 완료 행 없음 — to_correct.csv 의 corrected_text 칸을 먼저 채우세요")
        return
    log.info("교정셋 %d 행 로드", len(rows))

    try:
        import torch  # noqa: F401
        from datasets import Audio, Dataset  # type: ignore
        from peft import LoraConfig, get_peft_model  # type: ignore
        from transformers import (  # type: ignore
            Seq2SeqTrainer, Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration, WhisperProcessor,
        )
    except Exception as e:  # noqa: BLE001
        log.error("학습 의존성 미설치(%s). 설치: pip install -e \".[heavy]\"", e)
        log.error("H100 학습 노드에서 실행 권장. 이 단계는 데이터 풍부 시 핵심(§13).")
        return

    base = cfg.get("base_model", "openai/whisper-large-v3")
    out_dir = cfg.get("output_dir", "models/asr_dialect")
    processor = WhisperProcessor.from_pretrained(base, language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(base)

    ft = cfg.get("finetune", {})
    if ft.get("method", "lora") == "lora":
        lc = ft.get("lora", {})
        model = get_peft_model(model, LoraConfig(
            r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32),
            lora_dropout=lc.get("dropout", 0.05),
            target_modules=["q_proj", "v_proj"], bias="none",
        ))

    ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))

    def prep(b):
        a = b["audio"]
        b["input_features"] = processor(a["array"], sampling_rate=16000,
                                        return_tensors="pt").input_features[0]
        b["labels"] = processor.tokenizer(b["text"]).input_ids
        return b

    ds = ds.map(prep, remove_columns=ds.column_names)

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=ft.get("batch_size", 8),
        gradient_accumulation_steps=ft.get("grad_accum", 2),
        learning_rate=ft.get("lr", 1e-4), num_train_epochs=ft.get("epochs", 3),
        fp16=True, logging_steps=10, save_strategy="epoch", report_to=[],
    )
    Seq2SeqTrainer(model=model, args=args, train_dataset=ds,
                   tokenizer=processor.feature_extractor).train()
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    log.info("ASR 적응 완료 → %s", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Whisper 방언 적응")
    ap.add_argument("--config", default="asr_adapt")
    ap.add_argument("--csv", default="data/datasets/asr_correction/to_correct.csv")
    args = ap.parse_args()
    train(load_config(args.config), args.csv)


if __name__ == "__main__":
    main()
