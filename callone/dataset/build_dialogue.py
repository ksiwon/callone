"""S3(b) 대화 학습셋 빌드 (§12b, §7.6).

시간순 멀티턴 복원. assistant=클론 대상(spk), 상대=user(관계 name).
선택 TAU(Think-Aloud) 증강. → data/datasets/{spk}/dialogue/train.jsonl.
각 샘플 system=페르소나 카드. PII 마스킹 적용.

사용:
  callone-build-dlg --speakers A B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.io import data_dir, load_config, read_json
from ..common.logging import get_logger
from ..common.schemas import ChatMessage, DialogueSample, SpeakerProfile
from ..asr.pii import mask_text
from .persona_card import build_persona_card

log = get_logger("build_dlg")


def _ordered_segments():
    """전 통화 세그먼트를 (call_id, start) 순으로 정렬."""
    ga = data_dir() / "speakers" / "global_assignment.parquet"
    if ga.exists():
        import pandas as pd

        df = pd.read_parquet(ga).sort_values(["call_id", "start"])
        return df.to_dict("records")
    if ga.with_suffix(".json").exists():
        rows = read_json(ga.with_suffix(".json"))
        return sorted(rows, key=lambda r: (r["call_id"], r["start"]))
    return []


def _load_profile(spk: str) -> SpeakerProfile:
    pj = data_dir() / "speakers" / spk / "profile.json"
    if pj.exists():
        return SpeakerProfile(**read_json(pj))
    return SpeakerProfile(speaker_id=spk)


def run(cfg: dict, speakers: list[str]) -> None:
    dcfg = cfg.get("dialogue", {})
    max_gap = dcfg.get("max_turn_gap_sec", 30)
    min_turns = dcfg.get("min_turns", 2)
    use_tau = dcfg.get("use_tau", False)
    pii_tokens = cfg.get("pii", {}).get("mask_tokens")
    pii_on = cfg.get("pii", {}).get("enabled", True)
    rows = _ordered_segments()

    for spk in speakers:
        other = "B" if spk == "A" else "A"
        prof = _load_profile(spk)
        other_prof = _load_profile(other)
        persona = build_persona_card(prof)
        relation_name = other_prof.user.relation or other_prof.user.name or "상대"

        out_dir = data_dir() / "datasets" / spk / "dialogue"
        out_dir.mkdir(parents=True, exist_ok=True)
        jl = out_dir / "train.jsonl"

        # call 별로 turn 시퀀스 만들기
        samples = []
        cur_call, msgs, last_end = None, [], None
        for r in rows:
            gs = r["global_speaker"]
            if gs not in (spk, other):
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue
            if pii_on:
                text = mask_text(text, pii_tokens)
            # 새 통화 또는 큰 공백 → 대화 분할
            if cur_call != r["call_id"] or (last_end is not None and r["start"] - last_end > max_gap):
                if len([m for m in msgs if m.role == "assistant"]) >= 1 and len(msgs) >= min_turns:
                    samples.append(_finalize(persona, msgs))
                msgs, cur_call = [], r["call_id"]
            if gs == spk:
                content = text
                if use_tau:
                    content = f"<thinking>평소처럼 자연스럽게 반응</thinking>{text}"
                msgs.append(ChatMessage(role="assistant", content=content))
            else:
                msgs.append(ChatMessage(role="user", name=relation_name, content=text))
            last_end = r["end"]
        if len([m for m in msgs if m.role == "assistant"]) >= 1 and len(msgs) >= min_turns:
            samples.append(_finalize(persona, msgs))

        with open(jl, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.model_dump(exclude_none=True), ensure_ascii=False) + "\n")
        log.info("화자 %s 대화셋: %d 샘플 → %s", spk, len(samples), jl)

        # 페르소나 카드도 파일로 저장
        (out_dir.parent / "persona_card.txt").write_text(persona, encoding="utf-8")


def _finalize(persona: str, msgs: list[ChatMessage]) -> DialogueSample:
    return DialogueSample(messages=[ChatMessage(role="system", content=persona), *msgs])


def main() -> None:
    ap = argparse.ArgumentParser(description="S3(b) 대화 학습셋 빌드")
    ap.add_argument("--config", default="s3_dataset")
    ap.add_argument("--speakers", nargs="+", default=["A", "B"])
    args = ap.parse_args()
    run(load_config(args.config), args.speakers)


if __name__ == "__main__":
    main()
