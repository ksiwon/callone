"""S5 SFT 데이터 준비 (§15.2).

대화셋 train.jsonl(7.6) → Gemma 4 채팅 템플릿 형식으로 변환.
assistant=본인, 상대 role name, TAU(선택). "모르면 모른다" 예시 보강(§15.3).

출력: data/datasets/{spk}/dialogue/sft.jsonl (학습 직행).

사용:
  callone-llm-sft --speakers A
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.io import data_dir, load_config
from ..common.logging import get_logger

log = get_logger("sft")

# "모르면 모른다" 캘리브레이션 예시 (환각 억제, §15.3)
_DONTKNOW = [
    {"q": "그거 정확히 몇 년도였더라?", "a": "글쎄, 정확히는 기억이 안 나네."},
    {"q": "내 비밀번호 뭐였지?", "a": "그건 내가 알 수가 없지."},
]


def to_chat_text(messages: list[dict], template: str = "qwen3.5") -> dict:
    """채팅 메시지 → 학습용 dict. 실제 토크나이즈는 train_lora 에서 apply_chat_template."""
    return {"messages": messages}


def run(cfg: dict, speakers: list[str]) -> None:
    add_dk = cfg.get("use_tau", True)  # 캘리브레이션도 함께
    # 결정서 §1: Qwen3.5 채택 → 기본 템플릿 qwen3.5(ChatML).
    template = cfg.get("chat_template", "qwen3.5")
    for spk in speakers:
        src = data_dir() / "datasets" / spk / "dialogue" / "train.jsonl"
        if not src.exists():
            log.error("대화셋 없음: %s — callone-build-dlg 먼저", src)
            continue
        out = src.parent / "sft.jsonl"
        n = 0
        persona = (src.parent.parent / "persona_card.txt")
        persona_txt = persona.read_text(encoding="utf-8") if persona.exists() else ""
        with open(src, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
            for line in f:
                sample = json.loads(line)
                w.write(json.dumps(to_chat_text(sample["messages"], template),
                                   ensure_ascii=False) + "\n")
                n += 1
            # 캘리브레이션 예시 추가
            if add_dk and persona_txt:
                for ex in _DONTKNOW:
                    msgs = [
                        {"role": "system", "content": persona_txt},
                        {"role": "user", "content": ex["q"]},
                        {"role": "assistant", "content": ex["a"]},
                    ]
                    w.write(json.dumps(to_chat_text(msgs, template), ensure_ascii=False) + "\n")
                    n += 1
        log.info("SFT 준비 %s: %d 샘플 → %s", spk, n, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="S5 SFT 데이터 준비")
    ap.add_argument("--config", default="llm_server")
    ap.add_argument("--speakers", nargs="+", default=["A"])
    args = ap.parse_args()
    run(load_config(args.config), args.speakers)


if __name__ == "__main__":
    main()
