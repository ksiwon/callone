"""클론 응답 검증 배터리 — 말이 되는가 + 기억을 적재적소에만 쓰는가.

여러 유형(인사·일상·감정·기억질문·무관한질문)을 클론에 넣어 응답을 보여준다. 사람이
직접 읽고 ① 이해 가능한가 ② 일상 질문엔 기억 안 끌고 자연스럽게 대화하나 ③ 기억질문엔
관련 기억을 떠올리나 를 판단. 각 응답에 '주입된 기억'과 자동 경고(반복·파편)도 표시.

전제: llama-server 8080 가동. 사용:
  python scripts/eval_clone.py --speaker A
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (유형, 입력) — 기억질문은 추출된 실제 맥락(학생회/일정 등)에 맞춰 일반화
CASES = [
    ("인사", "여보세요"),
    ("인사", "나 왔어"),
    ("일상", "밥 먹었어?"),
    ("일상", "오늘 날씨 좋더라"),
    ("감정", "나 오늘 좀 힘들었어"),
    ("감정", "보고 싶었어"),
    ("무관", "내일 비 온대?"),
    ("무관", "주말에 뭐 할까"),
    ("기억질문", "그 학생회 일은 어떻게 됐어?"),
    ("기억질문", "요즘 일은 몇 시까지 해?"),
    ("기억질문", "동아리 애들은 잘 지내?"),
]


def _warn(text: str) -> str:
    flags = []
    if re.search(r"(.{1,6})\1{3,}", text):           # 같은 조각 4회+ 반복
        flags.append("반복")
    if len(text.strip()) < 3:
        flags.append("너무짧음")
    if text.count(" ") > 40:
        flags.append("너무김")
    return (" ⚠️ " + ",".join(flags)) if flags else ""


def main():
    ap = argparse.ArgumentParser(description="클론 응답 검증 배터리")
    ap.add_argument("--speaker", default="A")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = ap.parse_args()

    from callone.common.io import load_config
    from callone.serve.llama_llm import LlamaPersonaLLM

    c = load_config("serve").get("llm", {})       # 실제 배포 설정으로 검증
    m = LlamaPersonaLLM(args.speaker, base_url=args.base_url, use_rag=True,
                        max_new_tokens=int(c.get("max_new_tokens", 64)),
                        temperature=float(c.get("temperature", 0.4)),
                        rag_cfg=c)
    print(f"=== 클론 검증: 화자 {args.speaker} (use_rag=ON, 게이트 회상) ===\n")
    for kind, q in CASES:
        mem = m._rag.context(q, k=3) if m._rag else ""    # 게이트 통과한 기억(없으면 빈값)
        reply = m.chat(q)
        print(f"[{kind}] 나: {q}")
        if mem:
            print("   ↳ 떠올린 기억:")
            for line in mem.splitlines():
                print(f"      {line}")
        else:
            print("   ↳ (기억 안 씀 — 일상 대화)")
        print(f"   클론: {reply}{_warn(reply)}\n")

    print("판단 기준:")
    print("  ① 모든 응답이 '이해 가능'한가? (방언·반말이어도 뜻이 통해야)")
    print("  ② 인사·일상·무관 질문엔 '기억 안 씀'이 떠야 정상(게이트 작동)")
    print("  ③ 기억질문엔 관련 기억이 떠오르고 그걸 자연스럽게 녹였나")
    print("  ⚠️ 표시는 자동 경고 — 반복/파편이면 temperature·rag_min_score 조정")


if __name__ == "__main__":
    main()
