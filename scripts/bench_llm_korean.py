"""한국어 LLM A/B 벤치 — 한국어 품질 확인용(docs/REBUILD_PLAN.md §1 LLM).

여러 llama-server(서로 다른 모델/포트)를 **실 서빙 경로 그대로**(LlamaPersonaLLM = 작업2 프롬프트+
DRY 샘플러 포함) 고정 한국어 멀티턴 시나리오로 돌려 응답을 나란히 덤프한다. 사람이 A/B 채점
(반복·되묻기·맥락유지·자연스러움·비문). 예: A100에서 EXAONE-4.0-32B-abliterated vs EXAONE-3.5-7.8B.

프라이버시: 더미 시나리오만(실데이터 X). 디스크엔 벤치 결과 txt 만.

사용:
  # 두 서버를 각각 다른 포트로 띄운 뒤(예: 8090=7.8B, 8091=32B):
  python scripts/bench_llm_korean.py \
      --model 7.8B=http://127.0.0.1:8090 \
      --model 32B=http://127.0.0.1:8091 \
      --out bench_llm.txt
"""
from __future__ import annotations

import argparse
import sys
import time

# 고정 페르소나/상황(상황극) — 캐릭터 카드. 더미.
PERSONA = "이름은 정민, 너의 오랜 친구다. 무뚝뚝하지만 정 많은 30대. 경상도 사투리 살짝."
SITUATION = "오랜만에 전화로 안부를 묻는 상황. 편하게 반말로 대화한다."
USER_PERSONA = "고향 친구"

# 8턴 시나리오 — 맥락유지/반복/되묻기를 일부러 자극(이름·계획을 앞에서 줌 → 뒤에서 기억하나 본다).
TURNS = [
    "야 정민아 오랜만이다. 나 다음 달에 부산 내려간다.",
    "어어 그때 시간 되면 얼굴 좀 보자.",
    "요즘 일은 좀 어때? 바쁘나?",
    "나는 회사 옮겼다. 적응하느라 죽겠어.",
    "맞다, 너 저번에 이사한다 했잖아. 그건 어떻게 됐어?",
    "부산 가면 우리 그 옛날 그 국밥집 또 가자.",
    "아 맞다 내가 다음 달에 부산 간다고 했지? 며칠에 보는 게 좋겠노?",   # 1턴 정보 재확인(되묻기/반복 테스트)
    "그래 그때 보자. 몸 건강하고.",
]


def run_one(label: str, base_url: str) -> str:
    from callone.serve.llama_llm import LlamaPersonaLLM

    out = [f"\n{'='*70}\n[{label}]  {base_url}\n{'='*70}"]
    try:
        llm = LlamaPersonaLLM("bench", base_url=base_url, use_rag=False, probe=True,
                              max_new_tokens=96, temperature=0.6)
        llm.set_context(persona=PERSONA, situation=SITUATION, user_persona=USER_PERSONA)
    except Exception as e:  # noqa: BLE001
        return "\n".join(out + [f"  ✗ 서버 연결 실패: {e}"])

    history: list[dict] = []
    total_tok = 0
    total_s = 0.0
    for i, user in enumerate(TURNS, 1):
        t0 = time.time()
        parts = list(llm.chat_stream(user, history))
        dt = time.time() - t0
        reply = " ".join(p.strip() for p in parts if p.strip())
        approx_tok = max(1, len(reply))           # 한국어는 글자수 근사(정확 토큰X)
        total_tok += approx_tok
        total_s += dt
        out.append(f"\n[{i}] 나: {user}\n    {label}: {reply}\n    ({dt:.2f}s, {len(reply)}자)")
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
    out.append(f"\n  합계 {total_s:.1f}s, {total_tok}자 (~{total_tok/max(0.01,total_s):.0f}자/s)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=URL",
                    help="라벨=base_url (여러 번). 예: 32B=http://127.0.0.1:8091")
    ap.add_argument("--out", default="bench_llm.txt")
    args = ap.parse_args()
    if not args.model:
        ap.error("최소 1개 --model LABEL=URL 필요")

    results = []
    for spec in args.model:
        if "=" not in spec:
            print(f"무시(형식 LABEL=URL 아님): {spec}", file=sys.stderr)
            continue
        label, url = spec.split("=", 1)
        print(f"▶ {label} 벤치 중... ({url})", file=sys.stderr)
        results.append(run_one(label.strip(), url.strip()))

    report = "\n".join(results) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n저장: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
