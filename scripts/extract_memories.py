"""통화 전사 → 화자 '사실(기억)' 추출 (오프라인 1회). Mem0 식 2단계의 1단계(추출·정리).

날것 발화를 그대로 RAG에 넣으면 무관한 발화를 주워섞어 삼천포가 난다. 대신 LLM으로
**원자적 사실**(거주/가족/건강/취향/사건/관계/습관 등)만 뽑아 정리해 두면, 통화 시
의미검색으로 *관련 기억만* 정밀 회상할 수 있다(rag.py).

LLM은 OpenAI 호환 엔드포인트(기본 llama-server :8080)로 호출. 추출은 구조화 작업이라
firm 프롬프트로 페르소나 편향을 억제하지만, **깨끗한 instruct 모델**(LoRA 미적용)을
띄워서 하면 사실 품질이 더 좋다.

사용:
  python scripts/extract_memories.py --speaker A
  python scripts/extract_memories.py --speaker A --base-url http://127.0.0.1:8080 --chunk 30
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callone.common.io import data_dir, read_json, write_json  # noqa: E402
from callone.common.logging import get_logger  # noqa: E402

log = get_logger("extract_memories")

_SYS = (
    "너는 통화 기록에서 '화자에 대한 사실'만 추출하는 도구다. "
    "추측·창작 절대 금지 — 발화에 명시되거나 강하게 함의된 것만. "
    "각 사실은 짧고 독립적인 한국어 평서문(예: '경상도에 산다', '아들 건강을 걱정한다'). "
    "민감해도 발화 근거가 있으면 포함. 출력은 **JSON 문자열 배열만**(설명·코드펜스 없이)."
)
_USR = (
    "다음은 한 화자의 통화 발화 모음이다. 이 사람에 대한 사실"
    "(거주·가족·건강·직업·취향·사건·관계·습관·일정 등)을 뽑아 JSON 배열로:\n\n{chunk}\n\nJSON 배열:"
)


def _chat(base_url: str, sys_p: str, usr_p: str, timeout: float = 120) -> str:
    payload = {
        "messages": [{"role": "system", "content": sys_p},
                     {"role": "user", "content": usr_p}],
        "max_tokens": 700, "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.loads(r.read().decode())
    return obj["choices"][0]["message"]["content"]


def _parse_facts(text: str) -> list[str]:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = re.search(r"\[.*\]", text, re.DOTALL)        # 첫 JSON 배열
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for f in arr:
        if isinstance(f, str) and f.strip():
            out.append(re.sub(r"\s+", " ", f.strip()))
    return out


def _dedup(facts: list[str]) -> list[str]:
    seen, out = set(), []
    for f in facts:
        key = re.sub(r"[^가-힣a-z0-9]", "", f.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(f)
    # 다른 것의 부분문자열인 사실 제거(짧은 중복 흡수)
    out.sort(key=len, reverse=True)
    kept: list[str] = []
    for f in out:
        if not any(f != g and f in g for g in kept):
            kept.append(f)
    return kept


def main():
    ap = argparse.ArgumentParser(description="화자 사실(기억) 추출")
    ap.add_argument("--speaker", default="A")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--chunk", type=int, default=30, help="청크당 발화 수")
    ap.add_argument("--max-chunks", type=int, default=0, help="0=전체")
    args = ap.parse_args()

    up = data_dir() / "speakers" / args.speaker / "utterances.json"
    if not up.exists():
        log.error("발화 없음: %s (precompute_voice_emb.py 로 생성)", up)
        sys.exit(1)
    texts = [t for t in read_json(up) if isinstance(t, str) and t.strip()]
    chunks = [texts[i:i + args.chunk] for i in range(0, len(texts), args.chunk)]
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    log.info("화자 %s: 발화 %d개 → 청크 %d개 추출 시작", args.speaker, len(texts), len(chunks))

    facts: list[str] = []
    for i, ch in enumerate(chunks, 1):
        body = "\n".join(f"- {t}" for t in ch)
        try:
            got = _parse_facts(_chat(args.base_url, _SYS, _USR.format(chunk=body)))
            facts.extend(got)
        except Exception as e:  # noqa: BLE001
            log.warning("청크 %d 실패(%s)", i, e)
        if i % 10 == 0:
            log.info("  %d/%d 청크, 누적 사실 %d", i, len(chunks), len(facts))

    facts = _dedup(facts)
    out = data_dir() / "speakers" / args.speaker / "memories.json"
    write_json(out, facts)
    log.info("완료: 사실 %d개 → %s", len(facts), out)
    # 임베딩 캐시는 rag.py 가 첫 로드 때 생성
    emb = out.with_name("memories_emb.npy")
    if emb.exists():
        emb.unlink()  # 갱신됐으니 캐시 무효화


if __name__ == "__main__":
    main()
