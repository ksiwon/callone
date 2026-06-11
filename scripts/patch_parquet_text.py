"""global_assignment.parquet 의 빈 text 를 diarized JSON 에서 백필.

원인: S2b link 가 S3 transcribe 보다 먼저 돌면 parquet 의 text 가 빈 채로 저장됨.
이 스크립트는 segment_uid(call_id#idx) 로 diarized 세그먼트 text 를 매핑해 채운다.
임베딩/A·B 귀속은 그대로(재계산 없음).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from callone.common.io import data_dir, read_json  # noqa: E402
from callone.common.logging import get_logger  # noqa: E402

log = get_logger("patch_text")


def main():
    ga = data_dir() / "speakers" / "global_assignment.parquet"
    df = pd.read_parquet(ga)
    diar_dir = data_dir() / "diarized"

    cache: dict[str, list] = {}

    def seg_text(uid: str) -> str:
        call_id, _, idx = uid.rpartition("#")
        if call_id not in cache:
            jp = diar_dir / f"{call_id}.json"
            cache[call_id] = read_json(jp)["segments"] if jp.exists() else []
        segs = cache[call_id]
        try:
            return (segs[int(idx)].get("text") or "").strip()
        except (ValueError, IndexError):
            return ""

    df["text"] = df["segment_uid"].map(seg_text)
    filled = (df["text"].str.strip() != "").sum()
    df.to_parquet(ga, index=False)
    log.info("parquet text 백필 완료: %d/%d 행에 텍스트", filled, len(df))


if __name__ == "__main__":
    main()
