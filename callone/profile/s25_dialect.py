"""S2.5 방언 자동 프로파일링 (§11.1).

⚠️ 방언 지역·세기를 미리 정의하지 않고 데이터에서 화자별 측정.

방법:
 1) 사투리 마커 사전(resources/dialect_markers/{region}.json) 로드
    (AIHub 방언↔표준어 대응쌍으로 확장 가능; 없으면 규칙 시드).
 2) 지역 추정: 화자 전사에서 지역 마커 출현 빈도 → 최빈 지역 + softmax confidence.
    제주는 데이터 없음 → 후보 기본 제외(폴백만).
 3) 세기: 마커 토큰 수 / 전체 토큰 수 → intensity_0to1. 같은 지역도 사람마다 다름.

출력: profile.json 의 auto.dialect. (CLI 는 화자별 전사 텍스트 집계 후 측정)
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from ..common.io import REPO_ROOT, data_dir, load_config, read_json, write_json
from ..common.logging import get_logger
from ..common.schemas import DialectAuto, DialectMarker

log = get_logger("s25_dialect")


def load_markers(markers_dir: str | Path, regions: list[str]) -> dict[str, list[dict]]:
    base = Path(markers_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    out = {}
    for r in regions:
        fp = base / f"{r}.json"
        if fp.exists():
            out[r] = read_json(fp).get("markers", [])
        else:
            log.warning("마커 사전 없음: %s", fp)
            out[r] = []
    return out


def _count_markers(text: str, markers: list[dict]) -> tuple[int, list[DialectMarker]]:
    total = 0
    found = []
    for m in markers:
        form = m["form"]
        # 어미(~) 패턴은 접미 매칭, 아니면 부분 매칭
        pat = re.escape(form.lstrip("~"))
        cnt = len(re.findall(pat, text))
        if cnt > 0:
            total += cnt
            found.append(DialectMarker(form=form, count=cnt, std=m.get("std", "")))
    return total, found


def profile_dialect(text: str, cfg: dict, examples_src: list[str] | None = None) -> DialectAuto:
    dcfg = cfg.get("dialect", {})
    regions = dcfg.get("regions", ["gyeongsang", "jeolla", "chungcheong", "gangwon"])
    markers = load_markers(dcfg.get("markers_dir", "resources/dialect_markers"), regions)

    n_tokens = max(1, len(text.split()))
    region_scores = {}
    region_markers = {}
    for r in regions:
        cnt, found = _count_markers(text, markers[r])
        region_scores[r] = cnt
        region_markers[r] = found

    if sum(region_scores.values()) == 0 or n_tokens < dcfg.get("min_tokens", 50):
        # 신호 부족 → 표준어 폴백
        return DialectAuto(region_est="standard", confidence=0.0, intensity_0to1=0.0)

    # softmax confidence
    best = max(region_scores, key=region_scores.get)
    vals = list(region_scores.values())
    exp = [math.exp(v - max(vals)) for v in vals]
    conf = exp[list(region_scores).index(best)] / sum(exp)

    # intensity = 마커 토큰 수 / 전체 토큰 수 (best 지역 기준)
    intensity = min(1.0, region_scores[best] / n_tokens)

    # 대표 예문
    examples = []
    if examples_src:
        marker_forms = [m.form.lstrip("~") for m in region_markers[best]]
        for s in examples_src:
            if any(mf and mf in s for mf in marker_forms):
                examples.append(s)
            if len(examples) >= 5:
                break

    return DialectAuto(
        region_est=best, confidence=round(conf, 3),
        intensity_0to1=round(intensity, 3),
        markers=sorted(region_markers[best], key=lambda m: -m.count)[:20],
        examples=examples,
    )


def collect_speaker_text(speaker_id: str) -> tuple[str, list[str]]:
    """global_assignment 에서 해당 화자 전사 텍스트 집계."""
    ga = data_dir() / "speakers" / "global_assignment.parquet"
    texts = []
    if ga.exists():
        import pandas as pd

        df = pd.read_parquet(ga)
        texts = df[(df["global_speaker"] == speaker_id) & (df["clean"])]["text"].dropna().tolist()
    else:
        jp = ga.with_suffix(".json")
        if jp.exists():
            for r in read_json(jp):
                if r["global_speaker"] == speaker_id and r.get("clean"):
                    texts.append(r.get("text", ""))
    texts = [t for t in texts if t.strip()]
    return " ".join(texts), texts


def run(cfg: dict, speakers: list[str]) -> None:
    for sid in speakers:
        text, utts = collect_speaker_text(sid)
        da = profile_dialect(text, cfg, examples_src=utts)
        out = data_dir() / "speakers" / sid / "dialect_auto.json"
        write_json(out, da)
        log.info("화자 %s: 지역=%s conf=%.2f 세기=%.2f (토큰 기반)",
                 sid, da.region_est, da.confidence, da.intensity_0to1)


def main() -> None:
    ap = argparse.ArgumentParser(description="S2.5 방언 자동 프로파일링")
    ap.add_argument("--config", default="s25_profile")
    ap.add_argument("--speakers", nargs="+", default=["A", "B"])
    args = ap.parse_args()
    run(load_config(args.config), args.speakers)


if __name__ == "__main__":
    main()
