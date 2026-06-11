"""S2 (b)(c)(d) 전역 화자 연결 CLI (§10).

(b) 세그먼트별 임베딩 → (c) 전 통화 A/B 센트로이드 → 전역 귀속 →
(d) 제3자 이상치 제거. 출력: data/speakers/global_assignment.parquet (§7.3).

사용:
  callone-link [--limit 50]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..common import db
from ..common.audio import cosine, load_wav
from ..common.io import data_dir, load_config, read_json, write_json
from ..common.logging import get_logger
from ..common.schemas import GlobalAssignment
from .embeddings import embed_waveform
from .filter_thirdparty import assign_AB, two_main_centroids

log = get_logger("s2b")


def _segment_embeddings(cfg: dict, limit: int | None):
    """모든 통화 세그먼트 → 임베딩 행렬 + 메타."""
    con = db.connect()
    calls = [c for c in db.all_calls(con) if c.status == "ok"]
    if limit:
        calls = calls[:limit]
    diar_dir = data_dir() / "diarized"

    rows, embs = [], []
    total = len(calls)
    step = max(1, total // 50)
    for ci, c in enumerate(calls, 1):
        if ci % step == 0 or ci == total:
            log.info("임베딩 진행 %d/%d 통화 (세그먼트 %d개)", ci, total, len(embs))
        jp = diar_dir / f"{c.call_id}.json"
        if not jp.exists():
            continue
        dc = read_json(jp)
        wav = c.restored_path or c.wav16k_path
        if not wav or not Path(wav).exists():
            continue
        y, sr = load_wav(wav, sr=16000)
        for i, seg in enumerate(dc["segments"]):
            s, e = seg["start"], seg["end"]
            clip = y[int(s * sr): int(e * sr)]
            if clip.size < sr * 0.3:
                continue
            try:
                emb = embed_waveform(clip, sr)
            except Exception:  # noqa: BLE001
                continue
            embs.append(emb)
            rows.append({
                "segment_uid": f"{c.call_id}#{i}",
                "call_id": c.call_id, "start": s, "end": e,
                "local_speaker": seg.get("local_speaker", ""),
                "text": seg.get("text", ""),
                "snr_db": seg.get("snr_db", 0.0),
                "is_overlap": seg.get("overlap", False),
            })
    if not embs:
        return rows, np.zeros((0, 1))
    # 차원 통일 위해 최소 길이 자르기 (폴백 임베딩 대비)
    dim = min(e.shape[0] for e in embs)
    mat = np.stack([e[:dim] for e in embs])
    return rows, mat


def run(cfg: dict, limit: int | None = None) -> None:
    rows, mat = _segment_embeddings(cfg, limit)
    if len(rows) == 0:
        log.warning("세그먼트 없음 — S2(a) 먼저 실행 필요")
        return

    link = cfg.get("link", {})
    cA, cB, labels = two_main_centroids(mat, method=link.get("cluster", "hdbscan"))
    sim_thr = float(link.get("sim_threshold", 0.55))
    min_snr = cfg.get("filters", {}).get("min_snr_db", 10)

    # ⚠️ A/B 귀속은 클러스터 라벨(0/1)을 직접 사용. 라벨 -1(HDBSCAN 이상치)만 제3자.
    #   (고정 코사인 임계값으로 재판정하면 평균 센트로이드 코사인이 낮아 대량 오분류됨)
    assigns = []
    for idx, (r, emb) in enumerate(zip(rows, mat)):
        lbl = int(labels[idx]) if idx < len(labels) else 0
        sa, sb = cosine(emb, cA), cosine(emb, cB)
        if lbl == -1:                       # HDBSCAN 이상치 = 제3자
            gs, is_3p = "UNK", True
        elif lbl == 1:
            gs, is_3p = "B", False
        else:                               # lbl == 0
            gs, is_3p = "A", False
        clean = (not is_3p) and (not r["is_overlap"]) and \
                (r["snr_db"] >= min_snr or r["snr_db"] == 0.0)
        assigns.append(GlobalAssignment(
            segment_uid=r["segment_uid"], call_id=r["call_id"],
            start=r["start"], end=r["end"], local_speaker=r["local_speaker"],
            global_speaker=gs, sim_A=round(sa, 3), sim_B=round(sb, 3),
            is_thirdparty=is_3p, is_overlap=r["is_overlap"], clean=clean,
            text=r["text"], snr_db=r["snr_db"],
        ))

    # 저장: parquet
    out = data_dir() / "speakers" / "global_assignment.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame([a.model_dump() for a in assigns]).to_parquet(out, index=False)
    except Exception as e:  # noqa: BLE001
        log.warning("parquet 저장 실패(%s) — json 폴백", e)
        write_json(out.with_suffix(".json"), [a.model_dump() for a in assigns])

    # 통계 리포트 (§10 출력)
    n_a = sum(1 for a in assigns if a.global_speaker == "A")
    n_b = sum(1 for a in assigns if a.global_speaker == "B")
    n_3p = sum(1 for a in assigns if a.is_thirdparty)
    stats = {"n_segments": len(assigns), "n_A": n_a, "n_B": n_b, "n_thirdparty": n_3p}
    write_json(data_dir().parent / "reports" / "s2_link_report.json", stats)
    log.info("S2 전역연결 완료: A=%d B=%d 제3자=%d", n_a, n_b, n_3p)


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 전역 화자 연결 + 제3자 제거")
    ap.add_argument("--config", default="s2_diarize")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
