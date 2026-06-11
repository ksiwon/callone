"""S0 — 적재 & 정규화 CLI (§8).

m4a → 16k mono wav + 메타 DB + manifest.parquet.
손상 파일은 status=error 로 격리.

사용:
  callone-ingest --config s0_ingest [--limit 50] [--raw data/raw]
  python -m callone.ingest.s0_convert --limit 50
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from ..common import db
from ..common.audio import to_wav16k
from ..common.io import data_dir, load_config
from ..common.logging import get_logger
from .manifest import call_id_from_path, probe_call

log = get_logger("s0")


def run(cfg: dict, limit: int | None = None) -> list:
    paths = sorted(glob.glob(cfg.get("input_glob", "data/raw/*.m4a")))
    if limit:
        paths = paths[:limit]
    log.info("S0: %d 통화 발견", len(paths))

    wav_dir = data_dir() / "wav16k"
    wav_dir.mkdir(parents=True, exist_ok=True)
    con = db.connect()
    metas = []

    for idx, sp in enumerate(paths):
        sp = Path(sp)
        cid = call_id_from_path(sp, idx)
        meta = probe_call(cid, sp)
        wav_path = wav_dir / f"{cid}.wav"

        if cfg.get("skip_existing", True) and wav_path.exists():
            meta.wav16k_path = str(wav_path)
            meta.status = "ok"
        else:
            ok = to_wav16k(sp, wav_path)
            if ok:
                meta.wav16k_path = str(wav_path)
                meta.status = "ok"
            else:
                meta.status = "error"
                meta.error = "ffmpeg 변환 실패 또는 손상"
                log.warning("격리: %s", cid)

        db.upsert_call(con, meta)
        metas.append(meta)

        n_total = len(paths)
        step = max(1, n_total // 50)
        if (idx + 1) % step == 0 or (idx + 1) == n_total:
            n_ok = sum(1 for m in metas if m.status == "ok")
            log.info("S0 변환 %d/%d (ok=%d)", idx + 1, n_total, n_ok)

    manifest = data_dir() / "manifest.parquet"
    try:
        db.write_manifest(metas, manifest)
        log.info("manifest 작성: %s", manifest)
    except Exception as e:  # noqa: BLE001
        log.warning("manifest parquet 작성 실패(%s) — DB 는 정상", e)

    ok = sum(1 for m in metas if m.status == "ok")
    log.info("S0 완료: ok=%d error=%d", ok, len(metas) - ok)
    return metas


def main() -> None:
    ap = argparse.ArgumentParser(description="S0 적재/정규화")
    ap.add_argument("--config", default="s0_ingest")
    ap.add_argument("--raw", default=None, help="raw 디렉토리 override")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    overrides = {}
    if args.raw:
        overrides["input_glob"] = str(Path(args.raw) / "*.m4a")
    cfg = load_config(args.config, overrides)
    run(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
