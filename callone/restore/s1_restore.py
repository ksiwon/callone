"""S1 — 음질 복원 CLI (§9).

denoise + 대역확장(8k→48k). ⚠️ 과복원 가드: 원본 대비 화자 임베딩
유사도 모니터 — 음색(=화자 정체성) 보존. 임계 미만이면 enhance 완화.

백엔드(DeepFilterNet/ClearerVoice/Resemble Enhance)는 무거우므로 선택 설치.
미설치 시 안전 폴백(원본 복사 + 경고) — 파이프라인은 끊기지 않음.

사용:
  callone-restore [--limit 50]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..common import db
from ..common.audio import cosine, estimate_snr_db, load_wav, save_wav
from ..common.io import data_dir, load_config
from ..common.logging import get_logger

log = get_logger("s1")


# ----- 백엔드 (선택 설치, 폴백 가능) ----------------------------------------
def _denoise(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    backend = cfg.get("denoise", {}).get("backend", "none")
    if backend == "deepfilternet":
        try:
            from df.enhance import enhance, init_df  # type: ignore
            import torch

            model, df_state, _ = init_df()
            t = torch.from_numpy(y).unsqueeze(0)
            out = enhance(model, df_state, t)
            return out.squeeze(0).cpu().numpy()
        except Exception as e:  # noqa: BLE001
            log.warning("DeepFilterNet 사용 불가(%s) — denoise 건너뜀", e)
    elif backend == "clearvoice":
        try:
            from clearvoice import ClearVoice  # type: ignore

            cv = ClearVoice(task="speech_enhancement",
                            model_names=[cfg["denoise"].get("clearvoice_model", "MossFormer2_SE_48K")])
            return np.asarray(cv(input_path=y))
        except Exception as e:  # noqa: BLE001
            log.warning("ClearerVoice 사용 불가(%s) — denoise 건너뜀", e)
    return y


def _enhance(y: np.ndarray, sr: int, cfg: dict, strength: float) -> tuple[np.ndarray, int]:
    ecfg = cfg.get("enhance", {})
    backend = ecfg.get("backend", "none")
    target_sr = int(ecfg.get("target_sr", 48000))
    if backend == "resemble":
        try:
            import torch
            from resemble_enhance.enhancer.inference import enhance  # type: ignore

            from ..common.io import resolve_device

            t = torch.from_numpy(y).float()
            wav, new_sr = enhance(t, sr, device=resolve_device(), solver="midpoint",
                                  nfe=64, tau=0.5, lambd=strength)
            return wav.cpu().numpy(), new_sr
        except Exception as e:  # noqa: BLE001
            log.warning("Resemble Enhance 사용 불가(%s) — 리샘플만", e)
    # 폴백: librosa 업샘플
    try:
        import librosa

        y2 = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y2, target_sr
    except Exception:
        return y, sr


# ----- 임베딩(음색 보존 가드) ------------------------------------------------
def _embed(y: np.ndarray, sr: int) -> np.ndarray | None:
    try:
        from ..diarize.embeddings import embed_waveform

        return embed_waveform(y, sr)
    except Exception:
        # 폴백: MFCC 평균 (간이 음색 프록시)
        try:
            import librosa

            m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            return m.mean(axis=1)
        except Exception:
            return None


def restore_one(wav16k: str | Path, cfg: dict) -> tuple[np.ndarray, int, dict]:
    y, sr = load_wav(wav16k, sr=None)
    snr_before = estimate_snr_db(y)
    emb_before = _embed(y, sr)

    guard = cfg.get("timbre_guard", {})
    strength = float(cfg.get("enhance", {}).get("strength", 0.6))
    min_cos = float(guard.get("min_cosine", 0.80))
    step = float(guard.get("step_down", 0.2))

    yd = _denoise(y, sr, cfg)
    out, out_sr = _enhance(yd, sr, cfg, strength)

    # 과복원 가드: 음색 유사도 체크 → 미달이면 강도 낮춰 재시도
    while guard.get("enabled", True) and strength > 0.0:
        emb_after = _embed(out, out_sr)
        if emb_before is None or emb_after is None:
            break
        # 길이 맞춰 비교 위해 임베딩 차원만 사용
        cos = cosine(emb_before, emb_after) if emb_before.shape == emb_after.shape else 1.0
        if cos >= min_cos:
            break
        strength = max(0.0, strength - step)
        log.warning("음색 유사도 %.2f < %.2f → enhance 강도 %.2f 로 완화", cos, min_cos, strength)
        out, out_sr = _enhance(yd, sr, cfg, strength)

    snr_after = estimate_snr_db(out)
    return out, out_sr, {"snr_before": snr_before, "snr_after": snr_after, "strength": strength}


def run(cfg: dict, limit: int | None = None) -> None:
    con = db.connect()
    calls = [c for c in db.all_calls(con) if c.status == "ok" and c.wav16k_path]
    if limit:
        calls = calls[:limit]
    out_dir = data_dir() / "restored"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for c in calls:
        dst = out_dir / f"{c.call_id}.wav"
        try:
            out, sr, stats = restore_one(c.wav16k_path, cfg)
            save_wav(dst, out, sr)
            c.restored_path = str(dst)
            db.upsert_call(con, c)
            report.append({"call_id": c.call_id, **stats})
            log.info("복원 %s SNR %.1f→%.1f (강도 %.2f)",
                     c.call_id, stats["snr_before"], stats["snr_after"], stats["strength"])
        except Exception as e:  # noqa: BLE001
            log.error("복원 실패 %s: %s", c.call_id, e)

    # 청취 샘플 리포트 (§9)
    n_rep = int(cfg.get("report_samples", 10))
    rep_path = data_dir().parent / "reports" / "s1_restore_report.json"
    from ..common.io import write_json

    write_json(rep_path, {"samples": report[:n_rep], "n_total": len(report)})
    log.info("S1 완료: %d 복원, 리포트 %s", len(report), rep_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="S1 음질 복원")
    ap.add_argument("--config", default="s1_restore")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
