"""S2.5 화자 프로필 빌드 + 라벨링 편집기 백엔드 (§11.2).

자동 추출(성별/나이대 추정 + 말투 통계 + 방언) → profile.json 초안.
사람이 라벨링 편집기(UI)로 user 필드 확정. REST: GET/PUT /api/speakers/{id}/profile.

CLI 는 초안 생성 + 대표 발화 샘플 추출.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..common.io import data_dir, load_config, read_json, write_json
from ..common.logging import get_logger
from ..common.schemas import (
    DialectAuto, ProfileAuto, SpeakerProfile, SpeechAuto,
)
from .s25_dialect import collect_speaker_text, profile_dialect

log = get_logger("s25_profile")

_FILLERS = ["아이고", "마", "그", "어", "음", "뭐", "이제", "그니까", "아니", "참"]
_BANMAL_END = re.compile(r"(아|어|지|냐|니|자|네|데이|나|노)$")
_JONDAE_END = re.compile(r"(요|니다|세요|십시오|습니까|까요)$")


def _speech_stats(utts: list[str]) -> SpeechAuto:
    if not utts:
        return SpeechAuto()
    lens = [len(u.split()) for u in utts]
    banmal = sum(1 for u in utts if _BANMAL_END.search(u.strip().rstrip(".?!")))
    jondae = sum(1 for u in utts if _JONDAE_END.search(u.strip().rstrip(".?!")))
    tot = max(1, banmal + jondae)
    questions = sum(1 for u in utts if "?" in u or u.strip().endswith(("나", "노", "니", "까")))
    fillers = sorted(_FILLERS, key=lambda f: -sum(u.count(f) for u in utts))
    top = [f for f in fillers if sum(u.count(f) for u in utts) > 0][:3]
    return SpeechAuto(
        avg_sentence_len=round(sum(lens) / len(lens), 2),
        banmal_ratio=round(banmal / tot, 2),
        top_fillers=top,
        question_rate=round(questions / len(utts), 3),
    )


def _gender_age_est(speaker_id: str) -> tuple[str, str]:
    """음향 기반 추정 자리. 모델 미설치 시 미상.

    실제 구현은 음높이(F0)/음향 분류기 사용. 여기선 보수적으로 미상 반환
    → 사람이 라벨링 편집기에서 확정 (요구사항: auto 는 초안일 뿐).
    """
    return "U", "unknown"


def build_profile(speaker_id: str, cfg: dict) -> SpeakerProfile:
    text, utts = collect_speaker_text(speaker_id)
    dialect: DialectAuto = profile_dialect(text, cfg, examples_src=utts)
    gender, age = _gender_age_est(speaker_id)
    auto = ProfileAuto(
        gender_est=gender, age_band_est=age,
        dialect=dialect, speech=_speech_stats(utts),
    )
    # 기존 user 필드 보존 (재실행 시 라벨 덮어쓰지 않음)
    pj = data_dir() / "speakers" / speaker_id / "profile.json"
    prof = SpeakerProfile(speaker_id=speaker_id, auto=auto)
    if pj.exists():
        old = SpeakerProfile(**read_json(pj))
        prof.user = old.user
        prof.tts = old.tts
        prof.llm = old.llm
    return prof


def sample_utterances(speaker_id: str, n: int = 8) -> list[dict]:
    """라벨링 편집기에서 재생할 대표 발화(텍스트+오디오 구간) 추출."""
    ga = data_dir() / "speakers" / "global_assignment.parquet"
    out = []
    if ga.exists():
        import pandas as pd

        df = pd.read_parquet(ga)
        df = df[(df["global_speaker"] == speaker_id) & (df["clean"])]
        df = df.sort_values("snr_db", ascending=False).head(n)
        out = df[["segment_uid", "call_id", "start", "end", "text"]].to_dict("records")
    return out


def run(cfg: dict, speakers: list[str]) -> None:
    n = cfg.get("profile", {}).get("n_sample_utterances", 8)
    for sid in speakers:
        prof = build_profile(sid, cfg)
        out = data_dir() / "speakers" / sid / "profile.json"
        write_json(out, prof)
        samples = sample_utterances(sid, n)
        write_json(data_dir() / "speakers" / sid / "sample_utterances.json", samples)
        log.info("프로필 초안 %s: 지역=%s 세기=%.2f 반말율=%.2f → %s",
                 sid, prof.auto.dialect.region_est,
                 prof.auto.dialect.intensity_0to1,
                 prof.auto.speech.banmal_ratio, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="S2.5 화자 프로필 초안 빌드")
    ap.add_argument("--config", default="s25_profile")
    ap.add_argument("--speakers", nargs="+", default=["A", "B"])
    args = ap.parse_args()
    run(load_config(args.config), args.speakers)


if __name__ == "__main__":
    main()
