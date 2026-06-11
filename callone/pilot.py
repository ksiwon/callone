"""파일럿 엔드투엔드 (§18) — 50통 + 한 사람(A)으로 전 과정 수직 관통.

원칙: 1,000통 전체 돌리기 전에 소량으로 M1~M7 검증.
각 스테이지 순차 호출 + 게이트 체크 + 리포트.

사용:
  callone-pilot --n 50 --speaker A
  callone-pilot --stages s0 s1 s2        # 일부만
"""
from __future__ import annotations

import argparse

from .common.io import load_config, write_json, data_dir
from .common.logging import get_logger

log = get_logger("pilot")

# ⚠️ 순서 중요(로컬 디버깅으로 확정):
#   분리(s2) → 전사(s3, diarized 에 text 채움) → 연결(s2b, parquet 에 text 복사)
#   → 프로필(s25) → 데이터셋. link 가 transcribe 보다 먼저면 parquet text 가 빔.
# S1(복원)은 denoise 모델 미설치 시 무용(CPU 업샘플뿐) → 기본 제외.
#   음질 강화 원하면 `pip install deepfilternet` 후 --stages 에 s1 추가.
STAGES = ["s0", "s2", "s3", "s2b", "s25", "build_tts", "build_dlg", "sft"]


def run(n: int, speaker: str, stages: list[str]) -> None:
    report = {"n_calls": n, "speaker": speaker, "stages": {}}

    if "s0" in stages:
        from .ingest.s0_convert import run as s0
        s0(load_config("s0_ingest"), limit=n)
        report["stages"]["s0"] = "done"

    if "s1" in stages:
        from .restore.s1_restore import run as s1
        s1(load_config("s1_restore"), limit=n)
        report["stages"]["s1"] = "done"

    if "s2" in stages:
        from .diarize.s2_diarize import run as s2
        s2(load_config("s2_diarize"), limit=n)
        report["stages"]["s2"] = "done"

    if "s3" in stages:
        from .asr.s3_transcribe import run as s3
        s3(load_config("asr"), limit=n)
        report["stages"]["s3"] = "done"

    if "s2b" in stages:
        from .diarize.s2b_link import run as s2b
        s2b(load_config("s2_diarize"), limit=n)
        report["stages"]["s2b"] = "done"

    if "s25" in stages:
        from .profile.s25_profile import run as s25
        s25(load_config("s25_profile"), [speaker])
        report["stages"]["s25"] = "done"

    if "build_tts" in stages:
        from .dataset.build_tts import run as bt
        bt(load_config("s3_dataset"), [speaker])
        report["stages"]["build_tts"] = "done"

    if "build_dlg" in stages:
        from .dataset.build_dialogue import run as bd
        bd(load_config("s3_dataset"), [speaker])
        report["stages"]["build_dlg"] = "done"

    if "sft" in stages:
        from .llm.prepare_sft import run as sft
        sft(load_config("llm_server"), [speaker])
        report["stages"]["sft"] = "done"

    write_json(data_dir().parent / "reports" / "pilot_report.json", report)
    log.info("파일럿 완료: %s", report["stages"])
    log.info("다음(무거운 학습): callone-asr-train / callone-tts-train / callone-llm-train")


def main() -> None:
    ap = argparse.ArgumentParser(description="파일럿 엔드투엔드")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--speaker", default="A")
    ap.add_argument("--stages", nargs="+", default=STAGES)
    args = ap.parse_args()
    run(args.n, args.speaker, args.stages)


if __name__ == "__main__":
    main()
