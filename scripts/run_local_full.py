"""로컬 전량 실행기 (재개 가능 + 절전 방지).

- Windows 절전/화면꺼짐 방지(SetThreadExecutionState) → 장시간 CPU 작업 중 노트북이
  잠들어 프로세스가 죽는 것을 막는다.
- 단일 프로세스로 남은 스테이지를 순서대로 실행. 이미 끝난 분리는 자동 건너뜀(재개).

사용:
  python scripts/run_local_full.py            # A,B 두 화자 데이터셋까지
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callone.common.io import load_config           # noqa: E402
from callone.common.logging import get_logger       # noqa: E402

log = get_logger("run_local")


def keep_awake():
    """실행 동안 절전 금지(Windows). 다른 OS는 무시."""
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        log.info("절전 방지 활성화(작업 중 잠들지 않음)")
    except Exception as e:  # noqa: BLE001
        log.info("절전 방지 미지원(%s) — 계속 진행", e)


def release_awake():
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


def main():
    keep_awake()
    try:
        from callone.diarize.s2_diarize import run as diarize
        from callone.diarize.s2b_link import run as link
        from callone.asr.s3_transcribe import run as transcribe
        from callone.profile.s25_profile import run as profile
        from callone.dataset.build_tts import run as build_tts
        from callone.dataset.build_dialogue import run as build_dlg
        from callone.llm.prepare_sft import run as prepare_sft

        log.info("=== [1/7] 화자 분리 (재개) ===")
        diarize(load_config("s2_diarize"))               # skip_existing 기본 True

        # ⚠️ 전사 → 연결 순서 중요: link 가 parquet 에 text 를 복사하므로
        #    전사가 먼저 diarized 에 text 를 채운 뒤 link 를 돌려야 한다.
        log.info("=== [2/7] 전사 (한국어, CPU) ===")
        transcribe(load_config("asr"))

        log.info("=== [3/7] 전역 화자 연결 ===")
        link(load_config("s2_diarize"))

        log.info("=== [4/7] 방언/프로필 ===")
        profile(load_config("s25_profile"), ["A", "B"])

        log.info("=== [5/7] TTS 학습셋 ===")
        build_tts(load_config("s3_dataset"), ["A", "B"])

        log.info("=== [6/7] 대화셋 + 페르소나 카드 ===")
        build_dlg(load_config("s3_dataset"), ["A", "B"])

        log.info("=== [7/7] SFT 준비 ===")
        prepare_sft(load_config("llm_server"), ["A", "B"])

        log.info("=== CALLONE LOCAL FULL DONE ===")
    finally:
        release_awake()


if __name__ == "__main__":
    main()
