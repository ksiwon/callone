"""파이프라인 순서 회귀 방지 (로컬 디버깅으로 확정된 버그).

전사(s3)는 반드시 연결(s2b)보다 먼저여야 global_assignment.parquet 에 text 가 들어간다.
link 가 먼저면 parquet text 가 비어 프로필/데이터셋이 전부 빈 결과가 됨.
"""
from callone.pilot import STAGES


def test_transcribe_before_link():
    assert STAGES.index("s3") < STAGES.index("s2b"), \
        "s3(전사)는 s2b(연결)보다 먼저여야 함 — parquet text 백필 보장"


def test_profile_after_link():
    assert STAGES.index("s2b") < STAGES.index("s25"), \
        "s25(프로필/방언)은 s2b(연결) 뒤 — parquet 텍스트 필요"


def test_datasets_last():
    for stage in ("build_tts", "build_dlg", "sft"):
        assert STAGES.index("s25") < STAGES.index(stage)
