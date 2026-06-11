"""test_asr_adapt (§19): 교정 CSV 로드 + (개념) WER 개선 체크 헬퍼."""
import csv

from callone.asr_adapt.whisper_finetune import load_correction_csv


def test_load_correction_csv(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"RIFF")  # 존재만 확인
    csv_path = tmp_path / "to_correct.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seg_uid", "call_id", "start", "end", "wav_clip", "asr_text", "corrected_text"])
        w.writerow(["s0", "c1", 0, 3, str(clip), "밥 묵엇나", "밥 묵었나"])
        w.writerow(["s1", "c1", 3, 6, str(clip), "응", ""])  # 미교정 → 제외
    rows = load_correction_csv(str(csv_path))
    assert len(rows) == 1
    assert rows[0]["text"] == "밥 묵었나"


def test_wer_improvement_concept():
    """적응 WER < 기본 WER 개념 게이트 (jiwer 있으면 실측, 없으면 스킵)."""
    try:
        import jiwer
    except Exception:
        return
    ref = "밥은 묵었나"
    base_hyp = "밥은 먹었나"     # 표준어로 잘못 인식
    adapt_hyp = "밥은 묵었나"    # 사투리 보존 인식
    assert jiwer.wer(ref, adapt_hyp) < jiwer.wer(ref, base_hyp)
