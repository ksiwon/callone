"""긴 통화 녹음 → 화자별 최적 제로샷 레퍼런스 추출 (UI 플로우 B 백엔드).

시나리오: 유저가 "몇 시간짜리 두 사람 통화 녹음"만 들고 옴 →
  ① 업로드(tmpfs) → ② 화자분리(pyannote 체인, s2_diarize 재사용) → ③ 겹침 제외 +
  구간 점수화(common.audio.ref_clip_score — CLI pick_ref_clip 과 동일 기준) →
  ④ 화자별 카드(들어볼 샘플 포함) 반환 → ⑤ 유저가 "그 사람" 선택 →
  best 클립을 data/voice_presets/<이름>.wav(+전사 .txt)로 저장 → UI 프리셋에 자동 노출.

작업이 분 단위(시간짜리 오디오 분리)라 **job 스레드 + 폴링** 구조.
프라이버시: 업로드 원본은 tmpfs, 분석 끝나면 즉시 삭제. RAM 에는 화자별 후보
클립(수 MB)만 유지. 유저가 저장을 눌러야만 프리셋(디스크)이 생긴다. job 은 1h TTL.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import time
import uuid

import numpy as np

from ..common.audio import estimate_snr_db, load_wav, ref_clip_score
from ..common.io import load_config
from ..common.logging import get_logger
from .voice_presets import preset_dir

log = get_logger("voice_analyze")

MIN_S, MAX_S = 6.0, 12.0     # ref 클립 길이 범위(ref_clip_score 와 합의된 기준)
SAMPLE_S = 4.0               # "누가 그 사람?" 청취 샘플 길이
TOP_CLIPS = 5                # 화자별 보관 후보 수
JOB_TTL_S = 3600
# 기억 추출용 오디오 창(원본은 분석 후 삭제되므로 분석 중에 잡아둠 — 시간순, 두 화자 모두).
# 10분 × 16k float32 ≈ 38MB RAM/job — 전시(잡 1개씩)에서 무리 없음.
MEMO_MAX_S = 600.0           # 전사 대상 총량 상한
MEMO_SEG_S = 30.0            # 창 하나 최대 길이(ASR 안정)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _cleanup_expired() -> None:
    now = time.time()
    with _lock:
        for jid in [j for j, v in _jobs.items() if now - v["t0"] > JOB_TTL_S]:
            _jobs.pop(jid, None)


def _wav_b64(y: np.ndarray, sr: int) -> str:
    """float32 → wav bytes b64 (브라우저 <audio> 재생용)."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _candidate_windows(y: np.ndarray, sr: int, s: float, e: float) -> list[dict]:
    """한 세그먼트(단일화자 턴) 안에서 MIN_S~MAX_S 창 후보 생성(2s 슬라이드)."""
    out = []
    if e - s < MIN_S:
        return out
    for w0 in np.arange(s, max(s + 0.01, e - MIN_S), 2.0):
        w1 = min(w0 + MAX_S, e)
        clip = y[int(w0 * sr): int(w1 * sr)]
        snr = estimate_snr_db(clip)
        sc = ref_clip_score(snr, w1 - w0, MIN_S, MAX_S)
        if sc >= 0:
            out.append({"start": float(w0), "end": float(w1), "snr": float(snr),
                        "score": float(sc)})
    return out


def _analyze(job: dict, path: str) -> None:
    """job 스레드 본체 — 단계: diarize → scoring → done. 실패 시 error."""
    try:
        job["stage"] = "diarize"
        from ..diarize.s2_diarize import diarize_dummy, diarize_one

        cfg = {}
        try:
            cfg = load_config("s2_diarize")
        except Exception:  # noqa: BLE001
            pass
        segs = diarize_one(path, cfg)
        # 더미 폴백(교번 라벨)이면 화자 구분이 무의미 → 명시적으로 알림(오분리 프리셋 방지).
        # diarize_one 은 폴백을 조용히 쓰므로 pyannote/whisperx 가용성으로 재판정.
        dummy = False
        try:
            import pyannote.audio  # type: ignore  # noqa: F401
        except ImportError:
            try:
                import whisperx  # type: ignore  # noqa: F401
            except ImportError:
                dummy = True
        _ = diarize_dummy  # (참조 유지 — 폴백 경로 문서화)

        job["stage"] = "scoring"
        y, sr = load_wav(path, sr=16000)
        speakers: dict[str, dict] = {}
        for seg in segs:
            spk = seg.local_speaker
            d = speakers.setdefault(spk, {"total": 0.0, "n": 0, "cands": []})
            d["total"] += seg.end - seg.start
            d["n"] += 1
            if getattr(seg, "overlap", False):
                continue                                   # 겹침발화 = ref 오염 → 제외
            d["cands"].extend(_candidate_windows(y, sr, seg.start, seg.end))

        result = []
        for spk, d in sorted(speakers.items(), key=lambda kv: -kv[1]["total"]):
            cands = sorted(d["cands"], key=lambda c: -c["score"])[:TOP_CLIPS]
            if not cands:
                continue
            best = cands[0]
            # 청취 샘플(최고 클립 앞 4초) + 후보 클립 오디오는 RAM 에 보관(저장 시 사용)
            clips = []
            for c in cands:
                clip = y[int(c["start"] * sr): int(c["end"] * sr)].copy()
                clips.append({**c, "audio": clip})
            n_samp = int(min(SAMPLE_S, best["end"] - best["start"]) * sr)
            sample = y[int(best["start"] * sr): int(best["start"] * sr) + n_samp]
            result.append({
                "id": spk, "total_sec": round(d["total"], 1), "n_segments": d["n"],
                "best_snr": round(best["snr"], 1),
                "sample_wav_b64": _wav_b64(sample, sr),
                "_clips": clips, "_sr": sr,               # 내부용(상태 응답에선 제거)
            })
        if not result:
            raise RuntimeError("쓸만한 구간을 못 찾음 — 녹음이 너무 짧거나 잡음이 큼")

        # 기억 추출용 창: 시간순으로 두 화자 모두, 총 MEMO_MAX_S 까지(원본 삭제 전에 확보).
        memo, memo_total = [], 0.0
        for seg in segs:
            if memo_total >= MEMO_MAX_S:
                break
            if getattr(seg, "overlap", False):
                continue
            e = min(seg.end, seg.start + MEMO_SEG_S, seg.start + (MEMO_MAX_S - memo_total))
            if e - seg.start < 1.0:
                continue
            memo.append({"speaker": seg.local_speaker, "start": float(seg.start),
                         "audio": y[int(seg.start * sr): int(e * sr)].copy()})
            memo_total += e - seg.start
        job["_memo"] = memo
        job["_memo_sr"] = sr

        job["speakers"] = result
        job["dummy_diarizer"] = dummy
        job["stage"] = "done"
        log.info("voice_analyze 완료: 화자 %d명(더미분리=%s)", len(result), dummy)
    except Exception as e:  # noqa: BLE001
        log.warning("voice_analyze 실패: %s", e)
        job["stage"] = "error"
        job["error"] = str(e)
    finally:
        try:
            os.remove(path)                                # 업로드 원본 즉시 폐기(ephemeral)
        except OSError:
            pass


def start_job(raw: bytes, suffix: str = ".m4a") -> str:
    """업로드 bytes → tmpfs 저장 → 분석 스레드 시작. job_id 반환."""
    _cleanup_expired()
    base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) \
        else tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix=suffix or ".bin", prefix="va_", dir=base)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    job = {"t0": time.time(), "stage": "loading"}
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = job
    threading.Thread(target=_analyze, args=(job, path), daemon=True).start()
    return jid


def job_status(jid: str) -> dict | None:
    """폴링 응답(내부 오디오 버퍼 제외)."""
    job = _jobs.get(jid)
    if job is None:
        return None
    out = {"stage": job["stage"]}
    if job["stage"] == "error":
        out["error"] = job.get("error", "")
    if job["stage"] == "done":
        out["dummy_diarizer"] = job.get("dummy_diarizer", False)
        out["speakers"] = [{k: v for k, v in s.items() if not k.startswith("_")}
                           for s in job["speakers"]]
    return out


def _safe_name(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in "-_가-힣") or "voice"


def list_jobs() -> list[dict]:
    """최근 job 목록(내용 무관 메타만) — 폰 업로드(/upload) 를 데스크 UI 가 이어받는 용도."""
    _cleanup_expired()
    now = time.time()
    with _lock:
        items = sorted(_jobs.items(), key=lambda kv: -kv[1]["t0"])[:10]
    return [{"job_id": jid, "stage": j["stage"], "age_s": round(now - j["t0"], 1)}
            for jid, j in items]


def remember_pick(jid: str, speaker_id: str, name: str, asr, base_url: str,
                  chat_fn=None) -> dict:
    """분석 job 의 통화 내용 → 프리셋 화자의 기억(memories.json) 자동 구축 (트랙②).

    분석 중 확보해 둔 시간순 오디오 창(_memo)을 전사해 '그 사람'(selected)=assistant,
    나머지 화자=user 로 이력을 재구성 → remember_from_history 재사용(같은 프롬프트·중복제거).
    asr: transcribe(audio, sr)->str 계약. chat_fn 은 테스트 주입용."""
    job = _jobs.get(jid)
    if job is None or job.get("stage") != "done":
        raise ValueError("job 없음/미완료")
    memo = job.get("_memo") or []
    if not memo or asr is None:
        return {"added": 0, "total": 0, "windows": 0}
    sr = job.get("_memo_sr", 16000)
    history = []
    for w in memo:
        try:
            text = (asr.transcribe(w["audio"], sr) or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("기억 전사 실패(창 %.1fs~): %s", w["start"], e)
            continue
        if text:
            role = "assistant" if w["speaker"] == speaker_id else "user"
            history.append({"role": role, "content": text})
    if not history:
        return {"added": 0, "total": 0, "windows": 0}
    from ..llm.memory_update import remember_from_history

    r = remember_from_history(_safe_name(name), history, base_url, chat_fn=chat_fn)
    return {**r, "windows": len(history)}


def save_pick(jid: str, speaker_id: str, name: str, asr=None) -> dict:
    """선택한 화자의 best 클립 → data/voice_presets/<name>.wav (+ 전사 .txt).
    asr: transcribe(audio, sr)->str 계약(선택) — 있으면 ref_text 도 저장(유사도↑)."""
    job = _jobs.get(jid)
    if job is None or job.get("stage") != "done":
        raise ValueError("job 없음/미완료")
    spk = next((s for s in job["speakers"] if s["id"] == speaker_id), None)
    if spk is None:
        raise ValueError(f"화자 없음: {speaker_id}")
    safe = _safe_name(name)
    d = preset_dir()
    d.mkdir(parents=True, exist_ok=True)
    best = spk["_clips"][0]
    sr = spk["_sr"]
    import soundfile as sf

    wav_path = d / f"{safe}.wav"
    sf.write(str(wav_path), best["audio"].astype(np.float32), sr)
    ref_text = ""
    if asr is not None:
        try:
            ref_text = asr.transcribe(best["audio"], sr) or ""
        except Exception as e:  # noqa: BLE001
            log.warning("프리셋 전사 실패(%s) — 서버가 통화 시 자동전사", e)
    if ref_text:
        (d / f"{safe}.txt").write_text(ref_text, encoding="utf-8")
    log.info("프리셋 저장: %s (%.1fs, SNR %.1f, 전사 %d자)",
             safe, best["end"] - best["start"], best["snr"], len(ref_text))
    return {"preset_id": safe, "ref_text": ref_text,
            "dur": round(best["end"] - best["start"], 1)}
