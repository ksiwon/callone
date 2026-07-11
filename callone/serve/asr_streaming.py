"""발화 중 스트리밍 전사(partial) — 백엔드 무관 재전사(rerun) 래퍼.

v2 지연 개선의 핵심(REBUILD_PLAN §2): 기존엔 말이 **끝난 뒤** 일괄 전사(턴마다 ASR 지연
수백 ms). 이 래퍼는 마이크 청크가 들어오는 **동안** 백그라운드 스레드가 누적 버퍼를
주기 재전사해 partial 을 만들고, 턴 종료(finalize) 시점엔:
  - 마지막 재전사 이후 새 오디오 없음 → partial 그대로 반환(**ASR 지연 0**)
  - 새 오디오 있음(꼬리) → 1회만 더 전사(짧음)

특징:
  - 어떤 ASR 백엔드든 감쌈(transcribe(audio, sr)->str 계약만 요구: whisper/Qwen3 공용).
  - 진짜 incremental 디코딩(vLLM 스트리밍)이 아니라 재전사라 발화가 아주 길면(>30s)
    회당 비용 커짐 — 통화 발화는 수 초라 실용상 문제 없음(max_window_s 로 상한).
  - 프라이버시: 버퍼는 인메모리 only, close/finalize 시 폐기. 본문 로그 없음.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from ..common.logging import get_logger

log = get_logger("asr_streaming")


class StreamingTranscriber:
    """한 발화(턴) 단위 세션. app 이 speech 시작 시 만들고 end_turn 에 finalize."""

    def __init__(self, asr, sr: int = 16000, interval_ms: int = 600,
                 max_window_s: float = 40.0,
                 on_partial: Callable[[str], None] | None = None):
        self.asr = asr
        self.sr = sr
        self.interval = max(0.1, interval_ms / 1000)
        self.max_samples = int(max_window_s * sr)
        self.on_partial = on_partial
        self._buf: list[np.ndarray] = []
        self._n = 0                    # 누적 샘플 수
        self._done_n = 0               # 마지막 전사가 소화한 샘플 수
        self._partial = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ── 입력 ──────────────────────────────────────────────────────────────
    def feed(self, chunk: np.ndarray) -> None:
        a = np.asarray(chunk, dtype=np.float32).flatten()
        if len(a) == 0:
            return
        with self._lock:
            self._buf.append(a)
            self._n += len(a)
        self._kick.set()

    # ── 백그라운드 재전사 ─────────────────────────────────────────────────
    def _snapshot(self) -> tuple[np.ndarray, int]:
        with self._lock:
            if not self._buf:
                return np.zeros(0, dtype=np.float32), 0
            a = np.concatenate(self._buf)
            n = self._n
        if len(a) > self.max_samples:               # 상한(재전사 비용 폭주 방지)
            a = a[-self.max_samples:]
        return a, n

    def _run(self):
        while not self._stop.is_set():
            self._kick.wait(timeout=self.interval)
            self._kick.clear()
            if self._stop.is_set():
                break
            audio, n = self._snapshot()
            if n <= self._done_n or len(audio) < int(0.4 * self.sr):
                continue                             # 새 오디오 없음/너무 짧음
            t0 = time.time()
            try:
                text = self.asr.transcribe(audio, self.sr)
            except Exception as e:  # noqa: BLE001
                log.warning("partial 전사 실패(%s)", e)
                continue
            self._done_n = n
            if text and text != self._partial:
                self._partial = text
                if self.on_partial:
                    try:
                        self.on_partial(text)
                    except Exception:  # noqa: BLE001
                        pass
            log.debug("partial %.0fms (%d샘플)", (time.time() - t0) * 1000, n)
            time.sleep(self.interval)                # 주기 하한(GPU 독점 방지)

    # ── 종료 ──────────────────────────────────────────────────────────────
    def finalize(self) -> str:
        """턴 종료 — 최종 전사 반환. 새 꼬리 오디오 있으면 1회만 더 전사."""
        self._stop.set()
        self._kick.set()
        self._worker.join(timeout=5.0)
        audio, n = self._snapshot()
        if n == 0:
            return ""
        if n > self._done_n or not self._partial:    # 꼬리 반영 필요
            try:
                self._partial = self.asr.transcribe(audio, self.sr)
            except Exception as e:  # noqa: BLE001
                log.warning("최종 전사 실패(%s) — 마지막 partial 사용", e)
        text = self._partial.strip()
        self.close()
        return text

    def close(self):
        """버퍼 즉시 폐기(ephemeral)."""
        self._stop.set()
        self._kick.set()
        with self._lock:
            self._buf = []
            self._n = 0
