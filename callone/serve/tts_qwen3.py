"""Qwen3-TTS-12Hz 백엔드 어댑터 — qwen-tts-server(:8093) HTTP 클라이언트.

qwen_tts_server 는 cosyvoice_server 와 **동일 API 계약**(/health /synth /synth_stream,
프레임 형식까지 동일)이라 CosyVoiceTTS 의 HTTP/ref 인메모리 로직을 그대로 상속한다.
차이는 서버 주소(:8093)와 기동 스크립트뿐.

⚠️ 게이트(docs/REBUILD_PLAN.md §1): 과거 Qwen3 계열 TTS 는 턴 간 음색 튐으로 기각됨.
   scripts/bench_v2.py 음색 안정성 A/B 통과 전엔 serve.yaml 기본 백엔드로 두지 말 것.
"""
from __future__ import annotations

import os

from ..common.logging import get_logger
from .tts_cosyvoice import CosyVoiceTTS

log = get_logger("tts_qwen3")


class Qwen3TTS(CosyVoiceTTS):
    def __init__(self, speaker: str, cfg: dict | None = None):
        scfg = dict(cfg or {})
        # 부모가 읽는 키(cosyvoice_url)에 qwen 주소를 넣어 클라이언트 로직 재사용.
        scfg["cosyvoice_url"] = str(scfg.get("qwen_tts_url")
                                    or os.environ.get("QWEN_TTS_URL", "http://127.0.0.1:8093"))
        super().__init__(speaker, scfg)
        log.info("Qwen3-TTS 백엔드: %s (speaker=%s)", self.base_url, speaker)

    def _probe(self):
        try:
            super()._probe()
        except Exception as e:  # noqa: BLE001  # 부모 에러 메시지는 cosyvoice 안내라 교체
            raise RuntimeError(
                f"qwen-tts-server 응답 없음({self.base_url}). 먼저 띄워라: "
                f"bash scripts/setup_qwen_tts_gpu.sh (최초) / "
                f"source .venv-qwentts/bin/activate && python qwen_tts_server/app.py") from e
