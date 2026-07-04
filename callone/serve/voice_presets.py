"""통화 프리셋 목소리 레지스트리 — data/voice_presets/<id>.wav (+ <id>.txt=전사) 자동 탐색.

프리셋 = 미리 준비한 레퍼런스 클립(‘내 목소리 업로드’ 대신 UI에서 고를 수 있는 것). breathy/ASMR 등.
드랍만 하면 UI 드롭다운에 뜬다(설정 수정 불필요).

⚠️ 클립은 **권리 있는 것만**(CC0/합성/본인 녹음). data/ 는 gitignore → **공개 레포에 안 올라간다**.
   pod 엔 scp 로 올린다(git 커밋 금지):
     scp -P <port> -i ~/.ssh/id_ed25519 clip.wav root@<ip>:/workspace/callone/data/voice_presets/
   전사(선택, 유사도↑)는 같은 이름 .txt: .../voice_presets/<id>.txt
"""
from __future__ import annotations

from pathlib import Path

from ..common.io import data_dir


def preset_dir() -> Path:
    return data_dir() / "voice_presets"


def list_presets() -> list[dict]:
    """탐색된 프리셋 [{id,label,has_text}]. 폴더/파일 없으면 빈 리스트."""
    d = preset_dir()
    if not d.is_dir():
        return []
    out: list[dict] = []
    for wav in sorted(d.glob("*.wav")):
        out.append({"id": wav.stem,
                    "label": wav.stem.replace("_", " "),
                    "has_text": wav.with_suffix(".txt").exists()})
    return out


def resolve(preset_id: str) -> tuple[str, str] | None:
    """preset_id → (wav_path, ref_text). ref_text 는 <id>.txt 있으면 그 내용, 없으면 ""(서버 자동전사).
    경로 이탈(../ 등) 차단 — 반드시 preset_dir 하위. 없으면 None."""
    if not preset_id or "/" in preset_id or "\\" in preset_id or ".." in preset_id:
        return None
    d = preset_dir()
    wav = d / f"{preset_id}.wav"
    try:                                    # preset_dir 밖으로 못 나가게
        wav.resolve().relative_to(d.resolve())
    except (ValueError, OSError):
        return None
    if not wav.is_file():
        return None
    txt = wav.with_suffix(".txt")
    ref_text = ""
    if txt.is_file():
        try:
            ref_text = txt.read_text(encoding="utf-8").strip()
        except OSError:
            ref_text = ""
    return str(wav), ref_text
