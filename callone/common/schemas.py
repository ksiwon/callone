"""callone 데이터 스키마 — §7 전부 pydantic 으로 구현.

모든 스테이지 산출물의 단일 진실원천(single source of truth).
중간 산출물은 디스크에 저장(재현성, §2.6).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# 7.1 통화 메타 (db `calls` 테이블 / manifest.parquet)
# --------------------------------------------------------------------------
CallStatus = Literal["ok", "error", "pending"]


class CallMeta(BaseModel):
    call_id: str
    src_path: str
    wav16k_path: Optional[str] = None
    restored_path: Optional[str] = None
    duration_sec: float = 0.0
    orig_sr: int = 0
    orig_channels: int = 0
    codec: str = ""
    status: CallStatus = "pending"
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# --------------------------------------------------------------------------
# 7.2 통화별 분리 결과 (data/diarized/{call_id}.json)
# --------------------------------------------------------------------------
class Word(BaseModel):
    word: str
    start: float
    end: float
    score: float = 0.0


class Segment(BaseModel):
    start: float
    end: float
    local_speaker: str               # SPK_00 / SPK_01 (통화 내부 라벨)
    text: str = ""
    words: list[Word] = Field(default_factory=list)
    asr_conf: float = 0.0
    snr_db: float = 0.0
    overlap: bool = False
    embedding_ref: Optional[str] = None


class DiarizedCall(BaseModel):
    call_id: str
    segments: list[Segment] = Field(default_factory=list)


# --------------------------------------------------------------------------
# 7.3 전역 화자 연결 결과 (data/speakers/global_assignment.parquet)
# --------------------------------------------------------------------------
GlobalSpeaker = Literal["A", "B", "UNK"]


class GlobalAssignment(BaseModel):
    segment_uid: str                 # call_00001#0
    call_id: str
    start: float
    end: float
    local_speaker: str = ""
    global_speaker: GlobalSpeaker = "UNK"
    sim_A: float = 0.0
    sim_B: float = 0.0
    is_thirdparty: bool = False
    is_overlap: bool = False
    clean: bool = True
    text: str = ""
    snr_db: float = 0.0


# --------------------------------------------------------------------------
# 7.4 화자 프로필 (data/speakers/{A|B}/profile.json) — S2.5
# --------------------------------------------------------------------------
class DialectMarker(BaseModel):
    form: str                        # "~카이"
    count: int = 0
    std: str = ""                    # 표준어 대응 "~니까"


class DialectAuto(BaseModel):
    region_est: str = "unknown"      # gyeongsang/jeolla/chungcheong/gangwon/standard
    confidence: float = 0.0
    intensity_0to1: float = 0.0      # 데이터에서 측정한 사투리 '세기'
    markers: list[DialectMarker] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class SpeechAuto(BaseModel):
    avg_sentence_len: float = 0.0
    banmal_ratio: float = 0.0
    top_fillers: list[str] = Field(default_factory=list)
    question_rate: float = 0.0


class ProfileAuto(BaseModel):
    """자동 추출 초안 (수정 가능)."""
    gender_est: str = "U"            # F/M/U
    age_band_est: str = "unknown"    # 60s 등
    dialect: DialectAuto = Field(default_factory=DialectAuto)
    speech: SpeechAuto = Field(default_factory=SpeechAuto)


class ProfileUser(BaseModel):
    """사람이 라벨링 편집기로 확정."""
    model_config = ConfigDict(protected_namespaces=())  # 'register' 필드 shadow 경고 억제

    name: str = ""
    age: Optional[int] = None
    gender: str = "U"
    relation: str = ""               # 어머니/친구 등
    register: str = "반말"            # 반말/존댓말
    traits: list[str] = Field(default_factory=list)
    catchphrases: list[str] = Field(default_factory=list)
    taboo: list[str] = Field(default_factory=list)
    dialect_confirmed: bool = False
    # 사람이 자동 추정 덮어쓰기 가능
    dialect_region_override: Optional[str] = None
    dialect_intensity_override: Optional[float] = None


class ProfileTTS(BaseModel):
    server_model: Optional[str] = None
    laptop_model: Optional[str] = None


class ProfileLLM(BaseModel):
    lora_server: Optional[str] = None
    lora_laptop: Optional[str] = None


class SpeakerProfile(BaseModel):
    speaker_id: str                  # A / B
    auto: ProfileAuto = Field(default_factory=ProfileAuto)
    user: ProfileUser = Field(default_factory=ProfileUser)
    tts: ProfileTTS = Field(default_factory=ProfileTTS)
    llm: ProfileLLM = Field(default_factory=ProfileLLM)

    def effective_region(self) -> str:
        return self.user.dialect_region_override or self.auto.dialect.region_est

    def effective_intensity(self) -> float:
        if self.user.dialect_intensity_override is not None:
            return self.user.dialect_intensity_override
        return self.auto.dialect.intensity_0to1


# --------------------------------------------------------------------------
# 7.5 TTS 학습셋 행 (metadata.csv, LJSpeech 류)  wav_path|text|duration|snr
# --------------------------------------------------------------------------
class TTSRow(BaseModel):
    wav_path: str
    text: str
    duration: float
    snr: float = 0.0

    def to_csv_line(self) -> str:
        return f"{self.wav_path}|{self.text}|{self.duration:.2f}|{self.snr:.1f}"


# --------------------------------------------------------------------------
# 7.6 대화 학습셋 (train.jsonl)
# --------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None       # 상대 발화의 관계 라벨 (화자 B/친구)


class DialogueSample(BaseModel):
    messages: list[ChatMessage]


__all__ = [
    "CallMeta", "Word", "Segment", "DiarizedCall",
    "GlobalAssignment", "DialectMarker", "DialectAuto", "SpeechAuto",
    "ProfileAuto", "ProfileUser", "ProfileTTS", "ProfileLLM", "SpeakerProfile",
    "TTSRow", "ChatMessage", "DialogueSample",
]
