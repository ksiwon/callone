// callone 로컬 백엔드 클라이언트 (§17.1).
// ⚠️ missvoice 의 외부 유료 API 레이어(audioProcessingAPI.ts) 제거 → 이걸로 대체.
// REST: 화자/프로필. WS: 실시간 통화.

export interface SpeakerSummary {
  speaker_id: string;
  name: string;
  relation: string;
  region: string;
}

export interface SpeakerProfile {
  speaker_id: string;
  auto: any;
  user: {
    name: string;
    age: number | null;
    gender: string;
    relation: string;
    register: string;
    traits: string[];
    catchphrases: string[];
    taboo: string[];
    dialect_confirmed: boolean;
    dialect_region_override?: string | null;
    dialect_intensity_override?: number | null;
  };
  tts: any;
  llm: any;
}

// 서브경로 배포(Elice /proxy/5173/) 대응: vite base(import.meta.env.BASE_URL)를 prefix 로.
// 기본 '/' → BASE='' (localhost/직접노출). VITE_BASE=/proxy/5173/ → BASE='/proxy/5173'.
const BASE = (import.meta.env.BASE_URL || "/").replace(/\/$/, "");

export async function listSpeakers(): Promise<SpeakerSummary[]> {
  const r = await fetch(`${BASE}/api/speakers`);
  if (!r.ok) return [];
  return r.json();
}

export async function getProfile(id: string): Promise<SpeakerProfile | null> {
  const r = await fetch(`${BASE}/api/speakers/${id}/profile`);
  if (!r.ok) return null;          // 프로필 없으면(404) null — 편집기가 빈 응답으로 크래시하던 것 방지
  return r.json();
}

export async function putProfile(id: string, profile: SpeakerProfile): Promise<void> {
  await fetch(`${BASE}/api/speakers/${id}/profile`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
}

export async function getSamples(id: string): Promise<any[]> {
  const r = await fetch(`${BASE}/api/speakers/${id}/samples`);
  return r.ok ? r.json() : [];
}

// 목소리 미리듣기 — 업로드한 참조로 복제 목소리를 통화 전에 확인(1순위 유사도).
// 서버는 인메모리만 쓰고 합성 직후 폐기. ref_text 는 자동 전사 결과(수정 가능) 반환.
export async function previewVoice(
  refAudioB64: string,
  opts?: { text?: string; refText?: string },
): Promise<{ refText: string; audio: Float32Array; sr: number }> {
  const r = await fetch(`${BASE}/api/voice/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref_audio_b64: refAudioB64, text: opts?.text, ref_text: opts?.refText }),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(e.error || "미리듣기 실패");
  }
  const j = await r.json();
  const bytes = Uint8Array.from(atob(j.audio_b64), (c) => c.charCodeAt(0));
  return { refText: j.ref_text, audio: new Float32Array(bytes.buffer), sr: j.sr };
}

// 대화 한 턴(클라가 소유·export/import. 서버엔 안 남음).
export interface Turn { role: "user" | "assistant"; content: string }

// 통화 시작 시 프론트가 보내는 개인데이터(전부 클라 소유 → 서버는 인메모리만).
export interface SessionInit {
  ref_audio_b64?: string;   // 화자 음성 파일 bytes(base64) — 목소리 복제
  ref_text?: string;
  portrait_b64?: string;    // 사진 파일 bytes(base64) — 얼굴
  // 캐릭터 카드(character card) 필드 — 상황극 페르소나
  persona?: string;          // 이름·관계 (이 사람은 누구)
  personality?: string;      // 성격·말투
  background?: string;       // 배경
  situation?: string;        // 지금 상황 (scenario)
  first_message?: string;    // 첫 마디 (greeting)
  example_dialogue?: string; // 예시 말투
  user_persona?: string;     // 나는 누구(상대 기준 = 관계)
  preset_id?: string;        // 준비된 목소리 선택(data/voice_presets/<id>) — 있으면 내 음성 업로드 대신 사용
  history?: Turn[];          // 이전 대화 복원(이어하기)
}

// 준비된 프리셋 목소리 목록('내 목소리 업로드' 대안). 클립은 서버 로컬, 여긴 id·label 만.
export interface VoicePreset { id: string; label: string; has_text: boolean }

export async function listVoicePresets(): Promise<VoicePreset[]> {
  const r = await fetch(`${BASE}/api/voice/presets`);
  return r.ok ? r.json() : [];
}

// ----- 긴 통화 녹음 → 화자 분석(플로우 B). 원본은 서버 tmpfs, 분석 끝나면 즉시 삭제 -----
export interface AnalyzeSpeaker {
  id: string; total_sec: number; n_segments: number; best_snr: number;
  sample_wav_b64: string;   // "누가 그 사람?" 청취 샘플(wav)
}
export interface AnalyzeStatus {
  stage: "loading" | "diarize" | "scoring" | "done" | "error";
  error?: string;
  dummy_diarizer?: boolean; // true = pyannote 미설치 → 화자 구분 신뢰 불가(설치 안내)
  speakers?: AnalyzeSpeaker[];
}
export async function analyzeVoiceStart(file: File): Promise<string> {
  const ext = (file.name.split(".").pop() || "m4a").toLowerCase();
  const r = await fetch(`${BASE}/api/voice/analyze?ext=${ext}`, { method: "POST", body: file });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `업로드 실패(${r.status})`);
  return j.job_id;
}
export async function analyzeVoiceStatus(jobId: string): Promise<AnalyzeStatus> {
  const r = await fetch(`${BASE}/api/voice/analyze/${jobId}`);
  if (!r.ok) throw new Error("분석 상태 조회 실패(만료 1h?)");
  return r.json();
}
export async function analyzeVoiceSave(jobId: string, speakerId: string, name: string):
  Promise<{ preset_id: string; ref_text: string; dur: number }> {
  const r = await fetch(`${BASE}/api/voice/analyze/${jobId}/save`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speaker_id: speakerId, name }),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || "프리셋 저장 실패");
  return j;
}

// 실시간 통화 WebSocket. 마이크 오디오(Float32) 업스트림, 음성 청크 다운스트림.
export class CallSocket {
  private ws: WebSocket;
  constructor(
    speakerId: string,
    private cb: {
      onReply: (text: string, latencyMs: number) => void;
      onAudio: (pcm: Float32Array) => void;
      onUser?: (text: string) => void;       // 내 발화 전사(이력 기록용, 최종)
      onPartial?: (text: string) => void;    // v2: 발화 중 실시간 부분 전사(자막)
      onTiming?: (stages: Record<string, number>) => void;  // v2: 단계별 ms(HUD)
      onInterrupted?: () => void;            // v2: 서버가 응답 중단 확인
      onFrame?: (jpegB64: string) => void;   // 토킹헤드 프레임
      onReady?: () => void;                  // session_init 완료
      onAudioEnd?: () => void;               // 한 턴 송출 완료 → A/V 동기 재생 트리거
      onClose?: () => void;                  // WS 종료(정상 stop 포함 — 키오스크 장애 감지용)
    },
  ) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}${BASE}/ws/call/${speakerId}`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onclose = () => this.cb.onClose?.();
    this.ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);
        if (msg.type === "reply") this.cb.onReply(msg.text, msg.latency_ms);
        else if (msg.type === "user") this.cb.onUser?.(msg.text);
        else if (msg.type === "partial") this.cb.onPartial?.(msg.text);
        else if (msg.type === "timing") this.cb.onTiming?.(msg.stages);
        else if (msg.type === "interrupted") this.cb.onInterrupted?.();
        else if (msg.type === "frame") this.cb.onFrame?.(msg.jpeg_b64);
        else if (msg.type === "audio_end") this.cb.onAudioEnd?.();
        else if (msg.type === "session_ready") this.cb.onReady?.();
      } else {
        this.cb.onAudio(new Float32Array(e.data));
      }
    };
  }
  // 통화 시작 — 개인데이터 전송(서버는 인메모리만, 끊으면 폐기).
  sessionInit(payload: SessionInit) {
    const send = () => this.ws.send(JSON.stringify({ type: "session_init", ...payload }));
    if (this.ws.readyState === WebSocket.OPEN) send();
    else this.ws.addEventListener("open", send, { once: true });
  }
  sendAudio(pcm: Float32Array) {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(pcm.buffer);
  }
  endTurn() {
    this.ws.send(JSON.stringify({ type: "end_turn" }));
  }
  // v2 barge-in: 재생/생성 중 응답을 즉시 중단(탭-투-인터럽트 버튼).
  interrupt() {
    try { this.ws.send(JSON.stringify({ type: "interrupt" })); } catch { /* noop */ }
  }
  // 안전한 끝맺음: 클론이 작별 인사 → 재생 후 클라가 끊음(급작스러운 종료의 심리적 해악 완화).
  // extra = 추가 연출 지시(전시 부메랑 — persona_from_survey 의 boomerang 문자열).
  farewell(extra?: string) {
    try { this.ws.send(JSON.stringify({ type: "farewell", extra })); } catch { /* noop */ }
  }
  stop() {
    try { this.ws.send(JSON.stringify({ type: "stop" })); } catch { /* noop */ }
    this.ws.close();
  }
}

// 통화 이력 → 기억 성장(유저 주도 영속화 — 누르면 서버 memories.json 에 사실 추가,
// 다음 통화부터 회상). 클라 소유 이력의 명시적 승격이라 프라이버시 원칙과 합치.
export async function rememberCall(speakerId: string, history: Turn[]):
  Promise<{ added: number; total: number }> {
  const r = await fetch(`${BASE}/api/speakers/${speakerId}/remember`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ history }),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || "기억 저장 실패");
  return j;
}

// ----- 전시 모드(call:one 키오스크) --------------------------------------
// 설문 답 → 캐릭터 카드+기억 시드+부메랑(서버 persona_from_survey — 단일 진실원, 디스크 기록 0).
export interface ExhibitPersona {
  card: Record<string, string>;   // SessionInit 캐릭터 카드 필드와 1:1
  memories: string[];
  boomerang?: string;             // 작별 직전 되돌려줄 지시문 → farewell(extra)
}
export async function exhibitPersona(
  name: string, answers: Record<string, unknown>, mode: "future_self" | "loved_one" = "future_self",
): Promise<ExhibitPersona> {
  const r = await fetch(`${BASE}/api/exhibit/persona`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, answers, mode }),
  });
  if (!r.ok) throw new Error("페르소나 생성 실패");
  return r.json();
}

// 소멸 카운터(개인 데이터 0 — 숫자만). 벽면 "오늘 N개의 목소리가 태어나고 사라졌습니다".
export interface ExhibitCount { day: string; today: number; total: number }
export async function exhibitCount(): Promise<ExhibitCount> {
  const r = await fetch(`${BASE}/api/exhibit/count`);
  if (!r.ok) return { day: "", today: 0, total: 0 };
  return r.json();
}
export async function exhibitDissolve(): Promise<ExhibitCount> {
  const r = await fetch(`${BASE}/api/exhibit/dissolve`, { method: "POST" });
  if (!r.ok) return { day: "", today: 0, total: 0 };
  return r.json();
}

// 마이크 Float32 PCM → 16bit WAV base64 — 키오스크 현장 녹음을 ref_audio_b64 로 보내는 용도.
export function pcmToWavB64(pcm: Float32Array, sr: number): string {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(bin);
}

// 파일 → base64(데이터URL 접두 제거). 음성/사진 전송용.
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1] ?? "");
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
