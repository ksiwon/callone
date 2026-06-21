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
  history?: Turn[];          // 이전 대화 복원(이어하기)
}

// 실시간 통화 WebSocket. 마이크 오디오(Float32) 업스트림, 음성 청크 다운스트림.
export class CallSocket {
  private ws: WebSocket;
  constructor(
    speakerId: string,
    private cb: {
      onReply: (text: string, latencyMs: number) => void;
      onAudio: (pcm: Float32Array) => void;
      onUser?: (text: string) => void;       // 내 발화 전사(이력 기록용)
      onFrame?: (jpegB64: string) => void;   // 토킹헤드 프레임
      onReady?: () => void;                  // session_init 완료
      onAudioEnd?: () => void;               // 한 턴 송출 완료 → A/V 동기 재생 트리거
    },
  ) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}${BASE}/ws/call/${speakerId}`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);
        if (msg.type === "reply") this.cb.onReply(msg.text, msg.latency_ms);
        else if (msg.type === "user") this.cb.onUser?.(msg.text);
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
  stop() {
    try { this.ws.send(JSON.stringify({ type: "stop" })); } catch { /* noop */ }
    this.ws.close();
  }
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
