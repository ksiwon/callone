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

const BASE = ""; // vite 프록시로 /api, /ws → localhost:8000

export async function listSpeakers(): Promise<SpeakerSummary[]> {
  const r = await fetch(`${BASE}/api/speakers`);
  if (!r.ok) return [];
  return r.json();
}

export async function getProfile(id: string): Promise<SpeakerProfile> {
  const r = await fetch(`${BASE}/api/speakers/${id}/profile`);
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

// 실시간 통화 WebSocket. 마이크 오디오(Float32) 업스트림, 음성 청크 다운스트림.
export class CallSocket {
  private ws: WebSocket;
  constructor(
    speakerId: string,
    private onReply: (text: string, latencyMs: number) => void,
    private onAudio: (pcm: Float32Array) => void,
    // 토킹헤드(선택): avatar 켜져 있으면 JPEG 프레임이 base64 로 온다. 영상통화 화면 렌더용.
    private onFrame?: (jpegB64: string) => void,
  ) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/call/${speakerId}`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);
        if (msg.type === "reply") this.onReply(msg.text, msg.latency_ms);
        else if (msg.type === "frame" && this.onFrame) this.onFrame(msg.jpeg_b64);
      } else {
        this.onAudio(new Float32Array(e.data));
      }
    };
  }
  sendAudio(pcm: Float32Array) {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(pcm.buffer);
  }
  endTurn() {
    this.ws.send(JSON.stringify({ type: "end_turn" }));
  }
  stop() {
    this.ws.send(JSON.stringify({ type: "stop" }));
    this.ws.close();
  }
}
