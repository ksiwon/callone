// CallScreen — 영상통화: 설정(음성·사진·페르소나·대화 불러오기) → 통화(음성+얼굴) → 내보내기.
// 프라이버시: 음성/사진/대화는 **브라우저(클라)가 소유**. 통화 시작 시 서버로 보내 인메모리만 쓰고,
// 끊기면 서버에서 즉시 폐기(디스크·로그에 안 남음). 대화 이력은 localStorage + 파일 export/import.
import { useEffect, useRef, useState } from "react";
import styled from "styled-components";
import { useParams, useNavigate } from "react-router-dom";
import { CallSocket, fileToBase64, type Turn, type SessionInit } from "../api/calloneClient";

const Screen = styled.div`
  min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  justify-content: space-between; padding: 32px 24px; gap: 16px;
  background: linear-gradient(180deg, #0e1726, #172234);
`;
const Who = styled.div`text-align: center; color: ${(p) => p.theme.colors.text};`;
const Big = styled.div`font-size: 28px; font-weight: 700;`;
const Status = styled.div`color: ${(p) => p.theme.colors.sub}; margin-top: 8px;`;
const Wave = styled.div<{ active: boolean }>`
  display: flex; gap: 4px; height: 48px; align-items: center;
  & span {
    width: 4px; background: ${(p) => p.theme.colors.accent}; border-radius: 2px;
    animation: ${(p) => (p.active ? "bounce 0.8s infinite" : "none")};
  }
  @keyframes bounce { 0%,100%{height:8px} 50%{height:40px} }
`;
/* ── 통화화면: 좌우 반반 split (좌=영상 꽉, 우=정보/채팅/버튼). 좁으면 세로 스택. ── */
const Split = styled.div`
  height: 100vh; display: flex; background: linear-gradient(180deg, #0e1726, #172234);
  @media (max-width: 760px) { flex-direction: column; }
`;
const VideoSide = styled.div`
  flex: 1; min-width: 0; min-height: 0; display: flex; align-items: center;
  justify-content: center; background: #000; padding: 12px;
`;
const Avatar = styled.canvas`
  /* 좌측 섹션을 꽉 채움: 비율 유지(contain)로 가로/세로 중 먼저 닿는 쪽까지 키움. */
  width: 100%; height: 100%; object-fit: contain; border-radius: 12px;
`;
const InfoSide = styled.div`
  flex: 1; min-width: 0; display: flex; flex-direction: column;
  padding: 24px 20px; gap: 16px; color: ${(p) => p.theme.colors.text};
`;
const Chat = styled.div`
  flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column;
  gap: 8px; padding-right: 4px;
`;
const Bubble = styled.div<{ me: boolean }>`
  align-self: ${(p) => (p.me ? "flex-end" : "flex-start")};
  max-width: 78%; padding: 9px 13px; border-radius: 16px; font-size: 14px; line-height: 1.45;
  white-space: pre-wrap; word-break: break-word;
  background: ${(p) => (p.me ? p.theme.colors.accent : p.theme.colors.surface)};
  color: ${(p) => (p.me ? "#06202b" : p.theme.colors.text)};
  ${(p) => (p.me ? "border-bottom-right-radius: 4px;" : "border-bottom-left-radius: 4px;")}
`;
const SysNote = styled.div`
  align-self: center; font-size: 12px; color: ${(p) => p.theme.colors.sub};
  padding: 2px 8px;
`;
const Controls = styled.div`display: flex; gap: 16px; flex-wrap: wrap; justify-content: center;`;
const Btn = styled.button<{ danger?: boolean }>`
  padding: 14px 20px; border-radius: 28px; border: none; cursor: pointer;
  color: #fff; font-size: 14px;
  background: ${(p) => (p.danger ? p.theme.colors.danger : p.theme.colors.surface)};
`;
const Setup = styled.div`
  width: 100%; max-width: 420px; display: flex; flex-direction: column; gap: 12px;
  color: ${(p) => p.theme.colors.text};
  & label { font-size: 13px; color: ${(p) => p.theme.colors.sub}; }
  & input[type="text"], & textarea {
    width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #2a3a52;
    background: #0c1422; color: #fff; font-size: 14px;
  }
`;

export default function CallScreen() {
  const { id = "A" } = useParams();
  const nav = useNavigate();
  const HKEY = `callone_history_${id}`;

  const [started, setStarted] = useState(false);
  const [sec, setSec] = useState(0);
  const [status, setStatus] = useState("준비");
  const [muted, setMuted] = useState(false);
  const mutedRef = useRef(false);
  const [chat, setChat] = useState<{ who: "me" | "them" | "sys"; text: string }[]>([]);

  // 클라가 소유하는 개인데이터(서버 영속 0)
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  // 캐릭터 카드(character card) 필드 — 실제 캐릭터 챗 사이트(Character.AI/SillyTavern) 표준.
  const [persona, setPersona] = useState("");          // 이름·관계 (description/who)
  const [personality, setPersonality] = useState("");  // 성격·말투 (personality)
  const [background, setBackground] = useState("");     // 배경
  const [situation, setSituation] = useState("");       // 지금 상황 (scenario)
  const [firstMessage, setFirstMessage] = useState(""); // 첫 마디 (greeting)
  const [exampleDialogue, setExampleDialogue] = useState(""); // 예시 말투 (example messages)
  const [userPersona, setUserPersona] = useState("");   // 나는 누구 (관계 기준)
  const historyRef = useRef<Turn[]>([]);

  const sockRef = useRef<CallSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);     // 마이크 캡처(16kHz)
  const playCtxRef = useRef<AudioContext | null>(null);      // 재생 전용(24kHz, TTS 출력 sr)
  const playheadRef = useRef<number>(0);                     // 다음 청크 재생 시작시각(누적)
  const turnAudioRef = useRef<Float32Array[]>([]);           // 한 턴 오디오 버퍼(A/V 동기 재생용)
  const turnFramesRef = useRef<string[]>([]);                // 한 턴 프레임 버퍼(같은 턴)
  const canvasRef = useRef<HTMLCanvasElement | null>(null);  // 프레임 그리기(canvas=디코딩 우회, 부드러움)
  const rafRef = useRef<number | null>(null);                // 프레임 재생 rAF 루프(중단 시 취소)
  const [hasVideo, setHasVideo] = useState(false);           // 영상 프레임 받은 적 있나(img vs 파형)
  const cleanupMicRef = useRef<() => void>(() => {});

  // 저장된 대화 불러오기(이어하기 편의)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(HKEY);
      if (raw) historyRef.current = JSON.parse(raw);
    } catch { /* noop */ }
  }, [HKEY]);

  useEffect(() => {
    if (!started) return;
    const t = setInterval(() => setSec((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [started]);

  function persist() {
    try { localStorage.setItem(HKEY, JSON.stringify(historyRef.current)); } catch { /* noop */ }
  }

  async function startCall() {
    setStarted(true);
    setStatus("연결 중…");
    // 이어하기: 저장된 이력을 채팅 말풍선으로 미리 표시(user=나, assistant=상대).
    setChat(historyRef.current.map((m) => ({ who: (m.role === "user" ? "me" : "them") as "me" | "them", text: m.content })));
    const sock = new CallSocket(id, {
      // A/V 동기: 오디오·프레임을 턴 버퍼에 모았다가 audio_end 에서 동시에 재생.
      onAudio: (pcm) => { turnAudioRef.current.push(pcm); },
      onReply: (text, latency) => {
        historyRef.current.push({ role: "assistant", content: text }); persist();
        setStatus("통화 중");
        void latency;
        setChat((c) => [...c, { who: "them", text }]);
      },
      onUser: (text) => {
        if (text.trim()) { historyRef.current.push({ role: "user", content: text }); persist();
          setChat((c) => [...c, { who: "me", text }]); }
      },
      onFrame: (jpegB64) => { turnFramesRef.current.push(jpegB64); },
      onAudioEnd: () => playTurn(),
      // 준비 완료(서버가 음성·사진·graph 다 세팅)되면 그때 마이크 켠다 — 연결 중엔 오디오 안 보냄
      // (안 그러면 서버가 init 처리 중에 오디오 폭주로 WS 수신큐 오버플로→끊김).
      onReady: () => { setStatus("통화 중"); startMic(sock); },
    });
    sockRef.current = sock;

    // 개인데이터 전송(클라 소유 → 서버 인메모리만). 캐릭터 카드 필드 포함.
    const init: SessionInit = {
      persona: persona || undefined,
      personality: personality || undefined,
      background: background || undefined,
      situation: situation || undefined,
      first_message: firstMessage || undefined,
      example_dialogue: exampleDialogue || undefined,
      user_persona: userPersona || undefined,
      history: historyRef.current.length ? historyRef.current : undefined,
    };
    if (voiceFile) init.ref_audio_b64 = await fileToBase64(voiceFile);
    if (photoFile) init.portrait_b64 = await fileToBase64(photoFile);
    sock.sessionInit(init);
  }

  async function startMic(sock: CallSocket) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = (e) => {
        if (mutedRef.current) return;
        sock.sendAudio(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      src.connect(proc); proc.connect(ctx.destination);
      cleanupMicRef.current = () => {
        proc.disconnect(); src.disconnect(); stream.getTracks().forEach((tr) => tr.stop());
      };
    } catch {
      setStatus("마이크 권한 필요(HTTPS/localhost)");
    }
  }

  // 한 턴의 오디오+프레임을 모았다가 **동시에** 재생(A/V 동기). 프레임은 오디오 길이에 균등 배치
  // → 입모양이 음성에 맞음. 영상이 Ditto 추론(~수 초)으로 늦게 오므로 audio_end 후 한 번에 재생.
  async function playTurn() {
    const chunks = turnAudioRef.current; turnAudioRef.current = [];
    const frames = turnFramesRef.current; turnFramesRef.current = [];
    if (rafRef.current != null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (!chunks.length && !frames.length) return;
    // 프레임을 **미리 전부 디코딩**(ImageBitmap) → rAF 에선 캔버스에 그리기만(디코딩 지연 0 = 부드러움).
    let bitmaps: ImageBitmap[] = [];
    if (frames.length) {
      try {
        bitmaps = await Promise.all(frames.map((b64) =>
          fetch(`data:image/jpeg;base64,${b64}`).then((r) => r.blob()).then((bl) => createImageBitmap(bl))));
      } catch { bitmaps = []; }
    }
    if (!chunks.length) {                        // 오디오 없으면 마지막 프레임만 표시
      if (bitmaps.length) drawBitmap(bitmaps[bitmaps.length - 1]);
      bitmaps.forEach((b) => b.close());
      return;
    }
    let ctx = playCtxRef.current;
    if (!ctx) { ctx = new AudioContext({ sampleRate: 24000 }); playCtxRef.current = ctx; }
    if (ctx.state === "suspended") ctx.resume();
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const audio = new Float32Array(total);
    let off = 0; for (const c of chunks) { audio.set(c, off); off += c.length; }
    const dur = audio.length / 24000;           // 초
    const buf = ctx.createBuffer(1, audio.length, 24000);
    buf.copyToChannel(audio, 0);
    const node = ctx.createBufferSource();
    node.buffer = buf; node.connect(ctx.destination);
    const startAt = ctx.currentTime + 0.08;
    node.start(startAt);
    console.log(`[A/V] frames=${frames.length} dur=${dur.toFixed(2)}s → ${(frames.length / Math.max(dur, 0.01)).toFixed(1)}fps`);
    if (bitmaps.length) {
      setHasVideo(true);
      drawBitmap(bitmaps[0]);
      const startPerf = performance.now() + (startAt - ctx.currentTime) * 1000;
      let last = -1;
      const tick = () => {
        const el = (performance.now() - startPerf) / 1000;   // 오디오 경과(초)
        if (el >= 0) {
          const idx = Math.min(bitmaps.length - 1, Math.max(0, Math.floor(el / dur * bitmaps.length)));
          if (idx !== last) { drawBitmap(bitmaps[idx]); last = idx; }
        }
        if (el < dur) { rafRef.current = requestAnimationFrame(tick); }
        else { rafRef.current = null; bitmaps.forEach((b) => b.close()); }   // 끝나면 메모리 해제
      };
      rafRef.current = requestAnimationFrame(tick);
    }
  }

  function drawBitmap(bmp: ImageBitmap) {
    const cv = canvasRef.current; if (!cv) return;
    if (cv.width !== bmp.width || cv.height !== bmp.height) { cv.width = bmp.width; cv.height = bmp.height; }
    const g = cv.getContext("2d"); if (g) g.drawImage(bmp, 0, 0);
  }
  function toggleMute() { mutedRef.current = !mutedRef.current; setMuted(mutedRef.current); }

  function endCall() {
    cleanupMicRef.current();
    sockRef.current?.stop();          // 서버가 인메모리 개인데이터 폐기
    if (rafRef.current != null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    try { playCtxRef.current?.close(); } catch { /* noop */ }
    playCtxRef.current = null; playheadRef.current = 0;
    nav("/");
  }

  function exportHistory() {
    const blob = new Blob([JSON.stringify(historyRef.current, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `callone_${id}_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
  }
  function importHistory(file: File) {
    file.text().then((t) => {
      try { historyRef.current = JSON.parse(t); persist(); setChat((c) => [...c, { who: "sys", text: `대화 ${historyRef.current.length}개 불러옴` }]); }
      catch { setChat((c) => [...c, { who: "sys", text: "불러오기 실패: JSON 아님" }]); }
    });
  }
  function clearHistory() {
    // 이전 대화 기억 전부 삭제(브라우저 보관분). 통화 시작 시 백엔드엔 빈 history 가 가므로 깨끗이 새 대화.
    historyRef.current = [];
    try { localStorage.removeItem(HKEY); } catch { /* noop */ }
    setChat([{ who: "sys", text: "기억 리셋됨 — 새 대화로 시작" }]);  // 상태변경 → 재렌더(턴수 0 반영)
  }

  const mm = String(Math.floor(sec / 60)).padStart(2, "0");
  const ss = String(sec % 60).padStart(2, "0");
  const turns = Math.floor(historyRef.current.length / 2);

  // ---- 설정 화면(통화 전) ----
  if (!started) {
    return (
      <Screen>
        <Who><Big>{id}</Big><Status>통화 준비 · 개인데이터는 내 브라우저만 보관</Status></Who>
        <Setup>
          <label>화자 음성 (목소리 복제, 7~10초 깨끗한 wav/mp3)</label>
          <input type="file" accept="audio/*" onChange={(e) => setVoiceFile(e.target.files?.[0] ?? null)} />
          <label>증명사진 (얼굴, jpg/png) — 선택</label>
          <input type="file" accept="image/*" onChange={(e) => setPhotoFile(e.target.files?.[0] ?? null)} />
          <label>이름·관계 (이 사람은 누구?)</label>
          <input type="text" value={persona} onChange={(e) => setPersona(e.target.value)} placeholder="예: 소꿉친구 나은" />
          <label>성격·말투 — 선택</label>
          <input type="text" value={personality} onChange={(e) => setPersonality(e.target.value)} placeholder="예: 밝고 장난기 많음. 반말로 짧고 편하게." />
          <label>배경 — 선택</label>
          <input type="text" value={background} onChange={(e) => setBackground(e.target.value)} placeholder="예: 초등학교 때부터 단짝, 지금도 같은 동네 살아." />
          <label>지금 상황 — 선택</label>
          <input type="text" value={situation} onChange={(e) => setSituation(e.target.value)} placeholder="예: 오랜만에 갑자기 전화함." />
          <label>나는 누구? (상대 기준) — 선택</label>
          <input type="text" value={userPersona} onChange={(e) => setUserPersona(e.target.value)} placeholder="예: 나은의 소꿉친구" />
          <label>첫 마디 — 선택</label>
          <input type="text" value={firstMessage} onChange={(e) => setFirstMessage(e.target.value)} placeholder="예: 야 오랜만이다! 살아있었네?" />
          <label>예시 말투 (이렇게 말함) — 선택</label>
          <textarea rows={3} value={exampleDialogue} onChange={(e) => setExampleDialogue(e.target.value)}
            placeholder={"예:\n나: 뭐해?\n나은: 그냥 침대에서 뒹굴뒹굴~ 넌 밥은 먹었어?"} />
          <label>이전 대화 불러오기 (이어하기) — 선택  {turns > 0 ? `· 저장된 ${turns}턴 있음` : ""}</label>
          <input type="file" accept="application/json" onChange={(e) => e.target.files?.[0] && importHistory(e.target.files[0])} />
        </Setup>
        <Controls>
          <Btn onClick={startCall} disabled={!voiceFile}>📞 통화 시작</Btn>
          {turns > 0 && <Btn onClick={exportHistory}>대화 내보내기</Btn>}
          {turns > 0 && <Btn danger onClick={clearHistory}>🗑 기억 리셋</Btn>}
          <Btn danger onClick={() => nav("/")}>취소</Btn>
        </Controls>
      </Screen>
    );
  }

  // ---- 통화 화면: 좌=영상 / 우=정보·채팅·버튼 ----
  return (
    <Split>
      <VideoSide>
        {hasVideo ? (
          <Avatar ref={canvasRef} />
        ) : (
          <Wave active={status === "통화 중"}>
            {Array.from({ length: 9 }).map((_, i) => <span key={i} style={{ animationDelay: `${i * 0.08}s` }} />)}
          </Wave>
        )}
      </VideoSide>
      <InfoSide>
        <Who><Big>{id}</Big><Status>{status} · {mm}:{ss}</Status></Who>
        <Chat>
          {chat.map((c, i) => c.who === "sys"
            ? <SysNote key={i}>{c.text}</SysNote>
            : <Bubble key={i} me={c.who === "me"}>{c.text}</Bubble>)}
        </Chat>
        <Controls>
          <Btn onClick={toggleMute}>{muted ? "음소거 해제" : "음소거"}</Btn>
          <Btn onClick={() => sockRef.current?.endTurn()}>응답 전송</Btn>
          <Btn onClick={exportHistory}>대화 내보내기</Btn>
          <Btn danger onClick={endCall}>종료</Btn>
        </Controls>
      </InfoSide>
    </Split>
  );
}
