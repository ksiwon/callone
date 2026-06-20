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
const Avatar = styled.img`
  /* 비율 유지(정사각 강제 X) + 크게. 긴 변 기준으로 화면에 맞춤. */
  width: auto; height: auto; max-width: min(48vw, 540px); max-height: 74vh;
  border-radius: 16px; object-fit: contain;
  background: #000; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
`;
/* 통화화면 본문: 좌(대화/버튼) | 우(얼굴) 2단. 좁은 화면이면 세로로 쌓임. */
const Stage = styled.div`
  flex: 1; width: 100%; display: flex; gap: 28px; align-items: center;
  justify-content: center; flex-wrap: wrap;
`;
const Panel = styled.div`
  display: flex; flex-direction: column; gap: 16px;
  flex: 1 1 320px; max-width: 480px; min-width: 280px;
`;
const Log = styled.div`
  width: 100%; max-height: 44vh; overflow-y: auto;
  color: ${(p) => p.theme.colors.sub}; font-size: 14px; line-height: 1.5;
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
  const [logLines, setLogLines] = useState<string[]>([]);
  const [frame, setFrame] = useState<string | null>(null);

  // 클라가 소유하는 개인데이터(서버 영속 0)
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  const [persona, setPersona] = useState("");
  const [situation, setSituation] = useState("");
  const historyRef = useRef<Turn[]>([]);

  const sockRef = useRef<CallSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);     // 마이크 캡처(16kHz)
  const playCtxRef = useRef<AudioContext | null>(null);      // 재생 전용(24kHz, TTS 출력 sr)
  const playheadRef = useRef<number>(0);                     // 다음 청크 재생 시작시각(누적)
  const turnAudioRef = useRef<Float32Array[]>([]);           // 한 턴 오디오 버퍼(A/V 동기 재생용)
  const turnFramesRef = useRef<string[]>([]);                // 한 턴 프레임 버퍼(같은 턴)
  const frameTimersRef = useRef<number[]>([]);               // 프레임 표시 타이머(중단 시 정리)
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
    const sock = new CallSocket(id, {
      // A/V 동기: 오디오·프레임을 턴 버퍼에 모았다가 audio_end 에서 동시에 재생.
      onAudio: (pcm) => { turnAudioRef.current.push(pcm); },
      onReply: (text, latency) => {
        historyRef.current.push({ role: "assistant", content: text }); persist();
        setStatus("통화 중");
        const lat = typeof latency === "number" ? `  (${latency.toFixed(0)}ms)` : "";
        setLogLines((l) => [...l, `${id}: ${text}${lat}`]);
      },
      onUser: (text) => {
        if (text.trim()) { historyRef.current.push({ role: "user", content: text }); persist();
          setLogLines((l) => [...l, `나: ${text}`]); }
      },
      onFrame: (jpegB64) => { turnFramesRef.current.push(jpegB64); },
      onAudioEnd: () => playTurn(),
      // 준비 완료(서버가 음성·사진·graph 다 세팅)되면 그때 마이크 켠다 — 연결 중엔 오디오 안 보냄
      // (안 그러면 서버가 init 처리 중에 오디오 폭주로 WS 수신큐 오버플로→끊김).
      onReady: () => { setStatus("통화 중"); startMic(sock); },
    });
    sockRef.current = sock;

    // 개인데이터 전송(클라 소유 → 서버 인메모리만)
    const init: SessionInit = {
      persona: persona || undefined, situation: situation || undefined,
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
  function playTurn() {
    const chunks = turnAudioRef.current; turnAudioRef.current = [];
    const frames = turnFramesRef.current; turnFramesRef.current = [];
    // 이전 턴 프레임 타이머 정리
    frameTimersRef.current.forEach((t) => clearTimeout(t)); frameTimersRef.current = [];
    if (!chunks.length) {                       // 오디오 없으면 프레임만이라도 표시
      if (frames.length) setFrame(frames[frames.length - 1]);
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
    // 프레임을 오디오 시작에 맞춰 균등 스케줄(frames.length 개를 dur 초에 분산).
    if (frames.length) {
      const leadMs = (startAt - ctx.currentTime) * 1000;
      const base = performance.now() + leadMs;
      frames.forEach((f, i) => {
        const at = base + (i / frames.length) * dur * 1000;
        const t = window.setTimeout(() => setFrame(f), Math.max(0, at - performance.now()));
        frameTimersRef.current.push(t);
      });
    }
  }
  function toggleMute() { mutedRef.current = !mutedRef.current; setMuted(mutedRef.current); }

  function endCall() {
    cleanupMicRef.current();
    sockRef.current?.stop();          // 서버가 인메모리 개인데이터 폐기
    frameTimersRef.current.forEach((t) => clearTimeout(t)); frameTimersRef.current = [];
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
      try { historyRef.current = JSON.parse(t); persist(); setLogLines((l) => [...l, `(대화 ${historyRef.current.length}개 불러옴)`]); }
      catch { setLogLines((l) => [...l, "(불러오기 실패: JSON 아님)"]); }
    });
  }
  function clearHistory() {
    // 이전 대화 기억 전부 삭제(브라우저 보관분). 통화 시작 시 백엔드엔 빈 history 가 가므로 깨끗이 새 대화.
    historyRef.current = [];
    try { localStorage.removeItem(HKEY); } catch { /* noop */ }
    setLogLines((l) => [...l, "(기억 리셋됨 — 새 대화로 시작)"]);  // 상태변경 → 재렌더(턴수 0 반영)
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
          <label>이 사람은 누구? (페르소나) — 선택</label>
          <input type="text" value={persona} onChange={(e) => setPersona(e.target.value)} placeholder="예: 어릴 적 친구 승호" />
          <label>지금 상황 — 선택</label>
          <input type="text" value={situation} onChange={(e) => setSituation(e.target.value)} placeholder="예: 오랜만에 전화" />
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

  // ---- 통화 화면 ----
  return (
    <Screen>
      <Who><Big>{id}</Big><Status>{status} · {mm}:{ss}</Status></Who>
      <Stage>
        <Panel>
          <Log>{logLines.map((l, i) => <div key={i}>{l}</div>)}</Log>
          <Controls>
            <Btn onClick={toggleMute}>{muted ? "음소거 해제" : "음소거"}</Btn>
            <Btn onClick={() => sockRef.current?.endTurn()}>응답 전송</Btn>
            <Btn onClick={exportHistory}>대화 내보내기</Btn>
            <Btn danger onClick={endCall}>종료</Btn>
          </Controls>
        </Panel>
        {frame ? (
          <Avatar src={`data:image/jpeg;base64,${frame}`} alt="avatar" />
        ) : (
          <Wave active={status === "통화 중"}>
            {Array.from({ length: 9 }).map((_, i) => <span key={i} style={{ animationDelay: `${i * 0.08}s` }} />)}
          </Wave>
        )}
      </Stage>
    </Screen>
  );
}
