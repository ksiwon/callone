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
  width: 256px; height: 256px; border-radius: 16px; object-fit: cover;
  background: #000; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
`;
const Log = styled.div`
  width: 100%; max-width: 480px; max-height: 26vh; overflow-y: auto;
  color: ${(p) => p.theme.colors.sub}; font-size: 14px;
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
  const audioCtxRef = useRef<AudioContext | null>(null);
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
    const sock = new CallSocket(id, {
      onAudio: (pcm) => playPcm(pcm),
      onReply: (text, latency) => {
        historyRef.current.push({ role: "assistant", content: text }); persist();
        setStatus("통화 중");
        setLogLines((l) => [...l, `${id}: ${text}  (${latency.toFixed(0)}ms)`]);
      },
      onUser: (text) => {
        if (text.trim()) { historyRef.current.push({ role: "user", content: text }); persist();
          setLogLines((l) => [...l, `나: ${text}`]); }
      },
      onFrame: (jpegB64) => setFrame(jpegB64),
      onReady: () => setStatus("통화 중"),
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

    // 마이크 캡처 → 업스트림
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = (e) => {
        if (mutedRef.current) return;     // ref 라 통화 중 음소거 즉시 반영(클로저 stale 회피)
        sock.sendAudio(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      src.connect(proc); proc.connect(ctx.destination);
      cleanupMicRef.current = () => {
        proc.disconnect(); src.disconnect(); stream.getTracks().forEach((tr) => tr.stop());
      };
      setStatus("통화 중");
    } catch {
      setStatus("마이크 권한 필요(HTTPS/localhost)");
    }
    setStarted(true);
  }

  function playPcm(pcm: Float32Array) {
    const ctx = audioCtxRef.current ?? new AudioContext();
    const buf = ctx.createBuffer(1, pcm.length, 24000);   // Qwen3-TTS 출력 sr
    buf.copyToChannel(pcm, 0);
    const node = ctx.createBufferSource();
    node.buffer = buf; node.connect(ctx.destination); node.start();
  }
  function toggleMute() { mutedRef.current = !mutedRef.current; setMuted(mutedRef.current); }

  function endCall() {
    cleanupMicRef.current();
    sockRef.current?.stop();          // 서버가 인메모리 개인데이터 폐기
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
          <input type="text" value={persona} onChange={(e) => setPersona(e.target.value)} placeholder="예: 내 친한 여동생, 반말로 짧게" />
          <label>지금 상황 — 선택</label>
          <input type="text" value={situation} onChange={(e) => setSituation(e.target.value)} placeholder="예: 오랜만에 전화" />
          <label>이전 대화 불러오기 (이어하기) — 선택  {turns > 0 ? `· 저장된 ${turns}턴 있음` : ""}</label>
          <input type="file" accept="application/json" onChange={(e) => e.target.files?.[0] && importHistory(e.target.files[0])} />
        </Setup>
        <Controls>
          <Btn onClick={startCall} disabled={!voiceFile}>📞 통화 시작</Btn>
          {turns > 0 && <Btn onClick={exportHistory}>대화 내보내기</Btn>}
          <Btn danger onClick={() => nav("/")}>취소</Btn>
        </Controls>
      </Screen>
    );
  }

  // ---- 통화 화면 ----
  return (
    <Screen>
      <Who><Big>{id}</Big><Status>{status} · {mm}:{ss}</Status></Who>
      {frame ? (
        <Avatar src={`data:image/jpeg;base64,${frame}`} alt="avatar" />
      ) : (
        <Wave active={status === "통화 중"}>
          {Array.from({ length: 9 }).map((_, i) => <span key={i} style={{ animationDelay: `${i * 0.08}s` }} />)}
        </Wave>
      )}
      <Log>{logLines.map((l, i) => <div key={i}>{l}</div>)}</Log>
      <Controls>
        <Btn onClick={toggleMute}>{muted ? "음소거 해제" : "음소거"}</Btn>
        <Btn onClick={() => sockRef.current?.endTurn()}>말끝</Btn>
        <Btn onClick={exportHistory}>대화 내보내기</Btn>
        <Btn danger onClick={endCall}>종료</Btn>
      </Controls>
    </Screen>
  );
}
