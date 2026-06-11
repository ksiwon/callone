// CallScreen (§17.1) — 통화 화면: 타이머·파형·음소거·종료 + 자막 로그.
// 마이크 캡처 → WS 업스트림 → 응답 음성 재생. 외부 API 없음.
import { useEffect, useRef, useState } from "react";
import styled from "styled-components";
import { useParams, useNavigate } from "react-router-dom";
import { CallSocket } from "../api/calloneClient";

const Screen = styled.div`
  height: 100vh; display: flex; flex-direction: column; align-items: center;
  justify-content: space-between; padding: 48px 24px;
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
const Log = styled.div`
  width: 100%; max-width: 480px; max-height: 30vh; overflow-y: auto;
  color: ${(p) => p.theme.colors.sub}; font-size: 14px;
`;
const Controls = styled.div`display: flex; gap: 24px;`;
const Btn = styled.button<{ danger?: boolean }>`
  width: 64px; height: 64px; border-radius: 50%; border: none; cursor: pointer;
  color: #fff; font-size: 13px;
  background: ${(p) => (p.danger ? p.theme.colors.danger : p.theme.colors.surface)};
`;

export default function CallScreen() {
  const { id = "A" } = useParams();
  const nav = useNavigate();
  const [sec, setSec] = useState(0);
  const [status, setStatus] = useState("전화 거는 중…");
  const [muted, setMuted] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const sockRef = useRef<CallSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    const t = setInterval(() => setSec((s) => s + 1), 1000);
    const sock = new CallSocket(
      id,
      (text, latency) => {
        setStatus("통화 중");
        setLog((l) => [...l, `${id}: ${text}  (${latency.toFixed(0)}ms)`]);
      },
      (pcm) => playPcm(pcm),
    );
    sockRef.current = sock;

    let cleanupMic = () => {};
    navigator.mediaDevices?.getUserMedia({ audio: true }).then((stream) => {
      const ctx = new AudioContext({ sampleRate: 16000 });
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = (e) => {
        if (muted) return;
        sock.sendAudio(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      src.connect(proc); proc.connect(ctx.destination);
      setStatus("통화 중");
      cleanupMic = () => { proc.disconnect(); src.disconnect(); stream.getTracks().forEach((t) => t.stop()); };
    }).catch(() => setStatus("마이크 권한 필요"));

    return () => { clearInterval(t); cleanupMic(); sock.stop(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  function playPcm(pcm: Float32Array) {
    const ctx = audioCtxRef.current ?? new AudioContext();
    const buf = ctx.createBuffer(1, pcm.length, 48000);
    buf.copyToChannel(pcm, 0);
    const node = ctx.createBufferSource();
    node.buffer = buf; node.connect(ctx.destination); node.start();
  }

  const mm = String(Math.floor(sec / 60)).padStart(2, "0");
  const ss = String(sec % 60).padStart(2, "0");

  return (
    <Screen>
      <Who>
        <Big>{id}</Big>
        <Status>{status} · {mm}:{ss}</Status>
      </Who>
      <Wave active={status === "통화 중"}>
        {Array.from({ length: 9 }).map((_, i) => (
          <span key={i} style={{ animationDelay: `${i * 0.08}s` }} />
        ))}
      </Wave>
      <Log>{log.map((l, i) => <div key={i}>{l}</div>)}</Log>
      <Controls>
        <Btn onClick={() => { setMuted((m) => !m); }}>{muted ? "음소거 해제" : "음소거"}</Btn>
        <Btn danger onClick={() => { sockRef.current?.endTurn(); }}>말끝</Btn>
        <Btn danger onClick={() => nav("/")}>종료</Btn>
      </Controls>
    </Screen>
  );
}
