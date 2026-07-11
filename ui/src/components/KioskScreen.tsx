// KioskScreen — 전시 《call:one》 트랙① 《나》 무인 키오스크 (docs/EXHIBIT_PLAN §2).
// 루프: 대기 → 동의 → 설문(7) → 녹음 10초 → 준비 → 벨 → 통화(시간제한→작별+부메랑) → 소멸 → 대기.
// 음성 전용(전화기 경험 — 화면에 대화 내용 표시 안 함). 개인데이터는 세션 인메모리, 종료 즉시 폐기.
// 운영: /kiosk 로 진입. 통화 시간(초)은 localStorage.callone_kiosk_limit 로 조정(기본 110).
import { useEffect, useRef, useState } from "react";
import styled, { css, keyframes } from "styled-components";
import { CallSocket, ExhibitEvents, exhibitEvent, exhibitPersona, exhibitCount, exhibitDissolve, pcmToWavB64, type ExhibitCount } from "../api/calloneClient";
import Wordmark from "./Wordmark";

/* ── 스타일: 전시 언어(종이·잉크·주홍) 그대로, 관람 거리용 큰 활자 ── */
const blink = keyframes`0%,100%{opacity:1} 50%{opacity:0.15}`;
const pulse = keyframes`0%,100%{opacity:1} 50%{opacity:0.55}`;

const Full = styled.div`
  min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 22px; padding: 48px 36px; text-align: center;
  cursor: default; user-select: none;
`;
const BigSerif = styled.div<{ size?: number }>`
  font-family: ${(p) => p.theme.font.display}; font-weight: 600;
  font-size: ${(p) => p.size ?? 44}px; line-height: 1.35; max-width: 720px;
`;
const Mono = styled.div`
  font-family: ${(p) => p.theme.font.mono}; font-size: 13px; letter-spacing: 0.1em;
  color: ${(p) => p.theme.colors.faint};
`;
const Hint = styled.div`font-size: 15px; color: ${(p) => p.theme.colors.faint}; line-height: 1.7; max-width: 560px;`;
const Colon = styled.span<{ $blink?: boolean }>`
  color: ${(p) => p.theme.colors.accent};
  ${(p) => p.$blink && css`animation: ${blink} 1.1s steps(1) infinite;`}
`;
const Pulsing = styled.div`animation: ${pulse} 2.4s ease-in-out infinite;`;
const BigBtn = styled.button<{ $accent?: boolean }>`
  padding: 20px 44px; cursor: pointer; font-size: 20px; font-weight: 600; border: none;
  border-radius: ${(p) => p.theme.radius};
  background: ${(p) => (p.$accent ? p.theme.colors.accent : p.theme.colors.ink)};
  color: ${(p) => p.theme.colors.paper};
`;
const QuietBtn = styled.button`
  padding: 14px 24px; cursor: pointer; font-size: 15px; background: transparent;
  border: 1px solid ${(p) => p.theme.colors.line}; border-radius: ${(p) => p.theme.radius};
  color: ${(p) => p.theme.colors.faint};
`;
const CornerBtn = styled.button`
  position: fixed; right: 22px; bottom: 22px; padding: 10px 16px; cursor: pointer;
  font-size: 12px; background: transparent; color: ${(p) => p.theme.colors.faint};
  border: 1px solid ${(p) => p.theme.colors.line}; border-radius: ${(p) => p.theme.radius};
`;
const QInput = styled.input`
  width: min(620px, 86vw); padding: 12px 4px; font-size: 24px; text-align: center;
  font-family: ${(p) => p.theme.font.display}; color: ${(p) => p.theme.colors.ink};
  background: transparent; border: none; border-bottom: 2px solid ${(p) => p.theme.colors.ink};
  border-radius: 0;
  &::placeholder { color: ${(p) => p.theme.colors.line}; }
  &:focus { outline: none; border-bottom-color: ${(p) => p.theme.colors.accent}; }
`;
const SliderRow = styled.div`
  display: flex; align-items: center; gap: 18px; width: min(560px, 84vw);
  & span { font-size: 15px; color: ${(p) => p.theme.colors.faint}; white-space: nowrap; }
  & input[type="range"] { flex: 1; accent-color: ${(p) => p.theme.colors.accent}; }
`;
const Level = styled.div<{ $v: number }>`
  width: min(420px, 70vw); height: 4px; background: ${(p) => p.theme.colors.line};
  & > div { height: 100%; width: ${(p) => Math.min(100, p.$v * 700)}%;
    background: ${(p) => p.theme.colors.accent}; transition: width 0.08s; }
`;
const Foot = styled.div`
  position: fixed; left: 0; right: 0; bottom: 26px; text-align: center;
  font-family: ${(p) => p.theme.font.mono}; font-size: 12px; color: ${(p) => p.theme.colors.faint};
`;

/* ── 상수 ── */
// 낭독문 = 제로샷 ref. 우리가 정하므로 ref_text 를 이미 안다(전사 생략, 정합 100%).
const SCRIPT = "여보세요, 나야. 잘 지냈어? 오늘은 내 목소리를 잠깐 빌려줄게.";
const RECORD_S = 10;
const DISSOLVE_S = 12;      // 소멸 화면 유지(초)
const IDLE_S = 90;          // 동의/설문 방치 → 대기 화면 복귀(초)
const MAX_RINGS = 3;        // 벨 3번이면 자동 수신(무인)
const limitS = () => {
  const v = parseInt(localStorage.getItem("callone_kiosk_limit") || "", 10);
  return Number.isFinite(v) && v >= 30 ? v : 110;
};

type Q = { key: string; ask: string; hint?: string; kind: "text" | "sliders" };
const QUESTIONS: Q[] = [
  { key: "name", ask: "당신을 뭐라고 부르면 될까요?", hint: "이름이나 별명 — 통화에서 그렇게 불려요", kind: "text" },
  { key: "worry", ask: "요즘 가장 큰 고민 하나.", kind: "text" },
  { key: "future", ask: "10년 뒤, 나는 어디서 무엇을 하고 있을까요?", kind: "text" },
  { key: "person", ask: "가장 아끼는 사람은 누구인가요?", hint: "호칭만 — 엄마, 동생, 지우…", kind: "text" },
  { key: "sliders", ask: "나는 어떤 사람인가요?", kind: "sliders" },
  { key: "joy", ask: "최근 나를 웃게 한 것.", kind: "text" },
  { key: "message", ask: "10년 뒤의 나에게, 한마디.", hint: "이 말은, 통화가 끝날 때 돌아옵니다", kind: "text" },
];

type Phase = "attract" | "consent" | "survey" | "record" | "prepare" | "ring" | "call" | "dissolve" | "error";

export default function KioskScreen() {
  const [phase, setPhase] = useState<Phase>("attract");
  const phaseRef = useRef<Phase>("attract");
  const setP = (p: Phase) => { phaseRef.current = p; setPhase(p); };

  const [count, setCount] = useState<ExhibitCount | null>(null);
  const [qi, setQi] = useState(0);                       // 설문 인덱스
  const [answers, setAnswers] = useState<Record<string, unknown>>({});
  const [qText, setQText] = useState("");
  const [ext, setExt] = useState(0.5);                   // 내향↔외향
  const [spo, setSpo] = useState(0.5);                   // 신중↔즉흥
  const [recLeft, setRecLeft] = useState(RECORD_S);
  const [recOn, setRecOn] = useState(false);             // 녹음 진행 중(버튼 중복 클릭 방지)
  const [level, setLevel] = useState(0);
  const [remain, setRemain] = useState(limitS());
  const [callState, setCallState] = useState<"listening" | "thinking" | "speaking">("listening");
  const callStateRef = useRef<"listening" | "thinking" | "speaking">("listening");
  const setCS = (s: "listening" | "thinking" | "speaking") => { callStateRef.current = s; setCallState(s); };

  const pcmRef = useRef<Float32Array | null>(null);      // 녹음된 ref(세션 동안만, 리셋 시 폐기)
  const boomerangRef = useRef<string>("");
  const sockRef = useRef<CallSocket | null>(null);
  const closingRef = useRef(false);                      // 우리가 끊는 중(onClose 를 장애로 안 봄)
  const endingRef = useRef(false);                       // 작별 인사 후 소멸로
  const micStopRef = useRef<() => void>(() => {});
  const playCtxRef = useRef<AudioContext | null>(null);
  const ringCtxRef = useRef<AudioContext | null>(null);
  const ringStopRef = useRef<() => void>(() => {});
  const turnAudioRef = useRef<Float32Array[]>([]);
  const speechRef = useRef(false);
  const lastVoiceRef = useRef(0);
  const startedAtRef = useRef(0);
  const idleAtRef = useRef(Date.now());

  /* ── 공용: 전부 리셋 → 대기 화면 ── */
  function resetAll() {
    closingRef.current = true;
    try { sockRef.current?.stop(); } catch { /* noop */ }
    sockRef.current = null;
    micStopRef.current(); micStopRef.current = () => {};
    ringStopRef.current(); ringStopRef.current = () => {};
    void exhibitEvent("ring_stop");                       // 물리 벨이 남아 울리는 것 방지
    try { playCtxRef.current?.close(); } catch { /* noop */ }
    try { ringCtxRef.current?.close(); } catch { /* noop */ }
    playCtxRef.current = null; ringCtxRef.current = null;
    pcmRef.current = null; boomerangRef.current = "";
    turnAudioRef.current = []; speechRef.current = false;
    endingRef.current = false; closingRef.current = false;
    setAnswers({}); setQi(0); setQText(""); setExt(0.5); setSpo(0.5);
    setRecLeft(RECORD_S); setRecOn(false); setLevel(0); setRemain(limitS()); setCS("listening");
    setP("attract");
  }

  /* ── 대기 화면: 카운터 로드 ── */
  useEffect(() => {
    if (phase === "attract") exhibitCount().then(setCount).catch(() => setCount(null));
  }, [phase]);

  /* ── 물리 전화기(GPIO 브리지) 이벤트: 후크 = 진짜 통제권 ── */
  useEffect(() => {
    const ev = new ExhibitEvents((e) => {
      const p = phaseRef.current;
      if (e === "hook_up") {
        if (p === "ring") answer();                       // 수화기 들면 통화 시작
        else if (p === "record") startRecording();        // 녹음 단계에선 들면 녹음 시작
      } else if (e === "hook_down") {
        if (p === "call") dissolve();                     // 통화 중 내려놓음 = 즉시 소멸
        else if (p === "prepare" || p === "ring") resetAll();
      }
    });
    return () => ev.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── 방치 감시: 동의/설문에서 IDLE_S 무입력 → 대기 복귀 ── */
  useEffect(() => {
    const t = setInterval(() => {
      const p = phaseRef.current;
      if ((p === "consent" || p === "survey") && Date.now() - idleAtRef.current > IDLE_S * 1000) resetAll();
    }, 5000);
    return () => clearInterval(t);
  }, []);
  const touch = () => { idleAtRef.current = Date.now(); };

  /* ── 설문 진행 ── */
  function nextQ() {
    touch();
    const q = QUESTIONS[qi];
    const a = { ...answers };
    if (q.kind === "text") a[q.key] = qText.trim();
    else { a.extraversion = ext; a.spontaneity = spo; }
    setAnswers(a); setQText("");
    if (qi + 1 < QUESTIONS.length) { setQi(qi + 1); return; }
    setP("record");
  }

  /* ── 녹음 10초: 수화기(마이크) → Float32 → ref ── */
  async function startRecording() {
    if (recOn) return;
    touch();
    setRecOn(true); setRecLeft(RECORD_S);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext({ sampleRate: 16000 });
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      const chunks: Float32Array[] = [];
      let n = 0;
      proc.onaudioprocess = (e) => {
        const pcm = new Float32Array(e.inputBuffer.getChannelData(0));
        chunks.push(pcm); n += pcm.length;
        let s = 0; for (let i = 0; i < pcm.length; i++) s += pcm[i] * pcm[i];
        setLevel(Math.sqrt(s / pcm.length));
        const left = Math.max(0, RECORD_S - n / 16000);
        setRecLeft(Math.ceil(left));
        if (left <= 0) {
          proc.disconnect(); src.disconnect(); stream.getTracks().forEach((t) => t.stop());
          void ctx.close();
          const all = new Float32Array(n);
          let off = 0; for (const c of chunks) { all.set(c, off); off += c.length; }
          pcmRef.current = all;
          void prepareCall();
        }
      };
      src.connect(proc); proc.connect(ctx.destination);
    } catch { setRecOn(false); setP("error"); }
  }

  /* ── 준비: 설문→페르소나(서버 템플릿) → 세션 init → 벨 ── */
  async function prepareCall() {
    setP("prepare");
    try {
      const name = String(answers.name || "").trim();
      const p = await exhibitPersona(name, answers, "future_self");
      boomerangRef.current = p.boomerang || "";
      // 기억 시드는 세션 한정 → 디스크 대신 카드 background 에 접어 넣는다(ephemeral).
      const bg = [p.card.background, p.memories.length ? `이미 아는 사실: ${p.memories.join(" / ")}` : ""]
        .filter(Boolean).join(" ");
      const sock = new CallSocket("kiosk", {
        onAudio: (pcm) => { turnAudioRef.current.push(pcm); },
        onReply: () => { /* 화면에 대화 내용 표시 안 함(전화기 경험·프라이버시) */ },
        onAudioEnd: () => playTurn(),
        onReady: () => { if (phaseRef.current === "prepare") { setP("ring"); startRing(); } },
        onClose: () => { if (!closingRef.current) setP("error"); },
      });
      sockRef.current = sock;
      sock.sessionInit({
        ref_audio_b64: pcmToWavB64(pcmRef.current!, 16000),
        ref_text: SCRIPT,                       // 낭독문 그대로 — 전사 불필요, 정합 100%
        ...p.card,
        background: bg || undefined,
      });
    } catch { setP("error"); }
  }

  /* ── 벨: 440+480Hz, 1초 울림/2초 쉼 — MAX_RINGS 후 자동 수신 ── */
  function startRing() {
    void exhibitEvent("ring_start");                      // GPIO 브리지 → 물리 벨(솔레노이드)
    const ctx = new AudioContext();
    ringCtxRef.current = ctx;
    const gain = ctx.createGain(); gain.gain.value = 0; gain.connect(ctx.destination);
    const o1 = ctx.createOscillator(); o1.frequency.value = 440;
    const o2 = ctx.createOscillator(); o2.frequency.value = 480;
    o1.connect(gain); o2.connect(gain); o1.start(); o2.start();
    let rings = 0;
    const ringOnce = () => {
      gain.gain.setTargetAtTime(0.12, ctx.currentTime, 0.01);
      gain.gain.setTargetAtTime(0, ctx.currentTime + 1.0, 0.02);
      rings += 1;
      if (rings >= MAX_RINGS) { setTimeout(() => { if (phaseRef.current === "ring") answer(); }, 1600); }
    };
    ringOnce();
    const iv = setInterval(() => { if (phaseRef.current === "ring" && rings < MAX_RINGS) ringOnce(); }, 3000);
    ringStopRef.current = () => { clearInterval(iv); try { o1.stop(); o2.stop(); void ctx.close(); } catch { /* noop */ } };
  }

  /* ── 수신 → 통화 ── */
  async function answer() {
    void exhibitEvent("ring_stop");                       // 물리 벨 정지
    ringStopRef.current(); ringStopRef.current = () => {};
    setRemain(limitS());
    startedAtRef.current = Date.now();
    setP("call"); setCS("listening");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext({ sampleRate: 16000 });
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = (e) => {
        if (callStateRef.current === "speaking") return;   // 에코 차단(CallScreen 과 동일)
        const pcm = new Float32Array(e.inputBuffer.getChannelData(0));
        sockRef.current?.sendAudio(pcm);
        let s = 0; for (let i = 0; i < pcm.length; i++) s += pcm[i] * pcm[i];
        const rms = Math.sqrt(s / pcm.length);
        const now = performance.now();
        if (rms > 0.015) { speechRef.current = true; lastVoiceRef.current = now; }
        else if (speechRef.current && callStateRef.current === "listening"
                 && now - lastVoiceRef.current > 900) {
          speechRef.current = false; setCS("thinking"); sockRef.current?.endTurn();
        }
      };
      src.connect(proc); proc.connect(ctx.destination);
      micStopRef.current = () => {
        proc.disconnect(); src.disconnect(); stream.getTracks().forEach((t) => t.stop());
        void ctx.close();
      };
    } catch { setP("error"); }
  }

  /* ── 시간제한: 매초 잔여 계산 → 0 이면 듣는 중일 때 작별(+부메랑) ── */
  useEffect(() => {
    if (phase !== "call") return;
    const t = setInterval(() => {
      const el = (Date.now() - startedAtRef.current) / 1000;
      const r = Math.max(0, Math.ceil(limitS() - el));
      setRemain(r);
      if (r <= 0 && !endingRef.current && callStateRef.current === "listening") {
        endingRef.current = true;
        sockRef.current?.farewell(boomerangRef.current || undefined);
        setCS("thinking");
      }
      if (el > limitS() + 60) dissolve();       // 안전핀: 어떤 이유로든 안 끝나면 강제 소멸
    }, 1000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  /* ── 재생(음성 전용) ── */
  function playTurn() {
    const chunks = turnAudioRef.current; turnAudioRef.current = [];
    if (!chunks.length) {
      if (endingRef.current) { dissolve(); return; }
      setCS("listening"); return;
    }
    let ctx = playCtxRef.current;
    if (!ctx) { ctx = new AudioContext({ sampleRate: 24000 }); playCtxRef.current = ctx; }
    if (ctx.state === "suspended") void ctx.resume();
    const total = chunks.reduce((n, c) => n + c.length, 0);
    const audio = new Float32Array(total);
    let off = 0; for (const c of chunks) { audio.set(c, off); off += c.length; }
    const buf = ctx.createBuffer(1, audio.length, 24000);
    buf.copyToChannel(audio, 0);
    const node = ctx.createBufferSource();
    node.buffer = buf; node.connect(ctx.destination);
    node.onended = () => {
      speechRef.current = false;
      if (endingRef.current) { dissolve(); return; }
      setCS("listening");
    };
    setCS("speaking");
    node.start();
  }

  /* ── 소멸: 세션 폐기 + 카운터 +1 → 잠시 후 대기 화면 ── */
  function dissolve() {
    if (phaseRef.current === "dissolve") return;
    closingRef.current = true;
    micStopRef.current(); micStopRef.current = () => {};
    try { sockRef.current?.stop(); } catch { /* noop */ }
    sockRef.current = null;
    setP("dissolve");
    exhibitDissolve().then(setCount).catch(() => { /* noop */ });
    setTimeout(() => { if (phaseRef.current === "dissolve") resetAll(); }, DISSOLVE_S * 1000);
  }

  /* ── 장애 화면: 8초 후 자동 복귀 ── */
  useEffect(() => {
    if (phase !== "error") return;
    const t = setTimeout(resetAll, 8000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  /* ─────────── 화면 ─────────── */
  if (phase === "attract") {
    return (
      <Full onPointerDown={() => { touch(); setP("consent"); }}>
        <Wordmark size={26} blink />
        <BigSerif size={56}>여보세요<Colon>,</Colon> 나야<Colon>.</Colon></BigSerif>
        <Hint>10초를 빌려주시면, 10년 뒤의 당신에게서 전화가 옵니다.</Hint>
        <Pulsing><Mono>화면을 눌러 시작</Mono></Pulsing>
        {count && count.today > 0 && (
          <Foot>오늘 {count.today}개의 목소리가 태어나고, 사라졌습니다</Foot>
        )}
      </Full>
    );
  }

  if (phase === "consent") {
    return (
      <Full onPointerDown={touch}>
        <Mono>시작하기 전에</Mono>
        <BigSerif size={34}>10초를 빌려주세요</BigSerif>
        <Hint>
          당신의 목소리 10초로 즉석에서 AI 클론을 만들어, 잠깐 통화합니다.<br />
          목소리와 대화는 이 방을 떠나지 않고, 수화기를 내려놓는 순간 삭제됩니다.<br />
          통화 상대는 AI 가 만든 클론 음성입니다.
        </Hint>
        <div style={{ display: "flex", gap: 16 }}>
          <BigBtn onClick={() => { touch(); setP("survey"); }}>동의하고 시작</BigBtn>
          <QuietBtn onClick={resetAll}>그만둘래요</QuietBtn>
        </div>
      </Full>
    );
  }

  if (phase === "survey") {
    const q = QUESTIONS[qi];
    return (
      <Full onPointerDown={touch}>
        <Mono>{String(qi + 1).padStart(2, "0")} / {String(QUESTIONS.length).padStart(2, "0")}</Mono>
        <BigSerif size={34}>{q.ask}</BigSerif>
        {q.hint && <Hint>{q.hint}</Hint>}
        {q.kind === "text" ? (
          <QInput autoFocus value={qText} onChange={(e) => { touch(); setQText(e.target.value); }}
            onKeyDown={(e) => e.key === "Enter" && nextQ()} placeholder="여기에 적어주세요" />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
            <SliderRow>
              <span>혼자가 편해요</span>
              <input type="range" min={0} max={1} step={0.01} value={ext}
                onChange={(e) => { touch(); setExt(parseFloat(e.target.value)); }} />
              <span>사람들이 좋아요</span>
            </SliderRow>
            <SliderRow>
              <span>계획대로</span>
              <input type="range" min={0} max={1} step={0.01} value={spo}
                onChange={(e) => { touch(); setSpo(parseFloat(e.target.value)); }} />
              <span>즉흥적으로</span>
            </SliderRow>
          </div>
        )}
        <div style={{ display: "flex", gap: 14 }}>
          <BigBtn onClick={nextQ}>{qi + 1 < QUESTIONS.length ? "다음" : "다 적었어요"}</BigBtn>
          {q.kind === "text" && <QuietBtn onClick={() => { setQText(""); nextQ(); }}>건너뛰기</QuietBtn>}
        </div>
        <CornerBtn onClick={resetAll}>처음으로</CornerBtn>
      </Full>
    );
  }

  if (phase === "record") {
    return (
      <Full>
        <Mono>수화기를 들고, 소리 내어 읽어주세요</Mono>
        <BigSerif size={40}>“{SCRIPT}”</BigSerif>
        {recOn ? (<>
          <BigSerif size={64}><Colon $blink>{recLeft}</Colon></BigSerif>
          <Level $v={level}><div /></Level>
        </>) : (
          <BigBtn $accent onClick={startRecording}>녹음 시작</BigBtn>
        )}
        <CornerBtn onClick={resetAll}>처음으로</CornerBtn>
      </Full>
    );
  }

  if (phase === "prepare") {
    return (
      <Full>
        <Pulsing><BigSerif size={34}>목소리를 만드는 중<Colon $blink>…</Colon></BigSerif></Pulsing>
        <Hint>잠시 후, 전화가 옵니다. 수화기 근처에서 기다려주세요.</Hint>
      </Full>
    );
  }

  if (phase === "ring") {
    return (
      <Full>
        <Pulsing><BigSerif size={44}>전화가 오고 있습니다</BigSerif></Pulsing>
        <Hint>받으면, 평소처럼 "여보세요" 하고 인사해보세요.</Hint>
        <BigBtn $accent onClick={answer}>받기</BigBtn>
      </Full>
    );
  }

  if (phase === "call") {
    const mm = String(Math.floor(remain / 60)).padStart(2, "0");
    const ss = String(remain % 60).padStart(2, "0");
    return (
      <Full>
        <Mono>10년 뒤의 나 — AI 클론 음성</Mono>
        <BigSerif size={30}>
          {callState === "listening" ? "듣고 있어요" : callState === "thinking" ? "생각하는 중" : "말하는 중"}
        </BigSerif>
        <Mono>{mm}<Colon $blink>:</Colon>{ss}</Mono>
        {remain <= 15 && !endingRef.current && <Hint>이제 곧, 마지막 인사를 나눌 시간이에요.</Hint>}
        <CornerBtn onClick={() => {
          if (!endingRef.current) {
            endingRef.current = true;
            sockRef.current?.farewell(boomerangRef.current || undefined);
            setCS("thinking");
          }
        }}>먼저 인사하고 끊기</CornerBtn>
      </Full>
    );
  }

  if (phase === "dissolve") {
    return (
      <Full>
        <BigSerif size={44}>목소리가 사라졌습니다<Colon>.</Colon></BigSerif>
        <Hint>당신의 10초는 이 방을 떠나지 않았습니다.</Hint>
        {count && <Mono>오늘 {count.today}번째 목소리</Mono>}
        <BigSerif size={24} style={{ marginTop: 18 }}>방금 당신과 통화한 것은, 누구였나요?</BigSerif>
      </Full>
    );
  }

  return (
    <Full>
      <BigSerif size={34}>잠시 쉬어갑니다</BigSerif>
      <Hint>연결이 고르지 않아요. 곧 처음 화면으로 돌아갑니다.</Hint>
    </Full>
  );
}
