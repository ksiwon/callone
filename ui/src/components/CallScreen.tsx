// CallScreen — 영상통화: 설정(음성·사진·페르소나·대화 불러오기) → 통화(음성+얼굴) → 내보내기.
// 프라이버시: 음성/사진/대화는 **브라우저(클라)가 소유**. 통화 시작 시 서버로 보내 인메모리만 쓰고,
// 끊기면 서버에서 즉시 폐기(디스크·로그에 안 남음). 대화 이력은 localStorage + 파일 export/import.
import { useEffect, useRef, useState } from "react";
import styled from "styled-components";
import { useParams, useNavigate } from "react-router-dom";
import { CallSocket, fileToBase64, previewVoice, type Turn, type SessionInit } from "../api/calloneClient";

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
  width: 100%; max-width: 460px; display: flex; flex-direction: column; gap: 12px;
  color: ${(p) => p.theme.colors.text};
  & label { font-size: 13px; color: ${(p) => p.theme.colors.sub}; }
  & input[type="text"], & textarea {
    width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #2a3a52;
    background: #0c1422; color: #fff; font-size: 14px;
  }
`;
/* ── 단계형 셋업(stepper): 상단 진행 점 + 단계별 본문 + 이전/다음 ── */
const Steps = styled.div`display: flex; gap: 8px; width: 100%; max-width: 460px; margin-bottom: 4px;`;
const StepDot = styled.div<{ on: boolean; done: boolean }>`
  flex: 1; height: 6px; border-radius: 3px;
  background: ${(p) => (p.done ? p.theme.colors.accent : p.on ? p.theme.colors.primary : p.theme.colors.border)};
`;
const StepTitle = styled.div`font-size: 18px; font-weight: 700; color: ${(p) => p.theme.colors.text};`;
const StepHint = styled.div`font-size: 13px; color: ${(p) => p.theme.colors.sub}; margin: 2px 0 8px;`;
const Preview = styled.button`
  align-self: flex-start; padding: 9px 14px; border-radius: 18px; border: none; cursor: pointer;
  background: ${(p) => p.theme.colors.accent}; color: #06202b; font-weight: 700; font-size: 13px;
  &:disabled { opacity: 0.5; cursor: default; }
`;
const Note = styled.div<{ err?: boolean }>`
  font-size: 13px; color: ${(p) => (p.err ? p.theme.colors.danger : p.theme.colors.sub)};
`;
const Thumb = styled.img`
  width: 120px; height: 120px; object-fit: cover; border-radius: 12px;
  border: 1px solid ${(p) => p.theme.colors.border};
`;
const Fold = styled.details`
  border: 1px solid ${(p) => p.theme.colors.border}; border-radius: 10px; padding: 8px 12px;
  & > summary { cursor: pointer; font-size: 13px; color: ${(p) => p.theme.colors.sub}; }
  & > div { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
`;
const Ghost = styled.button`
  padding: 12px 18px; border-radius: 24px; cursor: pointer; font-size: 14px;
  background: transparent; color: ${(p) => p.theme.colors.sub}; border: 1px solid ${(p) => p.theme.colors.border};
  &:disabled { opacity: 0.4; cursor: default; }
`;
/* ── 예시 캐릭터 칩(원클릭 프리셋) ── */
const PresetRow = styled.div`display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 4px;`;
const Chip = styled.button`
  padding: 7px 12px; border-radius: 16px; cursor: pointer; font-size: 13px;
  background: ${(p) => p.theme.colors.surface}; color: ${(p) => p.theme.colors.text};
  border: 1px solid ${(p) => p.theme.colors.border};
  &:hover { border-color: ${(p) => p.theme.colors.accent}; }
`;

// 캐릭터 카드 프리셋 — 누르면 설정 칸이 채워진다(수정 가능). example_dialogue 는 캐릭터챗 말투의
// 최대 레버라(웹조사: 말투·감정·관계가 드러나는 구체 예시대화일수록 몰입↑) 각 프리셋에 2~3 교환 수록.
// 실존 인물 복제 시 이름·관계만 바꾸고 말투 예시를 본인에 맞게 손보면 됨.
type CharCard = {
  persona: string; userPersona: string; personality: string;
  background: string; situation: string; firstMessage: string; exampleDialogue: string;
};
const PRESETS: { label: string; card: CharCard }[] = [
  {
    label: "소꿉친구",
    card: {
      persona: "소꿉친구 나은. 20대 후반.",
      userPersona: "나은의 오랜 소꿉친구",
      personality: "밝고 장난기 많음. 반말로 짧고 편하게, 말끝에 ㅋㅋ·~ 자주.",
      background: "초등학교 때부터 단짝. 지금도 가끔 연락하는 사이.",
      situation: "오랜만에 갑자기 전화함.",
      firstMessage: "야 오랜만이다! 살아있었네?",
      exampleDialogue:
        "나: 요즘 뭐하고 지내?\n나은: 그냥저냥~ 회사 다니고 주말엔 뒹굴뒹굴ㅋㅋ 넌 잘 지냈어?\n" +
        "나: 좀 힘들었어.\n나은: 아이고 무슨 일 있었어? 말해봐, 다 들어줄게.",
    },
  },
  {
    label: "엄마",
    card: {
      persona: "엄마. 60대.",
      userPersona: "엄마의 자식",
      personality: "다정하고 걱정 많음. 반말, 경상도 억양 살짝(~노/~나/마).",
      background: "객지에 나가 사는 자식을 늘 챙김.",
      situation: "밥은 먹었는지 안부 전화.",
      firstMessage: "어이구 내 새끼, 밥은 묵었나?",
      exampleDialogue:
        "나: 엄마 나 왔어.\n엄마: 아이고 우리 딸~ 얼굴이 영 안 좋다, 끼니는 챙기 묵나?\n" +
        "나: 요즘 바빠서 잘 못 먹어.\n엄마: 그라믄 안 된다, 밥 거르지 말고 꼭 챙기 묵어라이.",
    },
  },
  {
    label: "연인",
    card: {
      persona: "연인 지호. 20대 후반.",
      userPersona: "지호의 애인",
      personality: "다정하고 장난스러움. 편한 반말에 가끔 애교.",
      background: "1년 넘게 만난 사이. 자주 통화함.",
      situation: "자기 전 안부 전화.",
      firstMessage: "자기야 뭐해~ 보고 싶어서 전화했어.",
      exampleDialogue:
        "나: 오늘 좀 피곤하다.\n지호: 에구 많이 힘들었어? 오늘은 일찍 쉬어, 무리하지 말고.\n" +
        "나: 응 그럴게.\n지호: 그래~ 우리 자기 푹 자고 좋은 꿈 꿔. 사랑해.",
    },
  },
  {
    label: "오랜 친구",
    card: {
      persona: "고향 친구 정민. 30대.",
      userPersona: "정민의 고향 친구",
      personality: "무뚝뚝하지만 정 많음. 짧은 반말, 츤데레.",
      background: "고향에서 같이 자란 죽마고우.",
      situation: "오랜만에 연락.",
      firstMessage: "어 웬일이냐. 살아는 있었네.",
      exampleDialogue:
        "나: 잘 지냈어?\n정민: 뭐 그냥 똑같지. 넌 얼굴 보기 힘드네.\n" +
        "나: 다음 달에 한번 내려갈까 해.\n정민: ...오면 연락해라. 술이나 한잔 하자.",
    },
  },
];

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

  // 단계형 셋업 상태
  const [step, setStep] = useState(1);                  // 1 목소리 · 2 얼굴 · 3 캐릭터 · 4 시작
  const [refText, setRefText] = useState("");           // 참조 음성 전사(자동→수정 가능, 유사도↑)
  const [previewing, setPreviewing] = useState(false);
  const [previewMsg, setPreviewMsg] = useState("");     // 미리듣기 안내/에러
  const [photoUrl, setPhotoUrl] = useState("");         // 사진 미리보기 objectURL
  const [foldOpen, setFoldOpen] = useState(false);      // 캐릭터 '더 자세히' 펼침(프리셋 적용 시 자동)
  const previewCtxRef = useRef<AudioContext | null>(null);  // 미리듣기 재생 전용

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
      ref_text: refText.trim() || undefined,   // 미리듣기에서 확정/수정한 전사 → 유사도↑
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

  // 예시 캐릭터 프리셋 적용 — 칸 한 번에 채움(이후 자유 수정). 목소리/사진은 안 건드림.
  // 채워진 예시대화가 접힌 '더 자세히' 안에 있으니 자동으로 펼쳐 바로 보이게 한다.
  function applyPreset(c: CharCard) {
    setPersona(c.persona); setUserPersona(c.userPersona); setPersonality(c.personality);
    setBackground(c.background); setSituation(c.situation);
    setFirstMessage(c.firstMessage); setExampleDialogue(c.exampleDialogue);
    setFoldOpen(true);
  }

  // 사진 선택 → 미리보기 objectURL(이전 것 해제).
  function pickPhoto(f: File | null) {
    setPhotoFile(f);
    setPhotoUrl((old) => { if (old) URL.revokeObjectURL(old); return f ? URL.createObjectURL(f) : ""; });
  }

  // 복제 목소리 미리듣기 — 업로드 음성으로 짧은 문장 합성해 재생(통화 전 유사도 확인).
  async function runPreview() {
    if (!voiceFile) return;
    setPreviewing(true); setPreviewMsg("합성 중… (첫 회는 전사 포함 ~수 초)");
    try {
      const b64 = await fileToBase64(voiceFile);
      const { refText: rt, audio, sr } = await previewVoice(b64, { refText: refText.trim() || undefined });
      if (rt && !refText.trim()) setRefText(rt);   // 자동 전사 결과 채움(비어있을 때만)
      let ctx = previewCtxRef.current;
      if (!ctx || ctx.sampleRate !== sr) { try { ctx?.close(); } catch { /* noop */ } ctx = new AudioContext({ sampleRate: sr }); previewCtxRef.current = ctx; }
      if (ctx.state === "suspended") await ctx.resume();
      const buf = ctx.createBuffer(1, audio.length, sr);
      buf.getChannelData(0).set(audio);   // copyToChannel 대신 set — 버퍼 타입(ArrayBufferLike) 무관, 타입세이프
      const node = ctx.createBufferSource(); node.buffer = buf; node.connect(ctx.destination); node.start();
      setPreviewMsg("▶ 재생 중 — 본인 목소리 같으면 다음으로.");
    } catch (e: any) {
      setPreviewMsg(`⚠️ ${e?.message || "미리듣기 실패"} (cosyvoice-server 확인)`);
    } finally { setPreviewing(false); }
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
      // 립싱크 보정: Ditto 스트리밍 초반의 뉴트럴(무동작) 프레임 때문에 입이 음성보다 늦게 시작한다.
      // → 앞쪽 skip 프레임을 건너뛰고, **남은 프레임을 오디오 전체 길이에 펼쳐** 마지막 프레임이 음성
      //   끝과 일치하게 한다(앞은 당기고 끝은 안 비움 — 영상이 음성보다 일찍 끝나는 것까지 방지).
      // AV_LEAD_S = 건너뛸 앞 구간(초). 브라우저 즉시 튜닝: `localStorage.callone_av_lead = 0.35` 후 새로고침
      //   (입이 여전히 늦으면 값↑, 너무 앞서면 값↓). 미설정 시 기본 0.3s.
      const AV_LEAD_S = (() => {
        const v = parseFloat(localStorage.getItem("callone_av_lead") || "");
        return Number.isFinite(v) ? v : 0.3;
      })();
      const skip = Math.min(bitmaps.length - 1, Math.max(0, Math.round(AV_LEAD_S / dur * bitmaps.length)));
      const span = bitmaps.length - 1 - skip;                 // 남은 프레임 → 오디오 전체에 균등 배치(끝 일치)
      drawBitmap(bitmaps[skip]);
      const startPerf = performance.now() + (startAt - ctx.currentTime) * 1000;
      let last = -1;
      const tick = () => {
        const el = (performance.now() - startPerf) / 1000;   // 오디오 경과(초)
        if (el >= 0) {
          const idx = Math.min(bitmaps.length - 1, Math.max(0, skip + Math.floor(el / dur * span)));
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
    try { previewCtxRef.current?.close(); } catch { /* noop */ }
    if (photoUrl) URL.revokeObjectURL(photoUrl);
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

  // ---- 설정 화면(통화 전): 단계형 — ①목소리 ②얼굴 ③캐릭터 ④시작 ----
  if (!started) {
    const STEP_META = [
      ["목소리", "복제할 목소리를 올리고 미리 들어보세요 (필수)"],
      ["얼굴", "영상통화용 사진 — 없으면 음성통화 (선택)"],
      ["캐릭터", "누구이고 나와 무슨 사이인지 (선택)"],
      ["시작", "이전 대화 이어가기 · 통화 시작"],
    ];
    const next = () => setStep((s) => Math.min(4, s + 1));
    const prev = () => setStep((s) => Math.max(1, s - 1));
    return (
      <Screen>
        <Who><Big>{id}</Big><Status>통화 준비 · 개인데이터는 내 브라우저만 보관</Status></Who>
        <Steps>
          {STEP_META.map((_, i) => <StepDot key={i} on={step === i + 1} done={step > i + 1} />)}
        </Steps>
        <Setup>
          <StepTitle>{step}. {STEP_META[step - 1][0]}</StepTitle>
          <StepHint>{STEP_META[step - 1][1]}</StepHint>

          {step === 1 && (<>
            <label>화자 음성 (7~10초, 깨끗한 wav/mp3)</label>
            <input type="file" accept="audio/*" onChange={(e) => { setVoiceFile(e.target.files?.[0] ?? null); setPreviewMsg(""); }} />
            {voiceFile && <Preview onClick={runPreview} disabled={previewing}>{previewing ? "합성 중…" : "🔊 복제 목소리 미리듣기"}</Preview>}
            {previewMsg && <Note err={previewMsg.startsWith("⚠️")}>{previewMsg}</Note>}
            {voiceFile && (<>
              <label>참조 음성 내용 (전사 — 정확할수록 유사도↑, 수정 가능)</label>
              <input type="text" value={refText} onChange={(e) => setRefText(e.target.value)} placeholder="미리듣기를 누르면 자동으로 채워집니다" />
            </>)}
          </>)}

          {step === 2 && (<>
            <label>증명사진 (얼굴, jpg/png)</label>
            <input type="file" accept="image/*" onChange={(e) => pickPhoto(e.target.files?.[0] ?? null)} />
            {photoUrl
              ? <Thumb src={photoUrl} alt="얼굴 미리보기" />
              : <Note>사진을 올리면 영상통화(움직이는 얼굴), 없으면 음성통화로 진행돼요.</Note>}
          </>)}

          {step === 3 && (<>
            <label>예시 캐릭터 빠르게 넣기 (누르면 아래 칸이 채워져요 — 자유롭게 수정)</label>
            <PresetRow>
              {PRESETS.map((p) => (
                <Chip key={p.label} type="button" onClick={() => applyPreset(p.card)}>{p.label}</Chip>
              ))}
            </PresetRow>
            <label>이름·관계 (이 사람은 누구?)</label>
            <input type="text" value={persona} onChange={(e) => setPersona(e.target.value)} placeholder="예: 소꿉친구 나은" />
            <label>나는 누구? (상대 기준)</label>
            <input type="text" value={userPersona} onChange={(e) => setUserPersona(e.target.value)} placeholder="예: 나은의 소꿉친구" />
            <label>성격·말투</label>
            <input type="text" value={personality} onChange={(e) => setPersonality(e.target.value)} placeholder="예: 밝고 장난기 많음. 반말로 짧고 편하게." />
            <Fold open={foldOpen} onToggle={(e) => setFoldOpen((e.currentTarget as HTMLDetailsElement).open)}>
              <summary>더 자세히 (배경·상황·첫 마디·예시 말투)</summary>
              <div>
                <label>배경</label>
                <input type="text" value={background} onChange={(e) => setBackground(e.target.value)} placeholder="예: 초등학교 때부터 단짝, 지금도 같은 동네 살아." />
                <label>지금 상황</label>
                <input type="text" value={situation} onChange={(e) => setSituation(e.target.value)} placeholder="예: 오랜만에 갑자기 전화함." />
                <label>첫 마디</label>
                <input type="text" value={firstMessage} onChange={(e) => setFirstMessage(e.target.value)} placeholder="예: 야 오랜만이다! 살아있었네?" />
                <label>예시 말투 (이렇게 말함)</label>
                <textarea rows={3} value={exampleDialogue} onChange={(e) => setExampleDialogue(e.target.value)}
                  placeholder={"예:\n나: 뭐해?\n나은: 그냥 침대에서 뒹굴뒹굴~ 넌 밥은 먹었어?"} />
              </div>
            </Fold>
          </>)}

          {step === 4 && (<>
            <label>이전 대화 불러오기 (이어하기) {turns > 0 ? `· 저장된 ${turns}턴 있음` : ""}</label>
            <input type="file" accept="application/json" onChange={(e) => e.target.files?.[0] && importHistory(e.target.files[0])} />
            <Note>
              {voiceFile ? "✓ 목소리" : "· 목소리 없음"}{photoFile ? " · ✓ 얼굴" : " · 음성통화"}{persona ? ` · ✓ ${persona}` : ""}
            </Note>
            {turns > 0 && (
              <Controls style={{ justifyContent: "flex-start" }}>
                <Btn onClick={exportHistory}>대화 내보내기</Btn>
                <Btn danger onClick={clearHistory}>🗑 기억 리셋</Btn>
              </Controls>
            )}
          </>)}
        </Setup>

        <Controls>
          {step > 1 ? <Ghost onClick={prev}>← 이전</Ghost> : <Ghost onClick={() => nav("/")}>취소</Ghost>}
          {step < 4
            ? <Btn onClick={next} disabled={step === 1 && !voiceFile}>다음 →</Btn>
            : <Btn onClick={startCall} disabled={!voiceFile}>📞 통화 시작</Btn>}
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
