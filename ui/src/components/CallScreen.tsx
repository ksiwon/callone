// CallScreen — 영상통화: 설정(음성·사진·페르소나·대화 불러오기) → 통화(음성+얼굴) → 내보내기.
// 프라이버시: 음성/사진/대화는 **브라우저(클라)가 소유**. 통화 시작 시 서버로 보내 인메모리만 쓰고,
// 끊기면 서버에서 즉시 폐기(디스크·로그에 안 남음). 대화 이력은 localStorage + 파일 export/import.
// 디자인: call:one 전시 언어 — 종이·잉크·주홍, 서식형 셋업, 대본형 통화 기록. 에러 메시지는 "!" 접두.
import { useEffect, useRef, useState } from "react";
import styled from "styled-components";
import { useParams, useNavigate } from "react-router-dom";
import { CallSocket, fileToBase64, previewVoice, listVoicePresets, analyzeVoiceStart, analyzeVoiceStatus, analyzeVoiceSave, analyzeVoiceRemember, listVoiceJobs, exhibitInterviewer, rememberCall, type Turn, type SessionInit, type VoicePreset, type AnalyzeStatus } from "../api/calloneClient";
import Wordmark from "./Wordmark";

/* ── 셋업(서식) ── */
const Screen = styled.div`
  min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  padding: 44px 28px 36px; gap: 4px;
`;
const SetupHead = styled.div`
  width: 100%; max-width: 520px; display: flex; align-items: baseline;
  justify-content: space-between; margin-bottom: 6px;
`;
const HeadNote = styled.div`
  font-family: ${(p) => p.theme.font.mono}; font-size: 11px; letter-spacing: 0.14em;
  color: ${(p) => p.theme.colors.faint}; text-transform: uppercase;
`;
const StepIndex = styled.div`
  width: 100%; max-width: 520px; display: flex; gap: 22px; margin: 14px 0 2px;
  border-bottom: 2px solid ${(p) => p.theme.colors.ink}; padding-bottom: 10px;
`;
const StepItem = styled.div<{ $on: boolean; $done: boolean }>`
  font-family: ${(p) => p.theme.font.mono}; font-size: 12px; letter-spacing: 0.06em;
  color: ${(p) => (p.$on ? p.theme.colors.ink : p.theme.colors.faint)};
  font-weight: ${(p) => (p.$on ? 700 : 400)};
  & em { font-style: normal; color: ${(p) => (p.$done || p.$on ? p.theme.colors.accent : "inherit")}; }
`;
const Setup = styled.div`
  width: 100%; max-width: 520px; display: flex; flex-direction: column; gap: 14px; padding-top: 18px;
  & label { font-size: 13px; color: ${(p) => p.theme.colors.faint}; }
  & input[type="text"], & textarea {
    width: 100%; padding: 9px 2px; font-size: 15px; color: ${(p) => p.theme.colors.ink};
    background: transparent; border: none; border-bottom: 1px solid ${(p) => p.theme.colors.line};
    border-radius: 0;
    &::placeholder { color: ${(p) => p.theme.colors.line}; }
    &:focus { outline: none; border-bottom: 2px solid ${(p) => p.theme.colors.ink}; }
  }
`;
const StepTitle = styled.div`
  font-family: ${(p) => p.theme.font.display}; font-size: 24px; font-weight: 600;
  & span { font-family: ${(p) => p.theme.font.mono}; font-size: 14px; font-weight: 400;
    color: ${(p) => p.theme.colors.accent}; margin-right: 10px; }
`;
const StepHint = styled.div`font-size: 13px; color: ${(p) => p.theme.colors.faint}; margin: -6px 0 6px;`;
const Seg = styled.div`display: flex; border: 1px solid ${(p) => p.theme.colors.ink};`;
const SegBtn = styled.button<{ $on: boolean }>`
  flex: 1; padding: 10px 6px; cursor: pointer; font-size: 13px; font-weight: 600; border: none;
  background: ${(p) => (p.$on ? p.theme.colors.ink : "transparent")};
  color: ${(p) => (p.$on ? p.theme.colors.paper : p.theme.colors.faint)};
  & + & { border-left: 1px solid ${(p) => p.theme.colors.line}; }
`;
const FileBox = styled.label`
  display: block; border: 1px dashed ${(p) => p.theme.colors.faint};
  padding: 16px 14px; cursor: pointer; text-align: center;
  font-size: 13px; color: ${(p) => p.theme.colors.faint} !important;
  & input { display: none; }
  &:hover { border-color: ${(p) => p.theme.colors.ink}; color: ${(p) => p.theme.colors.ink} !important; }
  & b { color: ${(p) => p.theme.colors.ink}; font-weight: 600; }
`;
const Btn = styled.button<{ danger?: boolean }>`
  padding: 11px 18px; cursor: pointer; font-size: 14px; font-weight: 600;
  border-radius: ${(p) => p.theme.radius};
  border: 1px solid ${(p) => (p.danger ? p.theme.colors.accent : p.theme.colors.ink)};
  background: ${(p) => (p.danger ? p.theme.colors.accent : "transparent")};
  color: ${(p) => (p.danger ? p.theme.colors.onAccent : p.theme.colors.ink)};
  &:hover:not(:disabled) { background: ${(p) => (p.danger ? "#9e3524" : p.theme.colors.ink)};
    color: ${(p) => p.theme.colors.paper}; }
  &:disabled { opacity: 0.35; cursor: default; }
`;
const Solid = styled.button`
  padding: 13px 26px; cursor: pointer; font-size: 15px; font-weight: 600; border: none;
  border-radius: ${(p) => p.theme.radius};
  background: ${(p) => p.theme.colors.ink}; color: ${(p) => p.theme.colors.paper};
  &:hover:not(:disabled) { background: ${(p) => p.theme.colors.accent}; color: ${(p) => p.theme.colors.onAccent}; }
  &:disabled { opacity: 0.35; cursor: default; }
`;
const Ghost = styled.button`
  padding: 13px 18px; cursor: pointer; font-size: 14px; border: none; background: transparent;
  color: ${(p) => p.theme.colors.faint};
  &:hover:not(:disabled) { color: ${(p) => p.theme.colors.ink}; }
  &:disabled { opacity: 0.4; cursor: default; }
`;
const Preview = styled.button`
  align-self: flex-start; padding: 9px 16px; cursor: pointer; font-size: 13px; font-weight: 600;
  border: 1px solid ${(p) => p.theme.colors.ink}; background: transparent; border-radius: ${(p) => p.theme.radius};
  color: ${(p) => p.theme.colors.ink};
  &:hover:not(:disabled) { background: ${(p) => p.theme.colors.ink}; color: ${(p) => p.theme.colors.paper}; }
  &:disabled { opacity: 0.4; cursor: default; }
`;
const Note = styled.div<{ err?: boolean }>`
  font-size: 13px; line-height: 1.65;
  color: ${(p) => (p.err ? p.theme.colors.accent : p.theme.colors.faint)};
`;
const Thumb = styled.img`
  width: 128px; height: 128px; object-fit: cover;
  border: 1px solid ${(p) => p.theme.colors.ink}; padding: 3px; background: #fff;
`;
const Fold = styled.details`
  border-top: 1px solid ${(p) => p.theme.colors.line}; padding: 12px 0 0;
  & > summary { cursor: pointer; font-size: 13px; color: ${(p) => p.theme.colors.faint}; list-style: none; }
  & > summary::before { content: "+ "; color: ${(p) => p.theme.colors.accent}; }
  &[open] > summary::before { content: "− "; }
  & > div { display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
`;
const PresetRow = styled.div`display: flex; gap: 8px; flex-wrap: wrap;`;
const Chip = styled.button`
  padding: 7px 14px; cursor: pointer; font-size: 13px; background: transparent;
  border: 1px solid ${(p) => p.theme.colors.line}; border-radius: ${(p) => p.theme.radius};
  color: ${(p) => p.theme.colors.ink};
  &:hover { border-color: ${(p) => p.theme.colors.accent}; color: ${(p) => p.theme.colors.accent}; }
`;
const CandRow = styled.div<{ $on: boolean }>`
  display: flex; align-items: center; gap: 12px; padding: 12px 10px; cursor: pointer;
  border: 1px solid ${(p) => (p.$on ? p.theme.colors.ink : p.theme.colors.line)};
  background: ${(p) => (p.$on ? "#fff" : "transparent")};
`;
const CandMeta = styled.div`
  font-family: ${(p) => p.theme.font.mono}; font-size: 11.5px; color: ${(p) => p.theme.colors.faint};
  margin-top: 3px;
`;
const Consent = styled.label`
  display: flex; align-items: center; gap: 10px; cursor: pointer;
  font-size: 13px; color: ${(p) => p.theme.colors.ink} !important;
  border-top: 1px solid ${(p) => p.theme.colors.line}; padding-top: 12px;
`;
const NavRow = styled.div`
  width: 100%; max-width: 520px; display: flex; justify-content: space-between;
  border-top: 2px solid ${(p) => p.theme.colors.ink}; margin-top: 26px; padding-top: 14px;
`;

/* ── 통화화면: 좌=영상(밤) / 우=대본·상태(종이). 좁으면 세로 스택. ── */
const Split = styled.div`
  height: 100vh; display: flex;
  @media (max-width: 760px) { flex-direction: column; }
`;
const VideoSide = styled.div`
  flex: 1; min-width: 0; min-height: 0; display: flex; align-items: center;
  justify-content: center; background: ${(p) => p.theme.colors.night}; padding: 12px;
`;
const Avatar = styled.canvas`
  /* 좌측 섹션을 꽉 채움: 비율 유지(contain)로 가로/세로 중 먼저 닿는 쪽까지 키움. */
  width: 100%; height: 100%; object-fit: contain;
`;
const Wave = styled.div<{ active: boolean }>`
  display: flex; gap: 5px; height: 48px; align-items: center;
  & span {
    width: 3px; background: ${(p) => p.theme.colors.onNight}; opacity: 0.9;
    animation: ${(p) => (p.active ? "bounce 0.8s infinite" : "none")};
  }
  @keyframes bounce { 0%,100%{height:8px} 50%{height:40px} }
`;
const InfoSide = styled.div`
  flex: 1; min-width: 0; display: flex; flex-direction: column;
  padding: 28px 26px 20px; gap: 14px;
`;
const CallHead = styled.div`
  border-bottom: 2px solid ${(p) => p.theme.colors.ink}; padding-bottom: 14px;
`;
const CallName = styled.div`
  font-family: ${(p) => p.theme.font.display}; font-size: 30px; font-weight: 600;
`;
const StatusLine = styled.div`
  display: flex; align-items: baseline; gap: 12px; margin-top: 8px;
  font-family: ${(p) => p.theme.font.mono}; font-size: 12.5px; color: ${(p) => p.theme.colors.faint};
  & .colon { color: ${(p) => p.theme.colors.accent}; }
`;
const LiveDot = styled.span<{ $live: boolean }>`
  width: 8px; height: 8px; border-radius: 50%; align-self: center;
  background: ${(p) => (p.$live ? p.theme.colors.accent : p.theme.colors.line)};
`;
const CloneTag = styled.span`
  border: 1px solid ${(p) => p.theme.colors.line}; padding: 2px 8px;
  font-size: 10.5px; letter-spacing: 0.1em;
`;
/* 대본(transcript): 말풍선 대신 화자 라벨 + 본문 — 인쇄 대본의 결 */
const Script = styled.div`
  flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column;
  gap: 13px; padding: 6px 2px;
`;
const LineRow = styled.div`display: flex; gap: 14px; align-items: baseline;`;
const SpeakerTag = styled.div<{ $me: boolean }>`
  flex: none; width: 72px; text-align: right;
  font-family: ${(p) => p.theme.font.mono}; font-size: 11px; letter-spacing: 0.05em;
  color: ${(p) => (p.$me ? p.theme.colors.faint : p.theme.colors.accent)};
  padding-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
`;
const LineText = styled.div<{ $dim?: boolean }>`
  flex: 1; font-size: 14.5px; line-height: 1.65; white-space: pre-wrap; word-break: break-word;
  color: ${(p) => p.theme.colors.ink}; opacity: ${(p) => (p.$dim ? 0.45 : 1)};
`;
const SysNote = styled.div`
  align-self: center; font-family: ${(p) => p.theme.font.mono}; font-size: 11.5px;
  color: ${(p) => p.theme.colors.faint}; padding: 2px 8px;
`;
const Controls = styled.div`
  display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-start;
  border-top: 1px solid ${(p) => p.theme.colors.line}; padding-top: 14px;
`;
const Ctl = styled.button<{ $accent?: boolean }>`
  padding: 10px 14px; cursor: pointer; font-size: 13px; font-weight: 600;
  border-radius: ${(p) => p.theme.radius};
  border: 1px solid ${(p) => (p.$accent ? p.theme.colors.accent : p.theme.colors.line)};
  background: ${(p) => (p.$accent ? p.theme.colors.accent : "transparent")};
  color: ${(p) => (p.$accent ? p.theme.colors.onAccent : p.theme.colors.ink)};
  &:hover:not(:disabled) { border-color: ${(p) => (p.$accent ? "#9e3524" : p.theme.colors.ink)}; }
  &:disabled { opacity: 0.4; cursor: default; }
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
  // v2 통화 상태머신: 연결중 → 듣는중 → (엔드포인트) 생각중 → 말하는중 → 듣는중.
  const [callState, setCallState] = useState<"connecting" | "listening" | "thinking" | "speaking">("connecting");
  const callStateRef = useRef<"connecting" | "listening" | "thinking" | "speaking">("connecting");
  const [partial, setPartial] = useState("");           // 발화 중 실시간 자막(서버 partial 전사)
  const [lastTiming, setLastTiming] = useState<Record<string, number> | null>(null);
  const [showHud, setShowHud] = useState(false);        // 단계별 지연 HUD(개발자)
  const [autoTurn, setAutoTurn] = useState(true);       // 침묵 감지 자동 응답(끄면 버튼)
  const autoTurnRef = useRef(true);
  const speechRef = useRef(false);                      // 이번 턴에 말 감지됨?
  const lastVoiceRef = useRef(0);                       // 마지막 음성 시각(ms)
  const playNodeRef = useRef<AudioBufferSourceNode | null>(null);  // 재생 중 노드(끼어들기 정지용)
  const endingRef = useRef(false);                      // 작별 인사 재생 후 자동 종료 플래그
  const [consent, setConsent] = useState(false);        // 목소리 주인 동의 확인(윤리 게이트)
  const [memBusy, setMemBusy] = useState(false);        // 기억시키기 진행 중
  const [chat, setChat] = useState<{ who: "me" | "them" | "sys"; text: string }[]>([]);

  // 클라가 소유하는 개인데이터(서버 영속 0)
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  // 목소리 소스: 내 파일 업로드 vs 서버에 준비된 프리셋 선택
  const [voiceSource, setVoiceSource] = useState<"own" | "call" | "preset">("own");
  // 플로우 B: 긴 통화 녹음 → 서버 화자분리 → "그 사람" 선택 → 프리셋 저장
  const [anaJob, setAnaJob] = useState("");                 // 분석 job id("" = 없음)
  const [anaStatus, setAnaStatus] = useState<AnalyzeStatus | null>(null);
  const [anaSpeaker, setAnaSpeaker] = useState("");         // 고른 화자 id
  const [anaName, setAnaName] = useState("");               // 저장할 프리셋 이름
  const [anaMsg, setAnaMsg] = useState("");                 // 진행/에러 안내("!" 접두 = 에러)
  const [anaBusy, setAnaBusy] = useState(false);
  const [presets, setPresets] = useState<VoicePreset[]>([]);
  const [presetId, setPresetId] = useState("");
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
  const [step, setStep] = useState(1);                  // 1 목소리 · 2 얼굴 · 3 관계 · 4 연결
  const [refText, setRefText] = useState("");           // 참조 음성 전사(자동→수정 가능, 유사도↑)
  const [previewing, setPreviewing] = useState(false);
  const [previewMsg, setPreviewMsg] = useState("");     // 미리듣기 안내/에러("!" 접두 = 에러)
  const [photoUrl, setPhotoUrl] = useState("");         // 사진 미리보기 objectURL
  const [foldOpen, setFoldOpen] = useState(false);      // 캐릭터 '더 자세히' 펼침(프리셋 적용 시 자동)
  const previewCtxRef = useRef<AudioContext | null>(null);  // 미리듣기 재생 전용

  const sockRef = useRef<CallSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);     // 마이크 캡처(16kHz)
  const playCtxRef = useRef<AudioContext | null>(null);      // 재생 전용(24kHz, TTS 출력 sr)
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

  // 서버에 준비된 프리셋 목소리 목록(있으면 '등록된 목소리' 탭에 표시)
  useEffect(() => { listVoicePresets().then(setPresets).catch(() => setPresets([])); }, []);

  // 플로우 B: 분석 job 폴링(2s) — done/error 에서 멈춤
  useEffect(() => {
    if (!anaJob) return;
    const t = setInterval(async () => {
      try {
        const st = await analyzeVoiceStatus(anaJob);
        setAnaStatus(st);
        if (st.stage === "done" || st.stage === "error") clearInterval(t);
      } catch (e: any) {
        setAnaMsg(`! ${e?.message || "분석 조회 실패"}`); clearInterval(t);
      }
    }, 2000);
    return () => clearInterval(t);
  }, [anaJob]);

  async function startAnalyze(f: File | null) {
    if (!f) return;
    setAnaStatus(null); setAnaSpeaker(""); setAnaMsg("업로드 중 — 긴 파일은 수십 초");
    try {
      const jid = await analyzeVoiceStart(f);
      setAnaJob(jid); setAnaMsg("");
    } catch (e: any) { setAnaMsg(`! ${e?.message || "업로드 실패"}`); }
  }

  async function saveAnalyzed() {
    if (!anaSpeaker || !anaName.trim()) return;
    setAnaBusy(true); setAnaMsg("최적 구간 저장 + 전사 중…");
    try {
      const r = await analyzeVoiceSave(anaJob, anaSpeaker, anaName.trim());
      const list = await listVoicePresets(); setPresets(list);
      setPresetId(r.preset_id); setVoiceSource("preset");     // 저장 즉시 '등록된 목소리'로 합류
      setAnaMsg(`'${r.preset_id}' 저장됨 (${r.dur}s) — 아래에서 미리듣고 다음으로.`);
    } catch (e: any) { setAnaMsg(`! ${e?.message || "저장 실패"}`); }
    finally { setAnaBusy(false); }
  }

  // 트랙②: 방금 분석한 통화 내용 → 그 사람 기억 자동 구축(전사+LLM, memories.json).
  async function rememberAnalyzed() {
    if (!anaJob || !anaSpeaker || !presetId || anaBusy) return;
    setAnaBusy(true); setAnaMsg("통화 내용을 기억으로 옮기는 중 — 몇 분 걸릴 수 있어요…");
    try {
      const r = await analyzeVoiceRemember(anaJob, anaSpeaker, presetId);
      setAnaMsg(r.added
        ? `기억 ${r.added}개 구축 (총 ${r.total}) — '${presetId}' 이름으로 통화하면 회상해요.`
        : "기억할 만한 내용을 못 찾았어요 (전사 창 " + r.windows + "개)");
    } catch (e: any) { setAnaMsg(`! ${e?.message || "기억 구축 실패"}`); }
    finally { setAnaBusy(false); }
  }

  // 폰 업로드(/upload) 이어받기 — 가장 최근 job 을 가져온다(전시 접수 데스크 플로우).
  async function pickUploadedJob() {
    setAnaMsg("업로드된 파일을 찾는 중…");
    try {
      const jobs = await listVoiceJobs();
      const j = jobs.find((x) => x.stage !== "error");
      if (!j) { setAnaMsg("! 업로드된 파일이 없어요 — 폰에서 /upload 로 보내주세요."); return; }
      setAnaStatus(null); setAnaSpeaker("");
      setAnaJob(j.job_id);
      setAnaMsg(`업로드 이어받음 (코드 ${j.job_id.slice(0, 6).toUpperCase()})`);
    } catch (e: any) { setAnaMsg(`! ${e?.message || "job 조회 실패"}`); }
  }

  // 목소리 준비됨? (다음/시작 버튼 활성 조건) — 업로드했거나 프리셋 골랐거나.
  // own/call 플로우는 목소리 주인 동의 체크(윤리 게이트)까지 요구. 프리셋은 등록 시 확인됨.
  const hasVoice = (voiceSource === "own" ? !!voiceFile : !!presetId)
    && (voiceSource === "preset" || consent);

  useEffect(() => {
    if (!started) return;
    const t = setInterval(() => setSec((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [started]);

  function persist() {
    try { localStorage.setItem(HKEY, JSON.stringify(historyRef.current)); } catch { /* noop */ }
  }

  function setCS(s: "connecting" | "listening" | "thinking" | "speaking") {
    callStateRef.current = s;
    setCallState(s);
  }

  // 턴 확정(수동 버튼/자동 침묵감지 공용) — 서버는 발화 중 이미 전사해 둠(partial) → 즉시 응답 시작.
  function sendTurn() {
    if (callStateRef.current !== "listening") return;
    speechRef.current = false;
    setCS("thinking");
    sockRef.current?.endTurn();
  }

  // v2 barge-in: 재생/생성 중단(버튼). 서버 interrupt + 로컬 재생 즉시 정지.
  function bargeIn() {
    sockRef.current?.interrupt();
    try { playNodeRef.current?.stop(); } catch { /* noop */ }
    if (rafRef.current != null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    turnAudioRef.current = []; turnFramesRef.current = [];
    speechRef.current = false;
    setCS("listening");
  }

  // 안전한 끝맺음: 듣는 중이면 클론이 작별 인사 → 재생 끝나면 자동 종료. 그 외 상태면 바로 종료.
  function farewellAndEnd() {
    if (callStateRef.current !== "listening" || !sockRef.current) { endCall(); return; }
    endingRef.current = true;
    setCS("thinking");
    sockRef.current.farewell();
  }

  // 오늘 대화를 서버 기억으로 승격(유저 주도) — 다음 통화부터 회상(use_rag: auto).
  async function rememberThisCall() {
    if (!historyRef.current.length || memBusy) return;
    setMemBusy(true);
    try {
      const r = await rememberCall(id, historyRef.current);
      setChat((c) => [...c, { who: "sys", text: r.added
        ? `기억 ${r.added}개 저장 (총 ${r.total}) — 다음 통화부터 기억해요`
        : "새로 기억할 만한 내용이 없었어요" }]);
    } catch (e: any) {
      setChat((c) => [...c, { who: "sys", text: `! ${e?.message || "기억 저장 실패"}` }]);
    } finally { setMemBusy(false); }
  }

  async function startCall() {
    setStarted(true);
    setStatus("");
    setCS("connecting");
    // 이어하기: 저장된 이력을 대본으로 미리 표시(user=나, assistant=상대).
    setChat(historyRef.current.map((m) => ({ who: (m.role === "user" ? "me" : "them") as "me" | "them", text: m.content })));
    const sock = new CallSocket(id, {
      // A/V 동기: 오디오·프레임을 턴 버퍼에 모았다가 audio_end 에서 동시에 재생.
      onAudio: (pcm) => { turnAudioRef.current.push(pcm); },
      onReply: (text, latency) => {
        historyRef.current.push({ role: "assistant", content: text }); persist();
        void latency;
        setChat((c) => [...c, { who: "them", text }]);
      },
      onUser: (text) => {
        setPartial("");                                  // 최종 전사 도착 → 자막을 대본으로 승격
        if (text.trim()) { historyRef.current.push({ role: "user", content: text }); persist();
          setChat((c) => [...c, { who: "me", text }]); }
      },
      onPartial: (text) => { if (callStateRef.current === "listening") setPartial(text); },
      onTiming: (stages) => setLastTiming(stages),
      onInterrupted: () => bargeIn(),                    // 서버발 중단 확인 → 로컬 재생도 정리
      onFrame: (jpegB64) => { turnFramesRef.current.push(jpegB64); },
      onAudioEnd: () => playTurn(),
      // 준비 완료(서버가 음성·사진·graph 다 세팅)되면 그때 마이크 켠다 — 연결 중엔 오디오 안 보냄
      // (안 그러면 서버가 init 처리 중에 오디오 폭주로 WS 수신큐 오버플로→끊김).
      onReady: () => { setCS("listening"); startMic(sock); },
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
      preset_id: voiceSource === "preset" ? (presetId || undefined) : undefined,
      history: historyRef.current.length ? historyRef.current : undefined,
    };
    // 내 목소리 모드일 때만 파일 전송(프리셋 모드면 서버 로컬 클립 사용)
    if (voiceSource === "own" && voiceFile) init.ref_audio_b64 = await fileToBase64(voiceFile);
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
    setPreviewing(true); setPreviewMsg("합성 중 — 첫 회는 전사 포함 몇 초");
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
      setPreviewMsg("재생 중 — 본인 목소리 같으면 다음으로.");
    } catch (e: any) {
      setPreviewMsg(`! ${e?.message || "미리듣기 실패"} (cosyvoice-server 확인)`);
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
        // 말하는 중(재생)엔 마이크를 안 보낸다 — 스피커 에코가 다음 턴 버퍼/전사를 오염시키는 것 방지.
        // 음성 barge-in 대신 끼어들기 버튼(bargeIn) 사용(C.AI Calls 와 같은 UX — REBUILD_PLAN §0).
        if (callStateRef.current === "speaking") return;
        const pcm = new Float32Array(e.inputBuffer.getChannelData(0));
        sock.sendAudio(pcm);
        if (!autoTurnRef.current) return;
        // 클라 엔드포인팅: 말 감지 후 900ms 무음 → 자동 응답(서버는 발화 중 이미 전사 완료 상태).
        let sum = 0;
        for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
        const rms = Math.sqrt(sum / pcm.length);
        const now = performance.now();
        if (rms > 0.015) { speechRef.current = true; lastVoiceRef.current = now; }
        else if (speechRef.current && callStateRef.current === "listening"
                 && now - lastVoiceRef.current > 900) {
          sendTurn();
        }
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
      speechRef.current = false;
      if (endingRef.current) { endCall(); return; }   // 작별 턴이 빈손이어도 종료는 진행
      setCS("listening");                        // 빈 턴(빈 전사 등) → 다시 듣기
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
    playNodeRef.current = node;                  // 끼어들기 버튼이 정지할 수 있게 보관
    node.onended = () => {                       // 재생 끝(또는 stop) → 다시 듣기
      playNodeRef.current = null;
      speechRef.current = false;                 // 재생 잔향/에코를 말로 오인하는 것 방지
      if (endingRef.current) { endCall(); return; }   // 작별 인사였다면 재생 후 종료
      setCS("listening");
    };
    setCS("speaking");
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
  function toggleAutoTurn() { autoTurnRef.current = !autoTurnRef.current; setAutoTurn(autoTurnRef.current); }

  function endCall() {
    cleanupMicRef.current();
    sockRef.current?.stop();          // 서버가 인메모리 개인데이터 폐기
    if (rafRef.current != null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    try { playCtxRef.current?.close(); } catch { /* noop */ }
    try { previewCtxRef.current?.close(); } catch { /* noop */ }
    if (photoUrl) URL.revokeObjectURL(photoUrl);
    playCtxRef.current = null;
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
      catch { setChat((c) => [...c, { who: "sys", text: "! 불러오기 실패: JSON 아님" }]); }
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

  // ---- 설정 화면(통화 전): 서식형 — 01 목소리 · 02 얼굴 · 03 관계 · 04 연결 ----
  if (!started) {
    const STEP_META = [
      ["목소리", "복제할 목소리를 올리고 미리 들어보세요 (필수)"],
      ["얼굴", "영상통화용 사진 — 없으면 음성통화 (선택)"],
      ["관계", "누구이고 나와 무슨 사이인지 (선택)"],
      ["연결", "이전 대화 이어가기 · 통화 시작"],
    ];
    const next = () => setStep((s) => Math.min(4, s + 1));
    const prev = () => setStep((s) => Math.max(1, s - 1));
    return (
      <Screen>
        <SetupHead>
          <Wordmark />
          <HeadNote>새 통화 준비 — {id}</HeadNote>
        </SetupHead>
        <StepIndex>
          {STEP_META.map(([t], i) => (
            <StepItem key={t} $on={step === i + 1} $done={step > i + 1}>
              <em>{String(i + 1).padStart(2, "0")}</em> {t}
            </StepItem>
          ))}
        </StepIndex>
        <Setup>
          <StepTitle><span>{String(step).padStart(2, "0")}</span>{STEP_META[step - 1][0]}</StepTitle>
          <StepHint>{STEP_META[step - 1][1]}</StepHint>

          {step === 1 && (<>
            <label>어떤 자료를 갖고 있나요?</label>
            <Seg>
              {(["own", "call", "preset"] as const).map((m) => (
                <SegBtn key={m} type="button" $on={voiceSource === m} onClick={() => setVoiceSource(m)}>
                  {m === "own" ? "짧은 목소리 파일"
                    : m === "call" ? "긴 통화 녹음"
                    : `등록된 목소리${presets.length ? ` (${presets.length})` : ""}`}
                </SegBtn>
              ))}
            </Seg>

            {voiceSource === "own" && (<>
              <FileBox>
                {voiceFile ? <b>{voiceFile.name}</b> : <>목소리 파일 선택 — <b>5~15초</b>, 깨끗할수록 좋아요 · wav / mp3 / m4a</>}
                <input type="file" accept="audio/*" onChange={(e) => { setVoiceFile(e.target.files?.[0] ?? null); setPreviewMsg(""); }} />
              </FileBox>
              {voiceFile && <Preview onClick={runPreview} disabled={previewing}>{previewing ? "합성 중…" : "복제 목소리 미리듣기"}</Preview>}
              {previewMsg && <Note err={previewMsg.startsWith("!")}>{previewMsg}</Note>}
              {voiceFile && (<>
                <label>참조 음성 내용 (전사 — 정확할수록 유사도가 올라가요, 수정 가능)</label>
                <input type="text" value={refText} onChange={(e) => setRefText(e.target.value)} placeholder="미리듣기를 누르면 자동으로 채워집니다" />
              </>)}
              <Consent>
                <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
                이 목소리의 주인에게 사용 동의를 받았어요 (본인 목소리 포함)
              </Consent>
            </>)}

            {voiceSource === "call" && (<>
              <Note>서버가 화자를 분리해 "그 사람" 목소리의 가장 깨끗한 구간을 자동으로 찾아드려요.
                원본은 분석 후 즉시 삭제되고, 저장을 눌러야만 목소리가 등록돼요.</Note>
              <FileBox>
                통화 녹음 파일 선택 — <b>몇 시간짜리도 OK</b> · m4a / mp3 / wav
                <input type="file" accept="audio/*"
                  onChange={(e) => startAnalyze(e.target.files?.[0] ?? null)} />
              </FileBox>
              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <Note style={{ flex: 1 }}>폰에 파일이 있다면 — 같은 네트워크에서
                  <b> {location.origin}/upload </b>로 보내고,</Note>
                <Btn onClick={pickUploadedJob} disabled={anaBusy}>업로드 이어받기</Btn>
              </div>
              {anaJob && anaStatus?.stage !== "done" && anaStatus?.stage !== "error" && (
                <Note>{anaStatus?.stage === "diarize" ? "화자 분리 중 — 긴 녹음은 몇 분 걸려요"
                  : anaStatus?.stage === "scoring" ? "좋은 구간 고르는 중…"
                  : "분석 준비 중…"}</Note>
              )}
              {anaStatus?.stage === "error" && <Note err>! {anaStatus.error}</Note>}
              {anaStatus?.stage === "done" && (<>
                {anaStatus.dummy_diarizer &&
                  <Note err>! 정밀 화자분리 모델(pyannote)이 서버에 없어 구분이 부정확할 수 있어요 —
                    serve venv 에 pip install pyannote.audio + HF_TOKEN 설정 권장.</Note>}
                <label>누가 "그 사람"인가요? 들어보고 고르세요</label>
                {(anaStatus.speakers ?? []).map((s, i) => (
                  <CandRow key={s.id} $on={anaSpeaker === s.id} onClick={() => setAnaSpeaker(s.id)}>
                    <input type="radio" checked={anaSpeaker === s.id} onChange={() => setAnaSpeaker(s.id)} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 14 }}>화자 {i + 1}</div>
                      <CandMeta>
                        발화 {Math.round(s.total_sec / 60)}분 {Math.round(s.total_sec % 60)}초 · {s.n_segments}구간 · 음질 {s.best_snr}dB
                      </CandMeta>
                    </div>
                    <audio controls preload="none" style={{ height: 32, maxWidth: 180 }}
                      src={`data:audio/wav;base64,${s.sample_wav_b64}`} />
                  </CandRow>
                ))}
                {anaSpeaker && (<>
                  <label>이 목소리의 이름 (예: 엄마, 나은)</label>
                  <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                    <input type="text" value={anaName} onChange={(e) => setAnaName(e.target.value)}
                      placeholder="등록할 이름" style={{ flex: 1 }} />
                    <Btn onClick={saveAnalyzed} disabled={anaBusy || !anaName.trim() || !consent}>
                      {anaBusy ? "저장 중…" : "이 목소리 쓰기"}
                    </Btn>
                  </div>
                  <Consent>
                    <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
                    이 목소리의 주인에게 사용 동의를 받았어요
                  </Consent>
                </>)}
              </>)}
              {anaMsg && <Note err={anaMsg.startsWith("!")}>{anaMsg}</Note>}
            </>)}

            {voiceSource === "preset" && (<>
              {anaJob && anaSpeaker && presetId && (
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <Note style={{ flex: 1 }}>방금 분석한 통화의 내용을 '{presetId}'의 기억으로
                    만들 수 있어요 — 다음 통화부터 회상해요.</Note>
                  <Btn onClick={rememberAnalyzed} disabled={anaBusy}>
                    {anaBusy ? "기억 구축 중…" : "통화 내용 기억시키기"}</Btn>
                </div>
              )}
              <label>등록된 목소리 선택</label>
              {presets.length ? (
                <select value={presetId} onChange={(e) => setPresetId(e.target.value)}
                  style={{ width: "100%", padding: "10px 2px", border: "none", borderBottom: "1px solid #CBC3B0",
                    borderRadius: 0, background: "transparent", color: "#221E16", fontSize: 15 }}>
                  <option value="">— 목소리 고르기 —</option>
                  {presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                </select>
              ) : (
                <Note>등록된 목소리가 없어요. '긴 통화 녹음' 탭에서 추출하거나, 서버 data/voice_presets/ 에 wav 를 올리면(scp) 여기 떠요. (권리 있는 클립만)</Note>
              )}
              {anaMsg && <Note err={anaMsg.startsWith("!")}>{anaMsg}</Note>}
            </>)}
          </>)}

          {step === 2 && (<>
            <FileBox>
              {photoFile ? <b>{photoFile.name}</b> : <>증명사진 선택 — 얼굴이 잘 보이는 정면 · jpg / png</>}
              <input type="file" accept="image/*" onChange={(e) => pickPhoto(e.target.files?.[0] ?? null)} />
            </FileBox>
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
              <Chip type="button" title="트랙③ 제작용 — AVP 변형 질문지로 callone 이 인터뷰를 진행"
                onClick={async () => {
                  try {
                    const r = await exhibitInterviewer(userPersona || "");
                    applyPreset({
                      persona: r.card.persona || "", userPersona: r.card.user_persona || "",
                      personality: r.card.personality || "", background: r.card.background || "",
                      situation: r.card.situation || "", firstMessage: r.card.first_message || "",
                      exampleDialogue: "",
                    });
                  } catch { /* 서버 미기동 시 무시 */ }
                }}>인터뷰어 (제작)</Chip>
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
            <FileBox>
              대화 파일 선택 — callone_*.json
              <input type="file" accept="application/json" onChange={(e) => e.target.files?.[0] && importHistory(e.target.files[0])} />
            </FileBox>
            <Note>
              {hasVoice ? (voiceSource === "preset" ? `목소리 ✓ (${presetId})` : "목소리 ✓") : "목소리 — 아직"}
              {photoFile ? " · 얼굴 ✓" : " · 음성통화"}
              {persona ? ` · ${persona}` : ""}
            </Note>
            {turns > 0 && (
              <div style={{ display: "flex", gap: 10 }}>
                <Btn onClick={exportHistory}>대화 내보내기</Btn>
                <Btn danger onClick={clearHistory}>기억 리셋</Btn>
              </div>
            )}
          </>)}
        </Setup>

        <NavRow>
          {step > 1 ? <Ghost onClick={prev}>← 이전</Ghost> : <Ghost onClick={() => nav("/")}>취소</Ghost>}
          {step < 4
            ? <Solid onClick={next} disabled={step === 1 && !hasVoice}>다음 →</Solid>
            : <Solid onClick={startCall} disabled={!hasVoice}>통화 걸기</Solid>}
        </NavRow>
      </Screen>
    );
  }

  // ---- 통화 화면: 좌=영상 / 우=대본·상태·버튼 ----
  const CS_LABEL: Record<typeof callState, string> = {
    connecting: "연결 중", listening: "듣는 중",
    thinking: "생각 중", speaking: "말하는 중",
  };
  return (
    <Split>
      <VideoSide>
        {hasVideo ? (
          <Avatar ref={canvasRef} />
        ) : (
          <Wave active={callState === "speaking"}>
            {Array.from({ length: 9 }).map((_, i) => <span key={i} style={{ animationDelay: `${i * 0.08}s` }} />)}
          </Wave>
        )}
      </VideoSide>
      <InfoSide>
        <CallHead>
          <CallName>{id}</CallName>
          <StatusLine>
            <LiveDot $live={callState === "speaking" || callState === "listening"} />
            <span>{CS_LABEL[callState]}</span>
            <span>{mm}<span className="colon">:</span>{ss}</span>
            <CloneTag>AI 클론 음성</CloneTag>
          </StatusLine>
        </CallHead>
        {status && <SysNote>! {status}</SysNote>}
        {sec > 1800 && <SysNote>30분 넘게 통화 중이에요 — 잠깐 쉬어가도 좋아요.</SysNote>}
        <Script>
          {chat.map((c, i) => c.who === "sys"
            ? <SysNote key={i}>{c.text}</SysNote>
            : (
              <LineRow key={i}>
                <SpeakerTag $me={c.who === "me"}>{c.who === "me" ? "나" : id}</SpeakerTag>
                <LineText>{c.text}</LineText>
              </LineRow>
            ))}
          {partial && (
            <LineRow>
              <SpeakerTag $me>나</SpeakerTag>
              <LineText $dim>{partial} …</LineText>
            </LineRow>
          )}
        </Script>
        {showHud && lastTiming && (
          <SysNote>
            asr {Math.round(lastTiming.asr_ms || 0)} ·
            llm {Math.round(lastTiming.llm_first_ms || 0)}/{Math.round(lastTiming.llm_total_ms || 0)} ·
            tts {Math.round(lastTiming.tts_first_ms || 0)} ·
            첫음성 {Math.round(lastTiming.first_audio_ms || 0)}ms
          </SysNote>
        )}
        <Controls>
          {(callState === "speaking" || callState === "thinking") &&
            <Ctl onClick={bargeIn}>끼어들기</Ctl>}
          <Ctl onClick={toggleMute}>{muted ? "음소거 해제" : "음소거"}</Ctl>
          <Ctl onClick={toggleAutoTurn} title="말 끝나면(0.9s 무음) 자동으로 응답">
            {autoTurn ? "자동 응답 ✓" : "자동 응답 끔"}
          </Ctl>
          {!autoTurn && <Ctl onClick={sendTurn}>응답 전송</Ctl>}
          <Ctl onClick={() => setShowHud((v) => !v)} title="단계별 지연(ms)">지연</Ctl>
          <Ctl onClick={rememberThisCall} disabled={memBusy}
            title="오늘 대화의 사실을 기억에 저장 — 다음 통화부터 회상">
            {memBusy ? "기억 중…" : "기억하기"}</Ctl>
          <Ctl onClick={exportHistory}>내보내기</Ctl>
          <Ctl onClick={farewellAndEnd} title="클론이 작별 인사를 한 뒤 끊어요">작별하고 종료</Ctl>
          <Ctl $accent onClick={endCall}>종료</Ctl>
        </Controls>
      </InfoSide>
    </Split>
  );
}
