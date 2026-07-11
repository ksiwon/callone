// ProcessingView — 풀 클론(풀튜닝) 학습 파이프라인 안내.
// 통화는 5~10초 제로샷이 기본(셋업에서 바로). 1시간+ 녹음으로 화자 모델을 "학습"하려면
// 이 파이프라인을 서버에서 스크립트로 돌린다(브라우저에선 실행 X — 오프라인 GPU 작업).
import styled from "styled-components";
import { Link } from "react-router-dom";
import Wordmark from "./Wordmark";

const Wrap = styled.div`max-width: 680px; margin: 0 auto; padding: 48px 28px 40px;`;
const Title = styled.h2`
  font-family: ${(p) => p.theme.font.display}; font-weight: 600; font-size: 26px; margin: 26px 0 8px;
`;
const Lead = styled.p`
  color: ${(p) => p.theme.colors.faint}; font-size: 14px; line-height: 1.75; margin: 0 0 8px;
  & b { color: ${(p) => p.theme.colors.ink}; font-weight: 600; }
`;
const Rule = styled.hr<{ strong?: boolean }>`
  border: none; margin: 22px 0;
  border-top: ${(p) => (p.strong ? `2px solid ${p.theme.colors.ink}` : `1px solid ${p.theme.colors.line}`)};
`;
const SecLabel = styled.div`
  font-family: ${(p) => p.theme.font.mono}; font-size: 11px; letter-spacing: 0.14em;
  color: ${(p) => p.theme.colors.faint}; text-transform: uppercase; margin: 0 0 10px;
`;
const Step = styled.div`
  display: flex; gap: 16px; align-items: baseline; padding: 12px 0;
  border-bottom: 1px solid ${(p) => p.theme.colors.line};
  &:last-child { border-bottom: none; }
`;
const Num = styled.span`
  flex: none; font-family: ${(p) => p.theme.font.mono}; font-size: 13px;
  color: ${(p) => p.theme.colors.accent};
`;
const StepName = styled.div`font-family: ${(p) => p.theme.font.display}; font-size: 16px;`;
const Sub = styled.div`color: ${(p) => p.theme.colors.faint}; font-size: 13px; line-height: 1.6; margin-top: 2px;`;
const Code = styled.code`
  display: block; background: ${(p) => p.theme.colors.night}; color: ${(p) => p.theme.colors.onNight};
  border-radius: ${(p) => p.theme.radius}; padding: 14px 16px; margin-top: 10px;
  font-family: ${(p) => p.theme.font.mono}; font-size: 12.5px; line-height: 1.7; white-space: pre-wrap;
`;
const Tag = styled.span`
  display: inline-block; border: 1px solid ${(p) => p.theme.colors.line};
  border-radius: ${(p) => p.theme.radius}; padding: 3px 10px; margin: 2px 8px 2px 0;
  font-family: ${(p) => p.theme.font.mono}; font-size: 12px; color: ${(p) => p.theme.colors.faint};
`;
const Back = styled(Link)`
  color: ${(p) => p.theme.colors.accent}; font-size: 14px; text-decoration: none;
  &:hover { text-decoration: underline; }
`;

// 현재 스택(EXAONE / CosyVoice3 / Ditto)에 맞춘 파이프라인 단계.
const STAGES: [string, string][] = [
  ["적재·정규화", "녹음(m4a/mp3/wav) → 16k 모노 wav + 메타. 잡음·무음 정리."],
  ["화자 분리·정제", "2화자 분리 + 동일화자 통합 + 제3자/배경 제거(본인 음성만 남김)."],
  ["전사·데이터셋", "faster-whisper 한국어 전사(사투리 보존) → TTS·대화 학습셋 + PII 마스킹."],
  ["음성 학습(풀튜닝)", "대량 발화로 화자 전용 TTS 학습 → 제로샷보다 음색·운율 충실. 결과는 data/speakers/{id}."],
  ["페르소나·기억", "EXAONE 캐릭터 카드 + (선택) RAG/기억 추출로 말투·사실 일관성 강화."],
  ["실시간 통화", "VAD → 전사 → EXAONE → CosyVoice3 스트리밍 → Ditto 아바타. (= 통화 화면)"],
];

export default function ProcessingView() {
  return (
    <Wrap>
      <Wordmark />
      <Title>풀 클론 파이프라인</Title>
      <Lead>
        통화는 <b>5~10초 제로샷</b>이 기본이라 셋업에서 바로 됩니다. 더 높은 음색·운율 충실도가
        필요하면 <b>1시간 이상의 녹음</b>으로 화자 전용 모델을 학습합니다. 학습은 브라우저가 아니라
        <b> 서버(GPU)</b>에서 아래 스크립트로 한 번 돌리는 오프라인 작업입니다.
      </Lead>
      <Rule strong />

      <SecLabel>현재 스택</SecLabel>
      <div>
        <Tag>LLM · EXAONE-3.5-7.8B</Tag>
        <Tag>TTS · CosyVoice3-0.5B</Tag>
        <Tag>아바타 · Ditto</Tag>
        <Tag>ASR · faster-whisper</Tag>
      </div>
      <Rule />

      <SecLabel>여섯 단계</SecLabel>
      {STAGES.map(([t, d], i) => (
        <Step key={t}>
          <Num>{String(i + 1).padStart(2, "0")}</Num>
          <div><StepName>{t}</StepName><Sub>{d}</Sub></div>
        </Step>
      ))}
      <Rule />

      <SecLabel>서버에서 실행</SecLabel>
      <Sub>녹음을 <code>data/raw/</code>에 넣고 학습 → 학습된 화자로 통화하면 끝.</Sub>
      <Code>{`# 1) 데이터 처리 + 화자 학습(풀튜닝)
bash scripts/setup_train.sh        # 최초 1회: 학습 환경
bash scripts/run_full.sh           # 전 단계 데이터셋 생성
# 목소리는 학습 불필요 — 통화 시작 시 음성 업로드(제로샷 클론)

# 2) 서비스 기동 후 통화 화면에서 학습된 화자 선택
bash scripts/run_all.sh            # llama/cosyvoice/avatar/serve`}</Code>
      <Sub style={{ marginTop: 10 }}>
        빠른 검증만 하려면 <code>scripts/run_pilot.sh</code>(소량 샘플)로 파이프라인만 돌려볼 수 있어요.
      </Sub>

      <Rule />
      <Back to="/">← 홈으로</Back>
    </Wrap>
  );
}
