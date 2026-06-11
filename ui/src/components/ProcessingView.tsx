// ProcessingView (§17.1) — 데이터 업로드/처리 진행 화면(골격).
// 실제 처리는 callone CLI(서버 측)가 수행. 여기선 스테이지 진행 표시.
import styled from "styled-components";
import { Link } from "react-router-dom";

const Wrap = styled.div`max-width: 560px; margin: 0 auto; padding: 24px; color: ${(p) => p.theme.colors.text};`;
const Step = styled.div`
  display: flex; gap: 12px; align-items: center; padding: 12px;
  border-bottom: 1px solid ${(p) => p.theme.colors.border};
`;
const Dot = styled.span`width: 10px; height: 10px; border-radius: 50%; background: ${(p) => p.theme.colors.accent};`;
const Sub = styled.div`color: ${(p) => p.theme.colors.sub}; font-size: 13px;`;

const STAGES = [
  ["S0 적재/정규화", "m4a → 16k mono wav + 메타"],
  ["S1 음질 복원", "denoise + 대역확장 (음색 보존 가드)"],
  ["S2 화자 분리/전역연결", "2화자 분리 + A/B 통합 + 제3자 제거"],
  ["S2.5 방언/라벨링", "사투리 자동 측정 → 라벨링 편집기"],
  ["S3 전사/데이터셋", "사투리 보존 전사 → TTS/대화셋 + PII 마스킹"],
  ["S4 음성 클론", "서버 Qwen3-TTS / 폰 Piper"],
  ["S5 페르소나 두뇌", "Gemma 4 LoRA + RAG/메모리"],
  ["S6 실시간 대화", "VAD → ASR → LLM → 스트리밍 TTS"],
];

export default function ProcessingView() {
  return (
    <Wrap>
      <h2>처리 파이프라인</h2>
      <Sub>서버에서 <code>scripts/run_pilot.sh</code> 또는 각 CLI 로 실행됩니다.</Sub>
      {STAGES.map(([t, d]) => (
        <Step key={t}><Dot /><div><div>{t}</div><Sub>{d}</Sub></div></Step>
      ))}
      <p><Link to="/" style={{ color: "#7aa2f7" }}>← 연락처로</Link></p>
    </Wrap>
  );
}
