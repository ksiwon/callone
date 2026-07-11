// ContactList (§17.1) — 바로 통화 시작(프라이버시 흐름: 음성·사진은 통화화면서 업로드) + 화자 명부.
import { useEffect, useState } from "react";
import styled from "styled-components";
import { Link, useNavigate } from "react-router-dom";
import { listSpeakers, SpeakerSummary } from "../api/calloneClient";
import Wordmark from "./Wordmark";

const Wrap = styled.div`padding: 48px 28px 40px; max-width: 560px; margin: 0 auto;`;
const Tagline = styled.p`
  margin: 10px 0 0; font-family: ${(p) => p.theme.font.display};
  font-size: 15px; color: ${(p) => p.theme.colors.faint};
`;
const Rule = styled.hr<{ strong?: boolean }>`
  border: none; margin: 22px 0;
  border-top: ${(p) => (p.strong ? `2px solid ${p.theme.colors.ink}` : `1px solid ${p.theme.colors.line}`)};
`;
const FieldLabel = styled.div`
  font-family: ${(p) => p.theme.font.mono}; font-size: 11px; letter-spacing: 0.14em;
  color: ${(p) => p.theme.colors.faint}; text-transform: uppercase; margin-bottom: 8px;
`;
const LineInput = styled.input`
  width: 100%; padding: 8px 2px; font-size: 17px; color: ${(p) => p.theme.colors.ink};
  background: transparent; border: none; border-bottom: 1px solid ${(p) => p.theme.colors.line};
  border-radius: 0;
  &::placeholder { color: ${(p) => p.theme.colors.line}; }
  &:focus { outline: none; border-bottom: 2px solid ${(p) => p.theme.colors.ink}; }
`;
const CallBtn = styled.button`
  padding: 13px 26px; border: none; border-radius: ${(p) => p.theme.radius}; cursor: pointer;
  background: ${(p) => p.theme.colors.ink}; color: ${(p) => p.theme.colors.paper};
  font-size: 15px; font-weight: 600; white-space: nowrap;
  &:hover { background: ${(p) => p.theme.colors.accent}; color: ${(p) => p.theme.colors.onAccent}; }
`;
const Row = styled.div`
  display: flex; align-items: center; gap: 16px; padding: 16px 2px;
  border-bottom: 1px solid ${(p) => p.theme.colors.line};
`;
const Name = styled.div`font-family: ${(p) => p.theme.font.display}; font-size: 18px;`;
const Meta = styled.div`
  font-family: ${(p) => p.theme.font.mono}; font-size: 12px; color: ${(p) => p.theme.colors.faint};
  margin-top: 3px;
`;
const TextLink = styled(Link)<{ $accent?: boolean }>`
  font-size: 14px; text-decoration: none; white-space: nowrap; padding: 6px 2px;
  color: ${(p) => (p.$accent ? p.theme.colors.accent : p.theme.colors.faint)};
  border-bottom: 1px solid transparent;
  &:hover { border-bottom-color: currentColor; }
`;
const Empty = styled.div`
  color: ${(p) => p.theme.colors.faint}; margin-top: 36px; font-size: 14px; line-height: 1.7;
`;
const Foot = styled.div`
  margin-top: 44px; font-family: ${(p) => p.theme.font.mono}; font-size: 11px;
  color: ${(p) => p.theme.colors.faint}; line-height: 1.8;
`;

export default function ContactList() {
  const [speakers, setSpeakers] = useState<SpeakerSummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [label, setLabel] = useState("me");
  const nav = useNavigate();

  useEffect(() => {
    listSpeakers().then((s) => { setSpeakers(s); setLoaded(true); });
  }, []);

  const go = () => nav(`/call/${encodeURIComponent(label.trim() || "me")}`);

  return (
    <Wrap>
      <Wordmark size={34} />
      <Tagline>지금 없는 목소리와, 지금 통화하기</Tagline>
      <Rule strong />

      <FieldLabel>01 — 누구에게 걸까요</FieldLabel>
      <div style={{ display: "flex", gap: 14, alignItems: "flex-end" }}>
        <LineInput
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && go()}
          placeholder="이름 (대화 저장 라벨)"
        />
        <CallBtn onClick={go}>통화 걸기</CallBtn>
      </div>

      {speakers.length > 0 && (<>
        <Rule />
        <FieldLabel>등록된 화자 — 풀 클론 학습본</FieldLabel>
        {speakers.map((s) => (
          <Row key={s.speaker_id}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <Name>{s.name || `화자 ${s.speaker_id}`}</Name>
              <Meta>{s.relation || "관계 미입력"} · {s.region}</Meta>
            </div>
            <TextLink to={`/editor/${s.speaker_id}`}>편집</TextLink>
            <TextLink to={`/call/${s.speaker_id}`} $accent>통화 →</TextLink>
          </Row>
        ))}
      </>)}
      {loaded && speakers.length === 0 && (
        <Empty>등록된 화자가 없어도 됩니다 — 위에서 바로 거세요.<br />
          음성과 사진은 통화 화면에서 올립니다.</Empty>
      )}

      <Rule />
      <Row style={{ borderBottom: "none", padding: "6px 2px" }}>
        <div style={{ flex: 1 }}>
          <Name style={{ fontSize: 16 }}>풀 클론 파이프라인</Name>
          <Meta>한 시간 이상의 녹음으로 말투·기억까지 — 서버 학습 안내</Meta>
        </div>
        <TextLink to="/processing" $accent>보기 →</TextLink>
      </Row>

      <Foot>
        음성·사진·대화는 이 브라우저가 보관합니다. 서버는 통화 동안만 메모리에 두고,<br />
        수화기를 내려놓는 순간 지웁니다. — call:one
      </Foot>
    </Wrap>
  );
}
