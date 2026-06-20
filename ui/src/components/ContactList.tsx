// ContactList (§17.1) — 바로 통화 시작(프라이버시 흐름: 음성·사진은 통화화면서 업로드) + 화자 목록.
import { useEffect, useState } from "react";
import styled from "styled-components";
import { Link, useNavigate } from "react-router-dom";
import { listSpeakers, SpeakerSummary } from "../api/calloneClient";

const Wrap = styled.div`padding: 24px; max-width: 480px; margin: 0 auto;`;
const Title = styled.h1`color: ${(p) => p.theme.colors.text}; font-size: 22px;`;
const Card = styled(Link)`
  display: flex; align-items: center; gap: 16px;
  background: ${(p) => p.theme.colors.surface};
  border: 1px solid ${(p) => p.theme.colors.border};
  border-radius: ${(p) => p.theme.radius};
  padding: 16px; margin: 12px 0; text-decoration: none;
`;
const Avatar = styled.div`
  width: 48px; height: 48px; border-radius: 50%;
  background: ${(p) => p.theme.colors.primary};
  display: grid; place-items: center; color: #0e1726; font-weight: 700;
`;
const Name = styled.div`color: ${(p) => p.theme.colors.text}; font-weight: 600;`;
const Sub = styled.div`color: ${(p) => p.theme.colors.sub}; font-size: 13px;`;
const Empty = styled.div`color: ${(p) => p.theme.colors.sub}; margin-top: 40px; text-align: center;`;
const StartRow = styled.div`
  display: flex; align-items: center; gap: 12px;
  background: ${(p) => p.theme.colors.surface};
  border: 1px solid ${(p) => p.theme.colors.border};
  border-radius: ${(p) => p.theme.radius}; padding: 16px; margin: 12px 0;
`;

export default function ContactList() {
  const [speakers, setSpeakers] = useState<SpeakerSummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [label, setLabel] = useState("me");
  const nav = useNavigate();

  useEffect(() => {
    listSpeakers().then((s) => { setSpeakers(s); setLoaded(true); });
  }, []);

  return (
    <Wrap>
      <Title>callone</Title>
      <Sub>바로 통화 — 음성·사진은 다음 화면에서 올립니다(서버에 안 남음)</Sub>
      <StartRow>
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="상대 이름(대화 저장 라벨)"
          style={{ flex: 1, padding: 10, borderRadius: 8, border: "1px solid #2a3a52", background: "#0c1422", color: "#fff" }}
        />
        <button
          onClick={() => nav(`/call/${encodeURIComponent(label.trim() || "me")}`)}
          style={{ padding: "10px 16px", borderRadius: 20, border: "none", background: "#7aa2f7", color: "#0e1726", fontWeight: 700, cursor: "pointer", whiteSpace: "nowrap" }}
        >📞 통화 시작</button>
      </StartRow>
      {speakers.length > 0 && <Sub style={{ marginTop: 24 }}>저장된 화자</Sub>}
      {speakers.map((s) => (
        <Card key={s.speaker_id} to={`/call/${s.speaker_id}`}>
          <Avatar>{(s.name || s.speaker_id)[0]}</Avatar>
          <div>
            <Name>{s.name || `화자 ${s.speaker_id}`}</Name>
            <Sub>{s.relation || "관계 미입력"} · {s.region}</Sub>
          </div>
        </Card>
      ))}
      {loaded && speakers.length === 0 && (
        <Empty>저장된 화자가 없어도 됩니다 — 위에서 바로 통화하세요.<br />(음성·사진은 통화 화면에서 직접 업로드)</Empty>
      )}
    </Wrap>
  );
}
