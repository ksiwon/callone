// ContactList (§17.1) — A/B 를 라벨한 이름·관계로 표시 → 통화 시작.
import { useEffect, useState } from "react";
import styled from "styled-components";
import { Link } from "react-router-dom";
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

export default function ContactList() {
  const [speakers, setSpeakers] = useState<SpeakerSummary[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    listSpeakers().then((s) => { setSpeakers(s); setLoaded(true); });
  }, []);

  return (
    <Wrap>
      <Title>callone</Title>
      <Sub>전화하고 싶은 사람을 고르세요</Sub>
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
        <Empty>
          아직 화자가 없습니다.<br />
          파이프라인 실행 후 <Link to="/editor/A" style={{ color: "#7aa2f7" }}>라벨링 편집기</Link>에서 등록하세요.
        </Empty>
      )}
    </Wrap>
  );
}
