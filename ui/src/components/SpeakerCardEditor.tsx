// SpeakerCardEditor (§11.2, §17.1) — 화자 라벨링 편집기.
// auto(추정) 초안 표시 + 대표 발화 재생 + user 필드 확정 저장.
// 방언 지역/세기는 자동 측정값을 초안으로, 사람이 덮어쓰기 가능.
import { useEffect, useState } from "react";
import styled from "styled-components";
import { useParams } from "react-router-dom";
import { getProfile, putProfile, getSamples, SpeakerProfile } from "../api/calloneClient";
import Wordmark from "./Wordmark";

const Wrap = styled.div`max-width: 560px; margin: 0 auto; padding: 48px 28px 40px;`;
const Title = styled.h2`
  font-family: ${(p) => p.theme.font.display}; font-weight: 600; font-size: 26px; margin: 26px 0 8px;
`;
const Rule = styled.hr<{ strong?: boolean }>`
  border: none; margin: 22px 0;
  border-top: ${(p) => (p.strong ? `2px solid ${p.theme.colors.ink}` : `1px solid ${p.theme.colors.line}`)};
`;
const SecLabel = styled.div`
  font-family: ${(p) => p.theme.font.mono}; font-size: 11px; letter-spacing: 0.14em;
  color: ${(p) => p.theme.colors.faint}; text-transform: uppercase; margin: 0 0 12px;
`;
const Auto = styled.div`
  color: ${(p) => p.theme.colors.faint}; font-size: 13px; line-height: 1.8;
  & b { color: ${(p) => p.theme.colors.ink}; }
`;
const Row = styled.label`
  display: flex; justify-content: space-between; align-items: baseline; gap: 16px;
  padding: 9px 0; border-bottom: 1px solid ${(p) => p.theme.colors.line}; font-size: 14px;
`;
const Input = styled.input`
  flex: 1; max-width: 62%; background: transparent; color: ${(p) => p.theme.colors.ink};
  border: none; border-bottom: 1px solid transparent; border-radius: 0;
  padding: 4px 2px; font-size: 14px; text-align: right;
  &::placeholder { color: ${(p) => p.theme.colors.line}; }
  &:focus { outline: none; border-bottom-color: ${(p) => p.theme.colors.ink}; }
`;
const Save = styled.button`
  width: 100%; padding: 15px; margin-top: 26px; border: none; cursor: pointer;
  border-radius: ${(p) => p.theme.radius};
  background: ${(p) => p.theme.colors.ink}; color: ${(p) => p.theme.colors.paper};
  font-size: 15px; font-weight: 600;
  &:hover { background: ${(p) => p.theme.colors.accent}; color: ${(p) => p.theme.colors.onAccent}; }
`;
const Sample = styled.div`
  color: ${(p) => p.theme.colors.faint}; font-size: 13px; line-height: 1.7; padding: 4px 0;
  & span { font-family: ${(p) => p.theme.font.mono}; font-size: 11px; }
`;

export default function SpeakerCardEditor() {
  const { id = "A" } = useParams();
  const [prof, setProf] = useState<SpeakerProfile | null>(null);
  const [samples, setSamples] = useState<any[]>([]);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getProfile(id).then(setProf);
    getSamples(id).then(setSamples);
  }, [id]);

  if (!prof) return <Wrap>불러오는 중…</Wrap>;
  const u = prof.user;
  const d = prof.auto?.dialect ?? {};

  const setU = (k: string, v: any) => setProf({ ...prof, user: { ...u, [k]: v } });

  return (
    <Wrap>
      <Wordmark />
      <Title>화자 기록 — {id}</Title>
      <Rule strong />

      <SecLabel>자동 추정 · 초안</SecLabel>
      <Auto>
        성별 추정 {prof.auto?.gender_est} · 나이대 {prof.auto?.age_band_est}<br />
        방언 <b>{d.region_est}</b> (신뢰 {(d.confidence ?? 0).toFixed(2)}) ·
        세기 <b>{(d.intensity_0to1 ?? 0).toFixed(2)}</b><br />
        대표 어미 {(d.markers ?? []).slice(0, 6).map((m: any) => m.form).join(", ") || "없음"}<br />
        반말율 {(prof.auto?.speech?.banmal_ratio ?? 0).toFixed(2)} ·
        평균 문장 {(prof.auto?.speech?.avg_sentence_len ?? 0).toFixed(1)}어절
      </Auto>
      <Rule />

      <SecLabel>대표 발화</SecLabel>
      {samples.length === 0 && <Sample>샘플 없음</Sample>}
      {samples.map((s, i) => (
        <Sample key={i}>{s.text || "(텍스트 없음)"} <span>[{s.call_id} {s.start?.toFixed?.(1)}s]</span></Sample>
      ))}
      <Rule />

      <SecLabel>사람이 확정</SecLabel>
      <Row>이름 <Input value={u.name} onChange={(e) => setU("name", e.target.value)} /></Row>
      <Row>나이 <Input type="number" value={u.age ?? ""} onChange={(e) => setU("age", Number(e.target.value) || null)} /></Row>
      <Row>성별 <Input value={u.gender} onChange={(e) => setU("gender", e.target.value)} /></Row>
      <Row>관계 <Input value={u.relation} onChange={(e) => setU("relation", e.target.value)} /></Row>
      <Row>호칭 <Input value={u.register} onChange={(e) => setU("register", e.target.value)} placeholder="반말/존댓말" /></Row>
      <Row>특징 <Input value={u.traits.join(", ")} onChange={(e) => setU("traits", e.target.value.split(",").map((s) => s.trim()).filter(Boolean))} /></Row>
      <Row>입버릇 <Input value={u.catchphrases.join(", ")} onChange={(e) => setU("catchphrases", e.target.value.split(",").map((s) => s.trim()).filter(Boolean))} /></Row>
      <Row>금기 <Input value={u.taboo.join(", ")} onChange={(e) => setU("taboo", e.target.value.split(",").map((s) => s.trim()).filter(Boolean))} /></Row>
      <Row>방언 지역(덮어쓰기) <Input value={u.dialect_region_override ?? ""} onChange={(e) => setU("dialect_region_override", e.target.value || null)} placeholder={d.region_est} /></Row>
      <Row>방언 세기(덮어쓰기 0~1) <Input type="number" step="0.01" value={u.dialect_intensity_override ?? ""} onChange={(e) => setU("dialect_intensity_override", e.target.value === "" ? null : Number(e.target.value))} placeholder={String(d.intensity_0to1 ?? "")} /></Row>
      <Row>방언 확인됨 <input type="checkbox" checked={u.dialect_confirmed} onChange={(e) => setU("dialect_confirmed", e.target.checked)} /></Row>

      <Save onClick={async () => { await putProfile(id, prof); setSaved(true); }}>
        {saved ? "저장됨" : "프로필 저장"}
      </Save>
    </Wrap>
  );
}
