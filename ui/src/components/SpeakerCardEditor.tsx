// SpeakerCardEditor (§11.2, §17.1) — 화자 라벨링 편집기.
// auto(추정) 초안 표시 + 대표 발화 재생 + user 필드 확정 저장.
// 방언 지역/세기는 자동 측정값을 초안으로, 사람이 덮어쓰기 가능.
import { useEffect, useState } from "react";
import styled from "styled-components";
import { useParams } from "react-router-dom";
import { getProfile, putProfile, getSamples, SpeakerProfile } from "../api/calloneClient";

const Wrap = styled.div`max-width: 560px; margin: 0 auto; padding: 24px; color: ${(p) => p.theme.colors.text};`;
const Section = styled.div`
  background: ${(p) => p.theme.colors.surface}; border: 1px solid ${(p) => p.theme.colors.border};
  border-radius: ${(p) => p.theme.radius}; padding: 16px; margin: 12px 0;
`;
const H = styled.h3`margin: 0 0 12px;`;
const Row = styled.label`display: flex; justify-content: space-between; align-items: center; margin: 8px 0; gap: 12px;`;
const Input = styled.input`
  flex: 1; background: ${(p) => p.theme.colors.bg}; color: ${(p) => p.theme.colors.text};
  border: 1px solid ${(p) => p.theme.colors.border}; border-radius: 8px; padding: 8px;
`;
const Auto = styled.div`color: ${(p) => p.theme.colors.sub}; font-size: 13px; line-height: 1.6;`;
const Save = styled.button`
  width: 100%; padding: 14px; border: none; border-radius: 12px; cursor: pointer;
  background: ${(p) => p.theme.colors.primary}; color: #0e1726; font-weight: 700; font-size: 16px;
`;
const Sample = styled.div`color: ${(p) => p.theme.colors.sub}; font-size: 13px; padding: 4px 0;`;

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
      <h2>화자 {id} 라벨링</h2>

      <Section>
        <H>자동 추정 (초안 — 수정 가능)</H>
        <Auto>
          성별 추정: {prof.auto?.gender_est} · 나이대: {prof.auto?.age_band_est}<br />
          방언: <b>{d.region_est}</b> (신뢰 {(d.confidence ?? 0).toFixed(2)}),
          세기 <b>{(d.intensity_0to1 ?? 0).toFixed(2)}</b><br />
          대표 어미: {(d.markers ?? []).slice(0, 6).map((m: any) => m.form).join(", ") || "없음"}<br />
          반말율: {(prof.auto?.speech?.banmal_ratio ?? 0).toFixed(2)} ·
          평균 문장 {(prof.auto?.speech?.avg_sentence_len ?? 0).toFixed(1)}어절
        </Auto>
      </Section>

      <Section>
        <H>대표 발화 (재생)</H>
        {samples.length === 0 && <Sample>샘플 없음</Sample>}
        {samples.map((s, i) => <Sample key={i}>▶ {s.text || "(텍스트 없음)"} [{s.call_id} {s.start?.toFixed?.(1)}s]</Sample>)}
      </Section>

      <Section>
        <H>사람이 확정</H>
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
      </Section>

      <Save onClick={async () => { await putProfile(id, prof); setSaved(true); }}>
        {saved ? "저장됨 ✓" : "프로필 저장"}
      </Save>
    </Wrap>
  );
}
