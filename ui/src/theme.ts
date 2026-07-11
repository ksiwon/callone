// call:one(콜론) 전시 디자인 언어 — 종이·잉크·수화기 주홍.
// 원칙: 액센트는 주홍 하나. 상태·수치는 모노스페이스. 제목은 세리프(서식/인쇄물 결).
export const theme = {
  colors: {
    paper: "#F4F0E6",    // 바탕 — 따뜻한 종이
    panel: "#ECE7D9",    // 한 단계 가라앉은 면
    ink: "#221E16",      // 본문 잉크
    faint: "#776F5E",    // 바랜 잉크(보조 텍스트)
    line: "#CBC3B0",     // 헤어라인
    accent: "#B5402C",   // 수화기 주홍
    onAccent: "#F4F0E6", // 주홍 위 글자
    night: "#181510",    // 영상면·코드 배경
    onNight: "#E9E3D3",  // 어두운 면 위 글자
  },
  font: {
    display: `'Noto Serif KR', 'NanumMyeongjo', 'Nanum Myeongjo', 'Batang', 'AppleMyungjo', serif`,
    body: `'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', system-ui, sans-serif`,
    mono: `'IBM Plex Mono', 'JetBrains Mono', 'Consolas', 'Courier New', monospace`,
  },
  radius: "2px",
};

export type Theme = typeof theme;
