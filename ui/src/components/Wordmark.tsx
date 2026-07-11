// call:one 워드마크 — 콜론(:)이 로고. blink 시 신호등처럼 깜빡인다(연결/대기 상태 표시 겸용).
import styled, { css, keyframes } from "styled-components";

const blinkAnim = keyframes`0%,100%{opacity:1} 50%{opacity:0.15}`;

const Mark = styled.span<{ size?: number }>`
  font-family: ${(p) => p.theme.font.display};
  font-weight: 600;
  font-size: ${(p) => p.size ?? 20}px;
  letter-spacing: -0.01em;
  color: ${(p) => p.theme.colors.ink};
  user-select: none;
`;

const Colon = styled.span<{ blink?: boolean }>`
  color: ${(p) => p.theme.colors.accent};
  ${(p) => p.blink && css`animation: ${blinkAnim} 1.1s steps(1) infinite;`}
`;

export default function Wordmark({ size, blink }: { size?: number; blink?: boolean }) {
  return (
    <Mark size={size}>call<Colon blink={blink}>:</Colon>one</Mark>
  );
}
