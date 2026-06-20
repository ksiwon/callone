import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// callone 로컬 백엔드(callone-serve, :8000)로 프록시. 외부 API 없음(§2,§17).
// 서브경로 프록시(Elice 터널 /proxy/5173/ 등)서 띄울 때: VITE_BASE=/proxy/5173/ npm run dev
//   → 에셋·HMR·WS 가 그 prefix 로 나가 터널이 라우팅 + 프록시가 prefix 떼고 :8000 으로 전달.
//   기본 '/' (localhost/SSH터널/직접노출).
const base = process.env.VITE_BASE || "/";
const p = base.replace(/\/$/, "");   // '' 또는 '/proxy/5173'

export default defineConfig({
  plugins: [react()],
  base,
  server: {
    host: true,            // 0.0.0.0 — 원격(SSH 터널/포트노출)서 접근 허용
    port: 5173,
    allowedHosts: true,    // 원격 프록시 도메인 접근 시 vite host 체크 우회(dev 전용)
    proxy: {
      // base prefix 가 붙은 요청(/proxy/5173/api)을 받아 prefix 떼고 :8000 으로. base='/'면 그대로.
      [`${p}/api`]: { target: "http://localhost:8000", rewrite: (path) => path.replace(p, "") },
      [`${p}/ws`]: { target: "ws://localhost:8000", ws: true, rewrite: (path) => path.replace(p, "") },
    },
  },
});
