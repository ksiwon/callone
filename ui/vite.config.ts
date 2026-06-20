import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// callone 로컬 백엔드(callone-serve, :8000)로 프록시. 외부 API 없음(§2,§17).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,            // 0.0.0.0 — 원격(SSH 터널/포트노출)서 접근 허용
    port: 5173,
    allowedHosts: true,    // 원격 프록시 도메인 접근 시 vite host 체크 우회(dev 전용)
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
