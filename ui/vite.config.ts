import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// callone 로컬 백엔드(callone-serve, :8000)로 프록시. 외부 API 없음(§2,§17).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
