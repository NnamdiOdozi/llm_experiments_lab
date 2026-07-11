import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiProxy =
  process.env.VITE_API_PROXY ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiProxy,
        changeOrigin: true,
      },
      "/ws": {
        target: apiProxy.replace(/^http/, "ws"),
        changeOrigin: true,
        ws: true,
      },
    },
  },
});