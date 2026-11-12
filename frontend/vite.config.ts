import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend serves the built SPA from `frontend/dist`.
// During development Vite proxies API and media requests to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/media": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
