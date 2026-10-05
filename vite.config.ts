import { defineConfig } from "vite";
import { resolve } from "node:path";

const pages = [
  "HBMtoSM",
  "HBMtoSM-decode",
  "HBMtoSM-compare",
  "HBMtoSM-multigpu-decode",
  "HBMtoSM-multigpu-prefill",
  "SMtoResults",
  "SMtoResults-decode",
];

export default defineConfig({
  server: {
    port: 5173,
    // Transform every page's module graph at startup so the first visit is instant.
    warmup: { clientFiles: pages.map((p) => `./${p}/src/main.ts`) },
  },
  // Pre-bundle three.js up front instead of discovering it on first page load (which forces a reload).
  optimizeDeps: { entries: ["index.html", ...pages.map((p) => `${p}/index.html`)] },
  build: {
    rollupOptions: {
      input: {
        home: resolve(import.meta.dirname, "index.html"),
        hbmToSm: resolve(import.meta.dirname, "HBMtoSM/index.html"),
        hbmToSmDecode: resolve(import.meta.dirname, "HBMtoSM-decode/index.html"),
        hbmToSmCompare: resolve(import.meta.dirname, "HBMtoSM-compare/index.html"),
        hbmToSmMultiGpuDecode: resolve(import.meta.dirname, "HBMtoSM-multigpu-decode/index.html"),
        hbmToSmMultiGpuPrefill: resolve(import.meta.dirname, "HBMtoSM-multigpu-prefill/index.html"),
        smToResults: resolve(import.meta.dirname, "SMtoResults/index.html"),
        smToResultsDecode: resolve(import.meta.dirname, "SMtoResults-decode/index.html"),
      },
    },
  },
});
