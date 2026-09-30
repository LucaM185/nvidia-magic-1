import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  server: { port: 5173 },
  build: {
    rollupOptions: {
      input: {
        home: resolve(import.meta.dirname, "index.html"),
        hbmToSm: resolve(import.meta.dirname, "HBMtoSM/index.html"),
        hbmToSmDecode: resolve(import.meta.dirname, "HBMtoSM-decode/index.html"),
        smToResults: resolve(import.meta.dirname, "SMtoResults/index.html"),
      },
    },
  },
});
