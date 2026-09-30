/**
 * Square FP32 GEMM, decomposed into output tiles.
 *
 * Two scales of truth:
 * - This scene says which logical A row-panel and B column-panel an SM needs
 *   for the whole K dimension.
 * - The next scene, inside the SM window, walks that subset along K in smaller
 *   tiles through shared memory. `tileK` is reserved for that pass and is not
 *   streamed here.
 */

export const MODEL = {
  n: 256,
  elementBytes: 4,
  tileM: 32,
  tileN: 32,
  tileK: 32,
  smCount: 64,
} as const;

export const ARCH = {
  hbm: {
    capacity: "80 GB",
    bandwidth: "~2 TB/s",
    latency: "~400–600 cycles",
  },
  l2: {
    capacity: "40 MB",
    bandwidth: "~5 TB/s aggregate",
    latency: "~200 cycles",
    sharing: "shared across GPU",
  },
} as const;

export interface Rect {
  row: number;
  col: number;
  rows: number;
  cols: number;
}

export type SMState = "idle" | "assigned" | "loading" | "computing" | "complete";

export interface MatrixState {
  id: "A" | "B" | "C";
  n: number;
  bytes: number;
}

export interface GemmBlock {
  id: number;
  row: number;
  col: number;
  smId: number;
  cRegion: Rect;
  aRegion: Rect;
  bRegion: Rect;
  progress: number;
  state: SMState;
}

export interface GemmModel {
  n: number;
  tileM: number;
  tileN: number;
  tileK: number;
  gridM: number;
  gridN: number;
  elementBytes: number;
  matrices: { A: MatrixState; B: MatrixState; C: MatrixState };
  blocks: GemmBlock[];
}

export function matrixBytes(n = MODEL.n): number {
  return n * n * MODEL.elementBytes;
}

export function regionBytes(region: Rect): number {
  return region.rows * region.cols * MODEL.elementBytes;
}

/** On-chip L2 this scene measures the working set against. */
export const L2_BYTES = 40 * 1024 * 1024;

export function kb(bytes: number): string {
  return `${Math.round(bytes / 1024)} KB`;
}

/** Bytes plus the share of the 40 MB L2. */
export function cacheUsed(bytes: number): string {
  const pct = (bytes / L2_BYTES) * 100;
  const digits = pct >= 10 ? 0 : 2;
  return `${kb(bytes)} · ${pct.toFixed(digits)}% of L2`;
}

export function aPanelBytes(): number {
  return MODEL.tileM * MODEL.n * MODEL.elementBytes;
}

export function bPanelBytes(): number {
  return MODEL.n * MODEL.tileN * MODEL.elementBytes;
}

export function cTileBytes(): number {
  return MODEL.tileM * MODEL.tileN * MODEL.elementBytes;
}

export function residentBytes(): number {
  return matrixBytes() * 2;
}

export function createModel(): GemmModel {
  const { n, tileM, tileN, tileK, elementBytes, smCount } = MODEL;
  const gridM = n / tileM;
  const gridN = n / tileN;
  if (!Number.isInteger(gridM) || !Number.isInteger(gridN)) {
    throw new Error("tileM and tileN must divide N");
  }
  if (gridM * gridN !== smCount) {
    throw new Error("this demo maps one output tile onto each visible SM");
  }

  const bytes = matrixBytes(n);
  const blocks: GemmBlock[] = [];
  for (let row = 0; row < gridM; row++) {
    for (let col = 0; col < gridN; col++) {
      const id = row * gridN + col;
      blocks.push({
        id,
        row,
        col,
        smId: id,
        cRegion: { row: row * tileM, col: col * tileN, rows: tileM, cols: tileN },
        aRegion: { row: row * tileM, col: 0, rows: tileM, cols: n },
        bRegion: { row: 0, col: col * tileN, rows: n, cols: tileN },
        progress: 0,
        state: "idle",
      });
    }
  }

  const matrix = (id: "A" | "B" | "C"): MatrixState => ({ id, n, bytes });
  return {
    n,
    tileM,
    tileN,
    tileK,
    gridM,
    gridN,
    elementBytes,
    matrices: { A: matrix("A"), B: matrix("B"), C: matrix("C") },
    blocks,
  };
}

const MODEL_SINGLETON = createModel();

export function getModel(): GemmModel {
  return MODEL_SINGLETON;
}

export function blockAt(row: number, col: number): GemmBlock {
  const block = MODEL_SINGLETON.blocks.find((entry) => entry.row === row && entry.col === col);
  if (!block) throw new Error(`no block at ${row},${col}`);
  return block;
}
