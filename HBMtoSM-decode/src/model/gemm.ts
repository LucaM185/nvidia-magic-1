/**
 * Memory-bound matrix-vector product used during autoregressive decode.
 *
 * A single activation row x (1xK) is multiplied by the weight matrix W
 * (KxN). The result y is another single row (1xN). Unlike a square GEMM,
 * there is no batch dimension over which to reuse the weights.
 */

export const MODEL = {
  m: 1,
  k: 256,
  n: 256,
  elementBytes: 4,
  tileN: 32,
  smCount: 64,
  activeSmCount: 8,
  activeRow: 3,
} as const;

export const ARCH = {
  hbm: {
    capacity: "80 GB",
    bandwidth: "~2 TB/s",
    latency: "~400-600 cycles",
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
  id: "x" | "W" | "y";
  rows: number;
  cols: number;
  bytes: number;
}

export interface DecodeBlock {
  id: number;
  row: number;
  col: number;
  smId: number;
  active: boolean;
  yRegion: Rect;
  xRegion: Rect;
  wRegion: Rect;
}

export interface DecodeModel {
  m: number;
  k: number;
  n: number;
  tileN: number;
  gridN: number;
  elementBytes: number;
  matrices: { x: MatrixState; W: MatrixState; y: MatrixState };
  blocks: DecodeBlock[];
}

export const L2_BYTES = 40 * 1024 * 1024;

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${Math.round(bytes / 1024)} KB`;
}

export function cacheUsed(bytes: number): string {
  const pct = (bytes / L2_BYTES) * 100;
  const digits = pct >= 10 ? 0 : 2;
  return `${formatBytes(bytes)} · ${pct.toFixed(digits)}% of L2`;
}

export function dataBytes(): number {
  return MODEL.m * MODEL.k * MODEL.elementBytes;
}

export function weightBytes(): number {
  return MODEL.k * MODEL.n * MODEL.elementBytes;
}

export function outputBytes(): number {
  return MODEL.m * MODEL.n * MODEL.elementBytes;
}

export function weightPanelBytes(): number {
  return MODEL.k * MODEL.tileN * MODEL.elementBytes;
}

export function outputTileBytes(): number {
  return MODEL.m * MODEL.tileN * MODEL.elementBytes;
}

export function residentBytes(): number {
  return dataBytes() + weightBytes();
}

export function bytesPerActiveSm(): number {
  return dataBytes() + weightPanelBytes() + outputTileBytes();
}

export function flopCount(): number {
  return 2 * MODEL.m * MODEL.k * MODEL.n;
}

/** Ideal lower bound: read x and W once, then write y once. */
export function arithmeticIntensity(): number {
  return flopCount() / (dataBytes() + weightBytes() + outputBytes());
}

export function createModel(): DecodeModel {
  const { m, k, n, tileN, elementBytes, activeRow } = MODEL;
  const gridN = n / tileN;
  if (!Number.isInteger(gridN) || gridN !== MODEL.activeSmCount) {
    throw new Error("tileN must split the output row across the active SMs");
  }

  const blocks: DecodeBlock[] = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const id = row * 8 + col;
      blocks.push({
        id,
        row,
        col,
        smId: id,
        active: row === activeRow,
        yRegion: { row: 0, col: col * tileN, rows: m, cols: tileN },
        xRegion: { row: 0, col: 0, rows: m, cols: k },
        wRegion: { row: 0, col: col * tileN, rows: k, cols: tileN },
      });
    }
  }

  const matrix = (id: MatrixState["id"], rows: number, cols: number): MatrixState => ({
    id,
    rows,
    cols,
    bytes: rows * cols * elementBytes,
  });

  return {
    m,
    k,
    n,
    tileN,
    gridN,
    elementBytes,
    matrices: {
      x: matrix("x", m, k),
      W: matrix("W", k, n),
      y: matrix("y", m, n),
    },
    blocks,
  };
}

const MODEL_SINGLETON = createModel();

export function getModel(): DecodeModel {
  return MODEL_SINGLETON;
}

export function blockAt(row: number, col: number): DecodeBlock {
  const block = MODEL_SINGLETON.blocks.find((entry) => entry.row === row && entry.col === col);
  if (!block) throw new Error(`no block at ${row},${col}`);
  return block;
}
