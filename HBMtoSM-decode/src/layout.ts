export type Vec3 = [number, number, number];

/**
 * Floorplan, viewed from above.
 * HBM modules ring the package. L2 is a short bar the full width of the SM banks,
 * the same relationship as the GA100 floorplan.
 * A and B are squares. On L2 they cover a fraction of that bar, not the whole cache.
 * Row 0 is the far bank (−Z). Row 7 is the near bank (+Z).
 */
export const GRID = 8;
export const SM_PITCH = 0.68;
export const SM_BODY = 0.51;
export const SM_Y = 0.62;
/** Eight SMs split the 256-wide output row into 32-value segments. */
export const SM_TOUR: { row: number; col: number }[] = [
  { row: 3, col: 0 },
  { row: 3, col: 1 },
  { row: 3, col: 2 },
  { row: 3, col: 3 },
  { row: 3, col: 4 },
  { row: 3, col: 5 },
  { row: 3, col: 6 },
  { row: 3, col: 7 },
];
/** Edge length for the square weight matrix. */
export const MATRIX = 1.34;
/** One logical row: the same height as one of the 32 visible matrix cells. */
export const VECTOR_DEPTH = MATRIX / 32;

/** H100-class FHFL card proportions: about 268 × 111 mm. */
export const BOARD = { width: 28.0, depth: 11.62 };

/** SM bank plate. L2 uses the same width so the three bands share one edge. */
export const SM_BANK_WIDTH = (GRID - 1) * SM_PITCH + SM_BODY + 0.5;
export const SM_BANK_DEPTH = 3 * SM_PITCH + SM_BODY + 0.42;

/**
 * Full L2 bar. Width matches the SM banks. Depth is a short band between them,
 * just tall enough for the resident squares to sit inside the rim.
 */
export const L2_SLAB = { center: [0, 0.56, 0] as Vec3, width: SM_BANK_WIDTH, depth: 1.5 };
/** Clearance between the L2 rim and the inner edge of each SM bank plate. */
const L2_SM_GAP = 0.12;
export const BANK_CENTER_Z = L2_SLAB.depth / 2 + L2_SM_GAP + SM_BANK_DEPTH / 2;

/** Die hugs the SM banks. The old floor was much wider than the array. */
const DIE_MARGIN_X = 0.42;
const DIE_MARGIN_Z = 0.26;
const DIE_BEZEL = 0.28;
const contentDepth = BANK_CENTER_Z * 2 + SM_BANK_DEPTH;
export const DIE_FLOOR = {
  width: SM_BANK_WIDTH + DIE_MARGIN_X * 2,
  depth: contentDepth + DIE_MARGIN_Z * 2,
};
export const DIE_FRAME = {
  width: DIE_FLOOR.width + DIE_BEZEL * 2,
  depth: DIE_FLOOR.depth + DIE_BEZEL * 2,
};
/** Square cells, so a square matrix covers a square block of the cache. */
const L2_GRID_INSET = 0.08;
const L2_GRID_DEPTH = L2_SLAB.depth - L2_GRID_INSET * 2;
const L2_GRID_WIDTH = L2_SLAB.width - L2_GRID_INSET * 2;
const L2_CELL = L2_GRID_DEPTH / 4;
export const L2_GRID = {
  width: L2_GRID_WIDTH,
  depth: L2_GRID_DEPTH,
  cellsX: Math.round(L2_GRID_WIDTH / L2_CELL),
  cellsZ: 4,
};

export const L2_A = { center: [-1.85, 0.96, 0] as Vec3, width: MATRIX, depth: VECTOR_DEPTH };
export const L2_B = { center: [1.85, 0.96, 0] as Vec3, width: MATRIX, depth: MATRIX };
/** Staging pose for the finished 1x256 output row. */
export const C_FORM: Vec3 = [0, 1.62, 0];

export interface HbmModule {
  id: string;
  x: number;
  z: number;
  role: "A" | "B" | "C" | "idle";
  /** Stack height. Varied so the modules around the die are not copies of one part. */
  height: number;
}

const HBM_FOOT = 1.48;
/** Plinth-to-frame gap, so the stacks stay beside the smaller package. */
const HBM_X = DIE_FRAME.width / 2 + 0.52 + (HBM_FOOT + 0.28) / 2;

export const HBM_MODULES: HbmModule[] = [
  { id: "W0", x: -HBM_X, z: -2.85, role: "idle", height: 0.68 },
  { id: "W1", x: -HBM_X, z: 0, role: "A", height: 0.76 },
  { id: "W2", x: -HBM_X, z: 2.85, role: "idle", height: 0.7 },
  { id: "E0", x: HBM_X, z: -2.85, role: "idle", height: 0.7 },
  { id: "E1", x: HBM_X, z: 0, role: "B", height: 0.76 },
  { id: "E2", x: HBM_X, z: 2.85, role: "C", height: 0.72 },
];

export function hbmModule(role: "A" | "B" | "C"): HbmModule {
  const found = HBM_MODULES.find((mod) => mod.role === role);
  if (!found) throw new Error(`missing HBM ${role}`);
  return found;
}

/** Plane height for the square sitting on a module. */
export function hbmMatrixCenter(mod: HbmModule): Vec3 {
  return [mod.x, 0.3 + mod.height + 0.09, mod.z];
}

export function hbmFootprint(): number {
  return HBM_FOOT;
}

export function smWorld(row: number, col: number): Vec3 {
  const colSpan = (GRID - 1) * SM_PITCH;
  const bank = row < 4 ? 0 : 1;
  const rowInBank = row - bank * 4;
  const inner = (4 - 1) * SM_PITCH;
  const bankZ = (bank === 0 ? -1 : 1) * BANK_CENTER_Z;
  return [
    -colSpan / 2 + col * SM_PITCH,
    SM_Y,
    bankZ - inner / 2 + rowInBank * SM_PITCH,
  ];
}

/**
 * Planes with rotation.x = -PI/2 face up.
 * uv.y = 1 sits at −Z. Row 0 of a matrix is that far edge.
 */
export function rowBand(row: number, rows = GRID): [number, number] {
  return [1 - (row + 1) / rows, 1 - row / rows];
}

export function colBand(col: number, cols = GRID): [number, number] {
  return [col / cols, (col + 1) / cols];
}

export function planePoint(
  center: Vec3,
  width: number,
  depth: number,
  u: number,
  v: number,
): Vec3 {
  return [
    center[0] + (u - 0.5) * width,
    center[1],
    center[2] + (0.5 - v) * depth,
  ];
}

/** One 1x32 output segment. */
export const TILE_SIZE = MATRIX / GRID;

/** Slot of one SM's green segment inside the single output row. */
export function cTileSlot(_row: number, col: number, center: Vec3): Vec3 {
  const half = (GRID - 1) / 2;
  return [
    center[0] + (col - half) * TILE_SIZE,
    center[1],
    center[2],
  ];
}
