import { SM_TOUR, hbmMatrixCenter, hbmModule, type Vec3 } from "../layout";
import {
  MODEL,
  arithmeticIntensity,
  bytesPerActiveSm,
  dataBytes,
  flopCount,
  formatBytes,
  getModel,
  outputBytes,
  weightBytes,
  weightPanelBytes,
  type SMState,
} from "./gemm";

export const DURATION = 24;

const TOUR_START = 8;
const TOUR_STEP = 0.9;
const TOUR_END = TOUR_START + SM_TOUR.length * TOUR_STEP;
const FILL_START = TOUR_END + 0.2;
const FILL_STAGGER = 0.055;
const COMPUTE_START = FILL_START + 0.52;
const MATMUL_DURATION = 1.35;
const WRITE_START = 18.2;

export interface Phase {
  t: number;
  id: string;
  short: string;
  title: string;
  body: string;
}

export const PHASES: Phase[] = [
  {
    t: 0,
    id: "establish",
    short: "Decode",
    title: "One row meets a weight matrix",
    body: "x is only 1×256 (1 KB), while W is 256×256 (256 KB). This is the matrix-vector shape produced by one-token-at-a-time decoding.",
  },
  {
    t: 2,
    id: "fill",
    short: "To L2",
    title: "The weights dominate the transfer",
    body: "The thin blue data row and the orange weight matrix move from HBM to L2. Almost every transferred byte belongs to W.",
  },
  {
    t: 6.1,
    id: "cached",
    short: "In cache",
    title: "257 KB moved for one row",
    body: "x + W occupy 257 KB. The multiply performs only 131k FLOPs: an ideal arithmetic intensity of about 0.5 FLOP per byte.",
  },
  {
    t: TOUR_START,
    id: "sms",
    short: "Eight SMs",
    title: "Only one strip of SMs has work",
    body: "Eight SMs split the 256-wide output. Each receives the same 1 KB row and a different 32 KB weight panel; the other 56 SMs stay idle.",
  },
  {
    t: FILL_START,
    id: "compute",
    short: "Brief math",
    title: "Memory movement outweighs math",
    body: "The eight dot-product groups finish quickly. Weight bytes were streamed once, but each weight contributes to only one multiply-add for this token.",
  },
  {
    t: WRITE_START,
    id: "writeback",
    short: "Output",
    title: "Eight tiny results become one row",
    body: "Eight 1×32 segments assemble into y, a 1×256 row of only 1 KB, and return to HBM.",
  },
  {
    t: 21.7,
    id: "bound",
    short: "Memory-bound",
    title: "Low arithmetic intensity",
    body: "About 258 KB cross the memory boundary for 131k FLOPs: ≈0.5 FLOP/byte. Throughput is constrained by bandwidth, not peak compute.",
  },
];

export interface MatrixLook {
  opacity: number;
  fill: number;
  scan: number;
  region: [number, number, number, number] | null;
  regionStrength: number;
  panelEdges: number;
  lanes: number[];
  laneAxis: number;
}

function quietLanes(): number[] {
  return [0, 0, 0, 0, 0, 0, 0, 0];
}

export interface BlockVisual {
  row: number;
  col: number;
  state: SMState;
  progress: number;
  load: number;
  dim: number;
  hot: number;
}

export interface TimelineSample {
  t: number;
  phase: Phase;
  camera: { position: Vec3; target: Vec3 };
  hbmA: MatrixLook;
  hbmB: MatrixLook;
  hbmC: MatrixLook;
  l2A: MatrixLook;
  l2B: MatrixLook;
  l2C: MatrixLook;
  l2Pulse: number;
  channelA: { opacity: number; active: number };
  channelB: { opacity: number; active: number };
  returnFlow: { opacity: number; active: number };
  travelA: number;
  travelB: number;
  travelC: number;
  cMover: number;
  windowContent: number;
  focus: { row: number; col: number; amount: number; tour: number };
  assemble: number;
  tileOpacity: number;
  blocks: BlockVisual[];
  callouts: { workingSet: number; stored: number };
  meters: {
    a: string;
    b: string;
    c: string;
    aBar: number;
    bBar: number;
    cBar: number;
  };
}

interface CameraKey {
  t: number;
  position: Vec3;
  target: Vec3;
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function smooth(t: number): number {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

function ramp(t: number, a: number, b: number): number {
  return smooth((t - a) / (b - a));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function mix3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

function pulse(t: number, center: number, width: number): number {
  const x = (t - (center - width / 2)) / width;
  if (x <= 0 || x >= 1) return 0;
  return Math.sin(x * Math.PI) ** 2;
}

function look(partial: Partial<MatrixLook> & Pick<MatrixLook, "opacity" | "fill">): MatrixLook {
  return {
    scan: -1,
    region: null,
    regionStrength: 0,
    panelEdges: 0.14,
    lanes: quietLanes(),
    laneAxis: 1,
    ...partial,
  };
}

export function phaseAt(t: number): Phase {
  let current = PHASES[0];
  for (const phase of PHASES) if (t >= phase.t) current = phase;
  return current;
}

function cameraAt(t: number): { position: Vec3; target: Vec3 } {
  const cModule = hbmMatrixCenter(hbmModule("C"));
  const keys: CameraKey[] = [
    { t: 0, position: [0, 25.5, 14.5], target: [0, 0.25, 0] },
    { t: 2, position: [0, 22, 10.5], target: [0, 0.65, 0] },
    { t: 4.2, position: [0, 15.8, 5.7], target: [0, 0.82, 0] },
    { t: 6.2, position: [0.08, 15.5, 4.4], target: [0, 0.68, 0] },
    { t: TOUR_START, position: [0.04, 14.8, 5.2], target: [0, 0.58, -0.45] },
    { t: TOUR_END, position: [0.02, 14.5, 5.5], target: [0, 0.55, -0.45] },
    { t: WRITE_START, position: [0.15, 12.2, 8], target: [0, 1.35, 0] },
    { t: WRITE_START + 1.1, position: [0.35, 10.4, 6.6], target: [0, 1.5, 0] },
    { t: WRITE_START + 2.2, position: [5.4, 11.2, 7.2], target: [cModule[0], 0.82, cModule[2]] },
    { t: DURATION, position: [7.4, 10.4, 7], target: [cModule[0], 0.65, cModule[2]] },
  ];

  if (t <= keys[0].t) return { position: keys[0].position, target: keys[0].target };
  const last = keys[keys.length - 1];
  if (t >= last.t) return { position: last.position, target: last.target };
  let index = 0;
  while (index < keys.length - 2 && t > keys[index + 1].t) index += 1;
  const a = keys[index];
  const b = keys[index + 1];
  const u = smooth((t - a.t) / (b.t - a.t));
  return { position: mix3(a.position, b.position, u), target: mix3(a.target, b.target, u) };
}

interface TourHit {
  index: number;
  row: number;
  col: number;
  amount: number;
  tour: number;
  local: number;
}

function stepAmount(local: number): number {
  return smooth(local / 0.12) * (1 - smooth((local - 0.82) / 0.14));
}

function tourAt(t: number): TourHit | null {
  if (t < TOUR_START || t >= TOUR_END) return null;
  const elapsed = t - TOUR_START;
  const index = Math.min(SM_TOUR.length - 1, Math.floor(elapsed / TOUR_STEP));
  const local = (elapsed - index * TOUR_STEP) / TOUR_STEP;
  const step = SM_TOUR[index];
  const tour = ramp(t, TOUR_START, TOUR_START + 0.28) * (1 - ramp(t, TOUR_END - 0.18, TOUR_END));
  return { index, row: step.row, col: step.col, amount: stepAmount(local), tour, local };
}

function tourCopy(index: number): { title: string; body: string } {
  const { row, col } = SM_TOUR[index];
  const start = col * MODEL.tileN;
  return {
    title: `SM (${row}, ${col}) · output ${start}:${start + MODEL.tileN}`,
    body: `The same x row (${formatBytes(dataBytes())}) meets W[:, ${start}:${start + MODEL.tileN}] (${formatBytes(weightPanelBytes())}). ${formatBytes(bytesPerActiveSm())} of traffic enables only 16,384 FLOPs.`,
  };
}

function blockVisual(t: number, row: number, col: number, tour: TourHit | null): BlockVisual {
  const active = row === MODEL.activeRow;
  if (!active) {
    return { row, col, state: "idle", progress: 0, load: 0, dim: tour ? 0.12 : 0.42, hot: 0 };
  }

  const stepIndex = col;
  const stepStart = TOUR_START + stepIndex * TOUR_STEP;
  let load = ramp(t, stepStart + 0.1, stepStart + 0.42);
  let progress = 0;
  if (t >= FILL_START) {
    const stagger = col * FILL_STAGGER;
    load = Math.max(load, ramp(t, FILL_START + stagger, FILL_START + 0.34 + stagger));
    progress = ramp(t, COMPUTE_START + stagger, COMPUTE_START + stagger + MATMUL_DURATION);
  }

  const isCurrent = tour !== null && tour.col === col && tour.amount > 0.18;
  let state: SMState = "idle";
  if (t >= TOUR_START) state = "assigned";
  if (load > 0.22 && progress < 0.97) state = progress < 0.2 ? "loading" : "computing";
  if (progress >= 0.97) state = "complete";

  let dim = 1;
  if (tour?.tour) dim = isCurrent ? 1 : 0.36;
  return { row, col, state, progress, load, dim, hot: isCurrent ? 1 : 0 };
}

export function sampleTimeline(time: number): TimelineSample {
  const t = Math.min(DURATION, Math.max(0, time));
  const model = getModel();
  const tour = tourAt(t);
  const travelA = ramp(t, 2.1, 4.25);
  const travelB = ramp(t, 3.05, 6.05);
  const travelC = ramp(t, WRITE_START + 1.25, WRITE_START + 3.05);
  const cMover = ramp(t, WRITE_START + 0.65, WRITE_START + 0.95);
  const lanesB = quietLanes();
  if (tour) lanesB[tour.col] = tour.tour;
  const residentA = travelA >= 0.992;
  const residentB = travelB >= 0.992;
  const blocks = model.blocks.map((block) => blockVisual(t, block.row, block.col, tour));
  const activeBlocks = blocks.filter((block) => block.row === MODEL.activeRow);
  const avg = activeBlocks.reduce((sum, block) => sum + block.progress, 0) / activeBlocks.length;
  const streamA = ramp(t, 2.05, 2.4) * (1 - ramp(t, 4.1, 4.5));
  const streamB = ramp(t, 3, 3.35) * (1 - ramp(t, 5.9, 6.35));
  const returning = ramp(t, WRITE_START + 1.2, WRITE_START + 1.55) * (1 - ramp(t, WRITE_START + 2.9, WRITE_START + 3.3));
  const phase = { ...phaseAt(t) };
  if (tour) Object.assign(phase, tourCopy(tour.index));

  let cText = "waiting";
  let cBar = 0;
  if (tour) {
    cText = `1×32 · SM ${tour.row},${tour.col}`;
    cBar = ((tour.index + tour.local) / SM_TOUR.length) * 0.45;
  } else if (t >= FILL_START && avg < 0.98) {
    cText = "8 × 1×32";
    cBar = 0.45 + avg * 0.55;
  } else if (avg >= 0.98 && t < WRITE_START + 1.1) {
    cText = "8 segments → one row";
    cBar = 1;
  } else if (t >= WRITE_START + 1.1 && t < WRITE_START + 3.05) {
    cText = "1×256 · SMs → HBM";
    cBar = 1;
  } else if (t >= WRITE_START + 3.05) {
    cText = "1×256 · 1 KB in HBM";
    cBar = 1;
  }

  return {
    t,
    phase,
    camera: cameraAt(t),
    hbmA: look({ opacity: lerp(1, 0.2, Math.min(1, travelA / 0.08)), fill: 1, panelEdges: 0.35 }),
    hbmB: look({ opacity: lerp(1, 0.2, Math.min(1, travelB / 0.08)), fill: 1, panelEdges: 0.35 }),
    hbmC: look({ opacity: 0, fill: 1, panelEdges: 0.4 }),
    l2A: residentA
      ? look({
          opacity: 1,
          fill: 1,
          panelEdges: 0.72,
          region: tour ? [0, 0, 1, 1] : null,
          regionStrength: tour?.tour ?? 0,
        })
      : look({ opacity: 1, fill: 1, scan: travelA > 0.02 && travelA < 0.98 ? 1 - travelA : -1, panelEdges: 0.5 }),
    l2B: residentB
      ? look({ opacity: 1, fill: 1, panelEdges: 0.72, lanes: lanesB, laneAxis: 2 })
      : look({ opacity: 1, fill: 1, scan: travelB > 0.02 && travelB < 0.98 ? 1 - travelB : -1, panelEdges: 0.5, laneAxis: 2 }),
    l2C: look({ opacity: 1, fill: 1, panelEdges: 0.7 }),
    l2Pulse: Math.max(pulse(t, 4.35, 0.75), pulse(t, 6.15, 0.9)),
    channelA: { opacity: streamA, active: streamA },
    channelB: { opacity: streamB, active: streamB },
    returnFlow: { opacity: returning, active: returning },
    travelA,
    travelB,
    travelC,
    cMover,
    windowContent: 1 - ramp(t, WRITE_START + 0.05, WRITE_START + 0.85),
    focus: tour ? { row: tour.row, col: tour.col, amount: tour.amount, tour: tour.tour } : { row: -1, col: -1, amount: 0, tour: 0 },
    assemble: ramp(t, WRITE_START, WRITE_START + 0.85),
    tileOpacity: ramp(t, FILL_START, FILL_START + 0.4),
    blocks,
    callouts: {
      workingSet: ramp(t, 6.2, 6.65) * (1 - ramp(t, 7.7, 8.1)),
      stored: ramp(t, WRITE_START + 2.8, WRITE_START + 3.25),
    },
    meters: {
      a: t < 2.1 ? "x · 1×256 · 1 KB in HBM" : t > 4.3 ? "x · 1×256 · 1 KB in L2" : "x · 1×256 · HBM → L2",
      b: t < 3.05 ? "W · 256×256 · 256 KB in HBM" : t > 6.1 ? "W · 256×256 · 256 KB in L2" : "W · 256×256 · HBM → L2",
      c: cText,
      aBar: travelA,
      bBar: travelB,
      cBar,
    },
  };
}

export function workloadCaption(): string {
  return `DECODE · 1×256 · FP32 · ${formatBytes(weightBytes())} weights · ${formatBytes(outputBytes())} output · ${flopCount().toLocaleString("en-US")} FLOPs · ${arithmeticIntensity().toFixed(2)} FLOP/B`;
}
