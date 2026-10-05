import { GRID, SM_TOUR, hbmMatrixCenter, hbmModule, type Vec3 } from "../layout";
import {
  CYCLES_PER_SECOND,
  GPU,
  createClock,
  cyclesToSeconds,
  formatGpuTime,
  hbmTransferCycles,
  type KernelBudget,
} from "../../../src/gpu-timing";
import {
  MODEL,
  aPanelBytes,
  bPanelBytes,
  cTileBytes,
  cacheUsed,
  getModel,
  kb,
  matrixBytes,
  residentBytes,
  type SMState,
} from "./gemm";

/**
 * Physical steps run on the shared GPU clock (src/gpu-timing.ts), the same
 * one the decode scene uses. The tour and the tile assembly are narration:
 * the clock is stopped while they play.
 */
const HBM_START = 2;
/** A and B leave HBM together and share its bandwidth, so they land together. */
const HBM_IN_CYCLES = hbmTransferCycles(2 * matrixBytes());
const HBM_IN_END = HBM_START + cyclesToSeconds(HBM_IN_CYCLES);
const CACHED_T = HBM_IN_END + 0.2;

/** One output tile at a time, from SM (2, 3) through SM (3, 2). Clock stopped. */
const TOUR_START = CACHED_T + 2.2;
const TOUR_STEP = 1.5;
const TOUR_END = TOUR_START + SM_TOUR.length * TOUR_STEP;

/**
 * Per-SM budget. 8 K-steps; each brings a 32×32 A tile and a 32×32 B tile
 * (8 KB) and costs 32·32·32 FMAs on 128 FP32 lanes (256 cycles). Loads for
 * the next step overlap the math, so the SM is compute-bound: 2,048 cycles
 * of FMAs after one L2 latency and the first tile.
 */
const K_STEPS = MODEL.n / MODEL.tileK;
const TILE_LOAD_CYCLES = ((MODEL.tileM + MODEL.tileN) * MODEL.tileK * MODEL.elementBytes) / GPU.smBytesPerCycle;
const MATH_CYCLES = (MODEL.tileM * MODEL.tileN * MODEL.n) / GPU.fp32Lanes;
const STEP_MATH_CYCLES = MATH_CYCLES / K_STEPS;
const STORE_CYCLES = cTileBytes() / GPU.smBytesPerCycle;
const SM_CYCLES =
  2 * (GRID - 1) * GPU.launchStagger + GPU.l2Latency + TILE_LOAD_CYCLES + MATH_CYCLES + STORE_CYCLES;

const RUN_START = TOUR_END + 0.2;
const RUN = { t: RUN_START, cycle: HBM_IN_CYCLES, cycles: SM_CYCLES };
const RUN_END = RUN_START + cyclesToSeconds(SM_CYCLES);

/** Tiles gather into one matrix with the clock stopped, then C streams out. */
const WRITE_START = RUN_END + 0.3;
const OUT_START = WRITE_START + 1.35;
const HBM_OUT_CYCLES = hbmTransferCycles(matrixBytes());
const OUT = {
  t: OUT_START,
  cycle: HBM_IN_CYCLES + SM_CYCLES,
  cycles: HBM_OUT_CYCLES,
  prelude: { from: WRITE_START, share: 0.35 },
};
const OUT_END = OUT_START + cyclesToSeconds(HBM_OUT_CYCLES);
export const DURATION = OUT_END + 4;

const CLOCK = createClock([{ t: HBM_START, cycle: 0, cycles: HBM_IN_CYCLES }, RUN, OUT]);
/** Cycle budget for the side-by-side comparison, in kernel cycles from 0. */
export const BUDGET: KernelBudget = {
  hbmIn: HBM_IN_CYCLES,
  sm: SM_CYCLES,
  out: HBM_OUT_CYCLES,
  total: CLOCK.total,
  inBytes: 2 * matrixBytes(),
  outBytes: matrixBytes(),
  flops: 2 * MODEL.n ** 3,
  activeSms: MODEL.smCount,
  math: [[HBM_IN_CYCLES + GPU.l2Latency + TILE_LOAD_CYCLES, HBM_IN_CYCLES + GPU.l2Latency + TILE_LOAD_CYCLES + MATH_CYCLES]],
};

const cyc = (cycles: number) => Math.round(cycles).toLocaleString("en-US");

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
    short: "Architecture",
    title: "HBM around the die",
    body: `Six HBM sites ring the package. A and B are 256×256 FP32 squares, 256 KB each. One GPU clock drives this scene and the decode scene: ${CYCLES_PER_SECOND} cycles per second of playback.`,
  },
  {
    t: HBM_START,
    id: "fill",
    short: "To L2",
    title: "The squares leave HBM",
    body: `A and B leave together and share HBM's ~${Math.round(GPU.hbmBytesPerCycle).toLocaleString("en-US")} B/cycle. ~${GPU.hbmLatency} cycles of latency, then 512 KB of streaming: ~${cyc(HBM_IN_CYCLES)} cycles in all.`,
  },
  {
    t: CACHED_T,
    id: "cached",
    short: "In cache",
    title: "A fraction of L2",
    body: "Both 256×256 squares sit in L2. Together they are 512 KB, 1.25% of the 40 MB cache. The rest of the slab stays empty.",
  },
  {
    t: TOUR_START,
    id: "sms",
    short: "One SM",
    title: "One SM at a time",
    body: "Clock stopped. Each SM reads a 32×256 panel of A and a 256×32 panel of B. That is 64 KB from L2, and it writes a 32×32 tile.",
  },
  {
    t: RUN_START,
    id: "complete",
    short: "Compute",
    title: "Compute-bound: math dominates",
    body: `Clock running. All 64 SMs start together. After ~${GPU.l2Latency} cycles of L2 latency, each streams 8 K-steps of 8 KB while its ${GPU.fp32Lanes} FP32 lanes do 262,144 FMAs: ${cyc(MATH_CYCLES)} cycles of math, with loads hidden underneath.`,
  },
  {
    t: WRITE_START,
    id: "writeback",
    short: "Write back",
    title: "Eight by eight becomes one matrix",
    body: `Clock stopped while the 64 green tiles lock into one 256×256 matrix; on the GPU they are just stores to adjacent addresses. Then 256 KB returns to HBM: ~${cyc(HBM_OUT_CYCLES)} cycles.`,
  },
  {
    t: OUT_END + 0.4,
    id: "bound",
    short: "Compute-bound",
    title: `${cyc(CLOCK.total)} cycles, mostly math`,
    body: `HBM in ${cyc(HBM_IN_CYCLES)} · SMs ${cyc(SM_CYCLES)} (${cyc(MATH_CYCLES)} of FMAs) · HBM out ${cyc(HBM_OUT_CYCLES)}. 33.6 MFLOP over 768 KB is ≈ 43 FLOP/byte. Decode, on the same clock, does 1/256 of the math in about half the time.`,
  },
];

export interface MatrixLook {
  opacity: number;
  fill: number;
  scan: number;
  region: [number, number, number, number] | null;
  regionStrength: number;
  panelEdges: number;
  /** Eight row or column amplitudes. Axis 1 is rows (along Z). Axis 2 is columns (along X). */
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
  /** `amount` pulses per SM. `tour` stays up across the gaps between them. */
  focus: { row: number; col: number; amount: number; tour: number };
  /** Shared GPU clock. Stopped during narration. */
  gpu: { cycles: number; running: boolean; label: string };
  /** 0 = green tiles sit on their SMs. 1 = they occupy the 8×8 matrix. */
  assemble: number;
  tileOpacity: number;
  blocks: BlockVisual[];
  callouts: {
    workingSet: number;
    stored: number;
  };
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

function linear(t: number, a: number, b: number): number {
  return clamp01((t - a) / (b - a));
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
  for (const phase of PHASES) {
    if (t >= phase.t) current = phase;
  }
  return current;
}

function cameraAt(t: number): { position: Vec3; target: Vec3 } {
  const cModule = hbmMatrixCenter(hbmModule("C"));
  const keys: CameraKey[] = [
    { t: 0, position: [0, 25.5, 14.5], target: [0, 0.25, 0] },
    { t: HBM_START, position: [0, 22.0, 10.5], target: [0, 0.65, 0] },
    { t: HBM_START + 2.1, position: [0, 15.8, 5.7], target: [0, 0.82, 0] },
    { t: CACHED_T + 0.2, position: [0.08, 15.5, 4.4], target: [0, 0.68, 0] },
    { t: TOUR_START, position: [0.04, 15.2, 4.6], target: [0, 0.58, 0] },
    { t: TOUR_END, position: [0.02, 15.4, 4.8], target: [0, 0.55, 0] },
    { t: WRITE_START, position: [0.15, 12.2, 8.0], target: [0, 1.35, 0] },
    { t: WRITE_START + 1.15, position: [0.35, 10.4, 6.6], target: [0, 1.5, 0] },
    { t: WRITE_START + 2.15, position: [5.4, 11.2, 7.2], target: [cModule[0], 0.82, cModule[2]] },
    { t: DURATION, position: [7.4, 10.4, 7.0], target: [cModule[0], 0.65, cModule[2]] },
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
  /** Pulse for this SM: ribbons and the SM highlight. Matrix bands stay held. */
  amount: number;
  /** Stays up across the gaps so the rest of the chip stays dim. */
  tour: number;
  local: number;
}

function stepAmount(local: number): number {
  const fadeIn = smooth(local / 0.12);
  const fadeOut = 1 - smooth((local - 0.82) / 0.14);
  return fadeIn * fadeOut;
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
  const prev = index > 0 ? SM_TOUR[index - 1] : null;
  const model = getModel();
  const aShape = `${model.tileM}×${model.n}`;
  const bShape = `${model.n}×${model.tileN}`;
  const cShape = `${model.tileM}×${model.tileN}`;
  const panels = `A ${aShape} · ${kb(aPanelBytes())} and B ${bShape} · ${kb(bPanelBytes())}`;
  const cache = `Read ${cacheUsed(aPanelBytes() + bPanelBytes())}. A+B resident: ${cacheUsed(residentBytes())}. C tile ${cShape} · ${kb(cTileBytes())}.`;
  const title = `SM (${row}, ${col})`;
  let lead: string;
  if (!prev) lead = `${panels} stream into this SM.`;
  else if (prev.row === row) lead = `Same A panel, ${aShape}. B steps to the next ${bShape} column.`;
  else lead = `The row wraps. New ${aShape} panel of A, and B jumps to column ${col}.`;
  return { title, body: `${lead} ${cache}` };
}

function blockVisual(t: number, row: number, col: number, tour: TourHit | null): BlockVisual {
  const stepIndex = SM_TOUR.findIndex((step) => step.row === row && step.col === col);
  let load = 0;
  let progress = 0;

  if (stepIndex >= 0) {
    // Tour preview, with the clock stopped. It clears before the real run so
    // every SM starts that run from empty.
    const stepStart = TOUR_START + stepIndex * TOUR_STEP;
    load = ramp(t, stepStart + 0.12, stepStart + 0.5) * (1 - ramp(t, TOUR_END - 0.2, RUN_START));
  }

  if (t >= RUN_START) {
    // Constant-rate ramps on the GPU clock: panels stream in K-step by
    // K-step while the FMAs run, and the last tile lands one step before the math ends.
    const launch = HBM_IN_CYCLES + (row + col) * GPU.launchStagger;
    const firstByte = launch + GPU.l2Latency;
    const mathStart = firstByte + TILE_LOAD_CYCLES;
    const mathEnd = mathStart + MATH_CYCLES;
    load = Math.max(load, linear(t, CLOCK.timeOf(RUN, firstByte), CLOCK.timeOf(RUN, mathEnd - STEP_MATH_CYCLES)));
    progress = linear(t, CLOCK.timeOf(RUN, mathStart), CLOCK.timeOf(RUN, mathEnd));
  }

  const isCurrent = tour !== null && tour.row === row && tour.col === col && tour.amount > 0.18;
  const visited = stepIndex >= 0 && progress > 0.15 && !isCurrent;

  let state: SMState = "idle";
  if (t >= RUN_START && progress <= 0) state = "assigned";
  if (load > 0.02 && progress < 0.97) state = progress <= 0 ? "loading" : "computing";
  if (progress >= 0.97) state = "complete";

  let dim = 1;
  if (tour && tour.tour > 0) {
    dim = isCurrent ? 1 : visited ? lerp(1, 0.55, tour.tour) : lerp(1, 0.16, tour.tour);
  }

  return {
    row,
    col,
    state,
    progress,
    load,
    dim,
    hot: isCurrent ? 1 : 0,
  };
}

export function sampleTimeline(time: number): TimelineSample {
  const t = Math.min(DURATION, Math.max(0, time));
  const model = getModel();
  const tour = tourAt(t);
  const travelA = linear(t, HBM_START, HBM_IN_END);
  const travelB = travelA;
  const travelC = linear(t, OUT_START, OUT_END);
  const cMover = 0;
  const lanesA = quietLanes();
  const lanesB = quietLanes();
  if (tour) {
    // Hold the lit row and column across the gap. `amount` still pulses the
    // ribbons and the SM; dropping it here flashed both squares back to full brightness.
    lanesA[tour.row] = tour.tour;
    lanesB[tour.col] = tour.tour;
  }
  const residentA = travelA >= 0.992;
  const residentB = travelB >= 0.992;
  const blocks = model.blocks.map((block) => blockVisual(t, block.row, block.col, tour));
  const avg = blocks.reduce((sum, block) => sum + block.progress, 0) / blocks.length;
  const streamA = ramp(t, HBM_START - 0.05, HBM_START + 0.3) * (1 - ramp(t, HBM_IN_END - 0.15, HBM_IN_END + 0.35));
  const streamB = streamA;
  const returning = ramp(t, OUT_START, OUT_START + 0.35) * (1 - ramp(t, OUT_END - 0.2, OUT_END + 0.25));
  const phase = { ...phaseAt(t) };
  if (tour) {
    const copy = tourCopy(tour.index);
    phase.title = copy.title;
    phase.body = copy.body;
  }

  let cText = "waiting";
  let cBar = 0;
  if (t < TOUR_START) {
    cText = "waiting";
    cBar = 0;
  } else if (tour) {
    cText = `32×32 · SM ${tour.row},${tour.col}`;
    cBar = ((tour.index + tour.local) / SM_TOUR.length) * 0.4;
  } else if (avg < 0.98) {
    cText = "64 × 32×32";
    cBar = 0.4 + avg * 0.6;
  } else if (t < OUT_START - 0.2) {
    cText = "8×8 → one matrix";
    cBar = 1;
  } else if (t < OUT_START) {
    cText = "256×256 · 256 KB";
    cBar = 1;
  } else if (t < OUT_END) {
    cText = "256×256 · SMs → HBM";
    cBar = 1;
  } else {
    cText = "256×256 · in HBM";
    cBar = 1;
  }

  return {
    t,
    phase,
    camera: cameraAt(t),
    hbmA: look({
      opacity: lerp(1, 0.2, Math.min(1, travelA / 0.08)),
      fill: 1,
      panelEdges: 0.35,
    }),
    hbmB: look({
      opacity: lerp(1, 0.2, Math.min(1, travelB / 0.08)),
      fill: 1,
      panelEdges: 0.35,
    }),
    hbmC: look({
      opacity: 0,
      fill: 1,
      panelEdges: 0.4,
    }),
    l2A: residentA
      ? look({
          opacity: 1,
          fill: 1,
          panelEdges: 0.72,
          lanes: lanesA,
          laneAxis: 1,
        })
      : look({
          opacity: 1,
          fill: 1,
          scan: travelA > 0.02 && travelA < 0.98 ? 1 - travelA : -1,
          panelEdges: 0.5,
        }),
    l2B: residentB
      ? look({
          opacity: 1,
          fill: 1,
          panelEdges: 0.72,
          lanes: lanesB,
          laneAxis: 2,
        })
      : look({
          opacity: 1,
          fill: 1,
          scan: travelB > 0.02 && travelB < 0.98 ? 1 - travelB : -1,
          panelEdges: 0.5,
          laneAxis: 2,
        }),
    l2C: look({
      opacity: 1,
      fill: 1,
      panelEdges: 0.7,
    }),
    l2Pulse: pulse(t, HBM_IN_END + 0.1, 0.9),
    channelA: { opacity: streamA, active: streamA },
    channelB: { opacity: streamB, active: streamB },
    returnFlow: { opacity: returning, active: returning },
    travelA,
    travelB,
    travelC,
    cMover,
    windowContent: 1 - ramp(t, WRITE_START + 0.05, WRITE_START + 0.9),
    gpu: gpuAt(t),
    focus: tour
      ? { row: tour.row, col: tour.col, amount: tour.amount, tour: tour.tour }
      : { row: -1, col: -1, amount: 0, tour: 0 },
    assemble: ramp(t, WRITE_START, WRITE_START + 1.15),
    tileOpacity: ramp(t, TOUR_START + 0.25, TOUR_START + 0.7),
    blocks,
    callouts: {
      workingSet: ramp(t, CACHED_T + 0.1, CACHED_T + 0.6) * (1 - ramp(t, TOUR_START - 0.35, TOUR_START + 0.15)),
      stored: ramp(t, OUT_END - 0.3, OUT_END + 0.2),
    },
    meters: {
      a: t < HBM_START ? "256×256 · HBM" : travelA >= 1 ? "256×256 · 256 KB in L2" : "256×256 · HBM → L2",
      b: t < HBM_START ? "256×256 · HBM" : travelB >= 1 ? "256×256 · 256 KB in L2" : "256×256 · HBM → L2",
      c: cText,
      aBar: travelA,
      bBar: travelB,
      cBar,
    },
  };
}

function gpuAt(t: number): TimelineSample["gpu"] {
  const cycles = CLOCK.cycleAt(t);
  const running = CLOCK.running(t);
  let label = formatGpuTime(cycles);
  if (cycles <= 0) label = "kernel not started";
  else if (cycles >= CLOCK.total) label += " · done";
  else if (!running) label += " · paused to explain";
  return { cycles, running, label };
}

/**
 * Drives the scene by GPU cycle alone, skipping narration. Used by the
 * side-by-side comparison, where both scenes share one cycle counter.
 */
export function sampleAtCycle(cycle: number): TimelineSample {
  const sample = sampleTimeline(CLOCK.timeAtCycle(cycle));
  const done = cycle >= CLOCK.total;
  return { ...sample, gpu: { cycles: cycle, running: cycle > 0 && !done, label: formatGpuTime(cycle) } };
}

export function workloadCaption(): string {
  const n = getModel().n;
  const bytes = kb(matrixBytes());
  return `${n}×${n} · FP32 · ${bytes} each · cache ${cacheUsed(residentBytes())}`;
}
