import { SM_TOUR, hbmMatrixCenter, hbmModule, type Vec3 } from "../layout";
import {
  CYCLES_PER_SECOND,
  GPU,
  createClock,
  cyclesToSeconds,
  formatGpuTime,
  hbmTransferCycles,
  type KernelBudget,
} from "../../../src/gpu-timing";
import { CHUNK_COUNT, CYCLES_AFTER_LAST_CHUNK, FMA_TOTAL, K, SLICE_N } from "../../../SMtoResults-decode/src/timeline";
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

/**
 * Physical steps run on the shared GPU clock (src/gpu-timing.ts), the same
 * one the prefill scene uses. The tour and the concatenation are narration:
 * the clock is stopped while they play.
 */
const HBM_START = 2;
/** x and W share HBM bandwidth. The 1 KB row is pure latency; W adds 256 KB of streaming. */
const X_IN_CYCLES = hbmTransferCycles(dataBytes());
const W_IN_CYCLES = hbmTransferCycles(dataBytes() + weightBytes());
const X_IN_END = HBM_START + cyclesToSeconds(X_IN_CYCLES);
const W_IN_END = HBM_START + cyclesToSeconds(W_IN_CYCLES);
const CACHED_T = W_IN_END + 0.2;

const TOUR_START = CACHED_T + 2;
const TOUR_STEP = 0.9;
const TOUR_END = TOUR_START + SM_TOUR.length * TOUR_STEP;

/**
 * Per-SM budget: L2 latency, a 64 B/cycle stream of x and eight 4 KB W chunks,
 * then what the decode SM scene does after its last chunk lands (load it,
 * multiply it, shuffle, store). The FMAs hide inside the stream.
 */
const X_COPY_START = GPU.l2Latency;
const X_ARRIVAL = X_COPY_START + (K * 4) / GPU.smBytesPerCycle;
const CHUNK_COPY_CYCLES = (K / CHUNK_COUNT) * SLICE_N * 4 / GPU.smBytesPerCycle;
const chunkArrival = (chunk: number) => X_ARRIVAL + (chunk + 1) * CHUNK_COPY_CYCLES;
const END_CYCLE = chunkArrival(CHUNK_COUNT - 1) + CYCLES_AFTER_LAST_CHUNK;
const FMA_CYCLES = FMA_TOTAL / GPU.fp32Lanes;
const SM_CYCLES = (MODEL.activeSmCount - 1) * GPU.launchStagger + END_CYCLE;
const RUN_START = TOUR_END + 0.2;
const RUN = { t: RUN_START, cycle: W_IN_CYCLES, cycles: SM_CYCLES };
const RUN_END = RUN_START + cyclesToSeconds(SM_CYCLES);

export const CONCAT_START = RUN_END + 0.5;
export const CONCAT_DURATION = 6.3;
export const CONCAT_END = CONCAT_START + CONCAT_DURATION;
const WRITE_START = CONCAT_END + 0.4;
const OUT_START = WRITE_START + 1.25;
const HBM_OUT_CYCLES = hbmTransferCycles(outputBytes());
const OUT = {
  t: OUT_START,
  cycle: W_IN_CYCLES + SM_CYCLES,
  cycles: HBM_OUT_CYCLES,
  prelude: { from: CONCAT_START - 0.2, share: 0.4 },
};
const OUT_END = OUT_START + cyclesToSeconds(HBM_OUT_CYCLES);
export const DURATION = OUT_END + 4.5;

const CLOCK = createClock([{ t: HBM_START, cycle: 0, cycles: W_IN_CYCLES }, RUN, OUT]);
/** Cycle budget for the side-by-side comparison, in kernel cycles from 0. */
export const BUDGET: KernelBudget = {
  hbmIn: W_IN_CYCLES,
  sm: SM_CYCLES,
  out: HBM_OUT_CYCLES,
  total: CLOCK.total,
  inBytes: dataBytes() + weightBytes(),
  outBytes: outputBytes(),
  flops: flopCount(),
  activeSms: MODEL.activeSmCount,
  // One FMA burst per W chunk as it lands.
  math: Array.from({ length: CHUNK_COUNT }, (_, chunk): [number, number] => {
    const start = W_IN_CYCLES + chunkArrival(chunk);
    return [start, start + FMA_CYCLES / CHUNK_COUNT];
  }),
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
    short: "Decode",
    title: "One row meets a weight matrix",
    body: `x is only 1×256 (1 KB), while W is 256×256 (256 KB). One GPU clock drives this scene and the prefill scene: ${CYCLES_PER_SECOND} cycles per second of playback.`,
  },
  {
    t: HBM_START,
    id: "fill",
    short: "To L2",
    title: "The weights dominate the transfer",
    body: `x is only latency: it lands after ~${cyc(X_IN_CYCLES)} cycles. W adds 256 KB of streaming on top: ~${cyc(W_IN_CYCLES)} cycles. Almost every byte belongs to W.`,
  },
  {
    t: CACHED_T,
    id: "cached",
    short: "In cache",
    title: "257 KB moved for one row",
    body: "The input activation x and weight matrix W occupy 257 KB together. The multiply performs only 131k FLOPs: about 0.5 FLOP per byte.",
  },
  {
    t: TOUR_START,
    id: "sms",
    short: "Eight SMs",
    title: "Only one strip of SMs has work",
    body: "Clock stopped. Eight SMs split the 256-wide output. Each receives the same 1 KB row and a different 32 KB weight panel; the other 56 SMs stay idle.",
  },
  {
    t: RUN_START,
    id: "compute",
    short: "Waiting on memory",
    title: "Memory movement outweighs math",
    body: `Clock running. Each SM waits ~${GPU.l2Latency} cycles for L2, then streams 33 KB at 64 B/cycle. Its ${GPU.fp32Lanes} FP32 lanes need only ${cyc(FMA_CYCLES)} cycles of FMAs; the SM is done after ~${cyc(END_CYCLE)} cycles, almost all of it waiting.`,
  },
  {
    t: CONCAT_START,
    id: "concatenate",
    short: "Concatenate",
    title: "Place eight output slices side by side",
    body: "Clock stopped. Each SM owns a different 1×32 range of y. The ranges join into one 1×256 row without any cross-SM addition; on the GPU they are just stores to adjacent addresses.",
  },
  {
    t: WRITE_START,
    id: "writeback",
    short: "Output",
    title: "The concatenated row returns to HBM",
    body: `The eight slices form y, a 1×256 row of only 1 KB. Returning it is almost pure latency: ~${cyc(HBM_OUT_CYCLES)} cycles.`,
  },
  {
    t: OUT_END + 0.4,
    id: "bound",
    short: "Memory-bound",
    title: `${cyc(CLOCK.total)} cycles, almost no math`,
    body: `HBM in ${cyc(W_IN_CYCLES)} · SMs ${cyc(SM_CYCLES)} (${cyc(FMA_CYCLES)} of FMAs) · HBM out ${cyc(HBM_OUT_CYCLES)}. ≈0.5 FLOP/byte: memory latency and bandwidth set the time, not peak compute. Prefill does 256× the math in about twice the time.`,
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
  /** Shared GPU clock. Stopped during narration. */
  gpu: { cycles: number; running: boolean; label: string };
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
  concatenate: {
    active: number;
    complete: number;
    progress: number;
    status: string;
  };
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
  for (const phase of PHASES) if (t >= phase.t) current = phase;
  return current;
}

function cameraAt(t: number): { position: Vec3; target: Vec3 } {
  const cModule = hbmMatrixCenter(hbmModule("C"));
  const keys: CameraKey[] = [
    { t: 0, position: [0, 25.5, 14.5], target: [0, 0.25, 0] },
    { t: HBM_START, position: [0, 22, 10.5], target: [0, 0.65, 0] },
    { t: HBM_START + 1.8, position: [0, 15.8, 5.7], target: [0, 0.82, 0] },
    { t: CACHED_T + 0.2, position: [0.08, 15.5, 4.4], target: [0, 0.68, 0] },
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

  // Tour preview, with the clock stopped. It clears before the real run.
  const stepStart = TOUR_START + col * TOUR_STEP;
  let load = ramp(t, stepStart + 0.1, stepStart + 0.42) * (1 - ramp(t, TOUR_END - 0.2, RUN_START));
  let progress = 0;
  if (t >= RUN_START) {
    // Same cycle marks as the decode SM scene, offset by this SM's launch.
    const launch = W_IN_CYCLES + col * GPU.launchStagger;
    const at = (cycle: number) => CLOCK.timeOf(RUN, launch + cycle);
    load = Math.max(load, linear(t, at(X_COPY_START), at(chunkArrival(CHUNK_COUNT - 1))));
    progress = linear(t, at(chunkArrival(0)), at(END_CYCLE));
  }

  const isCurrent = tour !== null && tour.col === col && tour.amount > 0.18;
  let state: SMState = "idle";
  if (t >= TOUR_START) state = "assigned";
  if (load > 0.02 && progress < 0.97) state = progress <= 0 ? "loading" : "computing";
  if (progress >= 0.97) state = "complete";

  let dim = 1;
  if (tour?.tour) dim = isCurrent ? 1 : 0.36;
  return { row, col, state, progress, load, dim, hot: isCurrent ? 1 : 0 };
}

function concatenateAt(t: number): TimelineSample["concatenate"] {
  const progress = clamp01((t - CONCAT_START) / CONCAT_DURATION);
  const active = ramp(t, CONCAT_START - 0.18, CONCAT_START + 0.12) * (1 - ramp(t, CONCAT_END, CONCAT_END + 0.22));
  const complete = ramp(t, CONCAT_END - 0.08, CONCAT_END + 0.18);
  const placed = Math.min(8, Math.floor(progress * 8 + 0.001));
  const status = progress >= 1 ? "8 slices · one contiguous output row" : `${placed}/8 slices placed · no addition`;
  return { active, complete, progress, status };
}

export function sampleTimeline(time: number): TimelineSample {
  const t = Math.min(DURATION, Math.max(0, time));
  const model = getModel();
  const tour = tourAt(t);
  const travelA = linear(t, HBM_START, X_IN_END);
  const travelB = linear(t, HBM_START, W_IN_END);
  const travelC = linear(t, OUT_START, OUT_END);
  const cMover = ramp(t, WRITE_START + 0.65, WRITE_START + 0.95);
  const lanesB = quietLanes();
  if (tour) lanesB[tour.col] = tour.tour;
  const residentA = travelA >= 0.992;
  const residentB = travelB >= 0.992;
  const blocks = model.blocks.map((block) => blockVisual(t, block.row, block.col, tour));
  const activeBlocks = blocks.filter((block) => block.row === MODEL.activeRow);
  const avg = activeBlocks.reduce((sum, block) => sum + block.progress, 0) / activeBlocks.length;
  const streamA = ramp(t, HBM_START - 0.05, HBM_START + 0.3) * (1 - ramp(t, X_IN_END - 0.15, X_IN_END + 0.35));
  const streamB = ramp(t, HBM_START - 0.05, HBM_START + 0.3) * (1 - ramp(t, W_IN_END - 0.15, W_IN_END + 0.35));
  const returning = ramp(t, OUT_START, OUT_START + 0.35) * (1 - ramp(t, OUT_END - 0.2, OUT_END + 0.25));
  const phase = { ...phaseAt(t) };
  if (tour) Object.assign(phase, tourCopy(tour.index));
  const concatenate = concatenateAt(t);

  let cText = "waiting";
  let cBar = 0;
  if (tour) {
    cText = `1×32 · SM ${tour.row},${tour.col}`;
    cBar = ((tour.index + tour.local) / SM_TOUR.length) * 0.45;
  } else if (t >= RUN_START && avg < 0.98) {
    cText = "8 × 1×32";
    cBar = 0.45 + avg * 0.55;
  } else if (avg >= 0.98 && t < CONCAT_START) {
    cText = "8 output slices · ready to concatenate";
    cBar = 0.7;
  } else if (t >= CONCAT_START && t < CONCAT_END + 0.2) {
    cText = concatenate.status;
    cBar = 0.7 + clamp01((t - CONCAT_START) / CONCAT_DURATION) * 0.25;
  } else if (t >= CONCAT_END && t < OUT_START) {
    cText = "1 concatenated output row";
    cBar = 1;
  } else if (t >= OUT_START && t < OUT_END) {
    cText = "1×256 · SMs → HBM";
    cBar = 1;
  } else if (t >= OUT_END) {
    cText = "1×256 · 1 KB in HBM";
    cBar = 1;
  }

  return {
    t,
    phase,
    camera: cameraAt(t),
    gpu: gpuAt(t),
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
    l2Pulse: Math.max(pulse(t, X_IN_END + 0.1, 0.75), pulse(t, W_IN_END + 0.1, 0.9)),
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
    tileOpacity: ramp(t, RUN_START, RUN_START + 0.4),
    concatenate,
    blocks,
    callouts: {
      workingSet: ramp(t, CACHED_T + 0.1, CACHED_T + 0.55) * (1 - ramp(t, TOUR_START - 0.3, TOUR_START + 0.1)),
      stored: ramp(t, OUT_END - 0.25, OUT_END + 0.2),
    },
    meters: {
      a: t < HBM_START ? "x · 1×256 · 1 KB in HBM" : travelA >= 1 ? "x · 1×256 · 1 KB in L2" : "x · 1×256 · HBM → L2",
      b: t < HBM_START ? "W · 256×256 · 256 KB in HBM" : travelB >= 1 ? "W · 256×256 · 256 KB in L2" : "W · 256×256 · HBM → L2",
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
  return `DECODE · 1×256 · FP32 · ${formatBytes(weightBytes())} weights · ${formatBytes(outputBytes())} output · ${flopCount().toLocaleString("en-US")} FLOPs · ${arithmeticIntensity().toFixed(2)} FLOP/B`;
}
