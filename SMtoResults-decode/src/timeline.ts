export const K = 256;
export const TILE_K = 32;
export const CHUNK_COUNT = K / TILE_K;
export const WARP_COUNT = 8;

/** Pedagogical cycle model: slow enough to see issue, latency, and completion separately. */
export const VISUAL_CYCLES_PER_SECOND = 4;
export const LOAD_CYCLES = 2;
export const ISSUE_CYCLES = 2;
export const TENSOR_CYCLES = 8;
export const ACCUMULATE_CYCLES = 2;
export const WARP_SLOT_CYCLES = LOAD_CYCLES + ISSUE_CYCLES;
export const CHUNK_CYCLES = WARP_COUNT * WARP_SLOT_CYCLES + TENSOR_CYCLES + ACCUMULATE_CYCLES;
export const TOTAL_COMPUTE_CYCLES = CHUNK_COUNT * CHUNK_CYCLES;

export const PANEL_START = 3;
export const RESIDENT_START = 7;
export const COMPUTE_START = 10;
export const COMPUTE_END = COMPUTE_START + TOTAL_COMPUTE_CYCLES / VISUAL_CYCLES_PER_SECOND;
export const WRITE_START = COMPUTE_END + 1;
export const ASSEMBLE_START = WRITE_START + 4.5;
export const DURATION = ASSEMBLE_START + 4.5;

export interface Phase {
  t: number;
  id: "hardware" | "panel" | "resident" | "stream" | "accumulate" | "writeback" | "assemble";
  short: string;
  title: string;
  body: string;
}

export const PHASES: Phase[] = [
  {
    t: 0,
    id: "hardware",
    short: "Hardware",
    title: "One SM owns one output slice",
    body: "This SM receives the full 1×256 data row and one 256×32 weight panel. Its only result is y[0, j:j+32].",
  },
  {
    t: PANEL_START,
    id: "panel",
    short: "Panel arrives",
    title: "The weights dominate the working set",
    body: "The blue x row is 1 KB. The orange W panel is 32 KB. The eventual green output is only 128 bytes.",
  },
  {
    t: RESIDENT_START,
    id: "resident",
    short: "8 warps",
    title: "Eight warps divide the 32 outputs",
    body: "Each warp owns four W columns and four y values. All eight reuse the same x chunk, but no two warps update the same output.",
  },
  {
    t: COMPUTE_START,
    id: "stream",
    short: "Stream K",
    title: "Issue is not completion",
    body: "A warp loads and issues, then remains in flight for eight visible Tensor Core cycles. The scheduler can move on while earlier warps are still executing.",
  },
  {
    t: COMPUTE_START + (COMPUTE_END - COMPUTE_START) * 0.52,
    id: "accumulate",
    short: "Accumulate",
    title: "Partial dot products stay inside the SM",
    body: "Each chunk updates the same 32 register accumulators. This is the real reduction: along K, not across the eight SMs.",
  },
  {
    t: WRITE_START,
    id: "writeback",
    short: "Write 128 B",
    title: "K is complete; write one 1×32 slice",
    body: "The 32 final values leave the registers together. No other SM contributes to these output columns.",
  },
  {
    t: ASSEMBLE_START,
    id: "assemble",
    short: "Concatenate",
    title: "Eight disjoint slices form y",
    body: "The active SMs own adjacent output columns. Their eight 1×32 slices concatenate into y[1×256]; they are never added together.",
  },
];

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function smooth(value: number): number {
  const x = clamp01(value);
  return x * x * (3 - 2 * x);
}

export function phaseAt(t: number): Phase {
  let current = PHASES[0];
  for (const phase of PHASES) if (t >= phase.t) current = phase;
  return current;
}

export interface TimelineSample {
  t: number;
  phase: Phase;
  panelProgress: number;
  residentProgress: number;
  kProgress: number;
  chunk: number;
  chunkProgress: number;
  computeCycle: number;
  chunkCycle: number;
  activeWarp: number;
  activeWarpProgress: number;
  returnWarp: number;
  returnProgress: number;
  tensorActive: boolean;
  writeProgress: number;
  assembleProgress: number;
}

export function sampleTimeline(time: number): TimelineSample {
  const t = Math.min(DURATION, Math.max(0, time));
  const computing = t >= COMPUTE_START && t < COMPUTE_END;
  const rawCycle = clamp01((t - COMPUTE_START) / (COMPUTE_END - COMPUTE_START)) * TOTAL_COMPUTE_CYCLES;
  const computeCycle = Math.min(TOTAL_COMPUTE_CYCLES, rawCycle);
  const boundedCycle = Math.min(TOTAL_COMPUTE_CYCLES - Number.EPSILON, computeCycle);
  const chunk = Math.min(CHUNK_COUNT - 1, Math.floor(boundedCycle / CHUNK_CYCLES));
  const chunkCycle = computing ? boundedCycle - chunk * CHUNK_CYCLES : computeCycle >= TOTAL_COMPUTE_CYCLES ? CHUNK_CYCLES : 0;
  const issueWindow = Math.min(WARP_COUNT * WARP_SLOT_CYCLES - Number.EPSILON, chunkCycle);
  const issueWarp = Math.floor(issueWindow / WARP_SLOT_CYCLES);
  const issueLocal = issueWindow - issueWarp * WARP_SLOT_CYCLES;
  const activeWarp = computing && chunkCycle < WARP_COUNT * WARP_SLOT_CYCLES ? issueWarp : -1;
  const activeWarpProgress = activeWarp >= 0 ? smooth(issueLocal / WARP_SLOT_CYCLES) : 0;
  let returnWarp = -1;
  let returnProgress = 0;
  let tensorActive = false;
  for (let warp = 0; warp < WARP_COUNT; warp++) {
    const local = chunkCycle - warp * WARP_SLOT_CYCLES;
    const tensorStart = WARP_SLOT_CYCLES;
    const tensorEnd = tensorStart + TENSOR_CYCLES;
    if (local >= tensorStart && local < tensorEnd) tensorActive = true;
    if (local >= tensorEnd && local < tensorEnd + ACCUMULATE_CYCLES) {
      returnWarp = warp;
      returnProgress = smooth((local - tensorEnd) / ACCUMULATE_CYCLES);
    }
  }
  const chunkProgress = computeCycle >= TOTAL_COMPUTE_CYCLES ? 1 : clamp01(chunkCycle / CHUNK_CYCLES);
  const kProgress = computeCycle >= TOTAL_COMPUTE_CYCLES ? 1 : (chunk + chunkProgress) / CHUNK_COUNT;

  return {
    t,
    phase: phaseAt(t),
    panelProgress: smooth((t - PANEL_START) / 2.5),
    residentProgress: smooth((t - RESIDENT_START) / 1.5),
    kProgress,
    chunk,
    chunkProgress,
    computeCycle,
    chunkCycle,
    activeWarp,
    activeWarpProgress,
    returnWarp,
    returnProgress,
    tensorActive,
    writeProgress: smooth((t - WRITE_START) / 2.4),
    assembleProgress: smooth((t - ASSEMBLE_START) / 2.2),
  };
}

export type WarpStage = "queued" | "load" | "issue" | "tensor" | "accumulate" | "done";

export interface WarpTiming {
  stage: WarpStage;
  progress: number;
  tensorCycle: number;
}

export function warpTiming(sample: TimelineSample, warp: number): WarpTiming {
  if (sample.t < COMPUTE_START) return { stage: "queued", progress: 0, tensorCycle: 0 };
  if (sample.t >= COMPUTE_END) return { stage: "done", progress: 1, tensorCycle: TENSOR_CYCLES };
  const local = sample.chunkCycle - warp * WARP_SLOT_CYCLES;
  if (local < 0) return { stage: "queued", progress: 0, tensorCycle: 0 };
  if (local < LOAD_CYCLES) return { stage: "load", progress: smooth(local / LOAD_CYCLES), tensorCycle: 0 };
  if (local < WARP_SLOT_CYCLES) {
    return { stage: "issue", progress: smooth((local - LOAD_CYCLES) / ISSUE_CYCLES), tensorCycle: 0 };
  }
  if (local < WARP_SLOT_CYCLES + TENSOR_CYCLES) {
    const elapsed = local - WARP_SLOT_CYCLES;
    return {
      stage: "tensor",
      progress: elapsed / TENSOR_CYCLES,
      tensorCycle: Math.min(TENSOR_CYCLES, Math.floor(elapsed) + 1),
    };
  }
  if (local < WARP_SLOT_CYCLES + TENSOR_CYCLES + ACCUMULATE_CYCLES) {
    return {
      stage: "accumulate",
      progress: smooth((local - WARP_SLOT_CYCLES - TENSOR_CYCLES) / ACCUMULATE_CYCLES),
      tensorCycle: TENSOR_CYCLES,
    };
  }
  return { stage: "done", progress: 1, tensorCycle: TENSOR_CYCLES };
}
