import {
  CYCLES_PER_SECOND,
  ISSUE_CYCLES,
  L2_BYTES_PER_CYCLE,
  L2_LATENCY,
  SELECT_LEAD,
  SHARED_LATENCY,
  WRITEBACK_DURATION,
  arrivalOf,
  createPlayback,
  planCopy,
  planWarp,
  type WarpPlan,
} from "../../src/sm-timing";
import type { PagePhase } from "../../src/sm-page";

/**
 * One decode SM computing y[j:j+32] = x[1×256] · W[256×32] in FP32, on the
 * same timing rules as the prefill page (src/sm-timing.ts).
 *
 * At cycle 0 the TMA is asked for everything: all of x, then W in eight
 * 32×32 chunks, streamed back to back from L2 exactly as on the Memory → SMs
 * page. Each chunk has its own mbarrier, so the warps run K step k as soon as
 * chunk k lands, while later chunks are still in flight.
 *
 * With M = 1 there is no matrix tile for a Tensor Core, so each K step is
 * four FFMAs on the FP32 lanes: every lane multiplies its x[k] by four W
 * columns into four running sums held in registers. After K, warp shuffles
 * move partial sums between lanes' registers and FADDs on the lanes add them,
 * until lane 0 holds each column's total.
 */

export const K = 256;
export const SLICE_N = 32;
export const CHUNK_COUNT = 8;
export const WARP_COUNT = 8;
export const COLS_PER_WARP = SLICE_N / WARP_COUNT;
export const FMA_TOTAL = K * SLICE_N;

/** All of x arrives first, then one 32×32 W chunk per K step, FP32. */
export const X_BYTES = K * 4;
export const W_CHUNK_BYTES = (K / CHUNK_COUNT) * SLICE_N * 4;
export const CHUNK_CYCLES = W_CHUNK_BYTES / L2_BYTES_PER_CYCLE;
/** Cycle at which W chunk `chunk` starts arriving, and has fully landed. */
export const chunkStart = (chunk: number) => arrivalOf(X_BYTES + chunk * W_CHUNK_BYTES);
export const chunkLanded = (chunk: number) => arrivalOf(X_BYTES + (chunk + 1) * W_CHUNK_BYTES);
/** The first K step's copy: x and W chunk 0. */
export const COPY = planCopy(X_BYTES + W_CHUNK_BYTES);

/** 4 FFMAs issued back to back plus their latency on the FP32 lanes. */
export const FFMA_CYCLES = 6;
/** STEPS[k][w]: warp w's load and FFMAs for K step k, once chunk k has landed. */
export const STEPS: WarpPlan[][] = Array.from({ length: CHUNK_COUNT }, (_, chunk) =>
  Array.from({ length: WARP_COUNT }, (_, warp) => planWarp(warp, FFMA_CYCLES, chunkLanded(chunk) + 2 * SELECT_LEAD)),
);
/** K step 0, which the page narrates in detail. */
export const PLANS = STEPS[0];
const LAST = STEPS[CHUNK_COUNT - 1];

/** Each warp reduces as soon as its own last FFMAs are back, then stores. */
export const SHUFFLE_STEPS = 5;
export const SHUFFLE_STEP_CYCLES = 6;
export const SHUFFLE_CYCLES = SHUFFLE_STEPS * SHUFFLE_STEP_CYCLES;
export const shuffleStart = (warp: number) => LAST[warp].done;
export const storeStart = (warp: number) => shuffleStart(warp) + SHUFFLE_CYCLES;
export const TOTAL_CYCLES = storeStart(WARP_COUNT - 1) + WRITEBACK_DURATION;

/** Within each shuffle step: the shfl moves a partial between registers, then the lanes FADD it in. */
export const FADD_WINDOW = [0.4, 0.7] as const;

/** The warp the scheduler is issuing for at `cycle`, or -1. */
export const SELECT_CYCLE = PLANS[0].load - SELECT_LEAD;
export function issuingWarp(cycle: number): number {
  if (cycle >= SELECT_CYCLE && cycle < PLANS[0].load) return 0;
  const issuing = (start: number) => cycle >= start && cycle < start + ISSUE_CYCLES;
  for (const step of STEPS) {
    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const plan = step[warp];
      if (issuing(plan.load) || issuing(plan.ready) || issuing(plan.execEnd)) return warp;
    }
  }
  for (let warp = 0; warp < WARP_COUNT; warp++) {
    for (let s = 0; s < SHUFFLE_STEPS; s++) if (issuing(shuffleStart(warp) + s * SHUFFLE_STEP_CYCLES)) return warp;
    if (issuing(storeStart(warp))) return warp;
  }
  return -1;
}

/** The K step a warp is on or last finished at `cycle`, or -1 before its first load. */
export function stepOf(warp: number, cycle: number): number {
  let current = -1;
  for (let chunk = 0; chunk < CHUNK_COUNT; chunk++) if (cycle >= STEPS[chunk][warp].load) current = chunk;
  return current;
}

/*
 * Pacing: K step 0 (and the start of step 1) in slow motion with holds on
 * warp 0's short moments, steps 2–7 fast-forwarded, then the reduction and
 * stores at a readable speed. The timeline's axis squeezes the same stretches.
 */
export const DETAIL_END = STEPS[1][0].done;
const FFMA_HOLD = PLANS[0].execStart + FFMA_CYCLES / 2;
const SHUFFLE_HOLD = shuffleStart(0) + 1.5 * SHUFFLE_STEP_CYCLES;
const FAST = CYCLES_PER_SECOND * 10;
const TAIL = CYCLES_PER_SECOND * 2;

const segments: Array<[number, number]> = [[0, 3], [COPY.firstByte, 5], [COPY.landed, 4]];
let cursor = COPY.landed;
const run = (to: number, cyclesPerSecond: number) => {
  segments.push([to, (to - cursor) / cyclesPerSecond]);
  cursor = to;
};
const hold = (seconds: number) => segments.push([cursor, seconds]);
run(SELECT_CYCLE, CYCLES_PER_SECOND);
hold(2);
run(FFMA_HOLD, CYCLES_PER_SECOND);
hold(3.5);
run(DETAIL_END, CYCLES_PER_SECOND);
run(chunkLanded(CHUNK_COUNT - 1), FAST);
run(SHUFFLE_HOLD, TAIL);
hold(3);
run(TOTAL_CYCLES, TAIL);

export const PLAYBACK = createPlayback(segments);
export const ASSEMBLE_T = PLAYBACK.end + 1;
export const DURATION = ASSEMBLE_T + 5;

/** Timeline axis: the L2 wait, K step 0 in detail, the fast-forward, then the tail. */
export const SQUEEZE: Array<[number, number]> = [
  [chunkStart(0), 0.12],
  [DETAIL_END, 0.56],
  [chunkLanded(CHUNK_COUNT - 1), 0.7],
];

const at = (cycle: number) => PLAYBACK.timeAt(cycle);
const first = PLANS[0];
/** FP32 lanes per sub-partition: busy for 2 warps × FFMA_CYCLES of every chunk. */
const LANES_BUSY = (WARP_COUNT / 4) * FFMA_CYCLES;

/** Warps a step is about; the scene and timeline fade the others. Steps not listed show every warp. */
export const FOCUS: Record<string, number[]> = {
  selected: [0],
  load: [0],
  waiting: [0, 1],
  ffma: [0],
  shuffle: [0],
};

export const PHASES: PagePhase[] = [
  {
    t: 0,
    id: "hardware",
    short: "The SM",
    title: "Same SM, same three stops as prefill",
    body: "Shared memory is drawn as the product itself: x runs down the left, 32 values per cell, beside the rows of W it multiplies; y sits under the columns it sums. Then the register file, and four sub-partitions of 32 FP32 lanes and a Tensor Core.",
  },
  {
    t: 3,
    id: "request",
    short: "TMA copy",
    title: "One request for the whole slice",
    body: `One thread asks the TMA for all of x, then W in ${CHUNK_COUNT} chunks of 32 rows (${W_CHUNK_BYTES / 1024} KB each), each chunk with its own mbarrier. Warp w owns 4 output columns, n = 4w … 4w+3: one colored column of W, one band of registers, one cell of y.`,
  },
  {
    t: at(L2_LATENCY * 0.3),
    id: "latency",
    short: "L2 wait",
    title: `~${L2_LATENCY} cycles before the first byte`,
    body: "The request crosses the chip to L2 and back. No warp has anything to run: the whole SM waits, exactly as in prefill.",
  },
  {
    t: at(L2_LATENCY),
    id: "land",
    short: "Chunk 0",
    title: "x, then W chunk 0, stream into shared memory",
    body: `${L2_BYTES_PER_CYCLE} B per cycle: x in ${X_BYTES / L2_BYTES_PER_CYCLE} cycles, then ${CHUNK_CYCLES} cycles per W chunk. When chunk 0 is in, its mbarrier releases the warps; chunk 1 keeps streaming in behind it.`,
  },
  {
    t: at(SELECT_CYCLE),
    id: "selected",
    short: "Pick warp 0",
    title: "The scheduler picks a ready warp",
    body: "Chunk 0 is in, so warp 0 is eligible. Its token lights up in the scheduler, and its color marks the hardware it is about to use.",
  },
  {
    t: at(first.load),
    id: "load",
    short: "LDS",
    title: "Warp 0 loads its block of chunk 0",
    body: `Each of its 32 lanes takes one row k of the chunk: x[k] and W[k, 0…3]. Issuing takes one cycle, then ${SHARED_LATENCY} cycles until shared memory answers.`,
  },
  {
    t: at(first.load + 8),
    id: "waiting",
    short: "Warp 0 waits",
    title: "Warp 0 waits, so the scheduler moves on",
    body: "The scheduler never sits on a stalled warp. Eight cycles after warp 0's load it issues warp 1's (columns 4…7), while warp 0 is still waiting.",
  },
  {
    t: at(first.ready),
    id: "ffma",
    short: "FFMA",
    title: "Warp 0 issues 4 FFMAs, not an MMA",
    body: `Lane k multiplies x[k] by W[k, n] for its 4 columns and adds each product into a running sum: ${FFMA_CYCLES} cycles of math for a ${SHARED_LATENCY}-cycle wait. With M = 1 there is no tile for a Tensor Core.`,
  },
  {
    t: at(PLANS[1].ready + 4),
    id: "gaps",
    short: "Gaps",
    title: "Too little math to hide the waits",
    body: `Warps still start 8 cycles apart, but each brings only ${FFMA_CYCLES} cycles of work. Between them the FP32 lanes sit idle: decode is short on math per byte loaded, not on hardware.`,
  },
  {
    t: at(chunkLanded(1)),
    id: "next",
    short: "Chunk 1",
    title: `Chunk 1 lands ${CHUNK_CYCLES} cycles after chunk 0`,
    body: "Each warp repeats the step on rows 32…63, adding into the same 4 sums per lane. The sums never leave the registers between chunks.",
  },
  {
    t: at(DETAIL_END),
    id: "stream",
    short: "Chunks 2–7",
    title: "The warps keep pace with L2, and wait for it",
    body: `Chunks 2–7 at 10× speed. A new chunk lands every ${CHUNK_CYCLES} cycles; each sub-partition's lanes are busy for ${LANES_BUSY} of them. The rest of the time every warp is waiting on the next chunk.`,
  },
  {
    t: at(chunkLanded(CHUNK_COUNT - 1)),
    id: "end-k",
    short: "End of K",
    title: "Each warp has 32 × 4 partial sums",
    body: "One register-file row is one warp, not one output. Its 32 lanes each hold 4 FP32 partial sums: 128 values total. Each lane summed 8 of the 256 input rows, so the 32 lane results still have to be combined.",
  },
  {
    t: at(shuffleStart(0)),
    id: "shuffle",
    short: "Shuffle",
    title: "32 lanes become 4 output values",
    body: "Why this step exists: each of the 4 outputs is still split into 32 lane-sized pieces. Five add steps combine 32 → 16 → 8 → 4 → 2 → 1. At the end, lane 0 holds this warp's 4 FP32 outputs. Eight warps × 4 outputs = 32 outputs for the SM.",
  },
  {
    t: at(storeStart(0)),
    id: "store",
    short: "Store",
    title: "Write y, 4 values per warp",
    body: "Lane 0 of each warp stores its 4 totals as soon as its shuffles finish: 8 warps × 4 = 32 values, 128 B. No other SM touches these columns.",
  },
  {
    t: ASSEMBLE_T,
    id: "assemble",
    short: "Concat",
    title: "Eight disjoint slices form y",
    body: "Nothing is gathered in registers: each SM writes its own 128 B slice to a disjoint range of y[1×256] in global memory, so the eight slices simply land side by side. The math here is short; streaming W in from L2 takes far longer, and the warps spend most of it waiting. That is why decode is memory-bound.",
  },
];

/** Bytes this SM's W slice and x take to stream in from L2 (Memory → SMs page). */
export const OPERAND_BYTES = (K + K * SLICE_N) * 4;

/**
 * The Memory → SMs page streams the same chunks; after the last one lands,
 * the warps still load it, multiply it, shuffle and store.
 */
export const CYCLES_AFTER_LAST_CHUNK = TOTAL_CYCLES - chunkLanded(CHUNK_COUNT - 1);
