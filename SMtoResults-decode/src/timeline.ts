import {
  CYCLES_PER_SECOND,
  FIRST_LOAD,
  RESIDENT_CYCLE,
  SELECT_CYCLE,
  SHARED_LATENCY,
  WRITEBACK_DURATION,
  planWarp,
  staggered,
} from "../../src/sm-timing";
import type { PagePhase } from "../../src/sm-page";

/**
 * One decode SM computing y[j:j+32] = x[1×256] · W[256×32] in FP32, on the
 * same timing rules as the prefill page (src/sm-timing.ts).
 *
 * With M = 1 there is no matrix tile for a Tensor Core, so each K step is
 * four FFMAs on the FP32 lanes: every lane multiplies its x[k] by four W
 * columns into four running sums. After K, warp shuffles fold the 32 lane
 * partials into one value per column.
 */

export const K = 256;
export const SLICE_N = 32;
export const CHUNK_COUNT = 8;
export const WARP_COUNT = 8;
export const COLS_PER_WARP = SLICE_N / WARP_COUNT;
export const FMA_TOTAL = K * SLICE_N;

/** 4 FFMAs issued back to back plus their latency on the FP32 lanes. */
export const FFMA_CYCLES = 6;
export const PLANS = Array.from({ length: WARP_COUNT }, (_, warp) => planWarp(warp, FFMA_CYCLES));
export const END_K = PLANS[WARP_COUNT - 1].done + 2;
/** The other K steps repeat the one shown; they play out quickly between END_K and the shuffle. */
export const SHUFFLE_START = END_K + 12;
export const SHUFFLE_STEPS = 5;
export const SHUFFLE_STEP_CYCLES = 6;
export const SHUFFLE_CYCLES = SHUFFLE_STEPS * SHUFFLE_STEP_CYCLES;
export const WRITEBACK_START = SHUFFLE_START + SHUFFLE_CYCLES;
export const TOTAL_CYCLES = staggered(WRITEBACK_START, WARP_COUNT - 1) + WRITEBACK_DURATION;

const KERNEL_T = TOTAL_CYCLES / CYCLES_PER_SECOND;
export const ASSEMBLE_T = KERNEL_T + 1;
export const DURATION = ASSEMBLE_T + 5;

const at = (cycle: number) => cycle / CYCLES_PER_SECOND;

export const PHASES: PagePhase[] = [
  {
    t: 0,
    id: "hardware",
    short: "The SM",
    title: "Same SM, same three stops as prefill",
    body: "Shared memory, the register file, then four sub-partitions with 32 FP32 lanes and a Tensor Core each. As on the prefill page, x and the 256×32 slice of W are already in shared memory.",
  },
  {
    t: at(RESIDENT_CYCLE),
    id: "resident",
    short: "Warps",
    title: "8 warps, each owning 4 output columns",
    body: "Warp w owns y[4w : 4w+4]: column w of the W panel, band w of the register file and cell w of y. Every row of the timeline below is one warp.",
  },
  {
    t: at(SELECT_CYCLE),
    id: "selected",
    short: "Pick W0",
    title: "The scheduler picks a ready warp",
    body: "W0 is eligible. Its token lights up in the scheduler, and its color marks the path through the hardware it is about to use.",
  },
  {
    t: at(FIRST_LOAD),
    id: "load",
    short: "LDS",
    title: "W0 loads x and W into its registers",
    body: `Issuing the load takes one cycle. Then W0 waits ${SHARED_LATENCY} cycles for shared memory, exactly as on the prefill page.`,
  },
  {
    t: at(20),
    id: "waiting",
    short: "W0 waits",
    title: "W0 waits, so the scheduler moves on to W1",
    body: "The scheduler never sits on a stalled warp. Eight cycles after W0's load it issues W1's, while W0 is still waiting.",
  },
  {
    t: at(PLANS[0].ready),
    id: "ffma",
    short: "FFMA",
    title: "W0 issues 4 FFMAs, not an MMA",
    body: `With M = 1 there is no tile for a Tensor Core. W0's lanes multiply x[k] by four W columns: ${FFMA_CYCLES} cycles of math for a ${SHARED_LATENCY}-cycle wait, where prefill's MMA keeps a Tensor Core busy for 25.`,
  },
  {
    t: at(PLANS[1].ready + 4),
    id: "gaps",
    short: "Gaps",
    title: "Too little math to hide the waits",
    body: `Warps still start 8 cycles apart, but each brings only ${FFMA_CYCLES} cycles of work. Between them the FP32 lanes sit idle: decode is short on math per byte loaded, not on hardware.`,
  },
  {
    t: at(66),
    id: "accumulators",
    short: "Accumulate",
    title: "Running sums stay in registers",
    body: "Each lane keeps four sums, one per column. They do not go back to shared memory between K steps.",
  },
  {
    t: at(END_K),
    id: "end-k",
    short: "End of K",
    title: "After K, each lane holds 4 partial sums",
    body: `The timeline shows one K step per warp; the other ${CHUNK_COUNT - 1} repeat it into the same registers, one W chunk each. Each lane's sums now cover 8 of the 256 k values.`,
  },
  {
    t: at(SHUFFLE_START),
    id: "shuffle",
    short: "Shuffle",
    title: "Warp shuffles finish each dot product",
    body: "Five shfl.down steps (16, 8, 4, 2, 1) fold 32 partials into lane 0, register to register. This is work prefill never needs: its MMA already sums across k.",
  },
  {
    t: at(WRITEBACK_START),
    id: "store",
    short: "Store 128 B",
    title: "Write one 128-byte slice",
    body: "Lane 0 of each warp stores its four floats, two cycles apart: 8 warps × 16 B = y[j:j+32]. No other SM touches these columns.",
  },
  {
    t: ASSEMBLE_T,
    id: "assemble",
    short: "Concatenate",
    title: "Eight disjoint slices form y",
    body: "The active SMs' 1×32 slices concatenate into y[1×256]. The math here is short; streaming W in from L2 to get it here takes far longer (Memory → SMs). That is why decode is memory-bound.",
  },
];

/** Shuffle window for one warp. */
export const shuffleStart = (warp: number) => staggered(SHUFFLE_START, warp);

/** Bytes this SM's W slice and x take to stream in from L2 (Memory → SMs page). */
export const OPERAND_BYTES = (K + K * SLICE_N) * 4;

/**
 * When W streams in chunk by chunk (the Memory → SMs page), earlier chunks are
 * multiplied while later ones are in flight. After the last chunk lands, the
 * SM still loads it, multiplies it, shuffles and stores.
 */
export const CYCLES_AFTER_LAST_CHUNK = SHARED_LATENCY + 2 + FFMA_CYCLES + SHUFFLE_CYCLES + WRITEBACK_DURATION;
