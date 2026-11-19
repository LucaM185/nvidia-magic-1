import { L2_BYTES_PER_CYCLE, L2_LATENCY, SHARED_LATENCY, arrivalOf, planCopy, planWarp, smPlayback } from "../../src/sm-timing";
import type { PagePhase } from "../../src/sm-page";

/**
 * Prefill: the TMA copies this K step's A and B tiles from L2, then each warp
 * loads its fragments and runs one mma.sync on its sub-partition's Tensor Core.
 */
export const WARP_COUNT = 12;
export const TENSOR_DURATION = 25;
/** A 32×32 tile of A and one of B, FP32. */
export const TILE_BYTES = 32 * 32 * 4;
export const COPY = planCopy(2 * TILE_BYTES);
export const PLANS = Array.from({ length: WARP_COUNT }, (_, warp) => planWarp(warp, TENSOR_DURATION, COPY.firstLoad));
export const END_K = PLANS[WARP_COUNT - 1].done - 1;
export const WRITEBACK_START = END_K + 4;
export const TOTAL_CYCLES = WRITEBACK_START + 30;
export const PLAYBACK = smPlayback(COPY, TOTAL_CYCLES);
export const DURATION = PLAYBACK.end;

/** Cycle at which the `index`-th of `count` equal cells of a tile has landed; tile 0 is A, tile 1 is B. */
export const cellLanded = (tile: number, index: number, count: number) => arrivalOf(tile * TILE_BYTES + ((index + 1) * TILE_BYTES) / count);

const at = (cycle: number) => PLAYBACK.timeAt(cycle);
const first = PLANS[0];

export const PHASES: PagePhase[] = [
  {
    t: 0,
    id: "hardware",
    short: "Hardware",
    title: "Three stops inside the SM, fed from L2",
    body: "The TMA copies tiles from L2 into shared memory, the register file holds each warp's operands and accumulators, and four sub-partitions each bring 32 FP32 lanes and a Tensor Core.",
  },
  {
    t: 3,
    id: "request",
    short: "TMA copy",
    title: "One copy for twelve warps",
    body: `Twelve warps are resident, three per sub-partition. One thread asks the TMA to copy this K step's A and B tiles (${(2 * TILE_BYTES) / 1024} KB) from L2, and every warp parks on an mbarrier.`,
  },
  {
    t: at(L2_LATENCY * 0.3),
    id: "latency",
    short: "L2 wait",
    title: `~${L2_LATENCY} cycles before the first byte`,
    body: "The request crosses the chip to L2 and back. No warp has anything to run: the whole SM waits.",
  },
  {
    t: at(L2_LATENCY),
    id: "land",
    short: "Tiles land",
    title: "A and B stream into shared memory",
    body: `${(2 * TILE_BYTES) / 1024} KB at ${L2_BYTES_PER_CYCLE} B/cycle: ${COPY.landed - L2_LATENCY} cycles. When the last byte is in, the mbarrier releases the warps.`,
  },
  {
    t: at(first.load - 2),
    id: "selected",
    short: "Pick W0",
    title: "The scheduler picks a ready warp",
    body: "W0 is eligible. Its token lights up in the scheduler, and its color marks the path through the hardware it is about to use.",
  },
  {
    t: at(first.load),
    id: "load",
    short: "ldmatrix",
    title: "W0 loads A and B into its registers",
    body: `Issuing ldmatrix takes one cycle. Then W0 has to wait: shared memory needs ${SHARED_LATENCY} cycles to deliver the operands.`,
  },
  {
    t: at(first.load + 8),
    id: "waiting",
    short: "W0 waits",
    title: "W0 waits, so the scheduler moves on to W1",
    body: "The scheduler never sits on a stalled warp. Eight cycles after W0's load it issues W1's, while W0 is still waiting.",
  },
  {
    t: at(first.ready),
    id: "mma",
    short: "mma.sync",
    title: "W0 issues mma.sync",
    body: "W0's operands are ready. They reach its Tensor Core in one cycle, are processed for 25, and the result returns to the registers in one more.",
  },
  {
    t: at(first.ready + 8),
    id: "inflight",
    short: "In flight",
    title: "Many warps, one set of hardware",
    body: "Warps start 8 cycles apart, so their waits overlap. W0 is still in its Tensor Core when W1 issues its MMA: every warp's latency hides behind another warp's work.",
  },
  {
    t: at(first.load + 54),
    id: "accumulators",
    short: "Accumulate",
    title: "Partial results stay in registers",
    body: "Each MMA updates the warp's C accumulators in the register file. C does not go back to shared memory after every step.",
  },
  {
    t: at(END_K),
    id: "end-k",
    short: "End of K",
    title: "After K, the registers hold the final C",
    body: "The timeline shows one K step per warp; the others repeat it, each with its own copy from L2, into the same accumulators. Each warp's C tile is now complete.",
  },
  {
    t: at(WRITEBACK_START),
    id: "writeback",
    short: "Write back",
    title: "C goes back to shared memory",
    body: "The 16×8 fragments leave the register file two cycles apart. Pairs of them rebuild 16×16 tiles of C in shared memory.",
  },
];
