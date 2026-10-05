import { CYCLES_PER_SECOND, RESIDENT_CYCLE, SELECT_CYCLE, FIRST_LOAD, planWarp } from "../../src/sm-timing";
import type { PagePhase } from "../../src/sm-page";

/** Prefill: each warp loads A and B fragments, then one mma.sync runs on its sub-partition's Tensor Core. */
export const WARP_COUNT = 12;
export const TENSOR_DURATION = 25;
export const PLANS = Array.from({ length: WARP_COUNT }, (_, warp) => planWarp(warp, TENSOR_DURATION));
export const END_K = 156;
export const WRITEBACK_START = 160;
export const TOTAL_CYCLES = 190;
export const DURATION = TOTAL_CYCLES / CYCLES_PER_SECOND;

const at = (cycle: number) => cycle / CYCLES_PER_SECOND;

export const PHASES: PagePhase[] = [
  {
    t: 0,
    id: "hardware",
    short: "Hardware",
    title: "Three stops inside the SM",
    body: "Shared memory holds the A and B tiles, the register file holds each warp's operands and accumulators, and four sub-partitions each bring 32 FP32 lanes and a Tensor Core. None of it moves.",
  },
  {
    t: at(RESIDENT_CYCLE),
    id: "resident",
    short: "Warps",
    title: "Twelve resident warps",
    body: "W0–W11 are software contexts, not hardware: three per sub-partition. Each colored band is the slice of the register file one warp owns; every row of the timeline below is one warp.",
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
    short: "ldmatrix",
    title: "W0 loads A and B into its registers",
    body: "Issuing ldmatrix takes one cycle. Then W0 has to wait: shared memory needs 30 cycles to deliver the operands.",
  },
  {
    t: at(20),
    id: "waiting",
    short: "W0 waits",
    title: "W0 waits, so the scheduler moves on to W1",
    body: "The scheduler never sits on a stalled warp. Eight cycles after W0's load it issues W1's, while W0 is still waiting.",
  },
  {
    t: at(42),
    id: "mma",
    short: "mma.sync",
    title: "W0 issues mma.sync",
    body: "At cycle 42 W0's operands are ready. They reach its Tensor Core in one cycle, are processed for 25, and the result returns to the registers in one more.",
  },
  {
    t: at(50),
    id: "inflight",
    short: "In flight",
    title: "Many warps, one set of hardware",
    body: "Warps start 8 cycles apart, so their waits overlap. W0 is still in its Tensor Core when W1 issues its MMA: every warp's latency hides behind another warp's work.",
  },
  {
    t: at(66),
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
    body: "The timeline shows one K step per warp; the others repeat it into the same accumulators. Each warp's C tile is now complete and independent of the others.",
  },
  {
    t: at(WRITEBACK_START),
    id: "writeback",
    short: "Write back",
    title: "C goes back to shared memory",
    body: "The 16×8 fragments leave the register file two cycles apart. Pairs of them rebuild 16×16 tiles of C in shared memory.",
  },
];
