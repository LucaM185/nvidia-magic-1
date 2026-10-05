/**
 * Timing rules shared by the two "inside one SM" pages, so prefill and decode
 * run on the same clock and differ only in the math each warp does.
 *
 * Every warp: issue a load from shared memory (1 cycle), wait for it
 * (SHARED_LATENCY), issue the math (1 cycle), execute, and get the result back
 * in its registers (1 cycle). The scheduler starts one warp every
 * WARP_INTERVAL cycles. The timeline shows one K step per warp; the other K
 * steps repeat it into the same registers.
 */

export const CYCLES_PER_SECOND = 3;
export const ISSUE_CYCLES = 1;
export const SHARED_LATENCY = 30;
export const WARP_INTERVAL = 8;
export const FIRST_LOAD = 12;
/** Before FIRST_LOAD: cycle 8 shows the resident warps, cycle 10 the scheduler picking W0. */
export const RESIDENT_CYCLE = 8;
export const SELECT_CYCLE = 10;
export const WRITEBACK_SPACING = 2;
export const WRITEBACK_DURATION = 8;

export interface WarpPlan {
  /** Load issued. */
  load: number;
  /** Operands in registers; the math is issued this cycle. */
  ready: number;
  execStart: number;
  execEnd: number;
  /** Result back in registers. */
  done: number;
}

export function planWarp(warp: number, execCycles: number): WarpPlan {
  const load = FIRST_LOAD + warp * WARP_INTERVAL;
  const ready = load + SHARED_LATENCY;
  const execStart = ready + ISSUE_CYCLES;
  const execEnd = execStart + execCycles;
  return { load, ready, execStart, execEnd, done: execEnd + ISSUE_CYCLES };
}

/** Cycle at which a warp starts a stage that later warps enter WRITEBACK_SPACING cycles apart. */
export function staggered(start: number, warp: number): number {
  return start + warp * WRITEBACK_SPACING;
}

/** The warp the scheduler is issuing for at `cycle`, or -1. `writeStart` is when W0's write back begins. */
export function issuingWarp(cycle: number, plans: WarpPlan[], writeStart: number): number {
  if (cycle >= SELECT_CYCLE && cycle < FIRST_LOAD) return 0;
  for (let warp = 0; warp < plans.length; warp++) {
    const plan = plans[warp];
    for (const issue of [plan.load, plan.ready, plan.execEnd]) {
      if (cycle >= issue && cycle < issue + ISSUE_CYCLES) return warp;
    }
  }
  for (let warp = 0; warp < plans.length; warp++) {
    const start = staggered(writeStart, warp);
    if (cycle >= start && cycle < start + WRITEBACK_SPACING) return warp;
  }
  return -1;
}

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function smooth(value: number): number {
  const x = clamp01(value);
  return x * x * (3 - 2 * x);
}

/** Progress of `cycle` through [start, end), clamped to 0 → 1. */
export function through(cycle: number, start: number, end: number): number {
  return clamp01((cycle - start) / (end - start));
}
