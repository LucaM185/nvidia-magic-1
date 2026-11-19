/**
 * Timing rules shared by the two "inside one SM" pages, so prefill and decode
 * run on the same clock and differ only in the data they copy and the math
 * each warp does.
 *
 * At cycle 0 one thread asks the TMA to copy this K step's operands from L2
 * into shared memory, and every warp parks on an mbarrier. After L2_LATENCY
 * the bytes stream in at L2_BYTES_PER_CYCLE. Once they have all landed, every
 * warp: issues a load from shared memory (1 cycle), waits for it
 * (SHARED_LATENCY), issues the math (1 cycle), executes, and gets the result
 * back in its registers (1 cycle). The scheduler starts one warp every
 * WARP_INTERVAL cycles. The timeline shows one K step per warp; the other K
 * steps repeat it into the same registers.
 */

/** Playback rate once the warps run; the L2 wait and the copy play faster. */
export const CYCLES_PER_SECOND = 3;
export const L2_LATENCY = 250;
export const L2_BYTES_PER_CYCLE = 64;
export const ISSUE_CYCLES = 1;
export const SHARED_LATENCY = 30;
export const WARP_INTERVAL = 8;
/** The scheduler picks W0 this many cycles after the copy lands, and issues its load SELECT_LEAD later. */
export const SELECT_LEAD = 2;
export const WRITEBACK_SPACING = 2;
export const WRITEBACK_DURATION = 8;

export interface Copy {
  /** All bytes are requested at cycle 0; the first one arrives at L2_LATENCY. */
  firstByte: number;
  /** Last byte in shared memory; the mbarrier releases the warps. */
  landed: number;
  /** W0's first load from shared memory. */
  firstLoad: number;
}

export function planCopy(bytes: number): Copy {
  const landed = L2_LATENCY + bytes / L2_BYTES_PER_CYCLE;
  return { firstByte: L2_LATENCY, landed, firstLoad: landed + 2 * SELECT_LEAD };
}

/** Cycle at which the copy has delivered `bytes` (of everything it carries, in order). */
export function arrivalOf(bytes: number): number {
  return L2_LATENCY + bytes / L2_BYTES_PER_CYCLE;
}

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

export function planWarp(warp: number, execCycles: number, firstLoad: number): WarpPlan {
  const load = firstLoad + warp * WARP_INTERVAL;
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
  if (cycle >= plans[0].load - SELECT_LEAD && cycle < plans[0].load) return 0;
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

/**
 * Maps playback time to GPU cycles through segments of [cycle reached, seconds].
 * A segment that does not advance the cycle holds the clock (narration).
 */
export interface Playback {
  cycleAt(t: number): number;
  timeAt(cycle: number): number;
  /** Seconds until the last cycle. */
  end: number;
}

export function createPlayback(segments: Array<[cycleEnd: number, seconds: number]>): Playback {
  const end = segments.reduce((sum, [, seconds]) => sum + seconds, 0);
  return {
    end,
    cycleAt(t) {
      let time = 0;
      let cycle = 0;
      for (const [cycleEnd, seconds] of segments) {
        if (t < time + seconds) return cycle + ((t - time) / seconds) * (cycleEnd - cycle);
        time += seconds;
        cycle = cycleEnd;
      }
      return cycle;
    },
    timeAt(target) {
      let time = 0;
      let cycle = 0;
      for (const [cycleEnd, seconds] of segments) {
        if (cycleEnd > cycle && target <= cycleEnd) return time + Math.max(0, (target - cycle) / (cycleEnd - cycle)) * seconds;
        time += seconds;
        cycle = cycleEnd;
      }
      return end;
    },
  };
}

/** The shared pacing: an intro hold, the L2 wait, the copy, then the warps at CYCLES_PER_SECOND. */
export function smPlayback(copy: Copy, lastCycle: number): Playback {
  return createPlayback([
    [0, 3],
    [copy.firstByte, 5],
    [copy.landed, 4],
    [lastCycle, (lastCycle - copy.landed) / CYCLES_PER_SECOND],
  ]);
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
