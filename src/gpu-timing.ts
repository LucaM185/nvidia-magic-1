/**
 * One GPU clock shared by the Memory → SMs scenes, so prefill and decode can
 * be compared second for second.
 *
 * Every physical step (HBM → L2, L2 → SM, math, write-back) is timed in core
 * cycles from the figures below and played at CYCLES_PER_SECOND. Narration
 * (the per-SM tour, tile assembly, captions) happens with the clock stopped,
 * and the HUD says so.
 *
 * H100 PCIe-class figures, matching the SM scenes: ~1.75 GHz, ~2 TB/s HBM,
 * ~5 TB/s aggregate L2, ~250 cycles L2 latency, 64 B/cycle L2 → SM, 128 FP32
 * lanes per SM. Transfers cost latency + bytes / bandwidth.
 */
export const GPU = {
  clockGHz: 1.755,
  hbmLatency: 500,
  hbmBytesPerCycle: 2.0e12 / 1.755e9,
  l2Latency: 250,
  l2BytesPerCycle: 5.0e12 / 1.755e9,
  smBytesPerCycle: 64,
  fp32Lanes: 128,
  /** CTA launch spread between neighbouring SMs. */
  launchStagger: 4,
} as const;

/** Playback rate for every running segment, in both scenes. */
export const CYCLES_PER_SECOND = 300;

export function hbmTransferCycles(bytes: number): number {
  return GPU.hbmLatency + bytes / GPU.hbmBytesPerCycle;
}

export function cyclesToSeconds(cycles: number): number {
  return cycles / CYCLES_PER_SECOND;
}

export function formatGpuTime(cycles: number): string {
  const ns = cycles / GPU.clockGHz;
  const time = ns >= 1000 ? `${(ns / 1000).toFixed(2)} µs` : `${Math.round(ns)} ns`;
  return `${Math.round(cycles).toLocaleString("en-US")} cycles · ${time}`;
}

/** A kernel's cycle budget, in cycles from launch. Used to compare scenes. */
export interface KernelBudget {
  hbmIn: number;
  sm: number;
  out: number;
  total: number;
  inBytes: number;
  outBytes: number;
  flops: number;
  activeSms: number;
  /** Windows when the first SM's FP32 lanes are doing FMAs. */
  math: [number, number][];
}

export interface ClockRun {
  /** Animation time the clock starts running. */
  t: number;
  /** GPU cycle at that moment. */
  cycle: number;
  /** Cycles played before the clock stops again. */
  cycles: number;
  /**
   * Narration just before this run, from animation time `from` up to `t`.
   * When a scene is driven by cycles alone, the first `share` of the run's
   * cycles replays it, so the visuals it holds are not skipped.
   */
  prelude?: { from: number; share: number };
}

export interface GpuClock {
  /** Animation time at which `cycle` is reached inside `run`. */
  timeOf(run: ClockRun, cycle: number): number;
  cycleAt(t: number): number;
  /** Animation time to show for `cycle` when narration is skipped. */
  timeAtCycle(cycle: number): number;
  running(t: number): boolean;
  total: number;
}

export function createClock(runs: ClockRun[]): GpuClock {
  const end = (run: ClockRun) => run.t + cyclesToSeconds(run.cycles);
  const last = runs[runs.length - 1];
  return {
    timeOf: (run, cycle) => run.t + cyclesToSeconds(cycle - run.cycle),
    cycleAt(t) {
      let cycle = 0;
      for (const run of runs) {
        if (t <= run.t) break;
        cycle = run.cycle + Math.min(run.cycles, (t - run.t) * CYCLES_PER_SECOND);
      }
      return cycle;
    },
    timeAtCycle(cycle) {
      for (const run of runs) {
        if (cycle > run.cycle + run.cycles && run !== last) continue;
        const u = Math.min(1, Math.max(0, (cycle - run.cycle) / run.cycles));
        const share = run.prelude?.share ?? 0;
        if (run.prelude && u < share) return run.prelude.from + (u / share) * (run.t - run.prelude.from);
        return run.t + ((u - share) / (1 - share)) * cyclesToSeconds(run.cycles);
      }
      return runs[0].t;
    },
    running: (t) => runs.some((run) => t > run.t && t < end(run)),
    total: last.cycle + last.cycles,
  };
}
