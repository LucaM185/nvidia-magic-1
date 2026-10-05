import { runSmPage, must } from "../../src/sm-page";
import { ISSUE_CYCLES, WRITEBACK_DURATION, issuingWarp, staggered } from "../../src/sm-timing";
import type { Row, RowState, Segment } from "../../src/cycle-timeline";
import { WARP_COLORS } from "../../src/sm-scene";
import {
  DURATION,
  END_K,
  FFMA_CYCLES,
  PHASES,
  PLANS,
  SHUFFLE_CYCLES,
  SHUFFLE_START,
  SHUFFLE_STEPS,
  SHUFFLE_STEP_CYCLES,
  TOTAL_CYCLES,
  WARP_COUNT,
  WRITEBACK_START,
  shuffleStart,
} from "./timeline";
import { createWorld } from "./scene";

const warpHex = (warp: number) => `#${WARP_COLORS[warp].getHexString()}`;

/* Cycle timeline: the scheduler and execution units on top, one row per warp below. */

const schedulerIssues: Segment[] = [];
const lanesBusy: Segment[] = [];
const warpRows: Row[] = PLANS.map((plan, warp) => {
  const writeStart = staggered(WRITEBACK_START, warp);
  for (const start of [plan.load, plan.ready, plan.execEnd]) {
    schedulerIssues.push({ start, end: start + ISSUE_CYCLES, kind: "issue", color: warpHex(warp) });
  }
  lanesBusy.push({ start: plan.execStart, end: plan.execEnd, kind: "compute", color: warpHex(warp) });
  return {
    name: `W${warp}`,
    color: warpHex(warp),
    segments: [
      { start: plan.load, end: plan.load + ISSUE_CYCLES, kind: "issue" },
      { start: plan.load + ISSUE_CYCLES, end: plan.ready, kind: "wait", label: "shared memory" },
      { start: plan.ready, end: plan.execStart, kind: "issue" },
      { start: plan.execStart, end: plan.execEnd, kind: "compute", label: "FFMA" },
      { start: plan.execEnd, end: plan.done, kind: "issue" },
      { start: plan.done, end: shuffleStart(warp), kind: "hold", label: "sums in registers" },
      { start: shuffleStart(warp), end: shuffleStart(warp) + SHUFFLE_CYCLES, kind: "result", label: "shfl" },
      { start: writeStart, end: writeStart + WRITEBACK_DURATION, kind: "store", label: "y" },
    ],
  };
});
lanesBusy.push(
  { start: END_K, end: SHUFFLE_START, kind: "compute", label: "rest of K" },
  { start: SHUFFLE_START, end: shuffleStart(WARP_COUNT - 1) + SHUFFLE_CYCLES, kind: "result", label: "shuffle" },
);

function warpStates(cycle: number): RowState[] {
  const selected = issuingWarp(cycle, PLANS, WRITEBACK_START);
  return PLANS.map((plan, warp) => {
    const writeStart = staggered(WRITEBACK_START, warp);
    const shuffle = shuffleStart(warp);
    let text: string;
    if (cycle < 8) text = "—";
    else if (cycle >= writeStart + WRITEBACK_DURATION) text = "y stored";
    else if (cycle >= writeStart) text = "writing y";
    else if (cycle >= shuffle + SHUFFLE_CYCLES) text = "sum in lane 0";
    else if (cycle >= shuffle) text = `shfl.down ${16 >> Math.floor((cycle - shuffle) / SHUFFLE_STEP_CYCLES)}`;
    else if (cycle >= END_K) text = "rest of K";
    else if (cycle < plan.load) text = "ready";
    else if (cycle < plan.load + ISSUE_CYCLES) text = "issue LDS";
    else if (cycle < plan.ready) text = `wait · ${Math.ceil(plan.ready - cycle)} cyc`;
    else if (cycle < plan.execStart) text = "issue FFMA";
    else if (cycle < plan.execEnd) text = `FFMA · ${Math.ceil(plan.execEnd - cycle)} cyc`;
    else if (cycle < plan.done) text = "sums → registers";
    else text = "sums in registers";
    return { text, active: warp === selected };
  });
}

function hardwareStates(cycle: number): RowState[] {
  const selected = issuingWarp(cycle, PLANS, WRITEBACK_START);
  const ffma = PLANS.filter((plan) => cycle >= plan.execStart && cycle < plan.execEnd).length;
  const lanes = ffma ? `${ffma} warp FFMA` : cycle >= END_K && cycle < SHUFFLE_START ? "rest of K" : cycle >= SHUFFLE_START && cycle < shuffleStart(WARP_COUNT - 1) + SHUFFLE_CYCLES ? "shuffle" : "idle";
  return [
    { text: selected >= 0 ? `→ W${selected}` : "idle", active: selected >= 0 },
    { text: lanes, active: lanes !== "idle" },
    { text: "idle" },
  ];
}

must<HTMLElement>("#math-per-load").textContent = `${FFMA_CYCLES} cycles`;

runSmPage({
  ...createWorld(),
  phases: PHASES,
  duration: DURATION,
  totalCycles: TOTAL_CYCLES,
  timeline: {
    ticks: [0, 20, 40, 60, 80, 100, 120, 140, TOTAL_CYCLES],
    hardware: [
      { name: "Sched", color: "#c8f06a", segments: schedulerIssues },
      { name: "FP32", color: "#c8f06a", segments: lanesBusy },
      { name: "Tensor", color: "#6d7a8c", segments: [], empty: `idle · with M = 1 there is no matrix tile; ${SHUFFLE_STEPS} shuffles finish the sums instead` },
    ],
    warps: warpRows,
  },
  rowStates: (cycle) => [...hardwareStates(cycle), ...warpStates(cycle)],
});
