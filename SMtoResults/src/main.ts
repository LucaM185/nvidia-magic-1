import { runSmPage, must } from "../../src/sm-page";
import { ISSUE_CYCLES, WRITEBACK_DURATION, issuingWarp, staggered } from "../../src/sm-timing";
import type { Row, RowState, Segment } from "../../src/cycle-timeline";
import { WARP_COLORS } from "../../src/sm-scene";
import { DURATION, PHASES, PLANS, TENSOR_DURATION, TOTAL_CYCLES, WARP_COUNT, WRITEBACK_START } from "./timeline";
import { createWorld } from "./scene";

const warpHex = (warp: number) => `#${WARP_COLORS[warp].getHexString()}`;
const TC_START = PLANS[0].execStart;
const TC_END = PLANS[WARP_COUNT - 1].execEnd;

/* Cycle timeline: the scheduler and execution units on top, one row per warp below. */

const schedulerIssues: Segment[] = [];
const warpRows: Row[] = PLANS.map((plan, warp) => {
  const writeStart = staggered(WRITEBACK_START, warp);
  for (const start of [plan.load, plan.ready, plan.execEnd]) {
    schedulerIssues.push({ start, end: start + ISSUE_CYCLES, kind: "issue", color: warpHex(warp) });
  }
  return {
    name: `W${warp}`,
    color: warpHex(warp),
    segments: [
      { start: plan.load, end: plan.load + ISSUE_CYCLES, kind: "issue" },
      { start: plan.load + ISSUE_CYCLES, end: plan.ready, kind: "wait", label: "shared memory" },
      { start: plan.ready, end: plan.execStart, kind: "issue" },
      { start: plan.execStart, end: plan.execEnd, kind: "compute", label: "Tensor Core" },
      { start: plan.execEnd, end: plan.done, kind: "issue" },
      { start: plan.done, end: writeStart, kind: "hold", label: "C in registers" },
      { start: writeStart, end: writeStart + WRITEBACK_DURATION, kind: "store", label: "C" },
    ],
  };
});

function warpStates(cycle: number): RowState[] {
  const selected = issuingWarp(cycle, PLANS, WRITEBACK_START);
  return PLANS.map((plan, warp) => {
    const writeStart = staggered(WRITEBACK_START, warp);
    let text: string;
    if (cycle < 8) text = "—";
    else if (cycle >= writeStart + WRITEBACK_DURATION) text = "C stored";
    else if (cycle >= writeStart) text = "writing C";
    else if (cycle < plan.load) text = "ready";
    else if (cycle < plan.load + ISSUE_CYCLES) text = "issue ldmatrix";
    else if (cycle < plan.ready) text = `wait · ${Math.ceil(plan.ready - cycle)} cyc`;
    else if (cycle < plan.execStart) text = "issue mma.sync";
    else if (cycle < plan.execEnd) text = `Tensor · ${Math.ceil(plan.execEnd - cycle)} cyc`;
    else if (cycle < plan.done) text = "C → registers";
    else text = "C in registers";
    return { text, active: warp === selected };
  });
}

function hardwareStates(cycle: number): RowState[] {
  const selected = issuingWarp(cycle, PLANS, WRITEBACK_START);
  const inTensor = PLANS.filter((plan) => cycle >= plan.execStart && cycle < plan.execEnd).length;
  return [
    { text: selected >= 0 ? `→ W${selected}` : "idle", active: selected >= 0 },
    { text: inTensor ? `${inTensor} MMA${inTensor > 1 ? "s" : ""} in flight` : "idle", active: inTensor > 0 },
    { text: "idle" },
  ];
}

must<HTMLElement>("#math-per-load").textContent = `${TENSOR_DURATION} cycles`;

runSmPage({
  ...createWorld(),
  phases: PHASES,
  duration: DURATION,
  totalCycles: TOTAL_CYCLES,
  timeline: {
    ticks: [0, 20, 40, 60, 80, 100, 120, 140, 160, TOTAL_CYCLES],
    hardware: [
      { name: "Sched", color: "#c8f06a", segments: schedulerIssues },
      { name: "Tensor", color: "#c8f06a", segments: [{ start: TC_START, end: TC_END, kind: "compute", label: "busy · up to 4 MMAs at once, one per Tensor Core" }] },
      { name: "FP32", color: "#6d7a8c", segments: [], empty: "idle · the MMA runs on the Tensor Cores" },
    ],
    warps: warpRows,
  },
  rowStates: (cycle) => [...hardwareStates(cycle), ...warpStates(cycle)],
});
