import { runSmPage, must, type PagePhase } from "../../src/sm-page";
import { ISSUE_CYCLES, L2_LATENCY, WRITEBACK_DURATION } from "../../src/sm-timing";
import type { Row, RowState, Segment } from "../../src/cycle-timeline";
import { WARP_COLORS } from "../../src/sm-scene";
import {
  CHUNK_COUNT,
  CHUNK_CYCLES,
  COPY,
  DETAIL_END,
  DURATION,
  FADD_WINDOW,
  FFMA_CYCLES,
  FOCUS,
  PHASES,
  PLAYBACK,
  SHUFFLE_CYCLES,
  SHUFFLE_STEPS,
  SHUFFLE_STEP_CYCLES,
  SQUEEZE,
  STEPS,
  TOTAL_CYCLES,
  chunkLanded,
  chunkStart,
  issuingWarp,
  shuffleStart,
  stepOf,
  storeStart,
} from "./timeline";
import { createWorld } from "./scene";

const warpHex = (warp: number) => `#${WARP_COLORS[warp].getHexString()}`;
const LAST_LANDED = chunkLanded(CHUNK_COUNT - 1);

/* Cycle timeline: the TMA, scheduler and execution units on top, one row per warp below. */

const schedulerIssues: Segment[] = [];
const lanesBusy: Segment[] = [];
const warpRows: Row[] = STEPS[0].map((_, warp) => {
  const color = warpHex(warp);
  const segments: Segment[] = [{ start: 0, end: COPY.landed, kind: "wait", label: "waiting on L2" }];
  STEPS.forEach((plans, chunk) => {
    const plan = plans[warp];
    if (chunk > 0) segments.push({ start: STEPS[chunk - 1][warp].done, end: plan.load, kind: "wait", label: chunk === 1 ? "chunk 1" : undefined });
    segments.push(
      { start: plan.load, end: plan.load + ISSUE_CYCLES, kind: "issue" },
      { start: plan.load + ISSUE_CYCLES, end: plan.ready, kind: "wait", label: chunk === 0 ? "shared memory" : undefined },
      { start: plan.ready, end: plan.execStart, kind: "issue" },
      { start: plan.execStart, end: plan.execEnd, kind: "compute", label: "FFMA" },
      { start: plan.execEnd, end: plan.done, kind: "issue" },
    );
    for (const issue of [plan.load, plan.ready, plan.execEnd]) schedulerIssues.push({ start: issue, end: issue + ISSUE_CYCLES, kind: "issue", color });
    lanesBusy.push({ start: plan.execStart, end: plan.execEnd, kind: "compute", color });
  });
  for (let s = 0; s < SHUFFLE_STEPS; s++) {
    const start = shuffleStart(warp) + s * SHUFFLE_STEP_CYCLES;
    schedulerIssues.push({ start, end: start + ISSUE_CYCLES, kind: "issue", color });
    lanesBusy.push({ start: start + FADD_WINDOW[0] * SHUFFLE_STEP_CYCLES, end: start + FADD_WINDOW[1] * SHUFFLE_STEP_CYCLES, kind: "result", color });
  }
  schedulerIssues.push({ start: storeStart(warp), end: storeStart(warp) + ISSUE_CYCLES, kind: "issue", color });
  segments.push(
    { start: shuffleStart(warp), end: storeStart(warp), kind: "result", label: "shfl + FADD" },
    { start: storeStart(warp), end: storeStart(warp) + WRITEBACK_DURATION, kind: "store", label: "y" },
  );
  return { name: `warp ${warp}`, color, segments };
});

function warpStates(cycle: number, phase: PagePhase): RowState[] {
  const selected = issuingWarp(cycle);
  const focus = FOCUS[phase.id];
  return STEPS[0].map((_, warp) => {
    const shuffle = shuffleStart(warp);
    const store = storeStart(warp);
    const step = stepOf(warp, cycle);
    let text: string;
    if (cycle < COPY.landed) text = "waiting on L2";
    else if (cycle >= store + WRITEBACK_DURATION) text = "y stored";
    else if (cycle >= store) text = "writing y";
    else if (cycle >= shuffle) text = `shfl.down ${16 >> Math.floor((cycle - shuffle) / SHUFFLE_STEP_CYCLES)} + FADD`;
    else if (step < 0) text = "ready";
    else {
      const plan = STEPS[step][warp];
      if (cycle < plan.load + ISSUE_CYCLES) text = `issue LDS · k${step}`;
      else if (cycle < plan.ready) text = `wait · ${Math.ceil(plan.ready - cycle)} cyc`;
      else if (cycle < plan.execStart) text = "issue FFMA";
      else if (cycle < plan.execEnd) text = `FFMA · ${Math.ceil(plan.execEnd - cycle)} cyc`;
      else if (cycle < plan.done) text = "sums → registers";
      else if (step < CHUNK_COUNT - 1 && cycle < chunkLanded(step + 1)) text = `wait chunk ${step + 1}`;
      else text = "ready";
    }
    return { text, active: warp === selected, dim: focus && !focus.includes(warp) };
  });
}

function hardwareStates(cycle: number): RowState[] {
  const selected = issuingWarp(cycle);
  const ffma = STEPS.flat().filter((plan) => cycle >= plan.execStart && cycle < plan.execEnd).length;
  const fadd = STEPS[0].some((_, warp) => cycle >= shuffleStart(warp) && cycle < storeStart(warp));
  const lanes = ffma ? `${ffma} warp FFMA` : fadd ? "shuffle FADDs" : "idle";
  const chunk = Math.max(0, Math.ceil((cycle - chunkStart(0)) / CHUNK_CYCLES) - 1);
  const copy = cycle <= 0 ? "idle" : cycle < L2_LATENCY ? "L2 latency" : cycle < chunkStart(0) ? "copying x" : cycle < LAST_LANDED ? `copying chunk ${chunk}` : "done";
  return [
    { text: copy, active: cycle >= L2_LATENCY && cycle < LAST_LANDED },
    { text: selected >= 0 ? `→ warp ${selected}` : "idle", active: selected >= 0 },
    { text: lanes, active: lanes !== "idle" },
    { text: "idle" },
  ];
}

must<HTMLElement>("#math-per-load").textContent = `${FFMA_CYCLES} cycles`;
must<HTMLElement>("#chunk-interval").textContent = `${CHUNK_CYCLES} cycles`;

runSmPage({
  ...createWorld(),
  phases: PHASES,
  playback: PLAYBACK,
  duration: DURATION,
  totalCycles: TOTAL_CYCLES,
  timeline: {
    ticks: [0, L2_LATENCY, COPY.landed, chunkLanded(1), DETAIL_END, LAST_LANDED, storeStart(0), TOTAL_CYCLES],
    squeeze: SQUEEZE,
    hardware: [
      {
        name: "TMA",
        color: "#f2a24a",
        segments: [
          { start: 0, end: L2_LATENCY, kind: "latency", label: `L2 latency · ~${L2_LATENCY} cycles` },
          { start: L2_LATENCY, end: chunkStart(0), kind: "x" },
          ...STEPS.map((_, chunk): Segment => ({ start: chunkStart(chunk), end: chunkLanded(chunk), kind: "w", label: chunk < 2 ? `W chunk ${chunk}` : String(chunk) })),
        ],
      },
      { name: "Sched", color: "#c8f06a", segments: schedulerIssues },
      { name: "FP32", color: "#c8f06a", segments: lanesBusy },
      { name: "Tensor", color: "#6d7a8c", segments: [], empty: `idle · with M = 1 there is no matrix tile · the reduction is ${SHUFFLE_STEPS} shuffles + FADDs (${SHUFFLE_CYCLES} cycles)` },
    ],
    warps: warpRows,
  },
  rowStates: (cycle, phase) => [...hardwareStates(cycle), ...warpStates(cycle, phase)],
});
