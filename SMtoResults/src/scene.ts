import * as THREE from "three";
import {
  ACTIVE,
  IDLE,
  LANES,
  SUBPARTITIONS,
  WARP_COLORS,
  W,
  X,
  Y,
  arc,
  callout,
  createSmHardware,
  showLabels,
  subpartitionOf,
  tag,
} from "../../src/sm-scene";
import { RESIDENT_CYCLE, WRITEBACK_DURATION, issuingWarp, smooth, staggered, through } from "../../src/sm-timing";
import type { PagePhase } from "../../src/sm-page";
import { PLANS, WARP_COUNT, WRITEBACK_START } from "./timeline";

/** Shared memory holds A and B (one fragment per warp) and the C tiles on their way out. */
const SHARED = { a: 0, b: 1, c: 2 };
const PER_OPERAND = 2;
const ACCUMULATORS = 4;

export function createWorld() {
  const hw = createSmHardware({
    warps: WARP_COUNT,
    shared: [
      { letter: "A", color: X, rows: 2, cols: 6 },
      { letter: "B", color: W, rows: 2, cols: 6 },
      { letter: "C", color: Y, rows: 2, cols: 6 },
    ],
    operands: [{ color: X, count: PER_OPERAND }, { color: W, count: PER_OPERAND }],
    accumulators: ACCUMULATORS,
  });
  const { scene, at } = hw;

  const tags = [
    tag(scene, "<b>FP32 lanes · 4 × 32</b><span>idle: the MMA runs on Tensor Cores</span>", new THREE.Vector3(at.lanesX, 0.3, at.lanesBack - 0.1), "muted", true, ["mma", "inflight"]),
    tag(scene, "<b>Warp schedulers</b><span>3 warps each</span>", new THREE.Vector3(at.registerX, 0.3, at.schedulerZ - 0.9), "active", true, ["selected", "waiting"]),
    tag(scene, "<b>Shared memory</b><span>A · B tiles · C on the way out</span>", new THREE.Vector3(at.sharedX, 0.05, at.front), "x"),
    tag(scene, "<b>Register file</b><span>one band per warp · A B | C</span>", new THREE.Vector3(at.registerX, 0.05, at.front), "active"),
    tag(scene, "<b>Tensor Cores</b><span>one per sub-partition · mma.sync</span>", new THREE.Vector3(at.tensorX, 0.05, at.front), "y", false, ["mma", "inflight"]),
  ];
  const callouts = [
    callout(scene, "<b>W0 is ready</b><span>its token lights up</span>", new THREE.Vector3(at.registerX - 1.2, 0.55, at.schedulerZ), "active", ["selected"]),
    callout(scene, "<b>ldmatrix · back in 30 cycles</b><span>A and B fragments → W0's registers</span>", new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.a, 0)), "x", ["load"]),
    callout(scene, "<b>Scheduler moves on to W1</b><span>W0 is still waiting</span>", new THREE.Vector3(at.registerX - 0.4, 0.55, at.schedulerZ), "active", ["waiting"]),
    callout(scene, "<b>mma.sync · 25 cycles</b><span>on W0's Tensor Core</span>", new THREE.Vector3(at.tensorX, 0.45, at.lanesBack - 0.15), "active", ["mma"]),
    callout(scene, "<b>Up to 4 MMAs at once</b><span>one per Tensor Core</span>", new THREE.Vector3(at.tensorX, 0.45, at.lanesBack - 0.15), "active", ["inflight"]),
    callout(scene, "<b>C stays in registers</b><span>updated by every MMA</span>", new THREE.Vector3(at.registerX + 0.9, 0.7, 0), "y", ["accumulators", "end-k"]),
    callout(scene, "<b>C → shared memory</b><span>fragments 2 cycles apart</span>", new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.c, 0)), "y", ["writeback"]),
  ];

  const [sharedA, sharedB, sharedC] = hw.shared;
  const [regA, regB] = hw.operands;
  const loadsA = PLANS.map((_, warp) => arc(sharedA.tops[warp], regA.tops[warp * PER_OPERAND], 1.1));
  const loadsB = PLANS.map((_, warp) => arc(sharedB.tops[warp], regB.tops[warp * PER_OPERAND], 0.9));
  const toTensor = PLANS.map((_, warp) => arc(hw.at.bandOut(warp), at.tensorTop(subpartitionOf(warp)), 0.9));
  const fromTensor = PLANS.map((_, warp) => arc(at.tensorTop(subpartitionOf(warp)), hw.accumulators.tops[warp * ACCUMULATORS + 3], 0.9));
  const writes = PLANS.map((_, warp) => arc(hw.accumulators.tops[warp * ACCUMULATORS], sharedC.tops[warp], 2.0));

  function update(_t: number, cycle: number, phase: PagePhase): void {
    showLabels(tags, callouts, phase.id);
    const c = cycle;
    const visible = c >= RESIDENT_CYCLE ? 1 : 0;
    const selected = issuingWarp(c, PLANS, WRITEBACK_START);

    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const plan = PLANS[warp];
      const reading = c >= plan.load && c < plan.ready;
      sharedA.set(warp, X, reading ? 0.8 : 0.38);
      sharedB.set(warp, W, reading ? 0.8 : 0.38);
      const written = through(c, staggered(WRITEBACK_START, warp), staggered(WRITEBACK_START, warp) + WRITEBACK_DURATION);
      sharedC.set(warp, written > 0 ? Y : IDLE, written > 0 ? 0.3 + written * 0.5 : 0.12, 0.15 + written * 0.85);

      hw.owners.set(warp, WARP_COLORS[warp], visible * (warp === selected ? 0.18 : 0.05));
      const loaded = through(c, plan.load, plan.ready);
      const operand = c >= plan.done ? 0.2 : c >= plan.ready ? 0.6 : 0.1 + loaded * 0.5;
      for (let i = 0; i < PER_OPERAND; i++) {
        regA.set(warp * PER_OPERAND + i, loaded > 0 ? X : IDLE, operand);
        regB.set(warp * PER_OPERAND + i, loaded > 0 ? W : IDLE, operand);
      }
      const filled = through(c, plan.execStart, plan.execEnd);
      const writing = written > 0 && written < 1;
      for (let i = 0; i < ACCUMULATORS; i++) {
        hw.accumulators.set(warp * ACCUMULATORS + i, filled > 0 ? Y : IDLE, writing ? 0.6 : 0.1 + filled * 0.4, 0.12 + filled * 0.88);
      }
      hw.tokens[warp].emissiveIntensity = warp === selected ? 0.8 : visible * 0.2;
      hw.tokenMeshes[warp].scale.setScalar(Math.max(0.001, visible));
    }
    sharedA.commit();
    sharedB.commit();
    sharedC.commit();
    hw.owners.commit();
    regA.commit();
    regB.commit();
    hw.accumulators.commit();
    hw.scheduler.emissive.copy(selected >= 0 ? WARP_COLORS[selected] : ACTIVE);
    hw.scheduler.emissiveIntensity = selected >= 0 ? 0.12 : 0.02;

    // Each sub-partition's Tensor Core glows while one of its warps has an MMA in flight; the lanes stay idle.
    for (let sp = 0; sp < SUBPARTITIONS; sp++) {
      const busy = PLANS.some((plan, warp) => subpartitionOf(warp) === sp && c >= plan.execStart && c < plan.execEnd);
      hw.tensor[sp].emissiveIntensity = busy ? 0.22 : 0;
      for (let lane = 0; lane < LANES; lane++) hw.lanes[sp].set(lane, IDLE, 0.16);
      hw.lanes[sp].commit();
    }

    // Beads: operands in, the MMA out to the Tensor Core and back, C out to shared memory.
    const beads = hw.beads;
    beads.begin();
    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const plan = PLANS[warp];
      const color = WARP_COLORS[warp];
      if (c >= plan.load && c < plan.ready) {
        const u = smooth((c - plan.load) / (plan.ready - plan.load));
        beads.trail(loadsA[warp], u, X, 0.7, 0.75);
        beads.trail(loadsB[warp], u, W, 0.7, 0.75);
      } else if (c >= plan.ready && c < plan.execStart) {
        beads.trail(toTensor[warp], smooth(c - plan.ready), color, 0.8, 0.8);
      } else if (c >= plan.execStart && c < plan.execEnd) {
        beads.add(at.tensorTop(subpartitionOf(warp)).add(new THREE.Vector3(0, 0.12 + 0.03 * Math.sin(c + warp), 0)), color, 0.6, 0.8);
      } else if (c >= plan.execEnd && c < plan.done) {
        beads.trail(fromTensor[warp], smooth(c - plan.execEnd), Y, 0.8, 0.8);
      }
      const write = (c - staggered(WRITEBACK_START, warp)) / WRITEBACK_DURATION;
      beads.trail(writes[warp], smooth(write), Y, 0.8, 0.8);
    }
    beads.end();
  }

  return { scene, bounds: hw.bounds, center: hw.center, update };
}
