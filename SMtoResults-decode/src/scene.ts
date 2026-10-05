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
import { RESIDENT_CYCLE, WRITEBACK_DURATION, clamp01, issuingWarp, smooth, staggered, through } from "../../src/sm-timing";
import type { PagePhase } from "../../src/sm-page";
import {
  CHUNK_COUNT,
  COLS_PER_WARP,
  END_K,
  PLANS,
  SHUFFLE_CYCLES,
  SHUFFLE_START,
  SHUFFLE_STEPS,
  WARP_COUNT,
  WRITEBACK_START,
  shuffleStart,
} from "./timeline";

/*
 * Column w of every W chunk, band w of the register file and cell w of y all
 * belong to warp w. The K step on the timeline reads chunk 0; the rest of K
 * sweeps through chunks 1–7 between END_K and the shuffle.
 */
const SHARED = { x: 0, w: 1, y: 2 };
const wCell = (chunk: number, warp: number) => chunk * WARP_COUNT + warp;

export function createWorld() {
  const hw = createSmHardware({
    warps: WARP_COUNT,
    shared: [
      { letter: "x", color: X, rows: 1, cols: CHUNK_COUNT },
      { letter: "W", color: W, rows: CHUNK_COUNT, cols: WARP_COUNT },
      { letter: "y", color: Y, rows: 1, cols: WARP_COUNT },
    ],
    operands: [{ color: X, count: 1 }, { color: W, count: COLS_PER_WARP }],
    accumulators: COLS_PER_WARP,
  });
  const { scene, at } = hw;

  const tags = [
    tag(scene, "<b>FP32 lanes · 4 × 32</b><span>one row per sub-partition</span>", new THREE.Vector3(at.lanesX, 0.3, at.lanesBack - 0.1), "active", true, ["ffma", "gaps", "shuffle"]),
    tag(scene, "<b>Warp schedulers</b><span>2 warps each</span>", new THREE.Vector3(at.registerX, 0.3, at.schedulerZ - 0.9), "active", true, ["selected", "waiting"]),
    tag(scene, "<b>Shared memory</b><span>x · W in 8 chunks of 4 KB · y</span>", new THREE.Vector3(at.sharedX, 0.05, at.front), "w"),
    tag(scene, "<b>Register file</b><span>one band per warp · x W | 4 sums</span>", new THREE.Vector3(at.registerX, 0.05, at.front), "active"),
    tag(scene, "<b>Tensor Cores</b><span>idle: M = 1 has no tile for them</span>", new THREE.Vector3(at.tensorX, 0.05, at.front), "muted"),
  ];
  const callouts = [
    callout(scene, "<b>W0 is ready</b><span>its token lights up</span>", new THREE.Vector3(at.registerX - 1.2, 0.55, at.schedulerZ), "active", ["selected"]),
    callout(scene, "<b>LDS · back in 30 cycles</b><span>x[k] and W[k, 4 columns] → W0</span>", new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.w, 0)), "w", ["load"]),
    callout(scene, "<b>Scheduler moves on to W1</b><span>W0 is still waiting</span>", new THREE.Vector3(at.registerX - 0.4, 0.55, at.schedulerZ), "active", ["waiting"]),
    callout(scene, "<b>4 FFMA · 6 cycles</b><span>on W0's row of 32 lanes</span>", new THREE.Vector3(at.lanesX, 0.45, at.lanesBack - 0.15), "active", ["ffma"]),
    callout(scene, "<b>Lanes idle between warps</b><span>6 cycles of math per 30-cycle wait</span>", new THREE.Vector3(at.lanesX, 0.45, at.lanesBack - 0.15), "active", ["gaps"]),
    callout(scene, "<b>4 running sums per lane</b><span>one per output column</span>", new THREE.Vector3(at.registerX + 0.9, 0.7, 0), "y", ["accumulators"]),
    callout(scene, "<b>Chunks 1–7 repeat the step</b><span>into the same 4 sums</span>", new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.w, 3)), "w", ["end-k"]),
    callout(scene, "<b>shfl.down 16 → 8 → 4 → 2 → 1</b><span>32 partials fold into lane 0</span>", new THREE.Vector3(at.lanesX, 0.45, at.lanesBack - 0.15), "y", ["shuffle"]),
    callout(scene, "<b>y[j : j+32] · 128 B</b><span>8 warps × 16 B</span>", new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.y, 0)), "y", ["store", "assemble"]),
  ];

  const [sharedX, sharedW, sharedY] = hw.shared;
  const [regX, regW] = hw.operands;
  const loadsX = PLANS.map((_, warp) => arc(sharedX.tops[0], regX.tops[warp], 1.2));
  const loadsW = PLANS.map((_, warp) => arc(sharedW.tops[wCell(0, warp)], regW.tops[warp * COLS_PER_WARP + 1], 0.9));
  const toLanes = PLANS.map((_, warp) => arc(at.bandOut(warp), at.laneIn(subpartitionOf(warp)), 0.8));
  const fromLanes = PLANS.map((_, warp) => arc(at.laneIn(subpartitionOf(warp)), hw.accumulators.tops[warp * COLS_PER_WARP + 3], 0.8));
  const stores = PLANS.map((_, warp) => arc(hw.accumulators.tops[warp * COLS_PER_WARP], sharedY.tops[warp], 2.0));

  function update(_t: number, cycle: number, phase: PagePhase): void {
    showLabels(tags, callouts, phase.id);
    const c = cycle;
    const visible = c >= RESIDENT_CYCLE ? 1 : 0;
    const selected = issuingWarp(c, PLANS, WRITEBACK_START);
    /** 0 → CHUNK_COUNT - 1 as the rest of K plays out. */
    const sweep = through(c, END_K, SHUFFLE_START) * (CHUNK_COUNT - 1);

    // Shared memory: x and W are resident; a cell brightens while it is read, dims once copied.
    let xReading = false;
    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const plan = PLANS[warp];
      const reading = c >= plan.load && c < plan.ready;
      xReading ||= reading;
      sharedW.set(wCell(0, warp), W, reading ? 0.8 : c >= plan.ready ? 0.22 : 0.38);
      for (let chunk = 1; chunk < CHUNK_COUNT; chunk++) {
        const read = sweep - (chunk - 1);
        sharedW.set(wCell(chunk, warp), W, read > 0 && read < 1 ? 0.7 : read >= 1 ? 0.22 : 0.38);
      }
      const written = through(c, staggered(WRITEBACK_START, warp), staggered(WRITEBACK_START, warp) + WRITEBACK_DURATION);
      sharedY.set(warp, written > 0 ? Y : IDLE, written > 0 ? 0.3 + written * 0.5 : 0.12, 0.15 + written * 0.85);
    }
    for (let chunk = 0; chunk < CHUNK_COUNT; chunk++) {
      const read = chunk === 0 ? (xReading ? 0.5 : c >= PLANS[WARP_COUNT - 1].ready ? 1 : 0) : sweep - (chunk - 1);
      sharedX.set(chunk, X, read > 0 && read < 1 ? 0.75 : read >= 1 ? 0.22 : 0.4);
    }
    sharedX.commit();
    sharedW.commit();
    sharedY.commit();

    // Register file: operands arrive, get used, and the four sums grow one K step at a time.
    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const plan = PLANS[warp];
      hw.owners.set(warp, WARP_COLORS[warp], visible * (warp === selected ? 0.18 : 0.05));
      const loaded = through(c, plan.load, plan.ready);
      const operand = c >= plan.done ? 0.2 : c >= plan.ready ? 0.6 : 0.1 + loaded * 0.5;
      regX.set(warp, loaded > 0 ? X : IDLE, operand);
      for (let col = 0; col < COLS_PER_WARP; col++) regW.set(warp * COLS_PER_WARP + col, loaded > 0 ? W : IDLE, operand);

      const fraction = (through(c, plan.execStart, plan.execEnd) + sweep) / CHUNK_COUNT;
      const shuffling = through(c, shuffleStart(warp), shuffleStart(warp) + SHUFFLE_CYCLES);
      const writing = through(c, staggered(WRITEBACK_START, warp), staggered(WRITEBACK_START, warp) + WRITEBACK_DURATION);
      let glow = 0.1 + fraction * 0.35;
      if (shuffling > 0 && shuffling < 1) glow = 0.45 + 0.15 * Math.sin(shuffling * SHUFFLE_STEPS * Math.PI);
      if (writing > 0) glow = 0.6;
      for (let col = 0; col < COLS_PER_WARP; col++) {
        hw.accumulators.set(warp * COLS_PER_WARP + col, fraction > 0 ? Y : IDLE, glow, 0.12 + fraction * 0.88);
      }
      hw.tokens[warp].emissiveIntensity = warp === selected ? 0.8 : visible * 0.2;
      hw.tokenMeshes[warp].scale.setScalar(Math.max(0.001, visible));
    }
    hw.owners.commit();
    regX.commit();
    regW.commit();
    hw.accumulators.commit();
    hw.scheduler.emissive.copy(selected >= 0 ? WARP_COLORS[selected] : ACTIVE);
    hw.scheduler.emissiveIntensity = selected >= 0 ? 0.12 : 0.02;

    // FP32 lanes: a warp's row lights while its FFMAs run, flickers through the rest of K, then folds in the shuffle.
    for (let sp = 0; sp < SUBPARTITIONS; sp++) {
      const warps = PLANS.map((_, warp) => warp).filter((warp) => subpartitionOf(warp) === sp);
      const ffma = warps.find((warp) => c >= PLANS[warp].execStart && c < PLANS[warp].execEnd);
      const shfl = warps.find((warp) => c >= shuffleStart(warp) && c < shuffleStart(warp) + SHUFFLE_CYCLES);
      // Lane 0 holds a warp's finished sums from the end of its shuffle until its store completes.
      const holding = warps.some((warp) => c >= shuffleStart(warp) + SHUFFLE_CYCLES && c < staggered(WRITEBACK_START, warp) + WRITEBACK_DURATION);
      const lanes = hw.lanes[sp];
      for (let lane = 0; lane < LANES; lane++) {
        if (ffma !== undefined) {
          const p = through(c, PLANS[ffma].execStart, PLANS[ffma].execEnd);
          const wave = Math.sin(clamp01(p * 1.5 - lane / 64) * Math.PI);
          lanes.set(lane, WARP_COLORS[ffma], 0.35 + wave * 0.4, 1 + wave * 0.35);
        } else if (sweep > 0 && sweep < CHUNK_COUNT - 1) {
          const wave = 0.5 + 0.5 * Math.sin(c * 3 - lane * 0.4 + sp);
          lanes.set(lane, ACTIVE, 0.2 + wave * 0.25, 1 + wave * 0.2);
        } else if (shfl !== undefined) {
          const local = ((c - shuffleStart(shfl)) / SHUFFLE_CYCLES) * SHUFFLE_STEPS;
          const step = Math.min(SHUFFLE_STEPS - 1, Math.floor(local));
          const progress = local - step;
          const holding = LANES >> step;
          if (lane < holding >> 1) lanes.set(lane, Y, 0.4 + progress * 0.3, 1 + progress * 0.3);
          else if (lane < holding) lanes.set(lane, Y, 0.6 * (1 - progress), 1);
          else lanes.set(lane, IDLE, 0.13);
        } else if (holding) {
          lanes.set(lane, lane === 0 ? Y : IDLE, lane === 0 ? 0.6 : 0.13);
        } else {
          lanes.set(lane, IDLE, 0.16);
        }
      }
      lanes.commit();
      hw.tensor[sp].emissiveIntensity = 0;
    }

    // Beads: operands in, to the lanes and back as sums, y out to shared memory.
    const beads = hw.beads;
    beads.begin();
    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const plan = PLANS[warp];
      if (c >= plan.load && c < plan.ready) {
        const u = smooth((c - plan.load) / (plan.ready - plan.load));
        beads.trail(loadsX[warp], u, X, 0.7, 0.7);
        beads.trail(loadsW[warp], u, W, 0.7, 0.75);
      } else if (c >= plan.ready && c < plan.execStart) {
        beads.trail(toLanes[warp], smooth(c - plan.ready), WARP_COLORS[warp], 0.8, 0.75);
      } else if (c >= plan.execEnd && c < plan.done) {
        beads.trail(fromLanes[warp], smooth(c - plan.execEnd), Y, 0.8, 0.75);
      }
      const write = (c - staggered(WRITEBACK_START, warp)) / WRITEBACK_DURATION;
      beads.trail(stores[warp], smooth(write), Y, 0.8, 0.8);
    }
    beads.end();
  }

  return { scene, bounds: hw.bounds, center: hw.center, update };
}

