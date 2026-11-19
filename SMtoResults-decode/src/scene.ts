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
  animateCopy,
  arc,
  callout,
  createSmHardware,
  filled,
  type CopyCell,
  showLabels,
  sideTag,
  subpartitionOf,
  tag,
} from "../../src/sm-scene";
import { L2_BYTES_PER_CYCLE, WRITEBACK_DURATION, arrivalOf, clamp01, smooth, through } from "../../src/sm-timing";
import type { PagePhase } from "../../src/sm-page";
import {
  CHUNK_COUNT,
  CHUNK_CYCLES,
  COLS_PER_WARP,
  FADD_WINDOW,
  FFMA_CYCLES,
  FOCUS,
  K,
  PHASES,
  SLICE_N,
  SHUFFLE_CYCLES,
  SHUFFLE_STEPS,
  SHUFFLE_STEP_CYCLES,
  STEPS,
  WARP_COUNT,
  W_CHUNK_BYTES,
  X_BYTES,
  chunkStart,
  issuingWarp,
  shuffleStart,
  stepOf,
  storeStart,
} from "./timeline";

/*
 * Shared memory is laid out as the product: row k of W is chunk k (32 rows of
 * the real matrix) and x's cell k, x[32k : 32k+32], sits beside it; y runs
 * under W's columns. Column w of every chunk (4 real columns), band w of the
 * register file and cell w of y all belong to warp w. The TMA streams x, then
 * the chunks in order; each warp loads its column of chunk k as soon as that
 * chunk has landed.
 */
const SHARED = { x: 0, w: 1, y: 2 };
/** W cell for one real column n of one chunk: 32 rows × 1 column, 32 values. Warp n / 4 owns it. */
const wCell = (chunk: number, n: number) => chunk * SLICE_N + n;
const ownerOf = (n: number) => Math.floor(n / COLS_PER_WARP);

export function createWorld() {
  const hw = createSmHardware({
    warps: WARP_COUNT,
    shared: [
      { letter: "x", color: X, rows: 1, cols: CHUNK_COUNT },
      { letter: "W", color: W, rows: CHUNK_COUNT, cols: SLICE_N },
      { letter: "y", color: Y, rows: 1, cols: SLICE_N },
    ],
    // x down the left beside W's rows (k), y under W's columns (n). Every x and W cell holds
    // 32 values; each warp's 4 columns form a group labeled in its color.
    matvec: {
      rows: STEPS.map((_, chunk) => `${chunk === 0 ? "k " : ""}${chunk * (K / CHUNK_COUNT)}`),
      cols: STEPS[0].map((_, warp) => `${warp === 0 ? "n " : ""}${warp * COLS_PER_WARP}`),
      colColors: WARP_COLORS,
      groupCols: COLS_PER_WARP,
    },
    warpLabel: (warp) => `warp ${warp}`,
    operands: [{ color: X, count: 1 }, { color: W, count: COLS_PER_WARP }],
    accumulators: COLS_PER_WARP,
  });
  const { scene, at } = hw;
  const [sharedX, sharedW, sharedY] = hw.shared;
  const wHeaderX = (sharedW.tops[0].x + sharedW.tops[SLICE_N - 1].x) / 2;
  // Above the scheduler, pointing down at warp 0's band of sums.
  const sumsAt = new THREE.Vector3(at.registerX + 0.9, 0.55, at.schedulerZ + 0.9);

  const tags = [
    tag(scene, "<b>FP32 lanes · 4 × 32</b><span>one row per sub-partition</span>", new THREE.Vector3(at.lanesX, 0.3, at.lanesBack - 0.1), "active", true, ["ffma", "gaps", "stream", "shuffle"]),
    tag(scene, "<b>Warp schedulers</b><span>2 warps each</span>", new THREE.Vector3(at.registerX, 0.3, at.schedulerZ - 0.9), "active", true, ["selected", "waiting", "next", "shuffle"]),
    tag(scene, "<b>Shared memory</b><span>one row of W per chunk · 4 KB each</span>", new THREE.Vector3(at.sharedX, 0.05, at.front), "w"),
    tag(scene, "<b>Register file · 1 row = 1 warp</b><span>32 lanes × 4 sums = 128 FP32 partials per row</span>", new THREE.Vector3(at.registerX, 0.05, at.front), "active"),
    tag(scene, "<b>Tensor Cores</b><span>idle: M = 1 has no tile for them</span>", new THREE.Vector3(at.tensorX, 0.05, at.front), "muted"),
    tag(scene, "<b>TMA</b><span>copies from L2</span>", new THREE.Vector3(at.tmaX, 0.5, at.tmaZ - 0.4), "w", true, ["request", "latency", "land", "load"]),
    // The real shapes, next to the cells that stand for them.
    sideTag(scene, `<b>x · ${K}</b><span>${K / CHUNK_COUNT} values per cell</span>`, new THREE.Vector3(at.sharedX - 1.55, 0.3, at.sharedRowZ(SHARED.x, 5)), "x plain", "left", ["store", "assemble"]),
    tag(scene, `<b>W · ${K} × ${SLICE_N}</b><span>${K / CHUNK_COUNT} values per cell</span>`, new THREE.Vector3(wHeaderX, 0.3, at.sharedRowZ(SHARED.w, 0) - 0.6), "w plain", true, ["land", "load"]),
    sideTag(scene, `<b>y · ${SLICE_N}</b><span>1 value per cell</span>`, new THREE.Vector3(at.sharedX - 1.55, 0.3, at.sharedRowZ(SHARED.y, 0)), "y plain", "left"),
  ];
  const callouts = [
    callout(scene, `<b>One bulk request · ${Math.round((X_BYTES + CHUNK_COUNT * W_CHUNK_BYTES) / 1024)} KB</b><span>x, then W chunks 0–7</span>`, new THREE.Vector3(at.tmaX, 0.7, at.tmaZ), "w", ["request"]),
    callout(scene, "<b>~250 cycles</b><span>before the first byte is back from L2</span>", new THREE.Vector3(at.tmaX, 0.7, at.tmaZ), "x", ["latency"]),
    callout(scene, `<b>${L2_BYTES_PER_CYCLE} B per cycle</b><span>one W chunk every ${CHUNK_CYCLES} cycles</span>`, new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.w, 0)), "w", ["land"]),
    callout(scene, "<b>Warp 0 is ready</b><span>its token lights up</span>", new THREE.Vector3(at.registerX - 1.2, 0.55, at.schedulerZ), "active", ["selected"]),
    callout(scene, "<b>LDS · back in 30 cycles</b><span>lane k ← x[k], W[k, 0…3]</span>", new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.w, 0)), "w", ["load"]),
    callout(scene, "<b>Scheduler moves on to warp 1</b><span>warp 0 is still waiting</span>", new THREE.Vector3(at.registerX - 0.4, 0.55, at.schedulerZ), "active", ["waiting"]),
    callout(scene, `<b>4 FFMA · ${FFMA_CYCLES} cycles</b><span>on warp 0's row of 32 lanes</span>`, new THREE.Vector3(at.lanesX, 0.45, at.lanesBack - 0.15), "active", ["ffma"]),
    callout(scene, `<b>Lanes idle between warps</b><span>${FFMA_CYCLES} cycles of math per 30-cycle wait</span>`, new THREE.Vector3(at.lanesX, 0.45, at.lanesBack - 0.15), "active", ["gaps"]),
    callout(scene, "<b>Chunk 1 · same 4 sums</b><span>they stay in the registers</span>", sumsAt, "y", ["next"]),
    callout(scene, `<b>Busy ${2 * FFMA_CYCLES} of every ${CHUNK_CYCLES} cycles</b><span>the rest is waiting on L2</span>`, new THREE.Vector3(at.lanesX, 0.45, at.lanesBack - 0.15), "active", ["stream"]),
    callout(scene, `<b>Chunk ${CHUNK_COUNT - 1} is the last</b><span>8 chunks × 32 k = 256</span>`, new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.w, CHUNK_COUNT - 1)), "w", ["end-k"]),
    callout(scene, "<b>32 lanes → 1 lane</b><span>lane 0 ends with 4 output values</span>", sumsAt, "y", ["shuffle"]),
    callout(scene, "<b>y[j : j+32] · 128 B</b><span>8 warps × 16 B</span>", new THREE.Vector3(at.sharedX, 0.6, at.sharedRowZ(SHARED.y, 0)), "y", ["store", "assemble"]),
  ];

  const [regX, regW] = hw.operands;
  const loadsX = STEPS.map((_, chunk) => STEPS[chunk].map((__, warp) => arc(sharedX.tops[chunk], regX.tops[warp], 1.2)));
  const loadsW = STEPS.map((_, chunk) => STEPS[chunk].map((__, warp) => arc(sharedW.tops[wCell(chunk, warp * COLS_PER_WARP + 1)], regW.tops[warp * COLS_PER_WARP + 1], 0.9)));
  const toLanes = STEPS[0].map((_, warp) => arc(at.bandOut(warp), at.laneIn(subpartitionOf(warp)), 0.8));
  const fromLanes = STEPS[0].map((_, warp) => arc(at.laneIn(subpartitionOf(warp)), hw.accumulators.tops[warp * COLS_PER_WARP + 3], 0.8));
  const stores = STEPS[0].map((_, warp) => arc(hw.accumulators.tops[warp * COLS_PER_WARP], sharedY.tops[warp * COLS_PER_WARP + 1], 2.0));

  // The TMA delivers all of x (cell by cell), then W chunk by chunk, each chunk column by column.
  const xSpan = X_BYTES / CHUNK_COUNT / L2_BYTES_PER_CYCLE;
  const wSpan = W_CHUNK_BYTES / SLICE_N / L2_BYTES_PER_CYCLE;
  const xCells: CopyCell[] = Array.from({ length: CHUNK_COUNT }, (_, chunk) => ({
    target: sharedX.tops[chunk],
    color: X,
    land: arrivalOf(((chunk + 1) * X_BYTES) / CHUNK_COUNT),
    span: xSpan,
  }));
  const wCells: CopyCell[] = Array.from({ length: CHUNK_COUNT * SLICE_N }, (_, index) => ({
    target: sharedW.tops[index],
    color: W,
    land: chunkStart(Math.floor(index / SLICE_N)) + ((index % SLICE_N) + 1) * wSpan,
    span: wSpan,
  }));
  // One bead per warp's group of 4 columns is plenty to show the stream.
  const copyCells = [...xCells, ...wCells.filter((_, index) => index % COLS_PER_WARP === COLS_PER_WARP - 1)];
  const copyCurves = copyCells.map((cell) => hw.inbound(cell.target));

  /** Where a warp is within its shuffle: step index and progress through it, or null. */
  const shuffleAt = (warp: number, c: number) => {
    const local = (c - shuffleStart(warp)) / SHUFFLE_STEP_CYCLES;
    if (local < 0 || local >= SHUFFLE_STEPS) return null;
    const step = Math.floor(local);
    return { step, p: local - step };
  };

  function update(t: number, cycle: number, phase: PagePhase): void {
    showLabels(tags, callouts, phase.id);
    const c = cycle;
    const visible = t >= PHASES[1].t ? 1 : 0;
    const selected = issuingWarp(c);
    // Warps the step is not about fade, so the one being narrated stands out.
    const focus = FOCUS[phase.id];
    const lit = (warp: number) => (!focus || focus.includes(warp) ? 1 : 0.22);

    // Shared memory: cells fill as the copy lands, brighten while read, and dim once copied to registers.
    const cell = (fill: number, color: THREE.Color, glow: number) => [fill > 0 ? color : IDLE, fill > 0 ? glow : 0.12, 0.15 + fill * 0.85] as const;
    for (let chunk = 0; chunk < CHUNK_COUNT; chunk++) {
      let xReading = false;
      for (let n = 0; n < SLICE_N; n++) {
        const warp = ownerOf(n);
        const plan = STEPS[chunk][warp];
        const reading = c >= plan.load && c < plan.ready;
        xReading ||= reading;
        const glow = reading ? 0.8 : c >= plan.ready ? 0.22 : 0.38;
        sharedW.set(wCell(chunk, n), ...cell(filled(c, wCells[wCell(chunk, n)]), W, glow * lit(warp)));
      }
      const used = c >= STEPS[chunk][WARP_COUNT - 1].ready;
      sharedX.set(chunk, ...cell(filled(c, xCells[chunk]), X, xReading ? 0.75 : used ? 0.22 : 0.4));
    }
    // y: each warp's lane 0 writes its 4 values one after another.
    const perValue = WRITEBACK_DURATION / COLS_PER_WARP;
    for (let n = 0; n < SLICE_N; n++) {
      const start = storeStart(ownerOf(n)) + (n % COLS_PER_WARP) * perValue;
      const written = through(c, start, start + perValue);
      // One value per cell: a thin slab, a fraction of the height of a 32-value cell.
      sharedY.set(n, written > 0 ? Y : IDLE, written > 0 ? 0.3 + written * 0.5 : 0.12, 0.04 + written * 0.06);
    }
    sharedX.commit();
    sharedW.commit();
    sharedY.commit();

    // Register file: operands arrive for each K step; the four sums grow by one chunk at a time and are reduced in place.
    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const l = lit(warp);
      const step = stepOf(warp, c);
      hw.owners.set(warp, WARP_COLORS[warp], visible * (warp === selected ? 0.18 : 0.05) * l);
      if (step < 0) {
        regX.set(warp, IDLE, 0.1 * l);
        for (let col = 0; col < COLS_PER_WARP; col++) regW.set(warp * COLS_PER_WARP + col, IDLE, 0.1 * l);
      } else {
        const plan = STEPS[step][warp];
        const loaded = through(c, plan.load, plan.ready);
        const operand = (c >= plan.done ? 0.2 : c >= plan.ready ? 0.6 : 0.1 + loaded * 0.5) * l;
        regX.set(warp, X, operand);
        for (let col = 0; col < COLS_PER_WARP; col++) regW.set(warp * COLS_PER_WARP + col, W, operand);
      }

      let fraction = 0;
      for (const plans of STEPS) fraction += through(c, plans[warp].execStart, plans[warp].execEnd);
      fraction /= CHUNK_COUNT;
      const shuffle = shuffleAt(warp, c);
      let glow = 0.1 + fraction * 0.35;
      // Each shuffle step pulses as its FADD result comes back into the registers.
      if (shuffle) glow = 0.45 + 0.25 * Math.sin(clamp01((shuffle.p - FADD_WINDOW[1]) / (1 - FADD_WINDOW[1])) * Math.PI);
      if (c >= shuffleStart(warp) + SHUFFLE_CYCLES) glow = 0.6;
      for (let col = 0; col < COLS_PER_WARP; col++) {
        hw.accumulators.set(warp * COLS_PER_WARP + col, fraction > 0 ? Y : IDLE, glow * l, 0.12 + fraction * 0.88);
      }
      hw.tokens[warp].emissiveIntensity = (warp === selected ? 0.8 : visible * 0.2) * l;
      hw.tokenMeshes[warp].scale.setScalar(Math.max(0.001, visible));
    }
    hw.owners.commit();
    regX.commit();
    regW.commit();
    hw.accumulators.commit();
    hw.scheduler.emissive.copy(selected >= 0 ? WARP_COLORS[selected] : ACTIVE);
    hw.scheduler.emissiveIntensity = selected >= 0 ? 0.12 : 0.02;

    // FP32 lanes: execution only, nothing is held here. A row lights while a warp's FFMAs or shuffle FADDs run.
    for (let sp = 0; sp < SUBPARTITIONS; sp++) {
      const warps = STEPS[0].map((_, warp) => warp).filter((warp) => subpartitionOf(warp) === sp);
      let ffma: { warp: number; p: number } | undefined;
      let fadd: { warp: number; step: number; p: number } | undefined;
      for (const warp of warps) {
        for (const plans of STEPS) {
          const plan = plans[warp];
          if (c >= plan.execStart && c < plan.execEnd) ffma = { warp, p: through(c, plan.execStart, plan.execEnd) };
        }
        const shuffle = shuffleAt(warp, c);
        if (shuffle && shuffle.p >= FADD_WINDOW[0] && shuffle.p < FADD_WINDOW[1]) {
          fadd = { warp, step: shuffle.step, p: (shuffle.p - FADD_WINDOW[0]) / (FADD_WINDOW[1] - FADD_WINDOW[0]) };
        }
      }
      const lanes = hw.lanes[sp];
      for (let lane = 0; lane < LANES; lane++) {
        if (ffma) {
          const wave = Math.sin(clamp01(ffma.p * 1.5 - lane / 64) * Math.PI);
          const l = lit(ffma.warp);
          lanes.set(lane, WARP_COLORS[ffma.warp], (0.35 + wave * 0.4) * l, 1 + wave * 0.35 * l);
        } else if (fadd) {
          // All 32 lanes run the FADD; only the lower half of the active range keeps a useful sum.
          const pulse = Math.sin(fadd.p * Math.PI);
          const useful = lane < LANES >> (fadd.step + 1);
          const l = lit(fadd.warp);
          lanes.set(lane, useful ? Y : IDLE, (useful ? 0.25 + pulse * 0.45 : 0.18 + pulse * 0.1) * l, 1 + (useful ? pulse * 0.3 : 0));
        } else {
          lanes.set(lane, IDLE, 0.16);
        }
      }
      lanes.commit();
      hw.tensor[sp].emissiveIntensity = 0;
    }

    // Beads: operands in, to the lanes and back as sums, shuffle partials out to the FADDs and back, y out to shared memory.
    const beads = hw.beads;
    beads.begin();
    animateCopy(hw, c, copyCells, copyCurves);
    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const l = lit(warp);
      const step = stepOf(warp, c);
      if (step >= 0) {
        const plan = STEPS[step][warp];
        if (c >= plan.load && c < plan.ready) {
          const u = smooth((c - plan.load) / (plan.ready - plan.load));
          beads.trail(loadsX[step][warp], u, X, 0.7 * l, 0.7 * l);
          beads.trail(loadsW[step][warp], u, W, 0.7 * l, 0.75 * l);
        } else if (c >= plan.ready && c < plan.execStart) {
          beads.trail(toLanes[warp], smooth(c - plan.ready), WARP_COLORS[warp], 0.8 * l, 0.75 * l);
        } else if (c >= plan.execEnd && c < plan.done) {
          beads.trail(fromLanes[warp], smooth(c - plan.execEnd), Y, 0.8 * l, 0.75 * l);
        }
      }
      const shuffle = shuffleAt(warp, c);
      if (shuffle) {
        beads.trail(toLanes[warp], smooth(shuffle.p / FADD_WINDOW[0]), Y, 0.7 * l, 0.6 * l);
        beads.trail(fromLanes[warp], smooth((shuffle.p - FADD_WINDOW[1]) / (1 - FADD_WINDOW[1])), Y, 0.8 * l, 0.6 * l);
      }
      beads.trail(stores[warp], smooth((c - storeStart(warp)) / WRITEBACK_DURATION), Y, 0.8, 0.8);
    }
    beads.end();
  }

  return { scene, bounds: hw.bounds, center: hw.center, update };
}
