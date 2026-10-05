import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createWorld as createPrefillWorld } from "../../HBMtoSM/src/scene/world";
import { createWorld as createDecodeWorld } from "../../HBMtoSM-decode/src/scene/world";
import { BUDGET as PREFILL, sampleAtCycle as samplePrefill } from "../../HBMtoSM/src/model/timeline";
import { BUDGET as DECODE, sampleAtCycle as sampleDecode } from "../../HBMtoSM-decode/src/model/timeline";
import { CYCLES_PER_SECOND, GPU, formatGpuTime, type KernelBudget } from "../../src/gpu-timing";
import { HUD_FRAME_INTERVAL, MAX_PIXEL_RATIO, MIN_FRAME_INTERVAL } from "../../src/render-performance";

/**
 * Prefill and decode side by side, both driven by one GPU cycle counter at
 * the same CYCLES_PER_SECOND as their own scenes. Narration is skipped, so
 * every second on screen is the same number of cycles on both sides.
 */
type Budget = KernelBudget;

interface Workload {
  id: "prefill" | "decode";
  name: string;
  shape: string;
  budget: Budget;
  sample: typeof samplePrefill;
  createWorld: typeof createPrefillWorld;
  stage(cycle: number): string;
}

const WORKLOADS: Workload[] = [
  {
    id: "prefill",
    name: "Prefill",
    shape: "C = A · B · 256×256 × 256×256",
    budget: PREFILL,
    sample: samplePrefill as Workload["sample"],
    createWorld: createPrefillWorld,
    stage: (c) =>
      c <= 0
        ? "ready"
        : c < PREFILL.hbmIn
          ? "HBM → L2 · A and B, 512 KB"
          : c < PREFILL.hbmIn + PREFILL.sm
            ? inMath(PREFILL, c)
              ? "64 SMs · FP32 math, loads hidden"
              : "64 SMs · waiting on L2"
            : c < PREFILL.total
              ? "C → HBM · 256 KB"
              : "done",
  },
  {
    id: "decode",
    name: "Decode",
    shape: "y = x · W · 1×256 × 256×256",
    budget: DECODE,
    sample: sampleDecode as unknown as Workload["sample"],
    createWorld: createDecodeWorld as unknown as typeof createPrefillWorld,
    stage: (c) =>
      c <= 0
        ? "ready"
        : c < DECODE.hbmIn
          ? "HBM → L2 · x and W, 257 KB"
          : c < DECODE.hbmIn + DECODE.sm
            ? "8 SMs · streaming W, lanes mostly idle"
            : c < DECODE.total
              ? "y → HBM · 1 KB"
              : "done",
  },
];

const LEAD = 1;
const TAIL = 5;
const TOTAL = Math.max(PREFILL.total, DECODE.total);
const DURATION = LEAD + TOTAL / CYCLES_PER_SECOND + TAIL;

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function cycleAt(t: number): number {
  return clamp((t - LEAD) * CYCLES_PER_SECOND, 0, TOTAL);
}

function mathCycles(budget: Budget): number {
  return budget.math.reduce((sum, [a, b]) => sum + (b - a), 0);
}

function mathDone(budget: Budget, cycle: number): number {
  return budget.math.reduce((sum, [a, b]) => sum + clamp(cycle - a, 0, b - a), 0);
}

function inMath(budget: Budget, cycle: number): boolean {
  return budget.math.some(([a, b]) => cycle >= a && cycle < b);
}

/** Bytes land after the HBM latency, then stream at a constant rate. */
function hbmBytes(budget: Budget, cycle: number): number {
  const stream = (start: number, span: number, bytes: number) =>
    bytes * clamp((cycle - start - GPU.hbmLatency) / (span - GPU.hbmLatency), 0, 1);
  return stream(0, budget.hbmIn, budget.inBytes) + stream(budget.hbmIn + budget.sm, budget.out, budget.outBytes);
}

function micros(cycles: number): number {
  return cycles / GPU.clockGHz / 1000;
}

function fmtBytes(bytes: number): string {
  return bytes < 1024 ? `${Math.round(bytes)} B` : `${Math.round(bytes / 1024)} KB`;
}

function fmtFlops(flops: number): string {
  if (flops >= 1e6) return `${(flops / 1e6).toFixed(1)}M`;
  if (flops >= 1e3) return `${(flops / 1e3).toFixed(flops >= 1e5 ? 0 : 1)}k`;
  return `${Math.round(flops)}`;
}

function tflops(budget: Budget): string {
  const value = budget.flops / (micros(budget.total) * 1e6);
  return value.toFixed(value >= 1 ? 1 : 2);
}

function must<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`missing ${selector}`);
  return node;
}

/* ---------- 3D: two scenes, one camera ---------- */

const stage = must<HTMLElement>("#stage");
const camera = new THREE.PerspectiveCamera(38, 1, 0.08, 200);
camera.position.set(0, 17.2, 7.4);
const controls = new OrbitControls(camera, stage);
controls.target.set(0, 0.4, 0.5);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 3;
controls.maxDistance = 40;

interface Panel {
  workload: Workload;
  host: HTMLElement;
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  world: ReturnType<typeof createPrefillWorld>;
  fields: Record<"stage" | "flops" | "bytes" | "math" | "time", HTMLElement>;
}

function createPanel(workload: Workload): Panel {
  const section = must<HTMLElement>(`#panel-${workload.id}`);
  const budget = workload.budget;
  section.innerHTML = `
    <div class="canvas"></div>
    <header class="panel-head ${workload.id}">
      <p class="eyebrow">${workload.name}</p>
      <h2>${workload.shape}</h2>
      <p class="stage" data-field="stage"></p>
    </header>
    <dl class="stats">
      <div><dt>FLOPs done</dt><dd><b data-field="flops"></b> / ${fmtFlops(budget.flops)}</dd></div>
      <div><dt>HBM traffic</dt><dd><b data-field="bytes"></b> / ${fmtBytes(budget.inBytes + budget.outBytes)}</dd></div>
      <div><dt>Lanes doing math</dt><dd><b data-field="math"></b> of cycles so far</dd></div>
      <div><dt>Working SMs</dt><dd><b>${budget.activeSms}</b> / 64</dd></div>
      <div class="finish"><dt>Finished</dt><dd><b data-field="time">—</b></dd></div>
    </dl>
  `;
  const field = (name: string) => must<HTMLElement>(`#panel-${workload.id} [data-field="${name}"]`);
  const host = section.querySelector<HTMLElement>(".canvas")!;

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance", stencil: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
  renderer.setClearColor(0x08090d, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x08090d);
  scene.add(new THREE.AmbientLight(0x8d97a8, 0.55));
  scene.add(new THREE.HemisphereLight(0xb7c6da, 0x101218, 0.5));
  const key = new THREE.DirectionalLight(0xfff6ee, 2.35);
  key.position.set(7, 12, 8);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x7f96b4, 0.75);
  fill.position.set(-8, 6, -3);
  scene.add(fill);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();

  const world = workload.createWorld(scene, false, false);
  return {
    workload,
    host,
    renderer,
    scene,
    world,
    fields: { stage: field("stage"), flops: field("flops"), bytes: field("bytes"), math: field("math"), time: field("time") },
  };
}

const panels = WORKLOADS.map(createPanel);

function resize(): void {
  const { clientWidth: width, clientHeight: height } = panels[0].host;
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
  for (const panel of panels) panel.renderer.setSize(panel.host.clientWidth, panel.host.clientHeight);
}
window.addEventListener("resize", resize);
resize();

/* ---------- Gantt: both kernels on one cycle axis ---------- */

const lanes = must<HTMLElement>("#lanes");
const axis = must<HTMLElement>("#axis");
const playhead = must<HTMLElement>("#playhead");
const pct = (cycle: number) => `${(cycle / TOTAL) * 100}%`;

for (let tick = 0; tick <= TOTAL; tick += 1000) {
  axis.insertAdjacentHTML("beforeend", `<span style="left:${pct(tick)}">${tick.toLocaleString("en-US")}<em>${micros(tick).toFixed(2)} µs</em></span>`);
}

const veils: HTMLElement[] = [];
for (const workload of WORKLOADS) {
  const b = workload.budget;
  const seg = (cls: string, a: number, z: number, label = "") =>
    `<div class="seg ${cls}" style="left:${pct(a)};width:${pct(z - a)}">${label}</div>`;
  const math = b.math.map(([a, z]) => seg("math", a, z)).join("");
  lanes.insertBefore(
    Object.assign(document.createElement("div"), {
      className: `lane ${workload.id}`,
      innerHTML: `
        <b>${workload.name}</b>
        <div class="track">
          ${seg("mem", 0, b.hbmIn, "HBM → L2")}
          ${seg("sm", b.hbmIn, b.hbmIn + b.sm)}
          ${math}
          ${seg("mem", b.hbmIn + b.sm, b.total, "→ HBM")}
          <div class="end" style="left:${pct(b.total)}"></div>
          <div class="veil"></div>
        </div>
        <span class="total">${formatGpuTime(b.total)}</span>`,
    }),
    lanes.querySelector(".overlay"),
  );
  veils.push(lanes.querySelector<HTMLElement>(`.lane.${workload.id} .veil`)!);
}

const verdict = must<HTMLElement>("#verdict");
const perFlop = micros(DECODE.total) / DECODE.flops / (micros(PREFILL.total) / PREFILL.flops);
const verdictText =
  `Decode finished first, ${formatGpuTime(DECODE.total).split(" · ")[1]} vs ${formatGpuTime(PREFILL.total).split(" · ")[1]}, ` +
  `but did ${Math.round(PREFILL.flops / DECODE.flops)}× less math. Per FLOP it is ~${Math.round(perFlop)}× slower: ` +
  `${tflops(DECODE)} vs ${tflops(PREFILL)} TFLOP/s.`;

/* ---------- Playback ---------- */

const playButton = must<HTMLButtonElement>("#play");
const scrub = must<HTMLInputElement>("#scrub");
const clock = must<HTMLElement>("#clock");
scrub.max = String(DURATION);

const params = new URLSearchParams(location.search);
const initialT = Number(params.get("t"));
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let t = Number.isFinite(initialT) ? clamp(initialT, 0, DURATION) : 0;
let playing = !reduceMotion && !params.has("t");
let scrubbing = false;

function toggle(): void {
  if (t >= DURATION) t = 0;
  playing = !playing;
}

playButton.addEventListener("click", toggle);
scrub.addEventListener("pointerdown", () => (scrubbing = true));
scrub.addEventListener("pointerup", () => (scrubbing = false));
scrub.addEventListener("input", () => {
  t = Number(scrub.value);
  playing = false;
});
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.code === "Space") {
    event.preventDefault();
    toggle();
  } else if (event.code === "ArrowRight" || event.code === "ArrowLeft") {
    event.preventDefault();
    const step = (event.shiftKey ? 1 : 0.1) * (event.code === "ArrowRight" ? 1 : -1);
    t = clamp(t + step, 0, DURATION);
    playing = false;
  }
});

function renderHud(cycle: number): void {
  if (!scrubbing) scrub.value = t.toFixed(2);
  playButton.textContent = t >= DURATION && !playing ? "Replay" : playing ? "Pause" : "Play";
  clock.textContent = formatGpuTime(cycle);
  playhead.style.left = pct(cycle);
  verdict.textContent = cycle >= TOTAL ? verdictText : `${CYCLES_PER_SECOND} cycles per second on both sides`;
  verdict.classList.toggle("done", cycle >= TOTAL);

  panels.forEach((panel, index) => {
    const b = panel.workload.budget;
    const c = Math.min(cycle, b.total);
    const done = cycle >= b.total;
    const fields = panel.fields;
    fields.stage.textContent = panel.workload.stage(c);
    fields.flops.textContent = fmtFlops((mathDone(b, c) / mathCycles(b)) * b.flops);
    fields.bytes.textContent = fmtBytes(hbmBytes(b, c));
    fields.math.textContent = c > 0 ? `${Math.round((mathDone(b, c) / c) * 100)}%` : "—";
    fields.time.textContent = done ? `${formatGpuTime(b.total)} · ${tflops(b)} TFLOP/s` : "—";
    panel.host.parentElement!.classList.toggle("done", done);
    veils[index].style.left = pct(c);
  });
}

let last = performance.now();
let lastFrame = 0;
let lastHud = -Infinity;
function frame(now: number): void {
  requestAnimationFrame(frame);
  if (document.hidden || now - lastFrame < MIN_FRAME_INTERVAL) return;
  lastFrame = now;
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (playing) {
    t = Math.min(DURATION, t + dt);
    if (t >= DURATION) playing = false;
  }

  const cycle = cycleAt(t);
  controls.update();
  for (const panel of panels) {
    panel.world.update(panel.workload.sample(Math.min(cycle, panel.workload.budget.total)), null);
    panel.renderer.render(panel.scene, camera);
  }
  if (now - lastHud >= HUD_FRAME_INTERVAL || scrubbing) {
    renderHud(cycle);
    lastHud = now;
  }
}

frame(performance.now());
