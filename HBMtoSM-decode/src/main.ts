import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { smWorld } from "./layout";
import {
  blockAt,
  bytesPerActiveSm,
  cacheUsed,
  dataBytes,
  formatBytes,
  getModel,
  outputTileBytes,
  weightPanelBytes,
} from "./model/gemm";
import {
  DURATION,
  PHASES,
  sampleTimeline,
  workloadCaption,
  type TimelineSample,
} from "./model/timeline";
import { createWorld, type PickHit } from "./scene/world";
import { HUD_FRAME_INTERVAL, MAX_PIXEL_RATIO, MIN_FRAME_INTERVAL } from "../../src/render-performance";

const viewport = document.querySelector<HTMLDivElement>("#viewport");
if (!viewport) throw new Error("missing viewport");

const params = new URLSearchParams(location.search);
const initialT = Number(params.get("t"));
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
  stencil: false,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
renderer.setClearColor(0x08090d, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
viewport.appendChild(renderer.domElement);

const labels = new CSS2DRenderer();
labels.domElement.id = "labels";
viewport.appendChild(labels.domElement);

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

const camera = new THREE.PerspectiveCamera(40, 1, 0.08, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 1.15;
controls.maxDistance = 40;
controls.target.set(0, 1, 0);

const world = createWorld(scene, params.has("debug"));

const playButton = must<HTMLButtonElement>("#play");
const followButton = must<HTMLButtonElement>("#follow");
const scrub = must<HTMLInputElement>("#scrub");
const clock = must<HTMLElement>("#clock");
const phaseKicker = must<HTMLElement>("#phase-kicker");
const phaseTitle = must<HTMLElement>("#phase-title");
const phaseBody = must<HTMLElement>("#phase-body");
const eyebrow = must<HTMLElement>("#eyebrow");
const inspect = must<HTMLElement>("#inspect");
const inspectTitle = must<HTMLElement>("#inspect-title");
const inspectState = must<HTMLElement>("#inspect-state");
const inspectFacts = must<HTMLElement>("#inspect-facts");
const inspectNote = must<HTMLElement>("#inspect-note");
const inspectBack = must<HTMLButtonElement>("#inspect-back");
const marks = must<HTMLElement>("#marks");

eyebrow.textContent = workloadCaption();
scrub.max = String(DURATION);

let t = Number.isFinite(initialT) ? Math.min(DURATION, Math.max(0, initialT)) : 0;
let playing = !reduceMotion && !params.has("t");
let mode: "timeline" | "free" | "inspect" = "timeline";
let inspectTarget: PickHit | null = null;
let snapInspect = false;
const smParam = params.get("sm");
if (smParam) {
  const [row, col] = smParam.split(",").map(Number);
  if (Number.isInteger(row) && Number.isInteger(col) && row >= 0 && col >= 0 && row < 8 && col < 8) {
    inspectTarget = { row, col };
    mode = "inspect";
    playing = false;
    snapInspect = true;
  }
}
let scrubbing = false;
let pointer = { x: 0, y: 0, down: false, moved: false };

const desiredPos = new THREE.Vector3();
const desiredTarget = new THREE.Vector3();
const ndc = new THREE.Vector2();

for (const phase of PHASES) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = phase.short;
  button.addEventListener("click", () => {
    t = phase.t;
    mode = "timeline";
    inspectTarget = null;
  });
  marks.appendChild(button);
}

playButton.addEventListener("click", () => {
  if (t >= DURATION) t = 0;
  playing = !playing;
  if (playing) {
    mode = "timeline";
    inspectTarget = null;
  }
});

followButton.addEventListener("click", () => {
  mode = "timeline";
  inspectTarget = null;
});

inspectBack.addEventListener("click", () => {
  mode = "timeline";
  inspectTarget = null;
});

scrub.addEventListener("pointerdown", () => {
  scrubbing = true;
});
scrub.addEventListener("pointerup", () => {
  scrubbing = false;
});
scrub.addEventListener("input", () => {
  t = Number(scrub.value);
  playing = false;
  mode = "timeline";
  inspectTarget = null;
});

renderer.domElement.addEventListener("pointerdown", (event) => {
  pointer = { x: event.clientX, y: event.clientY, down: true, moved: false };
});
renderer.domElement.addEventListener("pointermove", (event) => {
  if (!pointer.down) {
    const hit = pickAt(event.clientX, event.clientY);
    renderer.domElement.style.cursor = hit ? "pointer" : "grab";
    return;
  }
  if (Math.abs(event.clientX - pointer.x) + Math.abs(event.clientY - pointer.y) > 5) {
    pointer.moved = true;
    mode = "free";
  }
});
renderer.domElement.addEventListener("pointerup", (event) => {
  if (pointer.down && !pointer.moved) {
    const hit = pickAt(event.clientX, event.clientY);
    if (hit) openInspect(hit);
  }
  pointer.down = false;
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (t >= DURATION) t = 0;
    playing = !playing;
    if (playing) {
      mode = "timeline";
      inspectTarget = null;
    }
  } else if (event.code === "ArrowRight") {
    event.preventDefault();
    nudge(event.shiftKey ? 1 : 0.1);
  } else if (event.code === "ArrowLeft") {
    event.preventDefault();
    nudge(event.shiftKey ? -1 : -0.1);
  } else if (event.code === "Escape") {
    mode = "timeline";
    inspectTarget = null;
  }
});

if (params.has("debug")) {
  const original = console.error.bind(console);
  console.error = (...args: unknown[]) => {
    const pre = document.createElement("pre");
    pre.className = "gl-error";
    pre.textContent = args.map(String).join(" ");
    document.body.appendChild(pre);
    original(...args);
  };
}

function nudge(delta: number): void {
  t = Math.min(DURATION, Math.max(0, t + delta));
  playing = false;
  mode = "timeline";
  inspectTarget = null;
}

function openInspect(hit: PickHit): void {
  inspectTarget = hit;
  mode = "inspect";
  playing = false;
}

function pickAt(clientX: number, clientY: number): PickHit | null {
  const rect = renderer.domElement.getBoundingClientRect();
  ndc.set(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1,
  );
  return world.pick(ndc, camera);
}

function resize(): void {
  const width = viewport?.clientWidth || window.innerWidth;
  const height = viewport?.clientHeight || window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  labels.setSize(width, height);
}

window.addEventListener("resize", resize);
resize();

let last = performance.now();
let lastFrame = 0;
let lastHud = -Infinity;
function frame(now: number): void {
  requestAnimationFrame(frame);
  if (document.hidden || now - lastFrame < MIN_FRAME_INTERVAL) return;
  lastFrame = now;
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (playing && mode === "timeline") {
    t = Math.min(DURATION, t + dt);
    if (t >= DURATION) playing = false;
  }

  const sample = sampleTimeline(t);
  world.update(sample, inspectTarget);
  applyCamera(sample, dt);
  if (now - lastHud >= HUD_FRAME_INTERVAL || scrubbing) {
    renderHud(sample);
    lastHud = now;
  }
  renderer.render(scene, camera);
  labels.render(scene, camera);
}

function applyCamera(sample: TimelineSample, dt: number): void {
  if (mode === "timeline") {
    desiredPos.set(...sample.camera.position);
    desiredTarget.set(...sample.camera.target);
    camera.position.copy(desiredPos);
    controls.target.copy(desiredTarget);
  } else if (mode === "inspect" && inspectTarget) {
    const [x, y, z] = smWorld(inspectTarget.row, inspectTarget.col);
    desiredPos.set(x + 0.85, y + 1.35, z + 1.55);
    desiredTarget.set(x, y + 0.28, z);
    const blend = snapInspect ? 1 : 1 - Math.exp(-5.5 * dt);
    snapInspect = false;
    camera.position.lerp(desiredPos, blend);
    controls.target.lerp(desiredTarget, blend);
  }
  controls.update();
}

function renderHud(sample: TimelineSample): void {
  if (!scrubbing) scrub.value = sample.t.toFixed(2);
  clock.textContent = `${formatTime(sample.t)} / ${formatTime(DURATION)}`;
  phaseKicker.textContent = sample.phase.short;
  phaseTitle.textContent = sample.phase.title;
  phaseBody.textContent = sample.phase.body;
  playButton.textContent = t >= DURATION && !playing ? "Replay" : playing ? "Pause" : "Play";
  followButton.hidden = mode === "timeline";
  setText("#a-status", sample.meters.a);
  setText("#b-status", sample.meters.b);
  setText("#c-status", sample.meters.c);
  setText("#gpu-clock", sample.gpu.label);
  document.querySelector("#gpu-clock")?.classList.toggle("paused", !sample.gpu.running);
  setWidth("#a-bar", sample.meters.aBar);
  setWidth("#b-bar", sample.meters.bBar);
  setWidth("#c-bar", sample.meters.cBar);

  const markButtons = marks.querySelectorAll("button");
  PHASES.forEach((phase, index) => {
    const next = PHASES[index + 1]?.t ?? DURATION + 0.001;
    markButtons[index]?.setAttribute("aria-current", sample.t >= phase.t && sample.t < next ? "true" : "false");
  });

  if (inspectTarget && mode === "inspect") {
    const block = blockAt(inspectTarget.row, inspectTarget.col);
    const live = sample.blocks[block.id];
    inspect.hidden = false;
    inspectTitle.textContent = `SM(${block.row},${block.col})`;
    inspectState.textContent = block.active
      ? `${live.state} · output segment ${block.col}`
      : `${live.state} · no work for this decode row`;
    const model = getModel();
    const wStart = block.wRegion.col;
    inspectFacts.innerHTML = `
      <dt>y segment</dt><dd>1×${model.tileN} · ${formatBytes(outputTileBytes())}</dd>
      <dt>x row</dt><dd>[0, :] · 1×${model.k} · ${formatBytes(dataBytes())}</dd>
      <dt>W panel</dt><dd>[:, ${wStart}:${wStart + model.tileN}] · ${model.k}×${model.tileN} · ${formatBytes(weightPanelBytes())}</dd>
      <dt>Traffic</dt><dd>${cacheUsed(bytesPerActiveSm())}</dd>
    `;
    inspectNote.textContent = block.active
      ? "This SM performs 16,384 FLOPs from roughly 33 KB of traffic: about 0.49 FLOP/byte."
      : "Only eight SMs are needed for the eight 1×32 output segments. This SM remains idle.";
  } else {
    inspect.hidden = true;
  }
}

function formatTime(seconds: number): string {
  const clamped = Math.max(0, seconds);
  const mins = Math.floor(clamped / 60);
  const rest = clamped - mins * 60;
  return `${mins}:${rest.toFixed(1).padStart(4, "0")}`;
}

function must<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`missing ${selector}`);
  return node;
}

function setText(selector: string, value: string): void {
  const node = document.querySelector(selector);
  if (node) node.textContent = value;
}

function setWidth(selector: string, value: number): void {
  const node = document.querySelector<HTMLElement>(selector);
  if (node) node.style.width = `${Math.round(value * 100)}%`;
}

frame(performance.now());
