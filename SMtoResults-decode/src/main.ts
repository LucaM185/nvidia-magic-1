import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import {
  ASSEMBLE_START,
  CHUNK_COUNT,
  COMPUTE_END,
  COMPUTE_START,
  DURATION,
  K,
  PHASES,
  RESIDENT_START,
  TILE_K,
  TENSOR_CYCLES,
  TOTAL_COMPUTE_CYCLES,
  WARP_COUNT,
  WRITE_START,
  clamp01,
  sampleTimeline,
  warpTiming,
  type TimelineSample,
} from "./timeline";
import "./style.css";

const X = new THREE.Color("#4c8dff");
const W = new THREE.Color("#f2a24a");
const Y = new THREE.Color("#3ddc97");
const ACTIVE = new THREE.Color("#c8f06a");
const MUTED = new THREE.Color("#566270");
const WARP_COLORS = [
  "#c8f06a", "#6dc8ff", "#d28cff", "#ff7fa7",
  "#ffc857", "#58e1c1", "#ff8c69", "#8ea1ff",
].map((color) => new THREE.Color(color));
const WARP_Z = Array.from({ length: WARP_COUNT }, (_, warp) => -2.22 + warp * (4.44 / (WARP_COUNT - 1)));
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const viewport = must<HTMLDivElement>("#viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.setClearColor(0x08090d, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
viewport.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.id = "labels";
viewport.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x08090d);
scene.fog = new THREE.FogExp2(0x08090d, 0.026);
scene.add(new THREE.AmbientLight(0x8d97a8, 0.52));
scene.add(new THREE.HemisphereLight(0xb7c6da, 0x101218, 0.62));
const key = new THREE.DirectionalLight(0xfff6ee, 2.65);
key.position.set(5, 11, 8);
scene.add(key);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
pmrem.dispose();

const camera = new THREE.PerspectiveCamera(39, 1, 0.08, 100);
camera.position.set(10.8, 10.1, 14.4);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-0.45, 0.12, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 8;
controls.maxDistance = 28;
controls.maxPolarAngle = Math.PI * 0.49;

function metal(color: number, roughness = 0.55, emissive = 0x000000): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.38, emissive, emissiveIntensity: 0.12 });
}

function box(width: number, height: number, depth: number, material: THREE.Material, radius = 0.08): THREE.Mesh {
  return new THREE.Mesh(new RoundedBoxGeometry(width, height, depth, 3, radius), material);
}

function label(html: string, position: THREE.Vector3, className = ""): CSS2DObject {
  const element = document.createElement("div");
  element.className = `tag ${className}`.trim();
  element.innerHTML = html;
  const object = new CSS2DObject(element);
  object.position.copy(position);
  object.center.set(0.5, 0);
  return object;
}

interface CellBank {
  materials: THREE.MeshStandardMaterial[];
  meshes: THREE.Mesh[];
  cols: number;
  rows: number;
}

function createCellBank(
  color: THREE.Color,
  center: THREE.Vector3,
  cols: number,
  rows: number,
  width: number,
  depth: number,
  opacity = 1,
): CellBank {
  const materials: THREE.MeshStandardMaterial[] = [];
  const meshes: THREE.Mesh[] = [];
  const cellW = width / cols;
  const cellD = depth / rows;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const material = new THREE.MeshStandardMaterial({
        color: color.clone().multiplyScalar(0.2),
        emissive: color,
        emissiveIntensity: 0.12,
        roughness: 0.38,
        metalness: 0.22,
        transparent: true,
        opacity,
      });
      const cell = box(cellW * 0.82, 0.055, cellD * 0.78, material, 0.018);
      cell.position.set(
        center.x - width / 2 + cellW * (col + 0.5),
        center.y,
        center.z - depth / 2 + cellD * (row + 0.5),
      );
      scene.add(cell);
      materials.push(material);
      meshes.push(cell);
    }
  }
  return { materials, meshes, cols, rows };
}

function setGlow(materials: THREE.MeshStandardMaterial[], color: THREE.Color, strength: number, opacity = 1): void {
  for (const material of materials) {
    material.color.copy(color).multiplyScalar(0.16 + strength * 0.3);
    material.emissive.copy(color);
    material.emissiveIntensity = 0.06 + strength * 0.92;
    material.opacity = opacity;
  }
}

function rail(points: THREE.Vector3[], color: THREE.Color): THREE.MeshBasicMaterial {
  const curve = new THREE.CatmullRomCurve3(points);
  const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.025 });
  scene.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 30, 0.022, 6, false), material));
  return material;
}

function packet(color: THREE.Color, width = 0.3, depth = 0.3): THREE.Mesh {
  const material = new THREE.MeshStandardMaterial({
    color: color.clone().multiplyScalar(0.48),
    emissive: color,
    emissiveIntensity: 1.15,
    roughness: 0.28,
    transparent: true,
    opacity: 0,
  });
  const mesh = box(width, 0.18, depth, material, 0.035);
  mesh.visible = false;
  scene.add(mesh);
  return mesh;
}

function movePacket(mesh: THREE.Mesh, from: THREE.Vector3, to: THREE.Vector3, progress: number): void {
  const u = clamp01(progress);
  mesh.visible = true;
  (mesh.material as THREE.MeshStandardMaterial).opacity = 1;
  mesh.position.lerpVectors(from, to, u);
  mesh.position.y += Math.sin(u * Math.PI) * 0.45;
  mesh.rotation.y = u * Math.PI * 1.5;
}

function hidePacket(mesh: THREE.Mesh): void {
  mesh.visible = false;
  (mesh.material as THREE.MeshStandardMaterial).opacity = 0;
}

const floor = box(14.5, 0.16, 7.4, metal(0x171b20, 0.72), 0.16);
floor.position.set(-0.45, -0.14, 0);
scene.add(floor);

const sharedX = -5.25;
const registerX = -0.85;
const coreX = 3.75;

const sharedBase = box(2.45, 0.34, 5.75, metal(0x252e37, 0.6), 0.1);
sharedBase.position.set(sharedX, 0.2, 0);
scene.add(sharedBase);
scene.add(label("<b>Shared Memory</b><span>1 KB x · 32 KB W · 128 B y</span>", new THREE.Vector3(sharedX, 0.02, 3.16), "w"));

const sharedInput = createCellBank(X, new THREE.Vector3(sharedX, 0.43, -2.1), 8, 1, 1.8, 0.34, 0.08);
const sharedWeights = createCellBank(W, new THREE.Vector3(sharedX, 0.43, 0), 8, 8, 1.8, 3.15, 0.08);
const sharedOutput = createCellBank(Y, new THREE.Vector3(sharedX, 0.43, 2.1), 8, 1, 1.8, 0.34, 0.08);
scene.add(label("<b>x[1×256]</b><span>8 K chunks</span>", new THREE.Vector3(sharedX, 0.66, -2.55), "x"));
scene.add(label("<b>W[256×32]</b><span>this SM's column panel</span>", new THREE.Vector3(sharedX, 0.66, -0.62), "w"));
scene.add(label("<b>y[1×32]</b><span>128-byte output slice</span>", new THREE.Vector3(sharedX, 0.66, 2.42), "y"));

const registerBase = box(3.3, 0.34, 5.75, metal(0x303946, 0.55), 0.1);
registerBase.position.set(registerX, 0.2, 0);
scene.add(registerBase);
scene.add(label("<b>Register File</b><span>32 accumulators stay resident</span>", new THREE.Vector3(registerX, 0.02, 3.16), "active"));

interface WarpVisual {
  owner: THREE.MeshStandardMaterial;
  xFragment: THREE.MeshStandardMaterial[];
  wFragment: THREE.MeshStandardMaterial[];
  accumulators: THREE.MeshStandardMaterial[];
  schedulerToken: THREE.MeshStandardMaterial;
}

const warps: WarpVisual[] = [];
for (let warp = 0; warp < WARP_COUNT; warp++) {
  const z = WARP_Z[warp];
  const warpColor = WARP_COLORS[warp];
  const owner = new THREE.MeshStandardMaterial({
    color: warpColor.clone().multiplyScalar(0.25),
    emissive: warpColor,
    emissiveIntensity: 0.08,
    roughness: 0.5,
    transparent: true,
    opacity: 0,
  });
  const band = box(2.95, 0.035, 0.42, owner, 0.035);
  band.position.set(registerX, 0.4, z);
  scene.add(band);

  const xFragment = createCellBank(X, new THREE.Vector3(registerX - 1.12, 0.46, z), 1, 1, 0.32, 0.26).materials;
  const wFragment = createCellBank(W, new THREE.Vector3(registerX - 0.47, 0.46, z), 3, 1, 0.76, 0.26).materials;
  const accumulators = createCellBank(Y, new THREE.Vector3(registerX + 0.94, 0.46, z), 4, 1, 1.05, 0.26).materials;

  const schedulerToken = metal(0x29313a, 0.42, warpColor.getHex());
  const token = new THREE.Mesh(new THREE.SphereGeometry(0.105, 18, 18), schedulerToken);
  token.position.set(registerX - 1.25 + warp * 0.355, 0.65, -2.83);
  scene.add(token);
  warps.push({ owner, xFragment, wFragment, accumulators, schedulerToken });
}

const schedulerMat = metal(0x46515e, 0.42, 0x1a2114);
const scheduler = box(2.85, 0.31, 0.7, schedulerMat, 0.1);
scheduler.position.set(registerX, 0.34, -2.76);
scene.add(scheduler);
const schedulerLabel = label("<b>Warp Scheduler</b><span>interleaves 8 output groups</span>", new THREE.Vector3(registerX, 0.9, -3.16), "active");
schedulerLabel.center.set(0.5, 1);
scene.add(schedulerLabel);

const coreMat = metal(0x56687a, 0.4, 0x1c2732);
const core = box(2.55, 0.48, 3.8, coreMat, 0.14);
core.position.set(coreX, 0.29, 0);
scene.add(core);
const coreGrid = createCellBank(MUTED, new THREE.Vector3(coreX, 0.58, 0), 8, 16, 1.85, 2.95);
scene.add(label("<b>Tensor Core path</b><span>M=1 · 1 useful row / 16</span>", new THREE.Vector3(coreX, 0.02, 2.18), "active"));

const loadRail = rail([
  new THREE.Vector3(sharedX + 1.2, 0.46, -0.9),
  new THREE.Vector3(-3.25, 0.92, 0),
  new THREE.Vector3(registerX - 1.55, 0.5, 0),
], new THREE.Color(0x98b9ff));
const computeRail = rail([
  new THREE.Vector3(registerX + 1.55, 0.5, 0),
  new THREE.Vector3(1.5, 0.94, 0),
  new THREE.Vector3(coreX - 1.25, 0.5, 0),
], ACTIVE);
const writeRail = rail([
  new THREE.Vector3(registerX + 0.95, 0.5, 1.7),
  new THREE.Vector3(-3.2, 0.95, 2.15),
  new THREE.Vector3(sharedX + 1.2, 0.5, 2.1),
], Y);

const xPacket = packet(X, 0.3, 0.24);
const wPacket = packet(W, 0.5, 0.3);
const instructionPacket = packet(ACTIVE, 0.48, 0.38);
const resultPacket = packet(Y, 0.62, 0.28);
const writePacket = packet(Y, 0.72, 0.26);

function pulse(progress: number): number {
  return Math.sin(clamp01(progress) * Math.PI) ** 2;
}

function updateScene(sample: TimelineSample): void {
  const panelOpacity = 0.08 + sample.panelProgress * 0.92;
  setGlow(sharedInput.materials, X, 0.16 + sample.panelProgress * 0.28, panelOpacity);
  setGlow(sharedWeights.materials, W, 0.1 + sample.panelProgress * 0.2, panelOpacity);
  setGlow(sharedOutput.materials, Y, sample.writeProgress, 0.08 + sample.writeProgress * 0.92);

  const computeActive = sample.t >= COMPUTE_START && sample.t < COMPUTE_END;
  for (let index = 0; index < sharedWeights.materials.length; index++) {
    const row = Math.floor(index / sharedWeights.cols);
    if (computeActive && row === sample.chunk) {
      sharedWeights.materials[index].emissiveIntensity = 1.05;
      sharedWeights.materials[index].opacity = 1;
    }
  }
  for (let index = 0; index < sharedInput.materials.length; index++) {
    if (computeActive && index === sample.chunk) sharedInput.materials[index].emissiveIntensity = 1.15;
  }

  for (let warp = 0; warp < WARP_COUNT; warp++) {
    const visual = warps[warp];
    const timing = warpTiming(sample, warp);
    const active = warp === sample.activeWarp;
    const inFlight = timing.stage === "tensor" || timing.stage === "accumulate";
    visual.owner.opacity = sample.residentProgress * (active ? 0.46 : inFlight ? 0.3 : 0.13);
    visual.owner.emissiveIntensity = active ? 0.78 : inFlight ? 0.38 : 0.07;
    visual.schedulerToken.emissiveIntensity = active ? 1.3 : inFlight ? 0.68 : sample.residentProgress * 0.25;
    setGlow(visual.xFragment, X, sample.residentProgress * (active ? 1 : inFlight ? 0.55 : 0.25));
    setGlow(visual.wFragment, W, sample.residentProgress * (active ? 1 : inFlight ? 0.55 : 0.25));
    setGlow(visual.accumulators, Y, sample.kProgress * (timing.stage === "accumulate" ? 1 : 0.62));
  }
  schedulerMat.emissive.copy(sample.activeWarp >= 0 ? WARP_COLORS[sample.activeWarp] : ACTIVE);
  schedulerMat.emissiveIntensity = sample.activeWarp >= 0 ? 0.62 : 0.1;

  const coreHot = computeActive && sample.tensorActive;
  coreMat.emissive.copy(coreHot ? ACTIVE : new THREE.Color(0x1c2732));
  coreMat.emissiveIntensity = coreHot ? 0.48 : 0.12;
  for (let index = 0; index < coreGrid.materials.length; index++) {
    const row = Math.floor(index / coreGrid.cols);
    const useful = row === 0;
    const material = coreGrid.materials[index];
    material.color.copy(useful ? ACTIVE : MUTED).multiplyScalar(useful ? 0.28 : 0.13);
    material.emissive.copy(useful ? ACTIVE : MUTED);
    material.emissiveIntensity = useful && coreHot ? 0.95 : useful ? 0.18 : 0.025;
    material.opacity = useful ? 1 : 0.32;
  }

  hidePacket(xPacket);
  hidePacket(wPacket);
  hidePacket(instructionPacket);
  hidePacket(resultPacket);
  hidePacket(writePacket);
  loadRail.opacity = 0.025;
  computeRail.opacity = 0.025;
  writeRail.opacity = 0.02;

  if (computeActive) {
    const z = sample.activeWarp >= 0 ? WARP_Z[sample.activeWarp] : 0;
    if (sample.activeWarp >= 0) {
      const timing = warpTiming(sample, sample.activeWarp);
      if (timing.stage === "load") {
        movePacket(xPacket, new THREE.Vector3(sharedX + 0.75, 0.54, -2.1), new THREE.Vector3(registerX - 1.12, 0.56, z), timing.progress);
        movePacket(wPacket, new THREE.Vector3(sharedX + 0.75, 0.54, 0), new THREE.Vector3(registerX - 0.47, 0.56, z), timing.progress);
        loadRail.opacity = 0.14 + pulse(timing.progress) * 0.8;
      } else if (timing.stage === "issue") {
        movePacket(instructionPacket, new THREE.Vector3(registerX + 1.25, 0.58, z), new THREE.Vector3(coreX, 0.72, 0), timing.progress);
        computeRail.opacity = 0.14 + pulse(timing.progress) * 0.82;
      }
    }
    if (sample.returnWarp >= 0) {
      movePacket(
        resultPacket,
        new THREE.Vector3(coreX, 0.72, 0),
        new THREE.Vector3(registerX + 0.94, 0.56, WARP_Z[sample.returnWarp]),
        sample.returnProgress,
      );
      computeRail.opacity = Math.max(computeRail.opacity, 0.14 + pulse(sample.returnProgress) * 0.82);
    }
  }

  if (sample.t >= WRITE_START && sample.writeProgress < 1) {
    movePacket(
      writePacket,
      new THREE.Vector3(registerX + 0.94, 0.58, 1.75),
      new THREE.Vector3(sharedX, 0.56, 2.1),
      sample.writeProgress,
    );
    writeRail.opacity = 0.12 + pulse(sample.writeProgress) * 0.85;
  }
}

const playButton = must<HTMLButtonElement>("#play");
const scrub = must<HTMLInputElement>("#scrub");
const clock = must<HTMLElement>("#clock");
const phaseKicker = must<HTMLElement>("#phase-kicker");
const phaseTitle = must<HTMLElement>("#phase-title");
const phaseBody = must<HTMLElement>("#phase-body");
const marks = must<HTMLElement>("#marks");
const lanes = must<HTMLElement>("#warp-lanes");
const schedulerCopy = must<HTMLElement>("#scheduler-copy");
const kValue = must<HTMLElement>("#k-value");
const kProgressElement = must<HTMLElement>("#k-progress");
const chunkReadout = must<HTMLElement>("#chunk-readout");
const chunks = must<HTMLElement>("#chunks");
const assembly = must<HTMLElement>(".assembly");
const smSlices = must<HTMLElement>("#sm-slices");

for (const phase of PHASES) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = phase.short;
  button.addEventListener("click", () => {
    t = phase.t;
    playing = false;
  });
  marks.appendChild(button);
}

for (let warp = 0; warp < WARP_COUNT; warp++) {
  const start = warp * 4;
  const row = document.createElement("div");
  row.className = "warp-row";
  row.style.setProperty("--warp", `#${WARP_COLORS[warp].getHexString()}`);
  row.innerHTML = `<b><i></i>W${warp}</b><span>W[:,${start}:${start + 4}] → y[${start}:${start + 4}]</span><span class="warp-state">not resident</span>`;
  lanes.appendChild(row);
}

for (let chunk = 0; chunk < CHUNK_COUNT; chunk++) {
  const item = document.createElement("div");
  item.className = "chunk";
  item.innerHTML = `<span>${chunk * TILE_K}–${(chunk + 1) * TILE_K - 1}</span>`;
  chunks.appendChild(item);
}

for (let sm = 0; sm < 8; sm++) {
  const slice = document.createElement("i");
  slice.className = "sm-slice";
  slice.title = `SM ${sm} · y[${sm * 32}:${(sm + 1) * 32}]`;
  smSlices.appendChild(slice);
}

const params = new URLSearchParams(location.search);
const initial = Number(params.get("t"));
let t = Number.isFinite(initial) ? Math.min(DURATION, Math.max(0, initial)) : 0;
let playing = !reduceMotion && !params.has("t");
let scrubbing = false;
scrub.max = String(DURATION);

playButton.addEventListener("click", () => {
  if (t >= DURATION) t = 0;
  playing = !playing;
});
scrub.addEventListener("pointerdown", () => { scrubbing = true; });
scrub.addEventListener("pointerup", () => { scrubbing = false; });
scrub.addEventListener("input", () => {
  t = Number(scrub.value);
  playing = false;
});
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (t >= DURATION) t = 0;
    playing = !playing;
  }
  if (event.code === "ArrowLeft") {
    event.preventDefault();
    t = Math.max(0, t - 0.5);
    playing = false;
  }
  if (event.code === "ArrowRight") {
    event.preventDefault();
    t = Math.min(DURATION, t + 0.5);
    playing = false;
  }
});

function renderHud(sample: TimelineSample): void {
  const phaseIndex = PHASES.indexOf(sample.phase);
  phaseKicker.textContent = `Step ${phaseIndex + 1} / ${PHASES.length} · ${sample.phase.short}`;
  phaseTitle.textContent = sample.phase.title;
  phaseBody.textContent = sample.phase.body;
  if (!scrubbing) scrub.value = sample.t.toFixed(2);
  clock.textContent = `${Math.floor(sample.t / 60)}:${(sample.t % 60).toFixed(1).padStart(4, "0")}`;
  playButton.textContent = sample.t >= DURATION && !playing ? "Replay" : playing ? "Pause" : "Play";

  const markButtons = marks.querySelectorAll("button");
  PHASES.forEach((phase, index) => {
    const next = PHASES[index + 1]?.t ?? DURATION + 1;
    markButtons[index]?.setAttribute("aria-current", sample.t >= phase.t && sample.t < next ? "true" : "false");
  });

  const rows = lanes.querySelectorAll<HTMLElement>(".warp-row");
  rows.forEach((row, warp) => {
    const timing = warpTiming(sample, warp);
    const active = warp === sample.activeWarp;
    row.classList.toggle("active", active);
    row.classList.toggle("in-flight", timing.stage === "tensor");
    const state = row.querySelector<HTMLElement>(".warp-state");
    if (!state) return;
    if (sample.t < RESIDENT_START) state.textContent = "not resident";
    else if (sample.t < COMPUTE_START) state.textContent = "ready";
    else if (sample.t < COMPUTE_END) {
      if (timing.stage === "load") state.textContent = "load x + W";
      else if (timing.stage === "issue") state.textContent = "issue MMA";
      else if (timing.stage === "tensor") state.textContent = `Tensor ${timing.tensorCycle}/${TENSOR_CYCLES}`;
      else if (timing.stage === "accumulate") state.textContent = "return + accumulate";
      else if (timing.stage === "done") state.textContent = "chunk done";
      else state.textContent = "queued";
    }
    else if (sample.t < WRITE_START) state.textContent = "complete";
    else state.textContent = "stored";
  });
  const inFlightCount = Array.from({ length: WARP_COUNT }, (_, warp) => warpTiming(sample, warp))
    .filter((timing) => timing.stage === "tensor").length;
  schedulerCopy.textContent = sample.activeWarp >= 0
    ? `W${sample.activeWarp} · ${warpTiming(sample, sample.activeWarp).stage} · ${inFlightCount} warp${inFlightCount === 1 ? "" : "s"} in flight`
    : sample.t >= WRITE_START
      ? "All 32 accumulators complete"
      : sample.tensorActive
        ? `${inFlightCount} warps remain in the 8-cycle Tensor stage`
        : "Waiting for the panel";

  const kDone = Math.min(K, Math.round(sample.kProgress * K));
  kValue.textContent = `cycle ${Math.floor(sample.computeCycle)} / ${TOTAL_COMPUTE_CYCLES}`;
  kProgressElement.style.width = `${sample.kProgress * 100}%`;
  must<HTMLElement>("#k-note").textContent = `K ${kDone}/${K} · Tensor latency ${TENSOR_CYCLES} cycles`;
  chunkReadout.textContent = sample.t < COMPUTE_START
    ? "waiting"
    : sample.t >= COMPUTE_END
      ? "8 / 8 complete"
      : `chunk ${sample.chunk + 1} / ${CHUNK_COUNT}`;

  const chunkNodes = chunks.querySelectorAll<HTMLElement>(".chunk");
  chunkNodes.forEach((node, index) => {
    const done = sample.kProgress >= 1 || index < sample.chunk && sample.t >= COMPUTE_START;
    const active = sample.t >= COMPUTE_START && sample.t < COMPUTE_END && index === sample.chunk;
    node.classList.toggle("done", done);
    node.classList.toggle("active", active);
    const fill = done ? 100 : active ? sample.chunkProgress * 100 : 0;
    node.style.setProperty("--fill", `${fill}%`);
  });

  assembly.classList.toggle("live", sample.t >= ASSEMBLE_START);
  const sliceNodes = smSlices.querySelectorAll<HTMLElement>(".sm-slice");
  sliceNodes.forEach((slice, index) => {
    const threshold = (index + 1) / Math.max(1, sliceNodes.length);
    slice.classList.toggle("ready", sample.assembleProgress >= threshold || sample.t >= DURATION);
  });
}

function resize(): void {
  const width = viewport.clientWidth || innerWidth;
  const height = viewport.clientHeight || innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  labelRenderer.setSize(width, height);
}
window.addEventListener("resize", resize);
resize();

let last = performance.now();
function frame(now: number): void {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (playing && !document.hidden) {
    t = Math.min(DURATION, t + dt);
    if (t >= DURATION) playing = false;
  }
  const sample = sampleTimeline(t);
  updateScene(sample);
  renderHud(sample);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  requestAnimationFrame(frame);
}

function must<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`missing ${selector}`);
  return node;
}

frame(performance.now());
