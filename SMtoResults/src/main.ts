import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import {
  DURATION,
  ISSUE_CYCLES,
  LOAD_STARTS,
  PHASES,
  SHARED_LATENCY,
  TENSOR_DURATION,
  TENSOR_ISSUE,
  TOTAL_CYCLES,
  WARP_COUNT,
  clamp01,
  cycleAt,
  phaseAt,
  pulse,
} from "./timeline";
import "./style.css";

const A = new THREE.Color("#4c8dff");
const B = new THREE.Color("#f2a24a");
const C = new THREE.Color("#3ddc97");
const ACTIVE = new THREE.Color("#c8f06a");
const WARP_COLORS = [
  "#c8f06a", "#6dc8ff", "#d28cff", "#ff7fa7", "#ffc857", "#58e1c1",
  "#ff8c69", "#8ea1ff", "#9ee85f", "#57b8ff", "#e889ff", "#ffad5c",
].map((color) => new THREE.Color(color));
const WARP_Z = Array.from({ length: WARP_COUNT }, (_, warp) => -2.48 + warp * (4.96 / (WARP_COUNT - 1)));
const WRITEBACK_START = 160;
const WRITEBACK_SPACING = 2;
const WRITEBACK_DURATION = 8;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const viewport = must<HTMLDivElement>("#viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.4));
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
scene.add(new THREE.AmbientLight(0x8d97a8, 0.56));
scene.add(new THREE.HemisphereLight(0xb7c6da, 0x101218, 0.64));
const key = new THREE.DirectionalLight(0xfff6ee, 2.65);
key.position.set(5, 11, 8);
scene.add(key);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
pmrem.dispose();

const camera = new THREE.PerspectiveCamera(39, 1, 0.08, 100);
camera.position.set(10.6, 10.3, 14.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(-0.55, 0.15, 0);
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

const floor = box(13.7, 0.16, 7.6, metal(0x171b20, 0.72), 0.16);
floor.position.set(-0.7, -0.14, 0);
scene.add(floor);

const sharedX = -5.15;
const registerX = -1.05;
const coreX = 3.35;
const sharedBase = box(2.35, 0.34, 5.85, metal(0x252e37, 0.6), 0.1);
sharedBase.position.set(sharedX, 0.2, 0);
scene.add(sharedBase);
scene.add(label("<b>Shared Memory</b><span>hardware · A / B / C</span>", new THREE.Vector3(sharedX, 0.02, 3.2), "a"));

interface CellBank { materials: THREE.MeshStandardMaterial[]; }

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
  const cellW = width / cols;
  const cellD = depth / rows;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const material = new THREE.MeshStandardMaterial({
        color: color.clone().multiplyScalar(0.2), emissive: color, emissiveIntensity: 0.14,
        roughness: 0.38, metalness: 0.22, transparent: opacity < 1, opacity,
      });
      const cell = box(cellW * 0.82, 0.055, cellD * 0.78, material, 0.018);
      cell.position.set(center.x - width / 2 + cellW * (col + 0.5), center.y, center.z - depth / 2 + cellD * (row + 0.5));
      scene.add(cell);
      materials.push(material);
    }
  }
  return { materials };
}

const sharedA = createCellBank(A, new THREE.Vector3(sharedX, 0.42, -1.78), 6, 3, 1.75, 1.18);
const sharedB = createCellBank(B, new THREE.Vector3(sharedX, 0.42, 0), 6, 3, 1.75, 1.18);
const sharedC = createCellBank(C, new THREE.Vector3(sharedX, 0.42, 1.78), 6, 2, 1.75, 1.18, 0.25);

const registerBase = box(3.15, 0.34, 5.85, metal(0x303946, 0.55), 0.1);
registerBase.position.set(registerX, 0.2, 0);
scene.add(registerBase);
scene.add(label("<b>Register File</b><span>hardware condiviso · contesti separati</span>", new THREE.Vector3(registerX, 0.02, 3.2), "active"));

interface WarpVisual {
  owner: THREE.MeshStandardMaterial;
  operandsA: THREE.MeshStandardMaterial[];
  operandsB: THREE.MeshStandardMaterial[];
  accumulators: THREE.MeshStandardMaterial[];
  schedulerToken: THREE.MeshStandardMaterial;
  loadRail: THREE.MeshBasicMaterial;
  mmaRail: THREE.MeshBasicMaterial;
  writeRail: THREE.MeshBasicMaterial;
  packetA: THREE.Mesh;
  packetB: THREE.Mesh;
  instruction: THREE.Mesh;
  result: THREE.Mesh;
}

function rail(points: THREE.Vector3[], color: THREE.Color): THREE.MeshBasicMaterial {
  const curve = new THREE.CatmullRomCurve3(points);
  const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08 });
  scene.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 24, 0.022, 6, false), material));
  return material;
}

function packet(color: THREE.Color, scale = 1): THREE.Mesh {
  const material = new THREE.MeshStandardMaterial({
    color: color.clone().multiplyScalar(0.46), emissive: color, emissiveIntensity: 1.15,
    roughness: 0.3, metalness: 0.15, transparent: true, opacity: 0,
  });
  const mesh = box(0.3 * scale, 0.18, 0.3 * scale, material, 0.035);
  mesh.visible = false;
  scene.add(mesh);
  return mesh;
}

const warps: WarpVisual[] = [];
for (let warp = 0; warp < WARP_COUNT; warp++) {
  const z = WARP_Z[warp];
  const warpColor = WARP_COLORS[warp];
  const owner = new THREE.MeshStandardMaterial({
    color: warpColor.clone().multiplyScalar(0.25), emissive: warpColor, emissiveIntensity: 0.08,
    roughness: 0.5, metalness: 0.18, transparent: true, opacity: 0,
  });
  const ownerBand = box(2.82, 0.035, 0.36, owner, 0.035);
  ownerBand.position.set(registerX, 0.4, z);
  scene.add(ownerBand);

  const operandsA = createCellBank(A, new THREE.Vector3(registerX - 0.94, 0.46, z), 4, 1, 0.72, 0.24).materials;
  const operandsB = createCellBank(B, new THREE.Vector3(registerX - 0.08, 0.46, z), 4, 1, 0.72, 0.24).materials;
  const accumulators = createCellBank(C, new THREE.Vector3(registerX + 0.93, 0.46, z), 4, 1, 0.88, 0.24).materials;

  const schedulerToken = metal(0x29313a, 0.42, warpColor.getHex());
  const token = new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 20), schedulerToken);
  token.position.set(registerX - 0.98 + (warp % 6) * 0.39, 0.65, -3.27 + Math.floor(warp / 6) * 0.3);
  scene.add(token);

  const loadRail = rail([
    new THREE.Vector3(sharedX + 1.18, 0.39, -0.85), new THREE.Vector3(-3.45, 0.82, z),
    new THREE.Vector3(registerX - 1.58, 0.47, z),
  ], warpColor);
  const mmaRail = rail([
    new THREE.Vector3(registerX + 1.58, 0.47, z), new THREE.Vector3(1.45, 0.86, z * 0.34),
    new THREE.Vector3(coreX - 1.25, 0.48, 0),
  ], warpColor);
  const writeRail = rail([
    new THREE.Vector3(registerX - 0.1, 0.5, z), new THREE.Vector3(-3.25, 1.05, 2.1),
    new THREE.Vector3(sharedX + 1.18, 0.47, 1.78),
  ], warpColor);
  warps.push({
    owner, operandsA, operandsB, accumulators, schedulerToken, loadRail, mmaRail, writeRail,
    packetA: packet(A), packetB: packet(B), instruction: packet(warpColor, 1.15), result: packet(C, 1.25),
  });
}

const schedulerMat = metal(0x46515e, 0.42, 0x1a2114);
const scheduler = box(2.65, 0.32, 0.84, schedulerMat, 0.1);
scheduler.position.set(registerX, 0.34, -3.12);
scene.add(scheduler);
const schedulerLabel = label("<b>Warp Scheduler</b><span>hardware · seleziona contesti</span>", new THREE.Vector3(registerX, 0.95, -3.62), "active");
schedulerLabel.center.set(0.5, 1);
scene.add(schedulerLabel);

const tensorMat = metal(0x56687a, 0.4, 0x1c2732);
const tensor = box(2.35, 0.48, 3.15, tensorMat, 0.14);
tensor.position.set(coreX, 0.29, 0);
scene.add(tensor);
const tensorGrid = createCellBank(ACTIVE, new THREE.Vector3(coreX, 0.58, 0), 5, 5, 1.65, 2.18);
scene.add(label("<b>Tensor Core</b><span>hardware · mma.sync</span>", new THREE.Vector3(coreX, 0.02, 1.85), "active"));

function setCellGlow(materials: THREE.MeshStandardMaterial[], color: THREE.Color, strength: number): void {
  for (const material of materials) {
    material.color.copy(color).multiplyScalar(0.18 + strength * 0.28);
    material.emissive.copy(color);
    material.emissiveIntensity = 0.08 + strength * 0.95;
  }
}

function move(mesh: THREE.Mesh, from: THREE.Vector3, to: THREE.Vector3, progress: number, opacity = 1): void {
  const u = clamp01(progress);
  mesh.visible = opacity > 0.02;
  (mesh.material as THREE.MeshStandardMaterial).opacity = opacity;
  mesh.position.lerpVectors(from, to, u);
  mesh.position.y += Math.sin(u * Math.PI) * 0.5;
  mesh.rotation.y = u * Math.PI * 1.5;
}

function hidePackets(visual: WarpVisual): void {
  visual.packetA.visible = false;
  visual.packetB.visible = false;
  visual.instruction.visible = false;
  visual.result.visible = false;
}

function showLoad(warp: number, progress: number): void {
  const visual = warps[warp];
  const z = WARP_Z[warp];
  move(visual.packetA, new THREE.Vector3(sharedX + 0.75, 0.52, -1.78), new THREE.Vector3(registerX - 0.94, 0.55, z), progress);
  move(visual.packetB, new THREE.Vector3(sharedX + 0.75, 0.52, 0), new THREE.Vector3(registerX - 0.08, 0.55, z), progress);
  visual.loadRail.opacity = 0.12 + pulse(progress, 0, 1) * 0.88;
  setCellGlow(visual.operandsA, A, progress);
  setCellGlow(visual.operandsB, B, progress);
}

function tensorSlot(warp: number): THREE.Vector3 {
  const cols = 4;
  const rows = 3;
  const width = 1.4;
  const depth = 1.55;
  const col = warp % cols;
  const row = Math.floor(warp / cols);
  return new THREE.Vector3(
    coreX - width / 2 + (width / cols) * (col + 0.5),
    0.74,
    -depth / 2 + (depth / rows) * (row + 0.5),
  );
}

function park(mesh: THREE.Mesh, at: THREE.Vector3): void {
  mesh.visible = true;
  mesh.position.copy(at);
  mesh.rotation.set(0, 0, 0);
  (mesh.material as THREE.MeshStandardMaterial).opacity = 1;
}

function showMma(warp: number, cycle: number): void {
  const visual = warps[warp];
  const elapsed = cycle - (LOAD_STARTS[warp] + SHARED_LATENCY);
  const departFrom = new THREE.Vector3(registerX + 1.2, 0.58, WARP_Z[warp]);
  const arriveAt = new THREE.Vector3(registerX + 0.93, 0.58, WARP_Z[warp]);
  const slot = tensorSlot(warp);
  const holdEnd = TENSOR_ISSUE + TENSOR_DURATION;
  if (elapsed < TENSOR_ISSUE) {
    move(visual.instruction, departFrom, slot, elapsed / TENSOR_ISSUE);
    visual.mmaRail.opacity = 0.15 + pulse(elapsed, 0, TENSOR_ISSUE) * 0.85;
    return;
  }
  if (elapsed < holdEnd) {
    park(visual.instruction, slot);
    (visual.instruction.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.9 + Math.sin(cycle * 1.6 + warp) * 0.18;
    visual.mmaRail.opacity = 0.07;
    return;
  }
  move(visual.instruction, slot, arriveAt, (elapsed - holdEnd) / TENSOR_ISSUE);
  visual.mmaRail.opacity = 0.15 + pulse(elapsed - holdEnd, 0, TENSOR_ISSUE) * 0.85;
}

function selectedWarpAt(cycle: number): number {
  if (cycle >= 10 && cycle < 12) return 0;
  for (let warp = 0; warp < WARP_COUNT; warp++) {
    const mmaStart = LOAD_STARTS[warp] + SHARED_LATENCY;
    const returnStart = mmaStart + TENSOR_ISSUE + TENSOR_DURATION;
    if (cycle >= mmaStart && cycle < mmaStart + TENSOR_ISSUE) return warp;
    if (cycle >= returnStart && cycle < returnStart + TENSOR_ISSUE) return warp;
  }
  for (let warp = 0; warp < WARP_COUNT; warp++) {
    if (cycle >= LOAD_STARTS[warp] && cycle < LOAD_STARTS[warp] + ISSUE_CYCLES) return warp;
  }
  for (let warp = 0; warp < WARP_COUNT; warp++) {
    const writeStart = WRITEBACK_START + warp * WRITEBACK_SPACING;
    if (cycle >= writeStart && cycle < writeStart + 2) return warp;
  }
  return -1;
}

function updateScene(t: number): void {
  const phase = phaseAt(t);
  const phaseIndex = PHASES.indexOf(phase);
  const cycle = cycleAt(t);
  const contextsVisible = phaseIndex >= 1;
  let selected = selectedWarpAt(cycle);

  tensorMat.emissive.copy(new THREE.Color(0x1c2732));
  tensorMat.emissiveIntensity = 0.12;
  for (const material of tensorGrid.materials) material.emissiveIntensity = 0.08;
  for (const visual of warps) {
    hidePackets(visual);
    setCellGlow(visual.operandsA, A, 0);
    setCellGlow(visual.operandsB, B, 0);
    setCellGlow(visual.accumulators, C, 0);
    visual.owner.opacity = contextsVisible ? 0.13 : 0;
    visual.owner.emissiveIntensity = 0.07;
    visual.loadRail.opacity = 0.015;
    visual.mmaRail.opacity = 0.015;
    visual.writeRail.opacity = 0.01;
    visual.schedulerToken.emissiveIntensity = contextsVisible ? 0.28 : 0.02;
  }

  let tensorResident = false;
  for (let warp = 0; warp < WARP_COUNT; warp++) {
    const loadStart = LOAD_STARTS[warp];
    const ready = loadStart + SHARED_LATENCY;
    const mmaEnd = ready + TENSOR_ISSUE + TENSOR_DURATION + TENSOR_ISSUE;

    if (cycle >= loadStart && cycle < ready) showLoad(warp, (cycle - loadStart) / SHARED_LATENCY);
    if (cycle >= ready) {
      setCellGlow(warps[warp].operandsA, A, 1);
      setCellGlow(warps[warp].operandsB, B, 1);
    }
    if (cycle >= ready && cycle < mmaEnd) {
      showMma(warp, cycle);
      tensorResident = true;
    }
    if (cycle >= mmaEnd) setCellGlow(warps[warp].accumulators, C, 1);
  }
  if (tensorResident) {
    tensorMat.emissive.copy(ACTIVE);
    tensorMat.emissiveIntensity = 0.4;
    for (const material of tensorGrid.materials) {
      material.emissive.copy(ACTIVE);
      material.emissiveIntensity = 0.32;
    }
  }

  if (phase.id === "writeback") {
    for (let warp = 0; warp < WARP_COUNT; warp++) {
      const writeStart = WRITEBACK_START + warp * WRITEBACK_SPACING;
      const progress = clamp01((cycle - writeStart) / WRITEBACK_DURATION);
      const visual = warps[warp];
      move(
        visual.result,
        new THREE.Vector3(registerX + 0.93, 0.58, WARP_Z[warp]),
        new THREE.Vector3(sharedX - 0.72 + (warp % 6) * 0.29, 0.55, 1.55 + Math.floor(warp / 6) * 0.46),
        progress,
        progress > 0 && progress < 0.995 ? 1 : 0,
      );
      visual.writeRail.opacity = 0.08 + pulse(progress, 0, 1) * 0.9;
    }
  }

  for (let warp = 0; warp < WARP_COUNT; warp++) {
    const active = warp === selected;
    warps[warp].owner.opacity = contextsVisible ? active ? 0.42 : 0.13 : 0;
    warps[warp].owner.emissiveIntensity = active ? 0.72 : 0.07;
    warps[warp].schedulerToken.emissiveIntensity = active ? 1.25 : contextsVisible ? 0.28 : 0.02;
  }
  schedulerMat.emissive.copy(selected >= 0 ? WARP_COLORS[selected] : ACTIVE);
  schedulerMat.emissiveIntensity = selected >= 0 ? 0.58 : 0.1;

  const residentGlow = 0.18 + Math.sin(t * 2.1) * 0.05;
  for (const material of sharedA.materials) material.emissiveIntensity = residentGlow;
  for (const material of sharedB.materials) material.emissiveIntensity = residentGlow;
  for (let index = 0; index < sharedC.materials.length; index++) {
    const warp = index;
    const progress = phase.id === "writeback"
      ? clamp01((cycle - (WRITEBACK_START + warp * WRITEBACK_SPACING)) / WRITEBACK_DURATION)
      : 0;
    sharedC.materials[index].opacity = 0.2 + progress * 0.8;
    sharedC.materials[index].emissiveIntensity = 0.05 + progress * 0.82;
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
const kProgressEl = must<HTMLElement>("#k-progress");
const cycleTimeline = must<HTMLElement>(".cycle-timeline");
const cycleScale = must<HTMLElement>("#cycle-scale");
const cycleRows = must<HTMLElement>("#cycle-rows");
const cyclePlayhead = must<HTMLElement>("#cycle-playhead");
const cycleReadout = must<HTMLElement>("#cycle-readout");
const params = new URLSearchParams(location.search);
const initial = Number(params.get("t"));
let t = Number.isFinite(initial) ? clamp01(initial / DURATION) * DURATION : 0;
let playing = !reduceMotion && !params.has("t");
let scrubbing = false;
scrub.max = String(DURATION);

for (const phase of PHASES) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = phase.short;
  button.addEventListener("click", () => { t = phase.t; playing = false; });
  marks.appendChild(button);
}
for (let warp = 0; warp < WARP_COUNT; warp++) {
  const row = document.createElement("div");
  row.className = "warp-row";
  row.style.setProperty("--warp", `#${WARP_COLORS[warp].getHexString()}`);
  row.innerHTML = `<b><i></i>W${warp}</b><span class="warp-state">—</span><span class="warp-op">—</span>`;
  lanes.appendChild(row);
}
for (let tick = 0; tick <= TOTAL_CYCLES; tick += 20) {
  const label = document.createElement("span");
  label.textContent = String(tick);
  label.style.left = `${(tick / TOTAL_CYCLES) * 100}%`;
  cycleScale.appendChild(label);
}
if (TOTAL_CYCLES % 20 !== 0) {
  const label = document.createElement("span");
  label.textContent = String(TOTAL_CYCLES);
  label.style.left = "100%";
  cycleScale.appendChild(label);
}
for (let warp = 0; warp < WARP_COUNT; warp++) {
  const row = document.createElement("div");
  row.className = "cycle-row";
  row.style.setProperty("--warp", `#${WARP_COLORS[warp].getHexString()}`);
  row.innerHTML = `<b>W${warp}</b>`;
  const loadStart = LOAD_STARTS[warp];
  const segments = [
    { start: loadStart, duration: ISSUE_CYCLES, className: "issue", label: "issue" },
    { start: loadStart + ISSUE_CYCLES, duration: SHARED_LATENCY - ISSUE_CYCLES, className: "wait", label: "wait" },
    { start: loadStart + SHARED_LATENCY, duration: TENSOR_ISSUE, className: "issue", label: "" },
    { start: loadStart + SHARED_LATENCY + TENSOR_ISSUE, duration: TENSOR_DURATION, className: "mma", label: "TC" },
    { start: loadStart + SHARED_LATENCY + TENSOR_ISSUE + TENSOR_DURATION, duration: TENSOR_ISSUE, className: "issue", label: "" },
    { start: WRITEBACK_START + warp * WRITEBACK_SPACING, duration: WRITEBACK_DURATION, className: "write", label: "WB" },
  ];
  for (const segment of segments) {
    const bar = document.createElement("i");
    bar.className = `cycle-segment ${segment.className}`;
    bar.textContent = segment.label;
    bar.style.left = `${(segment.start / TOTAL_CYCLES) * 100}%`;
    bar.style.width = `${(segment.duration / TOTAL_CYCLES) * 100}%`;
    row.appendChild(bar);
  }
  cycleRows.appendChild(row);
}

playButton.addEventListener("click", () => { if (t >= DURATION) t = 0; playing = !playing; });
scrub.addEventListener("pointerdown", () => { scrubbing = true; });
scrub.addEventListener("pointerup", () => { scrubbing = false; });
scrub.addEventListener("input", () => { t = Number(scrub.value); playing = false; });
window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault(); if (t >= DURATION) t = 0; playing = !playing;
  } else if (event.code === "ArrowRight") {
    event.preventDefault(); playing = false; t = Math.min(DURATION, t + (event.shiftKey ? 1 : 0.1));
  } else if (event.code === "ArrowLeft") {
    event.preventDefault(); playing = false; t = Math.max(0, t - (event.shiftKey ? 1 : 0.1));
  }
});

interface WarpState { state: string; op: string; active?: boolean; }

function statesAt(time: number): WarpState[] {
  const cycle = cycleAt(time);
  if (cycle < 8) return Array.from({ length: WARP_COUNT }, () => ({ state: "not shown", op: "software overlay" }));
  const states: WarpState[] = [];
  for (let warp = 0; warp < WARP_COUNT; warp++) {
    const loadStart = LOAD_STARTS[warp];
    const issueEnd = loadStart + ISSUE_CYCLES;
    const ready = loadStart + SHARED_LATENCY;
    const holdEnd = ready + TENSOR_ISSUE + TENSOR_DURATION;
    const mmaEnd = holdEnd + TENSOR_ISSUE;
    let state: WarpState;
    if (cycle < loadStart) state = { state: "ready", op: "—" };
    else if (cycle < issueEnd) state = { state: "issuing", op: `ldmatrix · ${ISSUE_CYCLES} cyc` };
    else if (cycle < ready) state = { state: "waiting", op: `shared · ${Math.ceil(ready - cycle)} cyc` };
    else if (cycle < ready + TENSOR_ISSUE) state = { state: "issuing", op: "registri → TC" };
    else if (cycle < holdEnd) state = {
      state: "in flight",
      op: `Tensor · ${Math.ceil(holdEnd - cycle)} cyc`,
    };
    else if (cycle < mmaEnd) state = { state: "issuing", op: "TC → registri" };
    else state = { state: "complete", op: "C in registers" };

    const writeStart = WRITEBACK_START + warp * WRITEBACK_SPACING;
    if (cycle >= writeStart) state = {
      state: cycle < writeStart + WRITEBACK_DURATION ? "writing" : "stored",
      op: cycle < writeStart + WRITEBACK_DURATION
        ? `shared · ${Math.ceil(writeStart + WRITEBACK_DURATION - cycle)} cyc`
        : "C in shared",
    };
    states.push(state);
  }
  const selected = selectedWarpAt(cycle);
  if (selected >= 0) states[selected].active = true;
  return states;
}

function renderHud(): void {
  const phase = phaseAt(t);
  const phaseIndex = PHASES.indexOf(phase);
  document.querySelector<HTMLElement>(".pipeline")?.classList.toggle("software-hidden", phaseIndex === 0);
  phaseKicker.textContent = `STEP ${phaseIndex + 1} / ${PHASES.length} · ${phase.short}`;
  phaseTitle.textContent = phase.title;
  phaseBody.textContent = phase.body;
  phaseBody.hidden = cycleAt(t) > 0;
  if (!scrubbing) scrub.value = t.toFixed(2);
  const cycle = cycleAt(t);
  clock.textContent = `cyc ${Math.floor(cycle).toString().padStart(3, "0")} / ${TOTAL_CYCLES}`;
  playButton.textContent = t >= DURATION && !playing ? "Replay" : playing ? "Pause" : "Play";
  const buttons = marks.querySelectorAll("button");
  PHASES.forEach((item, index) => {
    const next = PHASES[index + 1]?.t ?? DURATION + 1;
    buttons[index]?.setAttribute("aria-current", t >= item.t && t < next ? "true" : "false");
  });

  const states = statesAt(t);
  const rows = lanes.querySelectorAll<HTMLElement>(".warp-row");
  states.forEach((state, warp) => {
    const row = rows[warp];
    row?.classList.toggle("active", Boolean(state.active));
    const stateEl = row?.querySelector<HTMLElement>(".warp-state");
    const opEl = row?.querySelector<HTMLElement>(".warp-op");
    if (stateEl) stateEl.textContent = state.state;
    if (opEl) opEl.textContent = state.op;
  });
  const activeWarp = states.findIndex((state) => state.active);
  schedulerCopy.textContent = phase.id === "hardware"
    ? "Hardware persistent · software ephemeral"
    : activeWarp >= 0 ? `Scheduler → W${activeWarp}` : "12 contesti residenti e indipendenti";

  const timelineProgress = cycle / TOTAL_CYCLES;
  kValue.textContent = `cycle ${Math.floor(cycle)} / ${TOTAL_CYCLES}`;
  kProgressEl.style.width = `${Math.round(timelineProgress * 100)}%`;
  cycleReadout.textContent = `cycle ${Math.floor(cycle)}`;
  const trackWidth = Math.max(0, cycleTimeline.clientWidth - 70);
  cyclePlayhead.style.left = `${58 + trackWidth * timelineProgress}px`;
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
  updateScene(t);
  renderHud();
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
