import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

/*
 * The SM hardware both "inside one SM" pages draw, left to right:
 * shared memory → register file (one band per warp) → execution units
 * (a row of 32 FP32 lanes per sub-partition, and one Tensor Core each).
 * Pages describe what sits in shared memory and in each warp's registers;
 * their own scene module animates it.
 */

export const X = new THREE.Color("#4c8dff");
export const W = new THREE.Color("#f2a24a");
export const Y = new THREE.Color("#3ddc97");
export const ACTIVE = new THREE.Color("#c8f06a");
export const IDLE = new THREE.Color("#6d7a8c");
export const WARP_COLORS = [
  "#c8f06a", "#6dc8ff", "#d28cff", "#ff7fa7", "#ffc857", "#58e1c1",
  "#ff8c69", "#8ea1ff", "#9ee85f", "#57b8ff", "#e889ff", "#ffad5c",
].map((color) => new THREE.Color(color));

export const SUBPARTITIONS = 4;
export const LANES = 32;

/* ------------------------------------------------------------------ layout */

const SHARED_X = -4.6;
const RF_X = -0.45;
const BLOCK_D = 5.7;
const BLOCK_Z = 0.3;
const Z0 = BLOCK_Z - BLOCK_D / 2;
const Z1 = BLOCK_Z + BLOCK_D / 2;
const BASE_H = 0.3;

const SHARED_W = 3.0;
const SMEM_X0 = SHARED_X - 0.95;
const SMEM_X1 = SHARED_X + 1.35;
const SMEM_Z0 = -2.3;
const SMEM_Z1 = 2.95;
const GROUP_GAP = 0.3;

const RF_W = 3.6;
const BAND_Z0 = -2.1;
const BAND_Z1 = 2.75;
const SCHED_Z = -3.3;

/** One strip per sub-partition: its label, its 32 FP32 lanes, then its Tensor Core. */
const EXEC_X0 = 2.65;
const EXEC_X1 = 6.95;
const EXEC_X = (EXEC_X0 + EXEC_X1) / 2;
const EXEC_W = EXEC_X1 - EXEC_X0;
const STRIP_PITCH = BLOCK_D / SUBPARTITIONS;
const laneRowZ = (sp: number) => Z0 + (sp + 0.5) * STRIP_PITCH;
const LANE_X0 = EXEC_X0 + 0.55;
const LANE_X1 = EXEC_X1 - 1.05;
const TC_X = EXEC_X1 - 0.55;
const STRIP_H = 0.02;

const FLOOR_X0 = -6.7;
const FLOOR_X1 = 7.3;
const FLOOR_Z0 = -4.0;
const FLOOR_Z1 = 3.9;

/* ------------------------------------------------------------- materials */

function metal(color: number, roughness = 0.5, metalness = 0.55): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

/** Lit like hardware, but every instance glows with its instance color. */
function glowMaterial(): THREE.MeshStandardMaterial {
  const material = new THREE.MeshStandardMaterial({ color: 0x0f1216, roughness: 0.38, metalness: 0.35 });
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <emissivemap_fragment>",
      "#include <emissivemap_fragment>\n#ifdef USE_COLOR\n  totalEmissiveRadiance += vColor.rgb;\n#endif",
    );
  };
  return material;
}

const GLOW = glowMaterial();
const scratch = new THREE.Vector3();

/* ---------------------------------------------------------- instanced cells */

export class Cells {
  readonly mesh: THREE.InstancedMesh;
  readonly tops: THREE.Vector3[];
  private readonly matrix = new THREE.Matrix4();
  private readonly color = new THREE.Color();
  private readonly scale = new THREE.Vector3(1, 1, 1);
  private readonly quaternion = new THREE.Quaternion();
  private readonly heights: Float32Array;
  private matricesDirty = false;

  constructor(
    parent: THREE.Object3D,
    private readonly positions: THREE.Vector3[],
    size: THREE.Vector3,
    radius: number,
  ) {
    const geometry = new RoundedBoxGeometry(size.x, size.y, size.z, 2, radius);
    geometry.translate(0, size.y / 2, 0);
    this.mesh = new THREE.InstancedMesh(geometry, GLOW, positions.length);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.heights = new Float32Array(positions.length).fill(1);
    this.tops = positions.map((p) => new THREE.Vector3(p.x, p.y + size.y, p.z));
    positions.forEach((position, index) => {
      this.mesh.setMatrixAt(index, this.matrix.makeTranslation(position));
      this.mesh.setColorAt(index, this.color.setRGB(0, 0, 0));
    });
    parent.add(this.mesh);
  }

  get count(): number {
    return this.positions.length;
  }

  set(index: number, color: THREE.Color, glow: number, height = 1): void {
    this.mesh.setColorAt(index, this.color.copy(color).multiplyScalar(glow));
    if (Math.abs(this.heights[index] - height) > 1e-3) {
      this.heights[index] = height;
      this.scale.set(1, Math.max(0.02, height), 1);
      this.matrix.compose(this.positions[index], this.quaternion, this.scale);
      this.mesh.setMatrixAt(index, this.matrix);
      this.matricesDirty = true;
    }
  }

  commit(): void {
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    if (this.matricesDirty) this.mesh.instanceMatrix.needsUpdate = true;
    this.matricesDirty = false;
  }
}

export class Beads {
  private readonly mesh: THREE.InstancedMesh;
  private readonly matrix = new THREE.Matrix4();
  private readonly color = new THREE.Color();
  private readonly scale = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();
  private count = 0;

  constructor(parent: THREE.Object3D, private readonly capacity: number) {
    this.mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.09, 14, 10), GLOW, capacity);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.setColorAt(0, this.color.setRGB(0, 0, 0));
    parent.add(this.mesh);
  }

  begin(): void {
    this.count = 0;
  }

  add(position: THREE.Vector3, color: THREE.Color, glow: number, size = 1): void {
    if (this.count >= this.capacity) return;
    this.scale.setScalar(size);
    this.matrix.compose(position, this.quaternion, this.scale);
    this.mesh.setMatrixAt(this.count, this.matrix);
    this.mesh.setColorAt(this.count, this.color.copy(color).multiplyScalar(glow));
    this.count++;
  }

  /** A bead plus a short fading tail along the same curve. */
  trail(curve: THREE.Curve<THREE.Vector3>, u: number, color: THREE.Color, glow: number, size = 1): void {
    if (u <= 0 || u >= 1) return;
    for (let i = 0; i < 4; i++) {
      const v = u - i * 0.035;
      if (v <= 0) break;
      this.add(curve.getPointAt(v, scratch), color, glow * (1 - i * 0.24), size * (1 - i * 0.2));
    }
  }

  end(): void {
    this.mesh.count = this.count;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}

/** A curve that lifts between two points, for data hopping from unit to unit. */
export function arc(from: THREE.Vector3, to: THREE.Vector3, lift: number): THREE.QuadraticBezierCurve3 {
  return new THREE.QuadraticBezierCurve3(
    from,
    new THREE.Vector3((from.x + to.x) / 2, Math.max(from.y, to.y) + lift, (from.z + to.z) / 2),
    to,
  );
}

/* ------------------------------------------------------------ silkscreen */

const engravings: Array<() => void> = [];
const PX = 150;

function engrave(
  parent: THREE.Object3D,
  lines: string[],
  width: number,
  depth: number,
  x: number,
  y: number,
  z: number,
  { size = 0.17, align = "left" as CanvasTextAlign, color = "rgba(186, 198, 214, .62)", weight = 500 } = {},
): void {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * PX);
  canvas.height = Math.round(depth * PX);
  const context = canvas.getContext("2d")!;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  const draw = () => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = color;
    context.font = `${weight} ${size * PX}px "IBM Plex Mono", ui-monospace, monospace`;
    context.textAlign = align;
    context.textBaseline = "middle";
    const lineHeight = size * PX * 1.28;
    const x0 = align === "left" ? 0 : align === "right" ? canvas.width : canvas.width / 2;
    const y0 = canvas.height / 2 - ((lines.length - 1) * lineHeight) / 2;
    lines.forEach((line, index) => context.fillText(line, x0, y0 + index * lineHeight));
    texture.needsUpdate = true;
  };
  draw();
  engravings.push(draw);
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depth),
    new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, toneMapped: false }),
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.set(x, y + 0.004, z);
  parent.add(plane);
}

/** Canvas text is drawn before web fonts arrive; repaint once they have. */
export function redrawEngravings(): void {
  for (const draw of engravings) draw();
}

function block(
  parent: THREE.Object3D,
  w: number,
  h: number,
  d: number,
  x: number,
  z: number,
  material: THREE.Material,
  y0 = 0,
  radius = 0.08,
): THREE.Mesh {
  const mesh = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 3, Math.min(radius, h / 2 - 1e-3)), material);
  mesh.position.set(x, y0 + h / 2, z);
  parent.add(mesh);
  return mesh;
}

function row(count: number, x0: number, x1: number, y: number, z: number): THREE.Vector3[] {
  const pitch = (x1 - x0) / count;
  return Array.from({ length: count }, (_, index) => new THREE.Vector3(x0 + pitch * (index + 0.5), y, z));
}

/* --------------------------------------------------------- labels in 3D */

export interface Label {
  element: HTMLDivElement;
  /** Callouts show during these phases; tags step aside during them. */
  phases: string[];
}

/** A name tag hanging below (or, with `above`, standing over) a block. It steps aside during `hiddenIn`. */
export function tag(parent: THREE.Object3D, html: string, position: THREE.Vector3, tone: string, above = false, hiddenIn: string[] = []): Label {
  const element = document.createElement("div");
  element.className = `tag ${tone}`;
  element.innerHTML = html;
  const object = new CSS2DObject(element);
  object.position.copy(position);
  object.center.set(0.5, above ? 1 : 0);
  parent.add(object);
  return { element, phases: hiddenIn };
}

/** A note with a leader line that only shows during `phases`. */
export function callout(parent: THREE.Object3D, html: string, position: THREE.Vector3, tone: string, phases: string[]): Label {
  const element = document.createElement("div");
  element.className = `callout ${tone}`;
  element.innerHTML = html;
  const object = new CSS2DObject(element);
  object.position.copy(position);
  object.center.set(0.5, 1);
  parent.add(object);
  return { element, phases };
}

export function showLabels(tags: Label[], callouts: Label[], phase: string): void {
  for (const item of callouts) item.element.classList.toggle("on", item.phases.includes(phase));
  for (const item of tags) item.element.classList.toggle("aside", item.phases.includes(phase));
}

/* -------------------------------------------------------------- hardware */

export interface SharedGroup {
  letter: string;
  color: THREE.Color;
  rows: number;
  cols: number;
}

export interface OperandGroup {
  color: THREE.Color;
  count: number;
}

export interface SmSpec {
  warps: number;
  /** Shared-memory contents, back to front. */
  shared: SharedGroup[];
  /** Operand registers in each warp's band, left to right; accumulators follow. */
  operands: OperandGroup[];
  accumulators: number;
}

export interface SmHardware {
  scene: THREE.Scene;
  /** One set of cells per shared group, indexed row * cols + col. */
  shared: Cells[];
  owners: Cells;
  /** One set per operand group, indexed warp * count + i. */
  operands: Cells[];
  /** Indexed warp * accumulators + i. */
  accumulators: Cells;
  tokens: THREE.MeshStandardMaterial[];
  tokenMeshes: THREE.Mesh[];
  scheduler: THREE.MeshStandardMaterial;
  /** One row of 32 lanes per sub-partition. */
  lanes: Cells[];
  tensor: THREE.MeshStandardMaterial[];
  beads: Beads;
  at: {
    sharedX: number;
    registerX: number;
    /** Middle of the FP32 lanes, and the column of Tensor Cores at the end of each strip. */
    lanesX: number;
    tensorX: number;
    /** Just in front of the three main blocks, for name tags. */
    front: number;
    schedulerZ: number;
    lanesBack: number;
    sharedRowZ: (group: number, row: number) => number;
    laneRowZ: (sp: number) => number;
    bandOut: (warp: number) => THREE.Vector3;
    laneIn: (sp: number) => THREE.Vector3;
    tensorTop: (sp: number) => THREE.Vector3;
  };
  bounds: THREE.Box3;
  center: THREE.Vector3;
}

export const subpartitionOf = (warp: number) => warp % SUBPARTITIONS;

export function createSmHardware(spec: SmSpec): SmHardware {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07080b);
  scene.fog = new THREE.Fog(0x07080b, 30, 60);

  scene.add(new THREE.HemisphereLight(0xc4d2e6, 0x0a0c10, 0.85));
  const key = new THREE.DirectionalLight(0xfff4e8, 2.5);
  key.position.set(6, 14, 9);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x7fa6ff, 1.0);
  rim.position.set(-9, 6, -12);
  scene.add(rim);

  // Ground: a soft pool of light under the SM, fading into the background.
  const groundCanvas = document.createElement("canvas");
  groundCanvas.width = groundCanvas.height = 256;
  const g = groundCanvas.getContext("2d")!;
  const gradient = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  gradient.addColorStop(0, "rgba(46, 58, 74, .55)");
  gradient.addColorStop(0.55, "rgba(22, 28, 38, .28)");
  gradient.addColorStop(1, "rgba(7, 8, 11, 0)");
  g.fillStyle = gradient;
  g.fillRect(0, 0, 256, 256);
  const groundTexture = new THREE.CanvasTexture(groundCanvas);
  groundTexture.colorSpace = THREE.SRGBColorSpace;
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(38, 26),
    new THREE.MeshBasicMaterial({ map: groundTexture, transparent: true, depthWrite: false }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set((FLOOR_X0 + FLOOR_X1) / 2, -0.32, 0);
  scene.add(ground);

  block(scene, FLOOR_X1 - FLOOR_X0, 0.22, FLOOR_Z1 - FLOOR_Z0, (FLOOR_X0 + FLOOR_X1) / 2, (FLOOR_Z0 + FLOOR_Z1) / 2, metal(0x12161c, 0.6, 0.4), -0.22, 0.16);
  const plate = metal(0x252e3a, 0.48, 0.5);

  /* ---------------------------------------------------- shared memory */

  block(scene, SHARED_W, BASE_H, BLOCK_D, SHARED_X, BLOCK_Z, plate, 0, 0.1);
  const totalRows = spec.shared.reduce((sum, group) => sum + group.rows, 0);
  const rowPitch = (SMEM_Z1 - SMEM_Z0 - GROUP_GAP * (spec.shared.length - 1)) / totalRows;
  const groupRowZ: number[][] = [];
  let z = SMEM_Z0 + rowPitch / 2;
  const shared = spec.shared.map((group) => {
    const rows = Array.from({ length: group.rows }, (_, index) => z + index * rowPitch);
    groupRowZ.push(rows);
    z += group.rows * rowPitch + GROUP_GAP;
    const hex = group.color.getHexString();
    engrave(scene, [group.letter], 0.4, 0.4, SHARED_X - 1.24, BASE_H, (rows[0] + rows[rows.length - 1]) / 2, { size: 0.24, align: "center", color: `#${hex}` });
    const pitch = (SMEM_X1 - SMEM_X0) / Math.max(...spec.shared.map((item) => item.cols));
    const width = pitch * group.cols;
    const x0 = (SMEM_X0 + SMEM_X1) / 2 - width / 2;
    const positions = rows.flatMap((rowZ) => row(group.cols, x0, x0 + width, BASE_H, rowZ));
    return new Cells(scene, positions, new THREE.Vector3(pitch * 0.8, 0.22, Math.min(0.38, rowPitch * 0.78)), 0.03);
  });

  /* -------------------------------------------------- register file */

  block(scene, RF_W, BASE_H, BLOCK_D, RF_X, BLOCK_Z, plate, 0, 0.1);
  const bandPitch = (BAND_Z1 - BAND_Z0) / Math.max(1, spec.warps - 1);
  const bandZ = (warp: number) => BAND_Z0 + warp * bandPitch;
  const cellD = Math.min(0.32, bandPitch * 0.66);
  const bandY = BASE_H + 0.03;
  const owners = new Cells(scene, Array.from({ length: spec.warps }, (_, warp) => new THREE.Vector3(RF_X, BASE_H, bandZ(warp))), new THREE.Vector3(RF_W - 0.3, 0.03, bandPitch * 0.8), 0.015);
  const operandCount = spec.operands.reduce((sum, group) => sum + group.count, 0);
  const operandPitch = 1.4 / operandCount;
  let first = 0;
  const operands = spec.operands.map((group) => {
    const x0 = RF_X - 1.45 + first * operandPitch;
    first += group.count;
    return new Cells(
      scene,
      Array.from({ length: spec.warps }, (_, warp) => row(group.count, x0, x0 + group.count * operandPitch, bandY, bandZ(warp))).flat(),
      new THREE.Vector3(operandPitch * 0.72, 0.12, cellD),
      0.03,
    );
  });
  const accumulators = new Cells(
    scene,
    Array.from({ length: spec.warps }, (_, warp) => row(spec.accumulators, RF_X + 0.3, RF_X + 1.52, bandY, bandZ(warp))).flat(),
    new THREE.Vector3((1.22 / spec.accumulators) * 0.8, 0.42, cellD),
    0.03,
  );
  for (let warp = 0; warp < spec.warps; warp++) {
    engrave(scene, [`W${warp}`], 0.5, 0.3, RF_X - 1.6, BASE_H, bandZ(warp), { size: spec.warps > 8 ? 0.13 : 0.15, align: "right", color: `#${WARP_COLORS[warp].getHexString()}` });
  }

  // Scheduler: one column of warp tokens per sub-partition.
  const scheduler = new THREE.MeshStandardMaterial({ color: 0x2c3542, roughness: 0.42, metalness: 0.5, emissive: ACTIVE, emissiveIntensity: 0 });
  block(scene, RF_W - 0.3, 0.26, 0.95, RF_X, SCHED_Z, scheduler, 0, 0.08);
  const slots = Math.ceil(spec.warps / SUBPARTITIONS);
  const tokens: THREE.MeshStandardMaterial[] = [];
  const tokenMeshes: THREE.Mesh[] = [];
  for (let warp = 0; warp < spec.warps; warp++) {
    const material = new THREE.MeshStandardMaterial({
      color: WARP_COLORS[warp].clone().multiplyScalar(0.3),
      emissive: WARP_COLORS[warp],
      emissiveIntensity: 0,
      roughness: 0.3,
      metalness: 0.2,
    });
    const token = new THREE.Mesh(new THREE.SphereGeometry(slots > 2 ? 0.1 : 0.12, 20, 14), material);
    const slot = Math.floor(warp / SUBPARTITIONS);
    token.position.set(RF_X - 1.2 + subpartitionOf(warp) * 0.8, 0.38, SCHED_Z - ((slots - 1) * 0.3) / 2 + slot * 0.3);
    scene.add(token);
    tokens.push(material);
    tokenMeshes.push(token);
  }

  /* ------------------------------------------------- execution units */

  block(scene, EXEC_W, BASE_H, BLOCK_D, EXEC_X, BLOCK_Z, plate, 0, 0.1);
  const strip = metal(0x2c3644, 0.5, 0.5);
  const lanePitch = (LANE_X1 - LANE_X0) / LANES;
  const top = BASE_H + STRIP_H;
  const lanes: Cells[] = [];
  const tensor: THREE.MeshStandardMaterial[] = [];
  for (let sp = 0; sp < SUBPARTITIONS; sp++) {
    const rowZ = laneRowZ(sp);
    block(scene, EXEC_W - 0.2, STRIP_H, STRIP_PITCH - 0.22, EXEC_X, rowZ, strip, BASE_H, 0.01);
    engrave(scene, [`SP${sp}`], 0.46, 0.3, EXEC_X0 + 0.3, top, rowZ, { size: 0.14, align: "center", color: "rgba(170, 182, 198, .6)" });
    lanes.push(new Cells(scene, row(LANES, LANE_X0, LANE_X1, top, rowZ), new THREE.Vector3(lanePitch * 0.68, 0.16, 0.72), 0.02));
    const material = new THREE.MeshStandardMaterial({ color: 0x29303a, roughness: 0.55, metalness: 0.5, emissive: ACTIVE, emissiveIntensity: 0 });
    block(scene, 0.72, 0.2, 0.95, TC_X, rowZ, material, top, 0.05);
    engrave(scene, [`TC${sp}`], 0.6, 0.3, TC_X, top + 0.2, rowZ, { size: 0.14, align: "center", color: "rgba(150, 162, 178, .55)" });
    tensor.push(material);
  }

  return {
    scene,
    shared,
    owners,
    operands,
    accumulators,
    tokens,
    tokenMeshes,
    scheduler,
    lanes,
    tensor,
    beads: new Beads(scene, 240),
    at: {
      sharedX: SHARED_X,
      registerX: RF_X,
      lanesX: (LANE_X0 + LANE_X1) / 2,
      tensorX: TC_X,
      front: Z1 + 0.12,
      schedulerZ: SCHED_Z,
      lanesBack: Z0,
      sharedRowZ: (group, rowIndex) => groupRowZ[group][rowIndex],
      laneRowZ,
      bandOut: (warp) => new THREE.Vector3(RF_X + 1.65, bandY + 0.1, bandZ(warp)),
      laneIn: (sp) => new THREE.Vector3(LANE_X0 - 0.1, top + 0.12, laneRowZ(sp)),
      tensorTop: (sp) => new THREE.Vector3(TC_X, top + 0.25, laneRowZ(sp)),
    },
    bounds: new THREE.Box3(new THREE.Vector3(FLOOR_X0, -0.3, FLOOR_Z0 - 0.5), new THREE.Vector3(FLOOR_X1, 0.9, FLOOR_Z1 + 0.5)),
    center: new THREE.Vector3((FLOOR_X0 + FLOOR_X1) / 2, 0, (FLOOR_Z0 + FLOOR_Z1) / 2),
  };
}
