import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { createWorld, type World } from "../../HBMtoSM/src/scene/world";
import { HUD_FRAME_INTERVAL, MAX_PIXEL_RATIO, MIN_FRAME_INTERVAL } from "../../src/render-performance";
import {
  DURATION as LOCAL_DURATION,
  PHASES as LOCAL_TIMELINE,
  sampleTimeline,
  type TimelineSample,
} from "../../HBMtoSM/src/model/timeline";

type LinkKind = "gen4" | "gen5";
type Algorithm = "ring" | "tree";
type Stage = "reduce" | "gather";
type Point = [number, number, number];

interface LinkSpec {
  label: string;
  bandwidth: number;
  latency: number;
}

interface Phase {
  t: number;
  short: string;
  title: string;
  body: string;
}

const LOCAL_END = LOCAL_DURATION;
/** A beat after every partial C lands, so the camera can pull back before PCIe traffic starts. */
const COLLECTIVE_START = LOCAL_END + 1;
const RING_STEP = 1.3;
const TREE_ROUND = 2.7;
/** Share of each step after a packet lands, while the receiver adds or stores it. */
const LANDING = 0.14;
const RANKS = 8;
const FOCUS_RANK = 1;
const GPU_SCALE = 0.19;
const GPU_POSITIONS: Point[] = [
  [-9.6, 0, -3.6], [-3.2, 0, -3.6], [3.2, 0, -3.6], [9.6, 0, -3.6],
  [-9.6, 0, 1.8], [-3.2, 0, 1.8], [3.2, 0, 1.8], [9.6, 0, 1.8],
];
const SWITCH_POSITION: Point = [0, 0.5, -0.9];
/** Ring order walks around the bench, so every hop goes to a physical neighbour. */
const RING = [0, 1, 2, 3, 7, 6, 5, 4];
const TREE_REDUCE: Array<Array<[number, number]>> = [
  [[1, 0], [3, 2], [5, 4], [7, 6]],
  [[2, 0], [6, 4]],
  [[4, 0]],
];
const TREE_GATHER = TREE_REDUCE.slice().reverse().map((pairs) => pairs.map(([source, destination]) => [destination, source] as [number, number]));
/** One rank's partial C: 2,048 tokens × 8,192 hidden × 2 bytes. */
const PAYLOAD_BYTES = 2048 * 8192 * 2;
const SLICE_BYTES = PAYLOAD_BYTES / RANKS;
const GREEN = new THREE.Color("#3ddc97");
const LINKS: Record<LinkKind, LinkSpec> = {
  gen4: { label: "PCIe 4.0", bandwidth: 28, latency: 3.0 },
  gen5: { label: "PCIe 5.0", bandwidth: 55, latency: 2.2 },
};
const LOCAL_START_OF = (id: string): number => LOCAL_TIMELINE.find((phase) => phase.id === id)?.t ?? 0;
const WRITE_START = LOCAL_START_OF("writeback");
const LOCAL_PHASES: Phase[] = [
  {
    t: 0,
    short: "Setup",
    title: "Each GPU holds one K-slice of the GEMM",
    body: "K is split eight ways: each rank holds 1,024 columns of A and the matching 1,024 rows of B.",
  },
  {
    t: LOCAL_START_OF("fill"),
    short: "HBM → L2",
    title: "Eight HBM transfers run in parallel",
    body: "Each GPU stages its activation slice and weight shard from its own HBM into its own L2. Nothing crosses PCIe yet.",
  },
  {
    t: LOCAL_START_OF("sms"),
    short: "L2 → SMs",
    title: "Every SM reads panels from local L2",
    body: "As on one GPU, each SM reads an A row-panel and a B column-panel, but only 1,024 deep in K.",
  },
  {
    t: LOCAL_START_OF("complete"),
    short: "Local GEMM",
    title: "512 SMs compute at once",
    body: "All eight SM arrays fill at once. Each tile sums only one eighth of K, so it is a partial result.",
  },
  {
    t: WRITE_START,
    short: "Writeback",
    title: "Eight partial C matrices land in HBM",
    body: "Each rank now holds a full-size 2,048×8,192 partial C, 32 MiB in BF16. The true C is the sum of all eight.",
  },
];

const params = new URLSearchParams(location.search);
let algorithm: Algorithm = params.get("algo") === "tree" ? "tree" : "ring";
let linkKind: LinkKind = "gen4";

function stageDuration(): number {
  return algorithm === "ring" ? (RANKS - 1) * RING_STEP : TREE_REDUCE.length * TREE_ROUND;
}

function stageSteps(): number {
  return algorithm === "ring" ? RANKS - 1 : TREE_REDUCE.length;
}

function reduceEnd(): number {
  return COLLECTIVE_START + stageDuration();
}

function gatherEnd(): number {
  return reduceEnd() + stageDuration();
}

function duration(): number {
  return gatherEnd() + 1.5;
}

interface Timing {
  fixed: number;
  wire: number;
  total: number;
}

/** Microseconds for one all-reduce of the 32 MiB partial C on the current link. */
function timing(kind: Algorithm): Timing {
  const link = LINKS[linkKind];
  const steps = kind === "ring" ? 2 * (RANKS - 1) : 2 * TREE_REDUCE.length;
  const bytes = kind === "ring" ? SLICE_BYTES : PAYLOAD_BYTES;
  const fixed = steps * link.latency;
  const wire = (steps * bytes) / (link.bandwidth * 1000);
  return { fixed, wire, total: fixed + wire };
}

function phases(): Phase[] {
  const ring = formatDuration(timing("ring").total);
  const tree = formatDuration(timing("tree").total);
  const link = LINKS[linkKind].label;
  const collective: Phase[] = algorithm === "ring"
    ? [
        {
          t: COLLECTIVE_START,
          short: "Reduce-scatter",
          title: "Seven steps, eight slices in flight",
          body: "C splits into eight 4 MiB slices. Each step, every GPU hands one slice to its neighbour, which adds its own.",
        },
        {
          t: reduceEnd(),
          short: "All-gather",
          title: "Seven more steps share the finished slices",
          body: "Each GPU now owns one fully reduced slice. The ring passes finished slices along until every GPU holds all eight.",
        },
        {
          t: gatherEnd(),
          short: "Ready",
          title: "All eight GPUs hold the same C",
          body: `Ring ${ring} vs tree ${tree} on ${link}: each GPU sends only 2 × 7/8 of C, the bandwidth minimum.`,
        },
      ]
    : [
        {
          t: COLLECTIVE_START,
          short: "Tree reduce",
          title: "Three rounds of full 32 MiB hops",
          body: "Pairs add whole 32 MiB partials: four transfers, then two, then one. Most links sit idle.",
        },
        {
          t: reduceEnd(),
          short: "Tree broadcast",
          title: "Three reverse rounds send C back out",
          body: "GPU 0 holds the full sum. The tree reverses until every rank has it, again moving the whole 32 MiB on each hop.",
        },
        {
          t: gatherEnd(),
          short: "Ready",
          title: "All eight GPUs hold the same C",
          body: `Tree ${tree} vs ring ${ring} on ${link}: six full 32 MiB hops in a row. Trees suit tiny decode messages.`,
        },
      ];
  return [...LOCAL_PHASES, ...collective];
}

const viewport = must<HTMLElement>("#viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, MAX_PIXEL_RATIO));
renderer.setClearColor(0x07090d, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
viewport.appendChild(renderer.domElement);

const labels = new CSS2DRenderer();
labels.domElement.id = "labels";
viewport.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
scene.fog = new THREE.FogExp2(0x07090d, 0.018);
scene.add(new THREE.AmbientLight(0x8997a9, 0.8));
scene.add(new THREE.HemisphereLight(0xb9c8db, 0x0d1117, 0.75));
const key = new THREE.DirectionalLight(0xfff6ee, 2.8);
key.position.set(-8, 18, 12);
scene.add(key);
const fill = new THREE.DirectionalLight(0x638ed2, 1.2);
fill.position.set(15, 8, -9);
scene.add(fill);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
pmrem.dispose();

const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 150);
camera.position.set(0, 22, 21);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.5, -0.9);
controls.minDistance = 1;
controls.maxDistance = 58;
controls.maxPolarAngle = Math.PI * 0.48;

const floor = new THREE.Mesh(
  new RoundedBoxGeometry(25.5, 0.18, 13.2, 4, 0.18),
  new THREE.MeshStandardMaterial({ color: 0x11171c, metalness: 0.48, roughness: 0.75 }),
);
floor.position.set(0, -0.26, 0.45);
scene.add(floor);
const grid = new THREE.GridHelper(25, 40, 0x27333c, 0x192128);
grid.position.set(0, -0.14, 0.45);
scene.add(grid);

interface RankVisual {
  world: World;
  card: CSS2DObject;
  role: HTMLElement;
  stateText: HTMLElement;
  cells: HTMLElement[];
}

const ranks: RankVisual[] = GPU_POSITIONS.map(([x, , z], rank) => {
  const host = new THREE.Group();
  host.position.set(x, 0, z);
  host.scale.setScalar(GPU_SCALE);
  if (rank >= 4) host.rotation.y = Math.PI;
  scene.add(host);
  // Only the camera's focus GPU needs its internal callouts. Rendering the
  // other seven sets added more than 90 needless DOM transforms per frame.
  const world = createWorld(host as unknown as THREE.Scene, false, rank === FOCUS_RANK);
  const card = makeLabel(
    `<header><b>GPU ${rank}</b><span></span></header><strong></strong><div class="chunks" aria-hidden="true">${"<i></i>".repeat(RANKS)}</div>`,
    [x, 1.45, z],
    "rank-state",
  );
  scene.add(card);
  return {
    world,
    card,
    role: card.element.querySelector("header span") as HTMLElement,
    stateText: card.element.querySelector("strong") as HTMLElement,
    cells: Array.from(card.element.querySelectorAll<HTMLElement>(".chunks i")),
  };
});

const switchBody = new THREE.Mesh(
  new RoundedBoxGeometry(22.4, 0.52, 1.08, 4, 0.1),
  new THREE.MeshStandardMaterial({ color: 0x394550, metalness: 0.7, roughness: 0.34, emissive: 0x111920 }),
);
switchBody.position.set(...SWITCH_POSITION);
scene.add(switchBody);
const switchFace = new THREE.Mesh(
  new THREE.PlaneGeometry(21.8, 0.68),
  new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0.12, side: THREE.DoubleSide }),
);
switchFace.rotation.x = -Math.PI / 2;
switchFace.position.set(0, 0.78, SWITCH_POSITION[2]);
scene.add(switchFace);
const switchLabel = makeLabel("<b>PCIe switch</b><span>peer-to-peer fabric</span>", [0, 1.38, SWITCH_POSITION[2]], "switch-label");
scene.add(switchLabel);

function connector(rank: number): THREE.Vector3 {
  const [x, , z] = GPU_POSITIONS[rank];
  const backRow = rank < 4;
  return new THREE.Vector3(x + (backRow ? -0.67 : 0.67), 0.42, z + (backRow ? 1.28 : -1.28));
}

function switchPort(rank: number): THREE.Vector3 {
  const from = connector(rank);
  const backRow = rank < 4;
  return new THREE.Vector3(from.x, 0.58, SWITCH_POSITION[2] + (backRow ? -0.54 : 0.54));
}

const linkCurves = GPU_POSITIONS.map((_, rank) => {
  const from = connector(rank);
  const to = switchPort(rank);
  const dz = to.z - from.z;
  return new THREE.CubicBezierCurve3(
    from,
    new THREE.Vector3(from.x, 0.34, from.z + dz * 0.34),
    new THREE.Vector3(to.x, 0.42, from.z + dz * 0.72),
    to,
  );
});

for (let rank = 0; rank < RANKS; rank += 1) {
  const port = new THREE.Mesh(
    new THREE.BoxGeometry(0.34, 0.05, 0.13),
    new THREE.MeshStandardMaterial({ color: 0xd8b35b, metalness: 0.72, roughness: 0.3 }),
  );
  port.position.copy(switchPort(rank));
  port.position.y = 0.79;
  scene.add(port);
}

const linkMaterials: THREE.MeshStandardMaterial[] = [];
for (const curve of linkCurves) {
  const material = new THREE.MeshStandardMaterial({
    color: 0x26343e,
    emissive: GREEN,
    emissiveIntensity: 0.1,
    metalness: 0.56,
    roughness: 0.38,
  });
  linkMaterials.push(material);
  scene.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 54, 0.09, 7, false), material));
}

interface LaneDots {
  mesh: THREE.InstancedMesh;
  material: THREE.MeshBasicMaterial;
  dummy: THREE.Object3D;
}

const laneDotGeometry = new THREE.SphereGeometry(0.035, 6, 6);
const laneDots: LaneDots[] = Array.from({ length: RANKS }, () => {
  const material = new THREE.MeshBasicMaterial({ color: GREEN, transparent: true, opacity: 0.1 });
  const mesh = new THREE.InstancedMesh(laneDotGeometry, material, 10);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(mesh);
  return { mesh, material, dummy: new THREE.Object3D() };
});

interface Packet {
  mesh: THREE.Mesh;
  label: CSS2DObject;
}

const packets: Packet[] = Array.from({ length: RANKS }, () => {
  const mesh = new THREE.Mesh(
    new RoundedBoxGeometry(0.42, 0.2, 0.28, 2, 0.05),
    new THREE.MeshStandardMaterial({ color: GREEN, emissive: GREEN, emissiveIntensity: 2.2, roughness: 0.25 }),
  );
  const label = makeLabel("", [0, 0.4, 0], "packet-label");
  mesh.add(label);
  mesh.visible = false;
  scene.add(mesh);
  return { mesh, label };
});

const playButton = must<HTMLButtonElement>("#play");
const followCameraButton = must<HTMLButtonElement>("#follow-camera");
const scrub = must<HTMLInputElement>("#scrub");
const clock = must<HTMLOutputElement>("#clock");
const marks = must<HTMLElement>("#marks");
const initialT = Number(params.get("t"));
let t = Number.isFinite(initialT) ? Math.max(0, initialT) : 0;
let playing = !matchMedia("(prefers-reduced-motion: reduce)").matches && !params.has("t");
let scrubbing = false;
let last = performance.now();
let lastFrame = 0;
let lastHud = -Infinity;
let guidedCamera = true;
/** A ?t= deep link jumps straight to that shot instead of easing in from the wide view. */
let snapCamera = params.has("t");

controls.addEventListener("start", () => {
  guidedCamera = false;
  followCameraButton.hidden = false;
});
followCameraButton.addEventListener("click", () => {
  guidedCamera = true;
  followCameraButton.hidden = true;
});

function rebuildMarks(): void {
  const items = phases();
  const total = duration();
  scrub.max = String(total);
  marks.replaceChildren();
  for (const phase of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = phase.short;
    button.style.left = `${(phase.t / total) * 100}%`;
    if (phase === items[0]) button.dataset.edge = "start";
    if (phase === items[items.length - 1]) button.dataset.edge = "end";
    button.addEventListener("click", () => {
      t = phase.t;
      playing = false;
    });
    marks.appendChild(button);
  }
}

playButton.addEventListener("click", togglePlay);
scrub.addEventListener("pointerdown", () => { scrubbing = true; });
scrub.addEventListener("pointerup", () => { scrubbing = false; });
scrub.addEventListener("input", () => {
  t = Number(scrub.value);
  playing = false;
});
must<HTMLButtonElement>("#gen4").addEventListener("click", () => setLink("gen4"));
must<HTMLButtonElement>("#gen5").addEventListener("click", () => setLink("gen5"));
must<HTMLButtonElement>("#algo-ring").addEventListener("click", () => setAlgorithm("ring"));
must<HTMLButtonElement>("#algo-tree").addEventListener("click", () => setAlgorithm("tree"));
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) return;
  if (event.code === "Space") {
    event.preventDefault();
    togglePlay();
  } else if (event.code === "ArrowRight" || event.code === "ArrowLeft") {
    event.preventDefault();
    t = clamp(t + (event.code === "ArrowRight" ? 0.2 : -0.2), 0, duration());
    playing = false;
  }
});

function phaseAt(time: number): Phase {
  const items = phases();
  let active = items[0];
  for (const phase of items) if (time >= phase.t) active = phase;
  return active;
}

interface CameraPose {
  position: Point;
  target: Point;
}

const WIDE: CameraPose = { position: [0, 22, 21], target: [0, 0.5, SWITCH_POSITION[2]] };

/** 0 = whole bench, 1 = riding the single-GPU prefill camera on the focus rank. */
function zoomAmount(time: number): number {
  return smooth((time - 0.3) / 2.6) * (1 - smooth((time - (LOCAL_END - 1.8)) / (COLLECTIVE_START - LOCAL_END + 1.8)));
}

/** The focus rank is in the back row and unrotated, so its local frame is a scale and an offset. */
function onFocusGpu(point: Point): Point {
  const [x, y, z] = GPU_POSITIONS[FOCUS_RANK];
  return [x + point[0] * GPU_SCALE, y + point[1] * GPU_SCALE, z + point[2] * GPU_SCALE];
}

function cameraPose(time: number, sample: TimelineSample): CameraPose {
  const zoom = zoomAmount(time);
  if (zoom <= 0) return WIDE;
  return {
    position: mixPoint(WIDE.position, onFocusGpu(sample.camera.position), zoom),
    target: mixPoint(WIDE.target, onFocusGpu(sample.camera.target), zoom),
  };
}

function mixPoint(from: Point, to: Point, amount: number): Point {
  return [
    from[0] + (to[0] - from[0]) * amount,
    from[1] + (to[1] - from[1]) * amount,
    from[2] + (to[2] - from[2]) * amount,
  ];
}

function updateCamera(dt: number, sample: TimelineSample): void {
  if (!guidedCamera) return;
  const pose = cameraPose(t, sample);
  const blend = scrubbing || snapCamera ? 1 : 1 - Math.exp(-6 * dt);
  snapCamera = false;
  camera.position.lerp(new THREE.Vector3(...pose.position), blend);
  controls.target.lerp(new THREE.Vector3(...pose.target), blend);
}

interface Transfer {
  source: number;
  destination: number;
  /** Ring moves one slice of C. Tree moves the whole matrix. */
  chunk: number | null;
}

interface CollectiveFrame {
  stage: Stage;
  /** Completed steps in this stage; also the index of the step in flight. */
  step: number;
  transfers: Transfer[];
  /** Eased 0..1 travel for the step in flight. */
  progress: number;
  waiting: boolean;
}

function collectiveAt(time: number): CollectiveFrame | null {
  if (time < COLLECTIVE_START || time >= gatherEnd()) return null;
  const stage: Stage = time < reduceEnd() ? "reduce" : "gather";
  const local = (time - (stage === "reduce" ? COLLECTIVE_START : reduceEnd())) / stageDuration();
  const steps = stageSteps();
  const scaled = Math.min(steps - 1e-6, local * steps);
  const step = Math.floor(scaled);
  const stepProgress = scaled - step;
  const link = LINKS[linkKind];
  const bytes = algorithm === "ring" ? SLICE_BYTES : PAYLOAD_BYTES;
  const latencyFraction = (1 - LANDING) * link.latency / (link.latency + bytes / (link.bandwidth * 1000));
  const travel = clamp((stepProgress - latencyFraction) / (1 - LANDING - latencyFraction), 0, 1);
  return {
    stage,
    step,
    transfers: transfersFor(stage, step),
    progress: smooth(travel),
    waiting: stepProgress < latencyFraction,
  };
}

function transfersFor(stage: Stage, step: number): Transfer[] {
  if (algorithm === "tree") {
    const pairs = (stage === "reduce" ? TREE_REDUCE : TREE_GATHER)[step];
    return pairs.map(([source, destination]) => ({ source, destination, chunk: null }));
  }
  // Position p sends slice p-s-1 while reducing (so position p ends owning slice p),
  // then forwards slice p-s while gathering. Slice ids follow ring positions.
  return RING.map((source, position) => ({
    source,
    destination: RING[(position + 1) % RANKS],
    chunk: RING[mod(stage === "reduce" ? position - step - 1 : position - step, RANKS)],
  }));
}

interface Cell {
  /** Contributions summed into this slice, out of eight. */
  count: number;
  hot: boolean;
}

/** How many partials position q has summed into slice k after `done` steps, plus the step in flight. */
function ringReduced(q: number, k: number, done: number, progress: number): number {
  const s = mod(q - k - 2, RANKS);
  if (s >= RANKS - 1) return 1;
  if (s < done) return s + 2;
  if (s === done) return 1 + (s + 1) * progress;
  return 1;
}

function ringCell(rank: number, chunk: number, frame: CollectiveFrame): Cell {
  const q = RING.indexOf(rank);
  const k = RING.indexOf(chunk);
  if (frame.stage === "reduce") {
    const s = mod(q - k - 2, RANKS);
    return { count: ringReduced(q, k, frame.step, frame.progress), hot: s === frame.step && frame.progress > 0 && frame.progress < 1 };
  }
  if (k === q) return { count: RANKS, hot: false };
  const stale = ringReduced(q, k, RANKS - 1, 0);
  const s = mod(q - 1 - k, RANKS);
  if (s < frame.step) return { count: RANKS, hot: false };
  if (s === frame.step) return { count: stale + (RANKS - stale) * frame.progress, hot: frame.progress > 0 && frame.progress < 1 };
  return { count: stale, hot: false };
}

function trailingZeros(rank: number): number {
  if (rank === 0) return TREE_REDUCE.length;
  let zeros = 0;
  while (((rank >> zeros) & 1) === 0) zeros += 1;
  return zeros;
}

function treeCell(rank: number, frame: CollectiveFrame): Cell {
  const zeros = trailingZeros(rank);
  const receiving = frame.transfers.some(({ destination }) => destination === rank);
  const moving = receiving && frame.progress > 0 && frame.progress < 1;
  if (frame.stage === "reduce") {
    const held = 2 ** Math.min(frame.step, zeros);
    return { count: receiving ? held * (1 + frame.progress) : held, hot: moving };
  }
  const stale = 2 ** zeros;
  const receiveRound = rank === 0 ? -1 : TREE_REDUCE.length - 1 - zeros;
  if (receiveRound < frame.step) return { count: RANKS, hot: false };
  if (receiveRound === frame.step) return { count: stale + (RANKS - stale) * frame.progress, hot: moving };
  return { count: stale, hot: false };
}

function cellAt(rank: number, chunk: number, frame: CollectiveFrame | null): Cell {
  if (t >= gatherEnd()) return { count: RANKS, hot: false };
  if (!frame) return { count: smooth(t - WRITE_START), hot: false };
  return algorithm === "ring" ? ringCell(rank, chunk, frame) : treeCell(rank, frame);
}

function linkPosition(rank: number, fromGpu: boolean, progress: number): THREE.Vector3 {
  const u = smooth(clamp(progress, 0, 1));
  const point = linkCurves[rank].getPoint(fromGpu ? u : 1 - u);
  point.y += 0.2 + Math.sin(u * Math.PI) * 0.5;
  return point;
}

function routeBetween(source: number, destination: number, progress: number): THREE.Vector3 {
  if (progress < 0.36) return linkPosition(source, true, progress / 0.36);
  if (progress > 0.64) return linkPosition(destination, false, (progress - 0.64) / 0.36);
  const u = smooth((progress - 0.36) / 0.28);
  const from = switchPort(source);
  const to = switchPort(destination);
  // Starts and ends where linkPosition leaves off, so a packet never jumps at a port.
  return new THREE.Vector3(
    from.x + (to.x - from.x) * u,
    from.y + 0.2 + Math.sin(Math.PI * u) * 0.14,
    from.z + (to.z - from.z) * u,
  );
}

function updateCollective(time: number, frame: CollectiveFrame | null): void {
  const moving = frame !== null && frame.progress > 0.005 && frame.progress < 0.995;
  packets.forEach((packet, index) => {
    const transfer = frame?.transfers[index];
    packet.mesh.visible = moving && transfer !== undefined;
    if (!frame || !transfer) return;
    packet.mesh.position.copy(routeBetween(transfer.source, transfer.destination, frame.progress));
    packet.mesh.rotation.y = time * 0.7 + index * 0.3;
    packet.mesh.scale.setScalar(transfer.chunk === null ? 1.55 : 0.85);
    const html = transfer.chunk === null
      ? `<b>${formatPayload(PAYLOAD_BYTES)}</b><span>GPU ${transfer.source} → GPU ${transfer.destination}</span>`
      : `<b>slice ${transfer.chunk}</b>`;
    if (packet.label.element.innerHTML !== html) packet.label.element.innerHTML = html;
  });

  const busy = new Set<number>();
  for (const transfer of frame?.transfers ?? []) busy.add(transfer.source).add(transfer.destination);
  linkMaterials.forEach((material, rank) => {
    material.emissiveIntensity = moving && busy.has(rank) ? 0.62 : 0.1;
  });
  laneDots.forEach(({ mesh, material, dummy }, rank) => {
    const speed = linkKind === "gen5" ? 1.45 : 0.78;
    const sending = frame?.transfers.some(({ source }) => source === rank) ?? false;
    const receiving = frame?.transfers.some(({ destination }) => destination === rank) ?? false;
    for (let dot = 0; dot < 10; dot += 1) {
      const forward = (dot / 10 + time * 0.13 * speed) % 1;
      // A ring link is full duplex: half the dots climb to the switch, half come back down.
      const inbound = receiving && (!sending || dot % 2 === 1);
      linkCurves[rank].getPoint(inbound ? 1 - forward : forward, dummy.position);
      dummy.position.y += 0.12;
      dummy.updateMatrix();
      mesh.setMatrixAt(dot, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
    material.opacity = moving && busy.has(rank) ? 0.8 : 0.08;
  });
  switchFace.material.opacity = moving ? 0.3 + Math.sin(time * 8) * 0.08 : 0.1;
}

function localState(sample: TimelineSample): string {
  if (sample.travelC >= 0.99) return "partial C in HBM";
  if (sample.travelC > 0.01 || sample.assemble > 0.01) return "partial C → HBM";
  if (sample.blocks.some((block) => block.progress > 0.02)) return "computing partial C";
  if (sample.focus.tour > 0 || sample.blocks.some((block) => block.load > 0.02)) return "L2 → SMs";
  if (sample.travelB >= 0.99) return "A + B in L2";
  if (sample.travelA > 0.01) return "HBM → L2";
  return "A slice + B shard in HBM";
}

function collectiveState(rank: number, frame: CollectiveFrame, cells: Cell[]): string {
  const final = cells.filter((cell) => cell.count >= RANKS - 0.001).length;
  if (algorithm === "ring") {
    if (frame.stage === "gather") return `${final}/8 slices final`;
    const incoming = frame.transfers.find(({ destination }) => destination === rank);
    return `step ${frame.step + 1}/7 · + slice ${incoming?.chunk ?? "–"}`;
  }
  const transfer = frame.transfers.find(({ source, destination }) => source === rank || destination === rank);
  if (frame.stage === "reduce") {
    if (transfer?.source === rank) return "sending partial C";
    if (transfer?.destination === rank) return "adding incoming C";
    return rank !== 0 && trailingZeros(rank) < frame.step ? "sent · link idle" : `holds ${Math.round(cells[0].count)} partials`;
  }
  if (transfer?.source === rank) return "sending final C";
  if (transfer?.destination === rank) return "receiving final C";
  return final === RANKS ? "final C ready" : "waiting · link idle";
}

function updateRanks(sample: TimelineSample, frame: CollectiveFrame | null, updateDom: boolean): void {
  const fade = (1 - zoomAmount(t)).toFixed(3);
  if (updateDom) switchLabel.element.style.opacity = fade;
  ranks.forEach((rank, index) => {
    rank.world.update(sample, null);
    if (!updateDom) return;
    const cells = Array.from({ length: RANKS }, (_, chunk) => cellAt(index, chunk, frame));
    let state: string;
    if (t < COLLECTIVE_START) state = localState(sample);
    else if (frame) state = collectiveState(index, frame, cells);
    else state = "C reduced · ready";
    if (rank.stateText.textContent !== state) rank.stateText.textContent = state;
    cells.forEach((cell, chunk) => {
      const element = rank.cells[chunk];
      element.style.setProperty("--v", (cell.count / RANKS).toFixed(3));
      element.classList.toggle("final", cell.count >= RANKS - 0.001);
      element.classList.toggle("hot", cell.hot);
      element.classList.toggle("owned", algorithm === "ring" && chunk === index && t >= reduceEnd());
    });
    rank.card.element.classList.toggle("complete", t >= gatherEnd());
    rank.card.element.style.opacity = fade;
  });
}

function updateHud(frame: CollectiveFrame | null): void {
  const phase = phaseAt(t);
  setText("#phase-kicker", phase.short);
  setText("#phase-title", phase.title);
  setText("#phase-body", phase.body);
  if (!scrubbing) scrub.value = t.toFixed(2);
  clock.value = `${formatTime(t)} / ${formatTime(duration())}`;
  playButton.textContent = t >= duration() && !playing ? "Replay" : playing ? "Pause" : "Play";

  const ring = timing("ring");
  const tree = timing("tree");
  const longest = Math.max(ring.total, tree.total);
  for (const [name, time] of [["ring", ring], ["tree", tree]] as const) {
    must<HTMLElement>(`#${name}-fixed`).style.width = `${(time.fixed / longest) * 100}%`;
    must<HTMLElement>(`#${name}-wire`).style.width = `${(time.wire / longest) * 100}%`;
    setText(`#${name}-time`, formatDuration(time.total));
    must<HTMLElement>(`#row-${name}`).classList.toggle("active", algorithm === name);
  }
  const current = algorithm === "ring" ? ring : tree;
  setText("#traffic-title", `${algorithm === "ring" ? "Ring" : "Tree"} all-reduce · ${formatPayload(PAYLOAD_BYTES)} · ${formatDuration(current.total)}`);
  setText("#traffic-detail", trafficStatus(frame));
  const ratio = (tree.total / ring.total).toFixed(1);
  setText("#algorithm-status", algorithm === "ring"
    ? `Bandwidth-optimal · ${ratio}× faster than tree here`
    : `Latency-optimal · ${ratio}× slower than ring here`);

  const markButtons = marks.querySelectorAll("button");
  const items = phases();
  items.forEach((item, index) => {
    const next = items[index + 1]?.t ?? duration() + 0.01;
    markButtons[index]?.setAttribute("aria-current", String(t >= item.t && t < next));
  });
}

function trafficStatus(frame: CollectiveFrame | null): string {
  if (!frame) {
    if (t >= gatherEnd()) return `Complete · every GPU holds the reduced C`;
    return algorithm === "ring" ? "7 reduce-scatter + 7 all-gather steps" : "3 reduce rounds + 3 broadcast rounds";
  }
  const links = new Set(frame.transfers.map(({ source }) => source)).size;
  const name = algorithm === "ring"
    ? `${frame.stage === "reduce" ? "Reduce-scatter" : "All-gather"} step ${frame.step + 1}/7`
    : `${frame.stage === "reduce" ? "Reduce" : "Broadcast"} round ${frame.step + 1}/3`;
  if (frame.waiting) return `${name} · waiting on fixed link latency`;
  return `${name} · ${links} of 8 GPUs sending`;
}

function setLink(next: LinkKind): void {
  linkKind = next;
  must<HTMLButtonElement>("#gen4").setAttribute("aria-pressed", String(next === "gen4"));
  must<HTMLButtonElement>("#gen5").setAttribute("aria-pressed", String(next === "gen5"));
  const spec = LINKS[next];
  setText("#link-summary", `${spec.bandwidth} GB/s effective · ${spec.latency.toFixed(1)} μs path latency`);
}

function setAlgorithm(next: Algorithm, restart = true): void {
  algorithm = next;
  must<HTMLButtonElement>("#algo-ring").setAttribute("aria-pressed", String(next === "ring"));
  must<HTMLButtonElement>("#algo-tree").setAttribute("aria-pressed", String(next === "tree"));
  ranks.forEach((rank, index) => {
    const position = RING.indexOf(index);
    rank.role.textContent = next === "ring"
      ? `ring → GPU ${RING[(position + 1) % RANKS]}`
      : index === 0 ? "tree root" : `rank ${index}`;
  });
  rebuildMarks();
  if (restart && t >= COLLECTIVE_START) {
    t = COLLECTIVE_START;
    playing = true;
  }
}

function togglePlay(): void {
  if (t >= duration()) t = 0;
  playing = !playing;
}

function makeLabel(html: string, position: Point, className: string): CSS2DObject {
  const element = document.createElement("div");
  element.className = `tag ${className}`;
  element.innerHTML = html;
  const object = new CSS2DObject(element);
  object.position.set(...position);
  object.center.set(0.5, 0.5);
  return object;
}

function resize(): void {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  labels.setSize(width, height);
}

function frame(now: number): void {
  requestAnimationFrame(frame);
  if (document.hidden || now - lastFrame < MIN_FRAME_INTERVAL) return;
  lastFrame = now;
  // The first rAF timestamp can predate `last`; a negative dt would fling the camera lerp.
  const dt = clamp((now - last) / 1000, 0, 0.05);
  last = now;
  if (playing) {
    t = Math.min(duration(), t + dt);
    if (t >= duration()) playing = false;
  }
  const sample = sampleTimeline(clamp(t, 0, LOCAL_DURATION));
  const collective = collectiveAt(t);
  const updateDom = now - lastHud >= HUD_FRAME_INTERVAL || scrubbing;
  updateRanks(sample, collective, updateDom);
  updateCollective(now / 1000, collective);
  if (updateDom) {
    updateHud(collective);
    lastHud = now;
  }
  updateCamera(dt, sample);
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

function mod(value: number, base: number): number {
  return ((value % base) + base) % base;
}

function smooth(value: number): number {
  const x = clamp(value, 0, 1);
  return x * x * (3 - 2 * x);
}

function formatPayload(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${Math.round(bytes / (1024 * 1024))} MiB` : `${Math.round(bytes / 1024)} KiB`;
}

function formatDuration(microseconds: number): string {
  return microseconds >= 1000 ? `${(microseconds / 1000).toFixed(2)} ms` : `${microseconds.toFixed(2)} μs`;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const rest = seconds - mins * 60;
  return `${mins}:${rest.toFixed(1).padStart(4, "0")}`;
}

function must<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`missing ${selector}`);
  return node;
}

function setText(selector: string, text: string): void {
  const node = must<HTMLElement>(selector);
  if (node.textContent !== text) node.textContent = text;
}

window.addEventListener("resize", resize);
resize();
setLink(params.get("link") === "gen5" ? "gen5" : "gen4");
setAlgorithm(algorithm, false);
t = Math.min(t, duration());
requestAnimationFrame(frame);
