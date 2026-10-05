import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { createCycleTimeline, type RowState, type TimelineSpec } from "./cycle-timeline";
import { fitCamera } from "./fit-camera";
import { redrawEngravings } from "./sm-scene";
import { CYCLES_PER_SECOND } from "./sm-timing";
import { HUD_FRAME_INTERVAL, MAX_PIXEL_RATIO, MIN_FRAME_INTERVAL } from "./render-performance";

/*
 * Everything the two "inside one SM" pages share around their scene:
 * renderer, camera framing, phase copy, transport, cycle timeline and loop.
 * Time t runs at CYCLES_PER_SECOND until `totalCycles`; a page may add
 * narration after the kernel (the clock then holds at its last cycle).
 */

export interface PagePhase {
  t: number;
  id: string;
  short: string;
  title: string;
  body: string;
}

export interface SmPageSpec {
  scene: THREE.Scene;
  bounds: THREE.Box3;
  center: THREE.Vector3;
  update(t: number, cycle: number, phase: PagePhase): void;
  phases: PagePhase[];
  duration: number;
  totalCycles: number;
  timeline: Omit<TimelineSpec, "onSeek" | "totalCycles">;
  rowStates(cycle: number): RowState[];
}

export const cycleAt = (t: number, totalCycles: number) => Math.min(totalCycles, Math.max(0, t * CYCLES_PER_SECOND));

export function phaseAt(phases: PagePhase[], t: number): PagePhase {
  let current = phases[0];
  for (const phase of phases) if (t >= phase.t) current = phase;
  return current;
}

export function runSmPage(spec: SmPageSpec): void {
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const viewport = must<HTMLDivElement>("#viewport");
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, MAX_PIXEL_RATIO));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  viewport.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.id = "labels";
  viewport.appendChild(labelRenderer.domElement);

  const scene = spec.scene;
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;
  pmrem.dispose();
  document.fonts?.ready.then(redrawEngravings);

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 160);
  camera.position.copy(spec.center).add(new THREE.Vector3(2.6, 10, 16.5));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(spec.center);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 10;
  controls.maxDistance = 60;
  controls.maxPolarAngle = Math.PI * 0.46;
  controls.update();

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.14, 0.4, 0.9);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  /* ----------------------------------------------------------------- HUD */

  const playButton = must<HTMLButtonElement>("#play");
  const scrub = must<HTMLInputElement>("#scrub");
  const clock = must<HTMLElement>("#clock");
  const phaseKicker = must<HTMLElement>("#phase-kicker");
  const phaseTitle = must<HTMLElement>("#phase-title");
  const phaseBody = must<HTMLElement>("#phase-body");
  const marks = must<HTMLElement>("#marks");

  const params = new URLSearchParams(location.search);
  const initial = Number(params.get("t"));
  let t = Number.isFinite(initial) ? Math.min(spec.duration, Math.max(0, initial)) : 0;
  let playing = !reduceMotion && !params.has("t");
  let scrubbing = false;
  scrub.max = String(spec.duration);

  for (const phase of spec.phases) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = phase.short;
    button.addEventListener("click", () => {
      t = phase.t;
      playing = false;
    });
    marks.appendChild(button);
  }

  const timeline = createCycleTimeline(must<HTMLElement>("#timeline"), {
    ...spec.timeline,
    totalCycles: spec.totalCycles,
    onSeek: (cycle) => {
      t = cycle / CYCLES_PER_SECOND;
      playing = false;
    },
  });

  function togglePlay(): void {
    if (t >= spec.duration) t = 0;
    playing = !playing;
  }
  playButton.addEventListener("click", togglePlay);
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
      togglePlay();
    } else if (event.code === "ArrowLeft" || event.code === "ArrowRight") {
      event.preventDefault();
      const step = (event.shiftKey ? 3 : 1) / CYCLES_PER_SECOND;
      t = Math.min(spec.duration, Math.max(0, t + (event.code === "ArrowLeft" ? -step : step)));
      playing = false;
    }
  });

  function renderHud(phase: PagePhase, cycle: number): void {
    const phaseIndex = spec.phases.indexOf(phase);
    phaseKicker.textContent = `Step ${phaseIndex + 1} / ${spec.phases.length} · ${phase.short}`;
    if (phaseTitle.textContent !== phase.title) {
      phaseTitle.textContent = phase.title;
      phaseBody.textContent = phase.body;
    }
    if (!scrubbing) scrub.value = t.toFixed(2);
    scrub.style.setProperty("--p", `${(t / spec.duration) * 100}%`);
    clock.textContent = `cyc ${Math.floor(cycle).toString().padStart(3, "0")} / ${spec.totalCycles}`;
    playButton.textContent = t >= spec.duration && !playing ? "Replay" : playing ? "Pause" : "Play";
    marks.querySelectorAll("button").forEach((button, index) => {
      button.setAttribute("aria-current", index === phaseIndex ? "true" : "false");
    });
    timeline.update(cycle, spec.rowStates(cycle));
  }

  /** Frame the SM inside the area between the header and the cycle timeline. */
  function resize(): void {
    const width = innerWidth;
    const height = innerHeight;
    renderer.setSize(width, height);
    composer.setSize(width, height);
    bloom.resolution.set(width, height);
    labelRenderer.setSize(width, height);
    const header = must<HTMLElement>(".top").getBoundingClientRect();
    const panel = must<HTMLElement>(".timeline").getBoundingClientRect();
    fitCamera(camera, spec.bounds, { left: 26, right: width - 26, top: header.bottom + 6, bottom: panel.top - 10 }, width, height);
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
    if (playing) {
      t = Math.min(spec.duration, t + dt);
      if (t >= spec.duration) playing = false;
    }
    const cycle = cycleAt(t, spec.totalCycles);
    const phase = phaseAt(spec.phases, t);
    spec.update(t, cycle, phase);
    if (now - lastHud >= HUD_FRAME_INTERVAL || scrubbing) {
      renderHud(phase, cycle);
      lastHud = now;
    }
    controls.update();
    composer.render();
    labelRenderer.render(scene, camera);
  }
  frame(performance.now());
}

export function must<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) throw new Error(`missing ${selector}`);
  return node;
}
