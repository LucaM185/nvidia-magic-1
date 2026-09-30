import * as THREE from "three";
import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import {
  BANK_CENTER_Z,
  BOARD,
  C_FORM,
  DIE_FLOOR,
  DIE_FRAME,
  GRID,
  HBM_MODULES,
  L2_A,
  L2_B,
  L2_GRID,
  L2_SLAB,
  MATRIX,
  SM_BANK_DEPTH,
  SM_BANK_WIDTH,
  SM_BODY,
  SM_TOUR,
  TILE_SIZE,
  cTileSlot,
  colBand,
  hbmFootprint,
  hbmMatrixCenter,
  hbmModule,
  planePoint,
  rowBand,
  smWorld,
  type Vec3,
} from "../layout";
import {
  aPanelBytes,
  bPanelBytes,
  cTileBytes,
  cacheUsed,
  kb,
  matrixBytes,
  residentBytes,
  getModel,
} from "../model/gemm";
import type { TimelineSample } from "../model/timeline";
import { STATE_COLOR } from "../theme";
import {
  applyMatrixLook,
  createCacheMaterial,
  createFootprintMaterial,
  createMatrixMaterial,
  createSliceMaterial,
  createTileMaterial,
  createWindowMaterial,
} from "./materials";
import { createRibbon, type Ribbon } from "./ribbon";

export interface PickHit {
  row: number;
  col: number;
}

export interface World {
  update(sample: TimelineSample, inspect: PickHit | null): void;
  pick(ndc: THREE.Vector2, camera: THREE.Camera): PickHit | null;
}

interface SquarePlate {
  object: THREE.Group;
  material: THREE.ShaderMaterial;
  plate: THREE.MeshStandardMaterial;
}

function v3(p: Vec3): THREE.Vector3 {
  return new THREE.Vector3(p[0], p[1], p[2]);
}

function placeHorizontal(mesh: THREE.Object3D, center: Vec3): void {
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(center[0], center[1], center[2]);
}

function tag(html: string, position: Vec3, className = ""): CSS2DObject {
  const element = document.createElement("div");
  element.className = `tag ${className}`.trim();
  element.innerHTML = html;
  const object = new CSS2DObject(element);
  object.position.set(position[0], position[1], position[2]);
  return object;
}

function l2TagHtml(compact: boolean): string {
  if (compact) return `<b>L2</b>`;
  return `<b>L2</b><span class="kicker">40 MB</span><span>A+B ${cacheUsed(residentBytes())}</span>`;
}

function metal(color: number, roughness = 0.58): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color,
    roughness,
    metalness: 0.42,
    envMapIntensity: 0.28,
  });
}

function smooth01(v: number): number {
  const x = Math.min(1, Math.max(0, v));
  return x * x * (3 - 2 * x);
}

function fly(from: Vec3, to: Vec3, t: number, lift: number, bow: number): Vec3 {
  const u = Math.min(1, Math.max(0, t));
  const arc = Math.sin(Math.PI * u);
  return [
    from[0] + (to[0] - from[0]) * u,
    from[1] + (to[1] - from[1]) * u + arc * lift,
    from[2] + (to[2] - from[2]) * u + arc * bow,
  ];
}

function flyPoints(from: Vec3, to: Vec3, lift: number, bow: number): THREE.Vector3[] {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= 12; i++) points.push(v3(fly(from, to, i / 12, lift, bow)));
  return points;
}

function aToSm(row: number, col: number): THREE.Vector3[] {
  const [v0, v1] = rowBand(row);
  const u = (col + 0.5) / GRID;
  const sm = smWorld(row, col);
  const vEdge = sm[2] < L2_A.center[2] ? 1.05 : -0.05;
  const onBand = planePoint(L2_A.center, MATRIX, MATRIX, u, (v0 + v1) / 2);
  const onEdge = planePoint(L2_A.center, MATRIX, MATRIX, u, vEdge);
  return [
    v3([onBand[0], onBand[1] + 0.1, onBand[2]]),
    v3([onEdge[0], onEdge[1] + 0.18, onEdge[2]]),
    v3([(onEdge[0] * 0.35 + sm[0] * 0.65), 1.18, (onEdge[2] * 0.35 + sm[2] * 0.65)]),
    v3([sm[0], sm[1] + 0.32, sm[2]]),
  ];
}

function bToSm(row: number, col: number): THREE.Vector3[] {
  const [u0, u1] = colBand(col);
  const u = (u0 + u1) / 2;
  const sm = smWorld(row, col);
  const vEdge = sm[2] < L2_B.center[2] ? 1.05 : -0.05;
  const onBand = planePoint(L2_B.center, MATRIX, MATRIX, u, 1 - (row + 0.5) / GRID);
  const onEdge = planePoint(L2_B.center, MATRIX, MATRIX, u, vEdge);
  return [
    v3([onBand[0], onBand[1] + 0.1, onBand[2]]),
    v3([onEdge[0], onEdge[1] + 0.18, onEdge[2]]),
    v3([(onEdge[0] * 0.35 + sm[0] * 0.65), 1.22, (onEdge[2] * 0.35 + sm[2] * 0.65)]),
    v3([sm[0], sm[1] + 0.32, sm[2]]),
  ];
}

const IDLE_CAPS = [0x6a7380, 0x4e5966, 0x7d8794, 0x3e4854, 0x5c6772, 0x8a94a1, 0x55606c, 0x454e5a, 0x6e7884];

export function createWorld(scene: THREE.Scene, debug = false): World {
  const gpu = new THREE.Group();
  gpu.name = "gpu";
  scene.add(gpu);

  const board = new THREE.Group();
  board.name = "board";
  const hbmGroup = new THREE.Group();
  hbmGroup.name = "hbmGroup";
  const gpuPackage = new THREE.Group();
  gpuPackage.name = "gpuPackage";
  const overlays = new THREE.Group();
  overlays.name = "overlays";
  gpu.add(board, hbmGroup, gpuPackage, overlays);

  // Extra PCB on the near edge so the gold fingers sit outside the package.
  const pcieLip = 1.45;
  const boardDepth = BOARD.depth + pcieLip;
  const boardCenterZ = pcieLip / 2;

  const pcb = new THREE.Mesh(
    new RoundedBoxGeometry(BOARD.width, 0.16, boardDepth, 3, 0.08),
    metal(0x24302a, 0.72),
  );
  pcb.position.set(0, 0.08, boardCenterZ);
  board.add(pcb);
  const solder = new THREE.Mesh(
    new RoundedBoxGeometry(BOARD.width - 0.8, 0.03, BOARD.depth - 0.9, 2, 0.04),
    metal(0x314238, 0.58),
  );
  solder.position.set(0, 0.17, 0);
  board.add(solder);

  // Full-height card bracket and a compact 16-pin auxiliary power socket make
  // the silhouette read as a PCIe accelerator rather than a square test board.
  const bracket = new THREE.Mesh(
    new RoundedBoxGeometry(0.22, 0.34, boardDepth + 0.72, 2, 0.04),
    metal(0xa8afb7, 0.3),
  );
  bracket.position.set(-BOARD.width / 2 - 0.08, 0.2, boardCenterZ);
  const powerSocket = new THREE.Mesh(
    new RoundedBoxGeometry(1.18, 0.56, 1.72, 2, 0.08),
    metal(0x10141a, 0.8),
  );
  powerSocket.position.set(BOARD.width / 2 - 0.82, 0.43, -2.6);
  board.add(bracket, powerSocket);

  const gold = new THREE.MeshStandardMaterial({
    color: 0xe0b85a,
    metalness: 0.55,
    roughness: 0.38,
    envMapIntensity: 0.4,
  });
  const connectorDepth = 0.82;
  const boardNearEdge = boardCenterZ + boardDepth / 2;
  // Most of the finger field hangs past the PCB lip (toward +Z), not on the package.
  const connectorOnBoard = 0.16;
  const connectorZ = boardNearEdge - connectorOnBoard + connectorDepth / 2;
  const connectorY = 0.1;
  const fingerY = connectorY + 0.045;
  // Scaled from the H100 card drawing: 81.88 mm contact field on a 267.7 mm PCB.
  const connectorWidth = 8.56;
  const connectorX = -3.54;
  const connector = new THREE.Mesh(new THREE.BoxGeometry(connectorWidth, 0.04, connectorDepth), metal(0x0c0e10, 0.5));
  connector.position.set(connectorX, connectorY, connectorZ);
  board.add(connector);
  const visiblePins = 80;
  const fingerPitch = connectorWidth / 84;
  const fingerGeo = new THREE.BoxGeometry(fingerPitch * 0.55, 0.055, 0.67);
  const fingers = new THREE.InstancedMesh(fingerGeo, gold, visiblePins);
  const pinDummy = new THREE.Object3D();
  let pinIndex = 0;
  for (let slot = 0; slot < 82; slot++) {
    // The key separates the short command/power section from the x16 lane field.
    if (slot === 17 || slot === 18) continue;
    pinDummy.position.set(connectorX - connectorWidth / 2 + (slot + 1.2) * fingerPitch, fingerY, connectorZ);
    pinDummy.updateMatrix();
    fingers.setMatrixAt(pinIndex++, pinDummy.matrix);
  }
  fingers.instanceMatrix.needsUpdate = true;
  board.add(fingers);

  const dieFrame = new THREE.Mesh(
    new RoundedBoxGeometry(DIE_FRAME.width, 0.2, DIE_FRAME.depth, 3, 0.08),
    metal(0x8e98a3, 0.38),
  );
  dieFrame.position.set(0, 0.28, 0);
  const dieFloor = new THREE.Mesh(
    new RoundedBoxGeometry(DIE_FLOOR.width, 0.06, DIE_FLOOR.depth, 2, 0.03),
    metal(0x171b20, 0.7),
  );
  dieFloor.position.set(0, 0.4, 0);
  gpuPackage.add(dieFrame, dieFloor);

  for (const z of [-BANK_CENTER_Z, BANK_CENTER_Z]) {
    const bank = new THREE.Mesh(
      new RoundedBoxGeometry(SM_BANK_WIDTH, 0.05, SM_BANK_DEPTH, 2, 0.03),
      metal(0x222830, 0.55),
    );
    bank.position.set(0, 0.46, z);
    gpuPackage.add(bank);
  }

  const foot = hbmFootprint();
  let idleCap = 0;
  for (const mod of HBM_MODULES) {
    const plinth = new THREE.Mesh(new RoundedBoxGeometry(foot + 0.28, 0.08, foot + 0.28, 2, 0.04), metal(0x1a2026, 0.6));
    plinth.position.set(mod.x, 0.22, mod.z);
    const body = new THREE.Mesh(new RoundedBoxGeometry(foot, mod.height, foot, 3, 0.06), metal(0x2c333c, 0.46));
    body.position.set(mod.x, 0.3 + mod.height / 2, mod.z);
    const capColor =
      mod.role === "A" ? 0x4c8dff : mod.role === "B" ? 0xf2a24a : mod.role === "C" ? 0x3ddc97 : IDLE_CAPS[idleCap++ % IDLE_CAPS.length];
    const cap = new THREE.Mesh(new THREE.BoxGeometry(foot * 0.62, 0.05, foot * 0.62), metal(capColor, 0.34));
    cap.position.set(mod.x, 0.3 + mod.height + 0.03, mod.z);
    const dx = -mod.x;
    const dz = -mod.z;
    const mag = Math.hypot(dx, dz) || 1;
    const ux = dx / mag;
    const uz = dz / mag;
    const bridgeLen = 0.62;
    const bridge = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.035, bridgeLen), metal(0x3a4550, 0.48));
    bridge.position.set(
      mod.x + ux * (foot * 0.5 + bridgeLen * 0.42),
      0.25,
      mod.z + uz * (foot * 0.5 + bridgeLen * 0.42),
    );
    bridge.rotation.y = Math.atan2(ux, uz);
    hbmGroup.add(plinth, body, cap, bridge);
  }

  const aColor = new THREE.Color("#4c8dff");
  const bColor = new THREE.Color("#f2a24a");
  const cColor = new THREE.Color("#3ddc97");

  function createSquare(hex: string): SquarePlate {
    const object = new THREE.Group();
    const tint = new THREE.Color(hex);
    const plate = new THREE.MeshStandardMaterial({
      color: tint.clone().multiplyScalar(0.28),
      roughness: 0.42,
      metalness: 0.34,
      emissive: tint,
      emissiveIntensity: 0.22,
      transparent: true,
      opacity: 1,
      depthWrite: false,
    });
    const slab = new THREE.Mesh(new THREE.BoxGeometry(MATRIX, 0.08, MATRIX), plate);
    slab.position.y = -0.05;
    slab.renderOrder = 3;
    const face = new THREE.Mesh(new THREE.PlaneGeometry(MATRIX * 0.96, MATRIX * 0.96), createMatrixMaterial(hex));
    face.rotation.x = -Math.PI / 2;
    face.position.y = 0.002;
    face.renderOrder = 4;
    object.add(slab, face);
    return { object, material: face.material as THREE.ShaderMaterial, plate };
  }

  const fromA = hbmMatrixCenter(hbmModule("A"));
  const fromB = hbmMatrixCenter(hbmModule("B"));
  const fromC = hbmMatrixCenter(hbmModule("C"));
  const ghostA = createSquare("#4c8dff");
  const ghostB = createSquare("#f2a24a");
  const moverA = createSquare("#4c8dff");
  const moverB = createSquare("#f2a24a");
  const moverC = createSquare("#3ddc97");
  ghostA.object.position.set(fromA[0], fromA[1], fromA[2]);
  ghostB.object.position.set(fromB[0], fromB[1], fromB[2]);
  hbmGroup.add(ghostA.object, ghostB.object);
  overlays.add(moverA.object, moverB.object, moverC.object);

  const l2 = new THREE.Group();
  l2.name = "l2";
  gpuPackage.add(l2);

  const l2RimMat = new THREE.MeshStandardMaterial({
    color: 0x5c7084,
    emissive: 0x8eaccc,
    emissiveIntensity: 0.18,
    roughness: 0.4,
    metalness: 0.45,
    envMapIntensity: 0.35,
  });
  const l2Rim = new THREE.Mesh(
    new RoundedBoxGeometry(L2_SLAB.width, 0.12, L2_SLAB.depth, 3, 0.04),
    l2RimMat,
  );
  l2Rim.position.set(L2_SLAB.center[0], L2_SLAB.center[1], L2_SLAB.center[2]);
  l2.add(l2Rim);

  const l2Grid = new THREE.Mesh(
    new THREE.PlaneGeometry(L2_GRID.width, L2_GRID.depth),
    createCacheMaterial(L2_GRID.cellsX, L2_GRID.cellsZ),
  );
  placeHorizontal(l2Grid, [0, L2_SLAB.center[1] + 0.08, 0]);
  l2Grid.renderOrder = 1;
  l2.add(l2Grid);

  const interconnect = new THREE.Group();
  interconnect.name = "interconnect";
  gpuPackage.add(interconnect);

  const flightA = createRibbon(flyPoints(fromA, L2_A.center, 2.9, -1.35), aColor, 0.28);
  const flightB = createRibbon(flyPoints(fromB, L2_B.center, 2.9, 1.35), bColor, 0.28);
  const flightC = createRibbon(flyPoints(C_FORM, fromC, 1.6, 0.35), cColor, 0.28);
  interconnect.add(flightA.mesh, flightB.mesh, flightC.mesh);

  const ribbonA: Ribbon[][] = Array.from({ length: GRID }, () => []);
  const ribbonB: Ribbon[][] = Array.from({ length: GRID }, () => []);
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      const a = createRibbon(aToSm(row, col), aColor, 0.16);
      const b = createRibbon(bToSm(row, col), bColor, 0.16);
      ribbonA[row].push(a);
      ribbonB[col].push(b);
      interconnect.add(a.mesh, b.mesh);
    }
  }

  const smArray = new THREE.Group();
  smArray.name = "smArray";
  gpuPackage.add(smArray);

  const count = GRID * GRID;
  const bodyGeo = new RoundedBoxGeometry(SM_BODY, 0.26, SM_BODY, 2, 0.05);
  const bodies = new THREE.InstancedMesh(
    bodyGeo,
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.62,
      metalness: 0.32,
      envMapIntensity: 0.22,
      emissive: 0x090b10,
      emissiveIntensity: 0.8,
    }),
    count,
  );
  bodies.frustumCulled = false;
  const windows = createWindowMaterial(count);
  const windowGeo = new THREE.PlaneGeometry(0.46, 0.46);
  windowGeo.setAttribute("aProgress", windows.progress);
  windowGeo.setAttribute("aLoad", windows.load);
  windowGeo.setAttribute("aDim", windows.dim);
  windowGeo.setAttribute("aHot", windows.hot);
  const windowMesh = new THREE.InstancedMesh(windowGeo, windows.material, count);
  windowMesh.frustumCulled = false;

  const tiles = createTileMaterial(count);
  const tileGeo = new THREE.PlaneGeometry(TILE_SIZE, TILE_SIZE);
  tileGeo.setAttribute("aFill", tiles.fill);
  tileGeo.setAttribute("aOpacity", tiles.opacity);
  tileGeo.setAttribute("aDim", tiles.dim);
  const tileMesh = new THREE.InstancedMesh(tileGeo, tiles.material, count);
  tileMesh.frustumCulled = false;
  tileMesh.renderOrder = 6;
  smArray.add(bodies, windowMesh, tileMesh);

  const sliceAMat = createSliceMaterial("#4c8dff", GRID, 1);
  const sliceBMat = createSliceMaterial("#f2a24a", 1, GRID);
  const sliceA = new THREE.Mesh(new THREE.PlaneGeometry(MATRIX, TILE_SIZE), sliceAMat);
  const sliceB = new THREE.Mesh(new THREE.PlaneGeometry(TILE_SIZE, MATRIX), sliceBMat);
  placeHorizontal(sliceA, [0, 0, 0]);
  placeHorizontal(sliceB, [0, 0, 0]);
  sliceA.renderOrder = 5;
  sliceB.renderOrder = 5;
  sliceA.visible = false;
  sliceB.visible = false;
  smArray.add(sliceA, sliceB);

  const footprintMat = createFootprintMaterial("#3ddc97");
  const footprint = new THREE.Mesh(new THREE.PlaneGeometry(MATRIX, MATRIX), footprintMat);
  placeHorizontal(footprint, C_FORM);
  footprint.renderOrder = 5;
  overlays.add(footprint);

  const dummy = new THREE.Object3D();
  const home: Vec3[] = [];
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      const id = row * GRID + col;
      const position = smWorld(row, col);
      home.push(position);
      dummy.position.set(position[0], position[1], position[2]);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      bodies.setMatrixAt(id, dummy.matrix);
      dummy.position.set(position[0], position[1] + 0.15, position[2]);
      dummy.rotation.x = -Math.PI / 2;
      dummy.updateMatrix();
      windowMesh.setMatrixAt(id, dummy.matrix);
      bodies.setColorAt(id, new THREE.Color(STATE_COLOR.idle));
    }
  }
  bodies.instanceMatrix.needsUpdate = true;
  windowMesh.instanceMatrix.needsUpdate = true;
  if (bodies.instanceColor) bodies.instanceColor.needsUpdate = true;

  const model = getModel();
  const bytes = kb(matrixBytes());
  const shape = `${model.n}×${model.n}`;
  const labels = {
    hbm: tag(`<b>HBM</b><span class="kicker">6 low stacks</span>`, [hbmModule("A").x - 1.65, 1.75, 0], "hw"),
    hbmRight: tag(`<b>HBM</b><span class="kicker">beside die</span>`, [hbmModule("B").x + 1.65, 1.75, 0], "hw"),
    a: tag(`<b>A · ${shape}</b><span>${bytes} · FP32</span>`, [fromA[0], fromA[1] + 0.7, fromA[2]], "a"),
    b: tag(`<b>B · ${shape}</b><span>${bytes} · FP32</span>`, [fromB[0], fromB[1] + 0.7, fromB[2]], "b"),
    l2: tag(l2TagHtml(false), [0, 1.05, L2_SLAB.depth / 2 + 0.55], "hw"),
    gpu: tag(`<b>GPU package</b>`, [-(DIE_FRAME.width / 2) - 1.15, 0.9, DIE_FRAME.depth / 2 - 0.55], "hw"),
    smTop: tag(`<b>SMs</b>`, [0, 1.2, -(BANK_CENTER_Z + SM_BANK_DEPTH * 0.28)], "hw"),
    smBottom: tag(`<b>SMs</b>`, [0, 1.2, BANK_CENTER_Z + SM_BANK_DEPTH * 0.28], "hw"),
    working: tag(
      `<b>A + B in L2</b><span>${shape} · ${bytes} each</span><span>${cacheUsed(residentBytes())}</span>`,
      [-(SM_BANK_WIDTH / 2) - 0.95, 1.75, 0.18],
    ),
    panelA: tag(`<b>A panel</b><span>into this SM</span>`, [0, 2, 0], "a"),
    panelB: tag(`<b>B panel</b><span>into this SM</span>`, [0, 2, 0], "b"),
    tileC: tag(`<b>C · 32×32</b><span>${kb(cTileBytes())}</span>`, [0, 2, 0], "c"),
    formed: tag(
      `<b>C · ${shape}</b><span>8×8 tiles · one matrix</span><span>${bytes}</span>`,
      [C_FORM[0], C_FORM[1] + 0.85, C_FORM[2]],
      "c",
    ),
    stored: tag(`<b>C · ${shape}</b><span>${bytes} · back in HBM</span>`, [fromC[0], fromC[1] + 0.55, fromC[2]], "c"),
  };
  labels.l2.center.set(0.5, 0);
  overlays.add(...Object.values(labels));

  if (debug) {
    const markers: [number, number, number][] = [
      [0, 0, 0x4c8dff],
      [0, 7, 0xf2a24a],
      [7, 0, 0x3ddc97],
      [SM_TOUR[0].row, SM_TOUR[0].col, 0xffffff],
    ];
    for (const [row, col, color] of markers) {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.12, 12, 12),
        new THREE.MeshBasicMaterial({ color }),
      );
      const position = smWorld(row, col);
      marker.position.set(position[0], position[1] + 0.8, position[2]);
      smArray.add(marker);
    }
  }

  const raycaster = new THREE.Raycaster();
  const color = new THREE.Color();
  const inspectTint = new THREE.Color("#d7deea");
  function setRibbon(ribbon: Ribbon, opacity: number, active: number, time: number): void {
    ribbon.material.uniforms.uOpacity.value = opacity;
    ribbon.material.uniforms.uActive.value = active;
    ribbon.material.uniforms.uTime.value = time;
  }

  function park(square: SquarePlate, at: Vec3, opacity: number, progress: number, phase = 0): void {
    square.object.position.set(at[0], at[1], at[2]);
    square.object.visible = opacity > 0.02;
    square.plate.opacity = opacity;
    const inFlight = Math.sin(Math.PI * Math.min(1, Math.max(0, progress)));
    const breathe = 1 + inFlight * 0.045;
    square.object.scale.setScalar(breathe);
    square.object.rotation.y = Math.sin(progress * Math.PI * 2 + phase) * inFlight * 0.055;
    square.object.rotation.z = Math.sin(progress * Math.PI + phase) * inFlight * 0.035;
  }

  function update(sample: TimelineSample, inspect: PickHit | null): void {
    applyMatrixLook(ghostA.material, sample.hbmA);
    applyMatrixLook(ghostB.material, sample.hbmB);
    ghostA.plate.opacity = sample.hbmA.opacity;
    ghostB.plate.opacity = sample.hbmB.opacity;
    ghostA.object.visible = sample.hbmA.opacity > 0.02;
    ghostB.object.visible = sample.hbmB.opacity > 0.02;

    const atA = fly(fromA, L2_A.center, sample.travelA, 2.9, -1.35);
    const atB = fly(fromB, L2_B.center, sample.travelB, 2.9, 1.35);
    const atC = fly(C_FORM, fromC, sample.travelC, 1.6, 0.35);
    park(moverA, atA, sample.travelA > 0.015 ? 1 : 0, sample.travelA, 0.0);
    park(moverB, atB, sample.travelB > 0.015 ? 1 : 0, sample.travelB, 1.2);
    park(moverC, atC, sample.cMover, sample.travelC, 2.1);
    applyMatrixLook(moverA.material, sample.l2A);
    applyMatrixLook(moverB.material, sample.l2B);
    applyMatrixLook(moverC.material, sample.l2C, sample.cMover);

    l2RimMat.emissiveIntensity = 0.16 + sample.l2Pulse * 0.9;
    const pulseScale = 1 + sample.l2Pulse * 0.012;
    l2Rim.scale.set(pulseScale, 1, pulseScale);

    setRibbon(flightA, sample.channelA.opacity, sample.channelA.active, sample.t);
    setRibbon(flightB, sample.channelB.opacity, sample.channelB.active, sample.t);
    setRibbon(flightC, sample.returnFlow.opacity, sample.returnFlow.active, sample.t);
    ribbonA.forEach((rowRibbons, row) => {
      rowRibbons.forEach((ribbon, col) => {
        const on = row === sample.focus.row && col === sample.focus.col ? sample.focus.amount : 0;
        setRibbon(ribbon, on, on, sample.t);
      });
    });
    ribbonB.forEach((colRibbons, col) => {
      colRibbons.forEach((ribbon, row) => {
        const on = row === sample.focus.row && col === sample.focus.col ? sample.focus.amount : 0;
        setRibbon(ribbon, on, on, sample.t + 0.35);
      });
    });

    windows.material.uniforms.uTime.value = sample.t;
    windows.material.uniforms.uContent.value = sample.windowContent;
    const progress = windows.progress.array as Float32Array;
    const load = windows.load.array as Float32Array;
    const dim = windows.dim.array as Float32Array;
    const hot = windows.hot.array as Float32Array;
    const tileFill = tiles.fill.array as Float32Array;
    const tileOpacity = tiles.opacity.array as Float32Array;
    const tileDim = tiles.dim.array as Float32Array;

    for (let i = 0; i < count; i++) {
      const block = sample.blocks[i];
      const row = Math.floor(i / GRID);
      const col = i % GRID;
      const inspected = inspect !== null && inspect.row === row && inspect.col === col;
      progress[i] = block.progress;
      load[i] = block.load;
      dim[i] = block.dim;
      hot[i] = inspected ? Math.max(block.hot, 0.65) : block.hot;
      const done = smooth01((block.progress - 0.5) / 0.42);
      tileFill[i] = 1;
      tileOpacity[i] = done * sample.tileOpacity;
      tileDim[i] = block.dim;

      color.set(STATE_COLOR[block.state]);
      if (inspected) color.lerp(inspectTint, 0.42);
      color.multiplyScalar(0.62 + 0.38 * block.dim);
      bodies.setColorAt(i, color);

      const isFocus = sample.focus.tour > 0 && row === sample.focus.row && col === sample.focus.col;
      let scale = 1;
      if (sample.focus.tour > 0) {
        scale = isFocus ? 1 + 0.09 * sample.focus.amount : 1 - 0.025 * sample.focus.tour;
      }
      if (inspected) scale = Math.max(scale, 1.08);
      const powerLift = isFocus ? sample.focus.amount * 0.095 : 0;

      const origin = home[i];
      dummy.position.set(origin[0], origin[1] + powerLift, origin[2]);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      bodies.setMatrixAt(i, dummy.matrix);

      dummy.position.set(origin[0], origin[1] + 0.15 + powerLift, origin[2]);
      dummy.rotation.set(-Math.PI / 2, 0, 0);
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      windowMesh.setMatrixAt(i, dummy.matrix);

      const onSm: Vec3 = [origin[0], origin[1] + 0.66, origin[2]];
      const slot = cTileSlot(row, col, atC);
      const gather = sample.assemble;
      const arc = Math.sin(Math.PI * gather) * 0.9;
      dummy.position.set(
        onSm[0] + (slot[0] - onSm[0]) * gather,
        onSm[1] + (slot[1] - onSm[1]) * gather + arc,
        onSm[2] + (slot[2] - onSm[2]) * gather,
      );
      dummy.rotation.set(-Math.PI / 2, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      tileMesh.setMatrixAt(i, dummy.matrix);
    }

    windows.progress.needsUpdate = true;
    windows.load.needsUpdate = true;
    windows.dim.needsUpdate = true;
    windows.hot.needsUpdate = true;
    tiles.fill.needsUpdate = true;
    tiles.opacity.needsUpdate = true;
    tiles.dim.needsUpdate = true;
    bodies.instanceMatrix.needsUpdate = true;
    windowMesh.instanceMatrix.needsUpdate = true;
    tileMesh.instanceMatrix.needsUpdate = true;
    if (bodies.instanceColor) bodies.instanceColor.needsUpdate = true;

    const callout = sample.callouts;
    const architecture = (1 - sample.focus.tour).toFixed(3);
    for (const name of ["hbm", "hbmRight", "gpu", "smTop", "smBottom"] as const) {
      labels[name].element.style.opacity = architecture;
    }
    const inFlightToL2 =
      (sample.travelA > 0.015 && sample.travelA < 0.992) ||
      (sample.travelB > 0.015 && sample.travelB < 0.992);
    const smWork = sample.blocks.some((block) => block.load > 0.04 || block.progress > 0.04);
    const cMotion = sample.travelC > 0.015 || sample.assemble > 0.02;
    const l2Html = l2TagHtml(inFlightToL2 || sample.focus.tour > 0 || smWork || cMotion);
    if (labels.l2.element.innerHTML !== l2Html) labels.l2.element.innerHTML = l2Html;
    labels.a.position.set(atA[0], atA[1] + 0.55, atA[2] - MATRIX * 0.78);
    labels.b.position.set(atB[0], atB[1] + 0.55, atB[2] - MATRIX * 0.78);
    labels.working.element.style.opacity = callout.workingSet.toFixed(3);
    labels.stored.element.style.opacity = callout.stored.toFixed(3);
    const matrixCenter = atC;
    labels.formed.position.set(matrixCenter[0], matrixCenter[1] + 0.72, matrixCenter[2] - MATRIX * 0.15);
    labels.formed.element.style.opacity = (sample.assemble * (1 - sample.travelC)).toFixed(3);
    footprint.position.set(C_FORM[0], C_FORM[1], C_FORM[2]);
    footprintMat.uniforms.uOpacity.value = sample.assemble * (1 - sample.travelC) * (1 - smooth01((sample.assemble - 0.72) / 0.28));
    const showPanels = sample.focus.amount;
    labels.panelA.element.style.opacity = showPanels.toFixed(3);
    labels.panelB.element.style.opacity = showPanels.toFixed(3);
    const focusIndex = sample.focus.row >= 0 ? sample.focus.row * GRID + sample.focus.col : -1;
    const focusBlock = focusIndex >= 0 ? sample.blocks[focusIndex] : null;
    const solid = focusBlock ? focusBlock.load * (1 - smooth01((focusBlock.progress - 0.58) / 0.32)) : 0;
    const ghost = focusBlock ? focusBlock.load * sample.focus.tour * (1 - Math.min(1, solid)) * 0.42 : 0;
    const panelFade = Math.max(solid, ghost);
    const showSlices = sample.focus.tour > 0 && panelFade > 0.02 && focusBlock !== null;
    sliceA.visible = showSlices;
    sliceB.visible = showSlices;
    if (showSlices && focusBlock) {
      const at = smWorld(sample.focus.row, sample.focus.col);
      const y = at[1] + 0.62;
      sliceA.position.set(at[0], y, at[2]);
      sliceB.position.set(at[0], y + 0.01, at[2]);
      sliceA.scale.set(Math.max(focusBlock.load, 0.04), 1, 1);
      sliceB.scale.set(1, Math.max(focusBlock.load, 0.04), 1);
      sliceAMat.uniforms.uOpacity.value = panelFade;
      sliceBMat.uniforms.uOpacity.value = panelFade;
    }
    const tileDone = focusBlock ? smooth01((focusBlock.progress - 0.5) / 0.42) : 0;
    labels.tileC.center.set(0, 0.5);
    labels.tileC.element.style.opacity = (tileDone * sample.focus.tour * sample.tileOpacity).toFixed(3);
    if (focusBlock && sample.focus.row >= 0) {
      const at = smWorld(sample.focus.row, sample.focus.col);
      labels.tileC.position.set(at[0] + MATRIX * 0.5 + 0.18, at[1] + 0.7, at[2]);
      labels.tileC.element.innerHTML = `<b>C · 32×32</b><span>${kb(cTileBytes())}</span>`;
    }
    if (sample.focus.row >= 0) {
      const { n, tileM, tileN } = model;
      const aStart = sample.focus.row * tileM;
      const bStart = sample.focus.col * tileN;
      const [v0, v1] = rowBand(sample.focus.row);
      const onA = planePoint(L2_A.center, MATRIX, MATRIX, 0.5, (v0 + v1) / 2);
      labels.panelA.position.set(L2_A.center[0] - MATRIX * 1.55, onA[1] + 0.4, onA[2]);
      labels.panelA.element.innerHTML = `<b>A[${aStart}:${aStart + tileM}, :] · ${tileM}×${n}</b><span>${cacheUsed(aPanelBytes())}</span>`;
      const [u0, u1] = colBand(sample.focus.col);
      const onB = planePoint(L2_B.center, MATRIX, MATRIX, (u0 + u1) / 2, 0);
      labels.panelB.position.set(onB[0], onB[1] + 0.4, onB[2] + 0.95);
      labels.panelB.element.innerHTML = `<b>B[:, ${bStart}:${bStart + tileN}] · ${n}×${tileN}</b><span>${cacheUsed(bPanelBytes())}</span>`;
    }
  }

  function pick(ndc: THREE.Vector2, camera: THREE.Camera): PickHit | null {
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObject(bodies, false);
    const hit = hits[0];
    if (!hit || hit.instanceId === undefined) return null;
    return { row: Math.floor(hit.instanceId / GRID), col: hit.instanceId % GRID };
  }

  return { update, pick };
}
