import * as THREE from "three";
import type { MatrixLook } from "../model/timeline";

const MATRIX_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const MATRIX_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform float uFill;
  uniform float uScan;
  uniform vec4 uRegion;
  uniform float uRegionStrength;
  uniform float uPanelEdges;
  uniform vec2 uCells;
  uniform vec2 uPanels;
  uniform float uLanes[8];
  uniform float uLaneAxis;

  float gridFactor(vec2 uv, vec2 cells) {
    vec2 q = uv * cells;
    vec2 fw = fwidth(q);
    float coverage = max(fw.x, fw.y);
    if (coverage > 0.48) return 0.0;
    vec2 dist = abs(fract(q - 0.5) - 0.5);
    vec2 aa = smoothstep(fw * 1.45, fw * 0.15, dist);
    float line = max(aa.x, aa.y);
    float fade = 1.0 - smoothstep(0.22, 0.48, coverage);
    return line * fade;
  }

  float inRegion(vec2 uv, vec4 r) {
    vec2 d = fwidth(uv) * 2.0;
    float x = smoothstep(r.x - d.x, r.x + d.x, uv.x) * (1.0 - smoothstep(r.z - d.x, r.z + d.x, uv.x));
    float y = smoothstep(r.y - d.y, r.y + d.y, uv.y) * (1.0 - smoothstep(r.w - d.y, r.w + d.y, uv.y));
    return x * y;
  }

  float laneAt(int i) {
    if (i == 0) return uLanes[0];
    if (i == 1) return uLanes[1];
    if (i == 2) return uLanes[2];
    if (i == 3) return uLanes[3];
    if (i == 4) return uLanes[4];
    if (i == 5) return uLanes[5];
    if (i == 6) return uLanes[6];
    return uLanes[7];
  }

  float lanePeak() {
    return max(
      max(max(uLanes[0], uLanes[1]), max(uLanes[2], uLanes[3])),
      max(max(uLanes[4], uLanes[5]), max(uLanes[6], uLanes[7]))
    );
  }

  void main() {
    float filled = smoothstep(1.0 - uFill - 0.02, 1.0 - uFill, vUv.y);
    float presence = mix(0.08, 1.0, filled);
    vec3 col = uColor * (0.16 + 0.62 * presence);

    float panels = gridFactor(vUv, uPanels);
    float cells = gridFactor(vUv, uCells);
    col += uColor * panels * (0.08 + 1.05 * uPanelEdges);
    col += vec3(0.9) * cells * 0.55;

    float reg = inRegion(vUv, uRegion);
    float shade = mix(1.0, mix(0.2, 1.0, reg), uRegionStrength);
    col *= shade;
    col += (uColor + vec3(0.25)) * reg * uRegionStrength;

    float q = uLaneAxis < 1.5 ? (1.0 - vUv.y) : vUv.x;
    int lane = int(min(floor(q * 8.0), 7.0));
    float amp = laneAt(lane);
    float peak = lanePeak();
    float along = abs(fract(q * 8.0) - 0.5);
    float band = smoothstep(0.48, 0.22, along);
    float lit = band * amp;
    float laneDim = mix(1.0, mix(0.3, 1.0, smoothstep(0.02, 0.2, lit)), smoothstep(0.05, 0.22, peak));
    col *= laneDim;
    col += (uColor + vec3(0.2)) * lit;
    float core = smoothstep(0.2, 0.02, along) * amp;
    col += vec3(0.95) * core * 0.22;

    if (uScan >= 0.0) {
      float band = smoothstep(0.055, 0.0, abs(vUv.y - uScan));
      col += vec3(0.9) * band;
    }

    float edge = min(min(vUv.x, vUv.y), min(1.0 - vUv.x, 1.0 - vUv.y));
    float border = 1.0 - smoothstep(0.0, fwidth(edge) * 2.2, edge);
    col += uColor * border * 0.95;

    float alpha = uOpacity * (0.22 + 0.78 * presence);
    alpha = max(alpha, uOpacity * max(panels * (0.4 + 0.55 * uPanelEdges), cells * 0.65));
    if (alpha < 0.012) discard;
    gl_FragColor = vec4(col, alpha);
  }
`;

export function createMatrixMaterial(
  color: string,
  cellsX = 32,
  cellsY = cellsX,
  panelsX = 8,
  panelsY = panelsX,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: 1 },
      uFill: { value: 1 },
      uScan: { value: -1 },
      uRegion: { value: new THREE.Vector4() },
      uRegionStrength: { value: 0 },
      uPanelEdges: { value: 0.14 },
      uCells: { value: new THREE.Vector2(cellsX, cellsY) },
      uPanels: { value: new THREE.Vector2(panelsX, panelsY) },
      uLanes: { value: [0, 0, 0, 0, 0, 0, 0, 0] },
      uLaneAxis: { value: 1 },
    },
    vertexShader: MATRIX_VERT,
    fragmentShader: MATRIX_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: true,
  });
}

export function applyMatrixLook(material: THREE.ShaderMaterial, look: MatrixLook, opacityScale = 1): void {
  material.uniforms.uOpacity.value = look.opacity * opacityScale;
  material.uniforms.uFill.value = look.fill;
  material.uniforms.uScan.value = look.scan;
  material.uniforms.uRegionStrength.value = look.regionStrength;
  material.uniforms.uPanelEdges.value = look.panelEdges;
  material.uniforms.uLaneAxis.value = look.laneAxis;
  const lanes = material.uniforms.uLanes.value as number[];
  for (let i = 0; i < 8; i++) lanes[i] = look.lanes[i];
  const region = material.uniforms.uRegion.value as THREE.Vector4;
  const rect = look.region ?? [0, 0, 0, 0];
  region.set(rect[0], rect[1], rect[2], rect[3]);
}

const CACHE_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const CACHE_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform vec2 uCells;

  void main() {
    vec2 q = vUv * uCells;
    vec2 fw = max(fwidth(q), vec2(0.02));
    vec2 dist = min(fract(q), 1.0 - fract(q));
    vec2 line = 1.0 - smoothstep(fw * 0.35, fw * 1.35, dist);
    float grid = max(line.x, line.y);
    vec2 cell = floor(q);
    float checker = mod(cell.x + cell.y, 2.0);
    vec3 base = mix(vec3(0.11, 0.14, 0.18), vec3(0.15, 0.18, 0.23), checker);
    vec3 col = base + vec3(0.55, 0.68, 0.82) * grid * 0.55;
    gl_FragColor = vec4(col, 0.92);
  }
`;

export function createCacheMaterial(cellsX: number, cellsZ: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uCells: { value: new THREE.Vector2(cellsX, cellsZ) },
    },
    vertexShader: CACHE_VERT,
    fragmentShader: CACHE_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
}

const WINDOW_VERT = /* glsl */ `
  attribute float aProgress;
  attribute float aLoad;
  attribute float aDim;
  attribute float aHot;
  varying vec2 vUv;
  varying float vProgress;
  varying float vLoad;
  varying float vDim;
  varying float vHot;
  void main() {
    vUv = uv;
    vProgress = aProgress;
    vLoad = aLoad;
    vDim = aDim;
    vHot = aHot;
    vec4 local = instanceMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * modelViewMatrix * local;
  }
`;

const WINDOW_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  varying float vProgress;
  varying float vLoad;
  varying float vDim;
  varying float vHot;
  uniform float uTime;
  uniform float uContent;
  uniform vec3 uColorA;
  uniform vec3 uColorB;
  uniform vec3 uColorC;

  void main() {
    vec2 uv = vUv;
    vec3 col = vec3(0.015, 0.017, 0.022);
    float edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
    float rim = smoothstep(0.72, 0.96, edge);
    col += vec3(0.55, 0.64, 0.76) * rim * 0.7;

    float load = clamp(vLoad, 0.0, 1.0) * uContent;
    float done = smoothstep(0.48, 0.92, clamp(vProgress, 0.0, 1.0));
    float inputs = load * (1.0 - done);
    // Decode: x is one row and W is one 256×32 panel. Their result is 1×32.
    float aBand = step(0.06, uv.x) * step(uv.x, 0.94) * step(0.44, uv.y) * step(uv.y, 0.56);
    float bBand = step(0.44, uv.x) * step(uv.x, 0.56) * step(0.06, uv.y) * step(uv.y, 0.94);
    float cTile = step(0.44, uv.x) * step(uv.x, 0.56) * step(0.44, uv.y) * step(uv.y, 0.56);
    col += uColorA * aBand * inputs;
    col += uColorB * bBand * inputs;
    col += uColorC * cTile * done * uContent;

    float pulse = 0.5 + 0.5 * sin(uTime * 5.2);
    col += vec3(0.45, 0.68, 0.95) * pulse * vHot * (0.12 + rim * 0.22);

    float dim = mix(0.42, 1.0, clamp(vDim, 0.0, 1.0));
    gl_FragColor = vec4(col * dim, 1.0);
  }
`;

export interface WindowMaterial {
  material: THREE.ShaderMaterial;
  progress: THREE.InstancedBufferAttribute;
  load: THREE.InstancedBufferAttribute;
  dim: THREE.InstancedBufferAttribute;
  hot: THREE.InstancedBufferAttribute;
}

export function createWindowMaterial(count: number): WindowMaterial {
  const progress = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  const load = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  const dim = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  const hot = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  dim.array.fill(1);

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uContent: { value: 1 },
      uColorA: { value: new THREE.Color("#4c8dff") },
      uColorB: { value: new THREE.Color("#f2a24a") },
      uColorC: { value: new THREE.Color("#3ddc97") },
    },
    vertexShader: WINDOW_VERT,
    fragmentShader: WINDOW_FRAG,
    side: THREE.DoubleSide,
    toneMapped: false,
  });

  return { material, progress, load, dim, hot };
}

const TILE_VERT = /* glsl */ `
  attribute float aFill;
  attribute float aOpacity;
  attribute float aDim;
  varying vec2 vUv;
  varying float vFill;
  varying float vOpacity;
  varying float vDim;
  void main() {
    vUv = uv;
    vFill = aFill;
    vOpacity = aOpacity;
    vDim = aDim;
    vec4 local = instanceMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * modelViewMatrix * local;
  }
`;

const TILE_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  varying float vFill;
  varying float vOpacity;
  varying float vDim;
  uniform vec3 uColor;

  // One 1×32 result segment: four visible groups across, exactly one row high.
  float gridFactor(vec2 uv, vec2 cells) {
    vec2 q = uv * cells;
    vec2 fw = fwidth(q);
    float coverage = max(fw.x, fw.y);
    if (coverage > 0.48) return 0.0;
    vec2 dist = abs(fract(q - 0.5) - 0.5);
    vec2 aa = smoothstep(fw * 1.45, fw * 0.15, dist);
    float line = max(aa.x, aa.y);
    float fade = 1.0 - smoothstep(0.22, 0.48, coverage);
    return line * fade;
  }

  void main() {
    vec3 col = uColor * 0.78;
    float cells = gridFactor(vUv, vec2(4.0, 1.0));
    col += vec3(0.9) * cells * 0.55;
    float alpha = vFill * vOpacity * mix(0.55, 1.0, vDim);
    alpha = max(alpha, vOpacity * cells * 0.65 * mix(0.55, 1.0, vDim));
    if (alpha < 0.02) discard;
    gl_FragColor = vec4(col, alpha);
  }
`;

export interface TileMaterial {
  material: THREE.ShaderMaterial;
  fill: THREE.InstancedBufferAttribute;
  opacity: THREE.InstancedBufferAttribute;
  dim: THREE.InstancedBufferAttribute;
}

export function createTileMaterial(count: number): TileMaterial {
  const fill = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  const opacity = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  const dim = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  dim.array.fill(1);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color("#3ddc97") },
    },
    vertexShader: TILE_VERT,
    fragmentShader: TILE_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: true,
  });
  return { material, fill, opacity, dim };
}

const SLICE_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const SLICE_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform vec2 uCells;

  void main() {
    vec2 q = vUv * uCells;
    vec2 fw = max(fwidth(q), vec2(0.02));
    vec2 dist = min(fract(q), 1.0 - fract(q));
    vec2 line = 1.0 - smoothstep(fw * 0.2, fw * 1.1, dist);
    float grid = max(line.x, line.y);
    vec2 edge = min(vUv, 1.0 - vUv);
    float rim = 1.0 - smoothstep(0.0, fwidth(edge.x + edge.y) * 1.6, min(edge.x, edge.y));
    vec3 col = uColor * (0.55 + 0.45 * grid) + vec3(0.95) * rim;
    float alpha = uOpacity * (0.72 + 0.28 * max(grid, rim));
    if (alpha < 0.02) discard;
    gl_FragColor = vec4(col, alpha);
  }
`;

const FOOT_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform vec3 uColor;
  uniform float uOpacity;

  void main() {
    vec2 q = vUv * 8.0;
    vec2 fw = max(fwidth(q), vec2(0.03));
    vec2 dist = min(fract(q), 1.0 - fract(q));
    float grid = max(
      1.0 - smoothstep(fw.x * 0.25, fw.x * 1.2, dist.x),
      1.0 - smoothstep(fw.y * 0.25, fw.y * 1.2, dist.y)
    );
    vec2 edge = min(vUv, 1.0 - vUv);
    float rim = 1.0 - smoothstep(0.0, 0.018, min(edge.x, edge.y));
    float mark = max(rim, grid * 0.55);
    if (mark < 0.04) discard;
    gl_FragColor = vec4(uColor, mark * uOpacity);
  }
`;

/** Empty 8×8 outline the green tiles lock into. */
export function createFootprintMaterial(color: string, cellsX = 8, cellsY = cellsX): THREE.ShaderMaterial {
  const fragmentShader = FOOT_FRAG.replace("vUv * 8.0", `vUv * vec2(${cellsX.toFixed(1)}, ${cellsY.toFixed(1)})`);
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: 0 },
    },
    vertexShader: SLICE_VERT,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
}

/** A 32×256 or 256×32 panel, gridded into the eight 32-wide cells. */
export function createSliceMaterial(color: string, cellsX: number, cellsY: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: 0 },
      uCells: { value: new THREE.Vector2(cellsX, cellsY) },
    },
    vertexShader: SLICE_VERT,
    fragmentShader: SLICE_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
}
