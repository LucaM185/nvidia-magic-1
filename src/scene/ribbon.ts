import * as THREE from "three";

export function ribbonGeometry(
  points: THREE.Vector3[],
  width: number,
  segments = 36,
): THREE.BufferGeometry {
  const curve = new THREE.CatmullRomCurve3(points, false, "catmullrom", 0.35);
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const up = new THREE.Vector3(0, 1, 0);
  const tangent = new THREE.Vector3();
  const side = new THREE.Vector3();
  const left = new THREE.Vector3();
  const right = new THREE.Vector3();

  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const point = curve.getPoint(t);
    curve.getTangent(t, tangent);
    side.crossVectors(up, tangent);
    if (side.lengthSq() < 1e-8) side.set(1, 0, 0);
    side.normalize();
    left.copy(point).addScaledVector(side, width * 0.5);
    right.copy(point).addScaledVector(side, -width * 0.5);
    positions.push(left.x, left.y, left.z, right.x, right.y, right.z);
    uvs.push(t, 0, t, 1);
  }

  for (let i = 0; i < segments; i++) {
    const a = i * 2;
    indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  return geometry;
}

const RIBBON_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const RIBBON_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform vec3 uColor;
  uniform float uOpacity;
  uniform float uActive;
  uniform float uTime;

  void main() {
    float across = abs(vUv.y - 0.5) * 2.0;
    float core = smoothstep(1.0, 0.08, across);
    float hot = smoothstep(0.48, 0.0, across);
    float edgeFade = smoothstep(0.0, 0.055, vUv.x) * (1.0 - smoothstep(0.94, 1.0, vUv.x));
    float stream = fract(vUv.x * 2.0 - uTime * 0.42);
    float packet = smoothstep(0.18, 0.0, abs(stream - 0.1));
    float streamB = fract(vUv.x * 2.0 - uTime * 0.42 + 0.5);
    float packetB = smoothstep(0.14, 0.0, abs(streamB - 0.1));
    float packets = (packet + packetB * 0.7) * uActive;
    float intensity = core * (0.08 + hot * 0.06 + packets * 0.68) * uOpacity * edgeFade;
    gl_FragColor = vec4(uColor * intensity * 1.25, 1.0);
  }
`;

export interface Ribbon {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
}

export function createRibbon(points: THREE.Vector3[], color: THREE.Color, width: number): Ribbon {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: color },
      uOpacity: { value: 0 },
      uActive: { value: 0 },
      uTime: { value: 0 },
    },
    vertexShader: RIBBON_VERT,
    fragmentShader: RIBBON_FRAG,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
  const mesh = new THREE.Mesh(ribbonGeometry(points, width), material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 5;
  return { mesh, material };
}
