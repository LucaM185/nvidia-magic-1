import * as THREE from "three";

export interface FreeArea {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/**
 * Zoom and shift a perspective camera so `bounds`, seen from the camera's
 * current pose, fills the screen rectangle the HUD cards leave free.
 */
export function fitCamera(camera: THREE.PerspectiveCamera, bounds: THREE.Box3, free: FreeArea, width: number, height: number): void {
  const freeW = Math.max(200, free.right - free.left);
  const freeH = Math.max(160, free.bottom - free.top);

  camera.aspect = width / height;
  camera.zoom = 1;
  camera.clearViewOffset();
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  const ndc = new THREE.Box2();
  const corner = new THREE.Vector3();
  for (const x of [bounds.min.x, bounds.max.x]) {
    for (const y of [bounds.min.y, bounds.max.y]) {
      for (const z of [bounds.min.z, bounds.max.z]) {
        corner.set(x, y, z).project(camera);
        ndc.expandByPoint(new THREE.Vector2(corner.x, corner.y));
      }
    }
  }
  const zoom = Math.min(freeW / (((ndc.max.x - ndc.min.x) / 2) * width), freeH / (((ndc.max.y - ndc.min.y) / 2) * height));
  const centerX = ((((ndc.min.x + ndc.max.x) / 2) * zoom + 1) / 2) * width;
  const centerY = ((1 - ((ndc.min.y + ndc.max.y) / 2) * zoom) / 2) * height;
  camera.zoom = zoom;
  camera.setViewOffset(width, height, centerX - (free.left + free.right) / 2, centerY - (free.top + free.bottom) / 2, width, height);
  camera.updateProjectionMatrix();
}
