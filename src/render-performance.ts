/**
 * The visualizations are animation-limited rather than resolution-limited.
 * Rendering at native Retina density almost doubles both dimensions on a
 * MacBook, while the extra pixels are barely visible in these geometric scenes.
 */
export const MAX_PIXEL_RATIO = 1;

/** Cap ProMotion/VRR displays at 60 fps instead of doing the scene work at 120 Hz. */
export const MIN_FRAME_INTERVAL = 1000 / 60 - 1;

/** DOM dashboards do not need to be rewritten for every WebGL frame. */
export const HUD_FRAME_INTERVAL = 1000 / 20;
