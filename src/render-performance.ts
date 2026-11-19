/**
 * Render at native Retina density: at 1× the lane rows, cell edges and
 * silkscreen text blur visibly. Capped at 2 so 3× displays don't pay for it.
 */
export const MAX_PIXEL_RATIO = 2;

/** Cap ProMotion/VRR displays at 60 fps instead of doing the scene work at 120 Hz. */
export const MIN_FRAME_INTERVAL = 1000 / 60 - 1;

/** DOM dashboards do not need to be rewritten for every WebGL frame. */
export const HUD_FRAME_INTERVAL = 1000 / 20;
