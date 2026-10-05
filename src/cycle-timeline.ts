/**
 * Cycle-level timeline shared by the two "inside one SM" pages: a few hardware
 * rows on top, one row per resident warp below, and a playhead you can drag.
 * Every bar is a fixed cycle window; per frame only its fill and state change.
 */

export type SegmentKind = "issue" | "wait" | "latency" | "x" | "w" | "compute" | "result" | "store" | "hold";

export interface Segment {
  start: number;
  end: number;
  kind: SegmentKind;
  label?: string;
  /** Overrides the kind's color, e.g. a scheduler issue tinted by its warp. */
  color?: string;
}

export interface Row {
  name: string;
  /** Row accent: the warp's color, or a hardware tone. */
  color: string;
  segments: Segment[];
  /** Text written across a track that has no segments. */
  empty?: string;
}

export interface RowState {
  text: string;
  /** Highlights the row, e.g. the warp the scheduler is issuing for. */
  active?: boolean;
}

export interface TimelineSpec {
  totalCycles: number;
  ticks: number[];
  hardware: Row[];
  warps: Row[];
  onSeek?: (cycle: number) => void;
}

export interface CycleTimeline {
  /** A negative cycle hides the playhead (the kernel has not started). States cover hardware rows, then warps. */
  update(cycle: number, states: RowState[]): void;
}

const KIND_TEXT: Record<SegmentKind, string> = {
  issue: "issue",
  wait: "waiting",
  latency: "latency",
  x: "copy",
  w: "copy",
  compute: "compute",
  result: "reduce",
  store: "store",
  hold: "held in registers",
};

interface LiveSegment {
  element: HTMLElement;
  start: number;
  end: number;
  fill: number;
  now: boolean;
}

export function createCycleTimeline(root: HTMLElement, spec: TimelineSpec): CycleTimeline {
  const percent = (cycle: number) => `${(cycle / spec.totalCycles) * 100}%`;
  const segments: LiveSegment[] = [];
  const rowNodes: HTMLElement[] = [];
  const stateNodes: HTMLElement[] = [];
  root.classList.add("tl");

  const grid = document.createElement("div");
  grid.className = "tl-grid";
  for (const tick of spec.ticks) {
    const line = document.createElement("i");
    line.style.left = percent(tick);
    grid.append(line);
  }
  root.append(grid);

  const divider = (text: string) => {
    const node = document.createElement("div");
    node.className = "tl-divider";
    node.innerHTML = `<span>${text}</span>`;
    root.append(node);
  };

  const addRow = (row: Row, group: "hw" | "warp") => {
    const node = document.createElement("div");
    node.className = `tl-row ${group}`;
    node.style.setProperty("--row", row.color);
    const name = document.createElement("span");
    name.className = "tl-name";
    name.innerHTML = `<i></i>${row.name}`;
    const state = document.createElement("span");
    state.className = "tl-state";
    const track = document.createElement("div");
    track.className = "tl-track";
    if (row.empty) {
      const empty = document.createElement("em");
      empty.textContent = row.empty;
      track.append(empty);
    }
    for (const segment of row.segments) {
      const bar = document.createElement("i");
      bar.className = `tl-seg ${segment.kind}`;
      bar.style.left = percent(segment.start);
      bar.style.width = percent(segment.end - segment.start);
      if (segment.color) bar.style.setProperty("--c", segment.color);
      const cycles = segment.end - segment.start;
      bar.title = `${row.name} · ${segment.label || KIND_TEXT[segment.kind]} · cycles ${segment.start}–${segment.end} (${cycles})`;
      if (segment.label) {
        const label = document.createElement("span");
        label.textContent = segment.label;
        bar.append(label);
      }
      track.append(bar);
      segments.push({ element: bar, start: segment.start, end: segment.end, fill: -1, now: false });
    }
    node.append(name, state, track);
    root.append(node);
    rowNodes.push(node);
    stateNodes.push(state);
  };

  divider("Hardware");
  spec.hardware.forEach((row) => addRow(row, "hw"));
  divider("Warps");
  spec.warps.forEach((row) => addRow(row, "warp"));

  const axis = document.createElement("div");
  axis.className = "tl-axis";
  for (const tick of spec.ticks) {
    const label = document.createElement("span");
    label.style.left = percent(tick);
    label.textContent = tick === spec.ticks[spec.ticks.length - 1] ? `${tick} cycles` : String(tick);
    axis.append(label);
  }
  root.append(axis);

  const playhead = document.createElement("div");
  playhead.className = "tl-playhead";
  playhead.setAttribute("aria-hidden", "true");
  const readout = document.createElement("b");
  playhead.append(readout);
  root.append(playhead);

  // Drag anywhere across the tracks to seek.
  if (spec.onSeek) {
    root.classList.add("seekable");
    const seek = (event: PointerEvent) => {
      const track = root.querySelector<HTMLElement>(".tl-track")!.getBoundingClientRect();
      const u = Math.min(1, Math.max(0, (event.clientX - track.left) / track.width));
      spec.onSeek?.(u * spec.totalCycles);
    };
    root.addEventListener("pointerdown", (event) => {
      const track = root.querySelector<HTMLElement>(".tl-track")!.getBoundingClientRect();
      if (event.clientX < track.left - 4) return;
      root.setPointerCapture(event.pointerId);
      seek(event);
    });
    root.addEventListener("pointermove", (event) => {
      if (root.hasPointerCapture(event.pointerId)) seek(event);
    });
  }

  // Labels only show where their bar is wide enough to hold them.
  const labels = Array.from(root.querySelectorAll<HTMLElement>(".tl-seg span"));
  const fitLabels = () => {
    for (const label of labels) label.style.visibility = label.scrollWidth > label.clientWidth + 1 ? "hidden" : "";
  };
  new ResizeObserver(fitLabels).observe(root);
  document.fonts?.ready.then(fitLabels);

  let lastReadout = "";
  return {
    update(cycle, states) {
      const started = cycle >= 0;
      const c = Math.max(0, cycle);
      root.classList.toggle("before", !started);
      for (const segment of segments) {
        const fill = Math.round(Math.min(1, Math.max(0, (c - segment.start) / (segment.end - segment.start))) * 1000) / 10;
        if (fill !== segment.fill) {
          segment.fill = fill;
          segment.element.style.setProperty("--fill", `${fill}%`);
        }
        const now = started && c >= segment.start && c < segment.end;
        if (now !== segment.now) {
          segment.now = now;
          segment.element.classList.toggle("now", now);
        }
      }
      root.style.setProperty("--p", String(c / spec.totalCycles));
      playhead.classList.toggle("flip", c / spec.totalCycles > 0.88);
      const text = `cycle ${Math.floor(c)}`;
      if (text !== lastReadout) readout.textContent = lastReadout = text;
      states.forEach((state, index) => {
        if (stateNodes[index].textContent !== state.text) stateNodes[index].textContent = state.text;
        rowNodes[index].classList.toggle("active", Boolean(state.active));
      });
    },
  };
}
