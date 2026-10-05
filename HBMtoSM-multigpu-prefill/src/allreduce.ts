/**
 * All-reduce schedules shared by the multi-GPU prefill and decode pages.
 * Every algorithm runs over the same PCIe switch, so any GPU can reach any other in one hop.
 */

export type Algorithm = "ring" | "twoshot" | "root" | "oneshot";
export type Stage = "reduce" | "gather" | "exchange";

export interface LinkSpec {
  label: string;
  /** GB/s, which is also kB/ms: multiplied by 1000 it gives bytes per microsecond. */
  bandwidth: number;
  /** Microseconds of fixed cost per sequential hop. */
  latency: number;
}

interface StageSpec {
  stage: Stage;
  steps: number;
  /** Animation seconds for the whole stage. */
  seconds: number;
}

interface AlgorithmSpec {
  name: string;
  tagline: string;
  stages: StageSpec[];
  /** Sequential link crossings, each paying the fixed path latency. */
  hops: number;
  /** Bytes the busiest link carries per hop, as a multiple of one rank's payload. */
  share: number;
}

export const RANKS = 8;
export const ROOT = 0;
/** Ring order walks around the bench, so every hop goes to a physical neighbour. */
export const RING = [0, 1, 2, 3, 7, 6, 5, 4];
const PEERS = RANKS - 1;
/** Share of each step after a packet lands, while the receiver adds or stores it. */
const LANDING = 0.14;

export const ALGORITHMS: Record<Algorithm, AlgorithmSpec> = {
  ring: {
    name: "Ring",
    tagline: "Bandwidth-optimal",
    stages: [
      { stage: "reduce", steps: PEERS, seconds: PEERS * 1.3 },
      { stage: "gather", steps: PEERS, seconds: PEERS * 1.3 },
    ],
    hops: 2 * PEERS,
    share: 1 / RANKS,
  },
  twoshot: {
    name: "Two-shot",
    tagline: "Bandwidth-optimal in 2 hops",
    stages: [
      { stage: "reduce", steps: 1, seconds: 3.6 },
      { stage: "gather", steps: 1, seconds: 3.6 },
    ],
    hops: 2,
    share: PEERS / RANKS,
  },
  root: {
    name: "Root",
    tagline: "2 hops · one link carries everything",
    stages: [
      { stage: "reduce", steps: 1, seconds: 4.8 },
      { stage: "gather", steps: 1, seconds: 4.8 },
    ],
    hops: 2,
    share: PEERS,
  },
  oneshot: {
    name: "One-shot",
    tagline: "Latency-optimal · 1 hop",
    stages: [{ stage: "exchange", steps: 1, seconds: 5.2 }],
    hops: 1,
    share: PEERS,
  },
};

export interface Timing {
  fixed: number;
  wire: number;
  total: number;
}

/** Microseconds for one all-reduce of `payload` bytes per rank. */
export function timing(algorithm: Algorithm, link: LinkSpec, payload: number): Timing {
  const { hops, share } = ALGORITHMS[algorithm];
  const fixed = hops * link.latency;
  const wire = (hops * share * payload) / (link.bandwidth * 1000);
  return { fixed, wire, total: fixed + wire };
}

export function collectiveSeconds(algorithm: Algorithm): number {
  return ALGORITHMS[algorithm].stages.reduce((sum, stage) => sum + stage.seconds, 0);
}

/** Animation seconds from the collective's start until its first stage ends. */
export function firstStageSeconds(algorithm: Algorithm): number {
  return ALGORITHMS[algorithm].stages[0].seconds;
}

export interface Transfer {
  source: number;
  destination: number;
  /** One slice of the payload, or null for the whole thing. */
  chunk: number | null;
  /** Slice of the step's travel time this packet moves in, so shared links visibly queue. */
  window: [number, number];
}

export interface CollectiveFrame {
  stage: Stage;
  /** Completed steps in this stage; also the index of the step in flight. */
  step: number;
  transfers: Transfer[];
  /** Linear 0..1 travel for the step in flight. */
  travel: number;
  /** Eased travel, shared by every transfer in a ring step. */
  progress: number;
  waiting: boolean;
}

/** `waitShare` is the fraction of each hop spent on fixed latency rather than bytes. */
export function frameAt(algorithm: Algorithm, elapsed: number, waitShare: number): CollectiveFrame | null {
  if (elapsed < 0) return null;
  let start = 0;
  for (const spec of ALGORITHMS[algorithm].stages) {
    if (elapsed < start + spec.seconds) {
      const scaled = Math.min(spec.steps - 1e-6, ((elapsed - start) / spec.seconds) * spec.steps);
      const step = Math.floor(scaled);
      const stepProgress = scaled - step;
      const wait = (1 - LANDING) * waitShare;
      const travel = clamp((stepProgress - wait) / (1 - LANDING - wait), 0, 1);
      return {
        stage: spec.stage,
        step,
        transfers: transfersFor(algorithm, spec.stage, step),
        travel,
        progress: smooth(travel),
        waiting: stepProgress < wait,
      };
    }
    start += spec.seconds;
  }
  return null;
}

/** Peers fan out a few beats apart, so seven packets leaving one GPU read as a stream. */
function peerWindow(order: number): [number, number] {
  return [order * 0.05, order * 0.05 + 0.7];
}

/** Seven full payloads share GPU 0's link, so they cross it one after another. */
function queueWindow(order: number): [number, number] {
  return [order * 0.12, order * 0.12 + 0.28];
}

function transfersFor(algorithm: Algorithm, stage: Stage, step: number): Transfer[] {
  if (algorithm === "ring") {
    // Position p sends slice p-s-1 while reducing (so position p ends owning slice p),
    // then forwards slice p-s while gathering. Slice ids follow ring positions.
    return RING.map((source, position) => ({
      source,
      destination: RING[(position + 1) % RANKS],
      chunk: RING[mod(stage === "reduce" ? position - step - 1 : position - step, RANKS)],
      window: [0, 1],
    }));
  }
  if (algorithm === "root") {
    return Array.from({ length: PEERS }, (_, order) => {
      const peer = order + 1;
      return stage === "reduce"
        ? { source: peer, destination: ROOT, chunk: null, window: queueWindow(order) }
        : { source: ROOT, destination: peer, chunk: null, window: queueWindow(order) };
    });
  }
  // Two-shot and one-shot: every GPU talks to all seven peers at once through the switch.
  const transfers: Transfer[] = [];
  for (let source = 0; source < RANKS; source += 1) {
    for (let order = 0; order < PEERS; order += 1) {
      const destination = (source + order + 1) % RANKS;
      const chunk = algorithm === "oneshot" ? null : stage === "reduce" ? destination : source;
      transfers.push({ source, destination, chunk, window: peerWindow(order) });
    }
  }
  return transfers;
}

/** Eased 0..1 progress of one transfer inside its window. */
export function arrival(frame: CollectiveFrame, transfer: Transfer): number {
  const [start, end] = transfer.window;
  return smooth((frame.travel - start) / (end - start));
}

export function inFlight(frame: CollectiveFrame, transfer: Transfer): boolean {
  const progress = arrival(frame, transfer);
  return progress > 0.005 && progress < 0.995;
}

export interface Cell {
  /** Contributions summed into this slice, out of eight. */
  count: number;
  hot: boolean;
}

/** One rank's view of one slice during the collective. */
export function cellFor(algorithm: Algorithm, rank: number, chunk: number, frame: CollectiveFrame): Cell {
  if (algorithm === "ring") return ringCell(rank, chunk, frame);
  const incoming = frame.transfers.filter(({ destination }) => destination === rank);
  if (algorithm === "oneshot") return summed(frame, incoming);
  if (algorithm === "twoshot") {
    if (frame.stage === "reduce") return chunk === rank ? summed(frame, incoming) : { count: 1, hot: false };
    if (chunk === rank) return { count: RANKS, hot: false };
    return copied(frame, incoming.find(({ source }) => source === chunk));
  }
  if (frame.stage === "reduce") return rank === ROOT ? summed(frame, incoming) : { count: 1, hot: false };
  return rank === ROOT ? { count: RANKS, hot: false } : copied(frame, incoming[0]);
}

/** A receiver adding each arriving partial to its own. */
function summed(frame: CollectiveFrame, incoming: Transfer[]): Cell {
  return {
    count: 1 + incoming.reduce((sum, transfer) => sum + arrival(frame, transfer), 0),
    hot: incoming.some((transfer) => inFlight(frame, transfer)),
  };
}

/** A receiver overwriting its partial with an already reduced copy. */
function copied(frame: CollectiveFrame, transfer: Transfer | undefined): Cell {
  if (!transfer) return { count: 1, hot: false };
  return { count: 1 + PEERS * arrival(frame, transfer), hot: inFlight(frame, transfer) };
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

/** Whether this rank finishes the first stage owning one reduced slice. */
export function ownsSlice(algorithm: Algorithm): boolean {
  return algorithm === "ring" || algorithm === "twoshot";
}

export function roleLabel(algorithm: Algorithm, rank: number): string {
  if (algorithm === "ring") return `ring → GPU ${RING[(RING.indexOf(rank) + 1) % RANKS]}`;
  if (algorithm === "twoshot") return `owns slice ${rank}`;
  if (algorithm === "root") return rank === ROOT ? "root" : "→ GPU 0";
  return "↔ 7 peers";
}

/** `noun` names the reduced tensor: C for prefill, y for decode. */
export function rankStatus(algorithm: Algorithm, rank: number, frame: CollectiveFrame, cells: Cell[], noun: string): string {
  const final = cells.filter((cell) => cell.count >= RANKS - 0.001).length;
  const whole = (count: number) => Math.floor(count + 1e-6);
  if (algorithm === "ring") {
    if (frame.stage === "gather") return `${final}/8 slices final`;
    const incoming = frame.transfers.find(({ destination }) => destination === rank);
    return `step ${frame.step + 1}/7 · + slice ${incoming?.chunk ?? "–"}`;
  }
  if (algorithm === "twoshot") {
    if (frame.stage === "gather") return `${final}/8 slices final`;
    return `slice ${rank} · ${whole(cells[rank].count)}/8 summed`;
  }
  if (algorithm === "oneshot") {
    return frame.waiting ? "writing to 7 peers" : `${whole(cells[0].count)}/8 partials summed`;
  }
  if (rank === ROOT) {
    return frame.stage === "reduce" ? `summing · ${whole(cells[0].count)}/8 partials` : `sending ${noun} to 7 GPUs`;
  }
  const mine = frame.transfers.find(({ source, destination }) => source === rank || destination === rank);
  const progress = mine ? arrival(frame, mine) : 0;
  if (frame.stage === "reduce") {
    if (progress >= 1) return "sent · link idle";
    return progress > 0 ? `sending partial ${noun}` : "queued for GPU 0 link";
  }
  if (progress >= 1) return `reduced ${noun} ready`;
  return progress > 0 ? `receiving reduced ${noun}` : "waiting on GPU 0 link";
}

/** What the traffic card says before the collective starts. */
export function plan(algorithm: Algorithm): string {
  switch (algorithm) {
    case "ring": return "7 reduce-scatter + 7 all-gather steps";
    case "twoshot": return "1 reduce-scatter shot + 1 all-gather shot";
    case "root": return "Fan in to GPU 0, then fan back out";
    case "oneshot": return "One all-to-all hop, then local sums";
  }
}

export function trafficNote(algorithm: Algorithm, frame: CollectiveFrame): string {
  const name = stepName(algorithm, frame);
  if (frame.waiting) return `${name} · waiting on fixed link latency`;
  if (algorithm === "ring") return `${name} · 8 of 8 GPUs sending`;
  if (algorithm === "root") return `${name} · GPU 0's link carries all 7 copies`;
  return `${name} · ${frame.transfers.length} transfers, all 8 links busy`;
}

function stepName(algorithm: Algorithm, frame: CollectiveFrame): string {
  const reduce = frame.stage === "reduce";
  switch (algorithm) {
    case "ring": return `${reduce ? "Reduce-scatter" : "All-gather"} step ${frame.step + 1}/7`;
    case "twoshot": return reduce ? "Reduce-scatter shot 1/2" : "All-gather shot 2/2";
    case "root": return reduce ? "Fan-in to GPU 0" : "Fan-out from GPU 0";
    case "oneshot": return "All-to-all exchange";
  }
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

export function mod(value: number, base: number): number {
  return ((value % base) + base) % base;
}

export function smooth(value: number): number {
  const x = clamp(value, 0, 1);
  return x * x * (3 - 2 * x);
}
