export const CYCLES_PER_SECOND = 3;
export const WARP_COUNT = 12;
export const TOTAL_CYCLES = 190;
export const DURATION = TOTAL_CYCLES / CYCLES_PER_SECOND;
export const SHARED_LATENCY = 30;
export const ISSUE_CYCLES = 1;
export const ISSUE_INTERVAL = 8;
export const TENSOR_ISSUE = 1;
export const TENSOR_DURATION = 25;
export const LOAD_STARTS = Array.from({ length: WARP_COUNT }, (_, warp) => 12 + warp * ISSUE_INTERVAL);

export function cycleAt(time: number): number {
  return Math.min(TOTAL_CYCLES, Math.max(0, time * CYCLES_PER_SECOND));
}

export interface Phase {
  t: number;
  id: string;
  short: string;
  title: string;
  body: string;
}

export const PHASES: Phase[] = [
  {
    t: 0,
    id: "hardware",
    short: "Hardware",
    title: "Questo è l’hardware reale dell’SM",
    body: "Shared memory, register file, warp scheduler e Tensor Core sono strutture fisiche persistenti. Restano sempre al loro posto.",
  },
  {
    t: 8 / CYCLES_PER_SECOND,
    id: "resident",
    short: "Warp residenti",
    title: "I warp sono stato software schedulabile",
    body: "W0–W11 compaiono nell’overlay, non come blocchi sul chip. Ogni colore identifica un contesto e la sua porzione simbolica del register file.",
  },
  {
    t: 10 / CYCLES_PER_SECOND,
    id: "selected",
    short: "Seleziona W0",
    title: "Lo scheduler sceglie un warp pronto",
    body: "W0 viene selezionato. Il suo token si accende nello scheduler e il colore di W0 evidenzia il percorso hardware che userà.",
  },
  {
    t: 12 / CYCLES_PER_SECOND,
    id: "load",
    short: "ldmatrix",
    title: "W0 carica A e B nei suoi registri",
    body: "W0 emette ldmatrix in un ciclo. Poi aspetta: gli operandi sono pronti dopo 30 cicli di shared memory.",
  },
  {
    t: 20 / CYCLES_PER_SECOND,
    id: "waiting",
    short: "W0 attende",
    title: "W0 aspetta, lo scheduler passa a W1",
    body: "Lo scheduler non resta occupato. Otto cicli dopo l’issue di W0 emette il load di W1, mentre W0 è ancora in attesa.",
  },
  {
    t: 42 / CYCLES_PER_SECOND,
    id: "mma",
    short: "mma.sync",
    title: "W0 emette mma.sync",
    body: "Al ciclo 42 gli operandi di W0 sono pronti. Vanno al Tensor Core in un ciclo, restano fermi lì per 25 cicli di processing, e tornano nei registri in un altro ciclo.",
  },
  {
    t: 50 / CYCLES_PER_SECOND,
    id: "inflight",
    short: "Warp in flight",
    title: "Più warp avanzano sullo stesso hardware",
    body: "I warp partono a 8 cicli di distanza, lo stesso scarto dei load. W0 è ancora nel Tensor Core quando W1 emette la sua MMA.",
  },
  {
    t: 66 / CYCLES_PER_SECOND,
    id: "accumulators",
    short: "Accumulatori",
    title: "I risultati parziali restano nei registri",
    body: "Il contributo di ogni MMA aggiorna la zona C del register file. Gli accumulatori non tornano in shared memory a ogni iterazione.",
  },
  {
    t: 156 / CYCLES_PER_SECOND,
    id: "end-k",
    short: "Fine K",
    title: "Dopo K, i registri contengono C finale",
    body: "I nuovi frammenti A/B hanno alimentato più MMA sugli stessi accumulatori. I tile C dei warp sono completi e indipendenti.",
  },
  {
    t: 160 / CYCLES_PER_SECOND,
    id: "writeback",
    short: "Writeback",
    title: "C finale torna in shared memory",
    body: "I frammenti 16×8 lasciano il register file. Due metà affiancate ricostruiscono un tile 16×16 nella shared memory.",
  },
];

export function phaseAt(t: number): Phase {
  let current = PHASES[0];
  for (const phase of PHASES) if (t >= phase.t) current = phase;
  return current;
}

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function smooth(value: number): number {
  const x = clamp01(value);
  return x * x * (3 - 2 * x);
}

export function ramp(t: number, start: number, end: number): number {
  return smooth((t - start) / (end - start));
}

export function pulse(t: number, start: number, duration: number): number {
  const x = (t - start) / duration;
  return x > 0 && x < 1 ? Math.sin(x * Math.PI) ** 2 : 0;
}
