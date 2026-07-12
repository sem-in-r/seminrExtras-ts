/**
 * Necessary Condition Analysis (NCA) and NCA-ESSE, ported from feature_nca.R
 * (via the Python port's nca.py).
 *
 * NCA (Dul 2016) complements PLS-SEM's sufficiency logic by testing whether a
 * predictor level is *necessary* (but not sufficient) for an outcome level. It
 * runs on construct scores: for each direct predictor of the target it fits a
 * ceiling envelope — CE-FDH (step function over the free-disposal hull) or
 * CR-FDH (OLS line through the CE-FDH peers) — and reports the ceiling effect
 * size `d` = empty-zone-area / scope, a permutation p-value, and a bottleneck
 * table of required predictor levels per outcome level. A predictor is a
 * necessary condition when `d >= 0.1` and (with permutations) `p < 0.05`.
 *
 * `assessNcaEsse` (Becker et al. 2026) is the effect-size sensitivity
 * extension: it varies an ECDF ceiling threshold, recomputing CE-FDH effect
 * sizes after dropping the most extreme observations, and compares them against
 * the analytical joint-uniform benchmark `d = t(1 - ln t)`.
 *
 * Parity notes (plan F13): NCA is deterministic given the permutations, so the
 * only stochastic seam is the permutation test. `perms` injects R's permutation
 * index streams (shape `(nStreams, n)`, 0-based, consumed flat in loop order:
 * ceiling outer, predictor, rep) for exact fixture parity; otherwise `seed`
 * seeds `mulberry32`, drawing one Fisher–Yates permutation per rep
 * (statistically equivalent, non-bit-identical to R). Effect sizes, benchmark
 * and delta match R to rel 1e-9 / abs 1e-12; bottleneck cells are rounded to
 * one decimal (R half-even). The R parameter `test.rep` is `testRep` here.
 */

import { mulberry32, namedMatrix, type NamedMatrix, type PlsModel } from "@seminr/core";
import { isNamedArgs, validateSeminrModel } from "./helpers.ts";
import { formatTable, gFormat } from "./records.ts";

/** Ceiling techniques implemented internally (R `INTERNAL_CEILINGS`). */
export const INTERNAL_CEILINGS: readonly string[] = ["ce_fdh", "cr_fdh"];

const EPS = Number.EPSILON;

// =============================================================================
// Small numeric utilities
// =============================================================================

function toNums(x: readonly number[]): number[] {
  return x as number[];
}

/** Sorted unique values (np.unique semantics, exact equality). */
function uniqueSorted(xs: readonly number[]): number[] {
  const sorted = [...xs].sort((a, b) => a - b);
  const out: number[] = [];
  for (const v of sorted) {
    if (out.length === 0 || out[out.length - 1] !== v) out.push(v);
  }
  return out;
}

function min(xs: readonly number[]): number {
  let m = xs[0]!;
  for (const v of xs) if (v < m) m = v;
  return m;
}

function max(xs: readonly number[]): number {
  let m = xs[0]!;
  for (const v of xs) if (v > m) m = v;
  return m;
}

/** Cumulative maximum (np.maximum.accumulate). */
function cummax(xs: readonly number[]): number[] {
  const out: number[] = new Array(xs.length);
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i]! > m) m = xs[i]!;
    out[i] = m;
  }
  return out;
}

/**
 * Round to one decimal, half-to-even (matches R `round` / numpy `round`).
 *
 * Real construct scores never land exactly on a `.05` boundary, so this only
 * differs from half-up on synthetic inputs; kept faithful regardless.
 */
function round1HalfEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  const scaled = x * 10;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let r: number;
  if (diff < 0.5) r = floor;
  else if (diff > 0.5) r = floor + 1;
  else r = floor % 2 === 0 ? floor : floor + 1;
  return r / 10;
}


/** `steps + 1` evenly spaced levels 0..100 (np.linspace(0, 100, steps + 1)). */
function outcomeLevels(steps: number): number[] {
  const out: number[] = new Array(steps + 1);
  for (let i = 0; i <= steps; i++) out[i] = (i / steps) * 100;
  out[steps] = 100; // pin the endpoint exactly, as np.linspace does.
  return out;
}

// =============================================================================
// Ceiling envelopes and effect sizes
// =============================================================================

/** CE-FDH ceiling: sorted unique x levels and the cumulative-max ceiling y. */
export function computeCeFdh(
  x: readonly number[],
  y: readonly number[],
): { ux: number[]; cy: number[] } {
  const xa = toNums(x);
  const ya = toNums(y);
  const ux = uniqueSorted(xa);
  const maxY = ux.map((xi) => {
    let m = -Infinity;
    for (let i = 0; i < xa.length; i++) if (xa[i] === xi && ya[i]! > m) m = ya[i]!;
    return m;
  });
  return { ux, cy: cummax(maxY) };
}

/**
 * CE-FDH peer coordinates: the unique-x levels that push the ceiling up.
 *
 * The first level is always a peer; a later level is a peer iff its max-y
 * strictly exceeds the previous cumulative maximum (flat steps are dropped).
 */
export function getCeFdhPeers(
  x: readonly number[],
  y: readonly number[],
): { px: number[]; py: number[] } {
  const xa = toNums(x);
  const ya = toNums(y);
  const ux = uniqueSorted(xa);
  const maxY = ux.map((xi) => {
    let m = -Infinity;
    for (let i = 0; i < xa.length; i++) if (xa[i] === xi && ya[i]! > m) m = ya[i]!;
    return m;
  });
  const cy = cummax(maxY);
  const px: number[] = [];
  const py: number[] = [];
  for (let i = 0; i < ux.length; i++) {
    const isPeer = i === 0 ? true : maxY[i]! > cy[i - 1]!;
    if (isPeer) {
      px.push(ux[i]!);
      py.push(cy[i]!);
    }
  }
  return { px, py };
}

/** CE-FDH effect size `d` = empty ceiling zone / scope (not clipped). */
export function ceFdhEffectSize(x: readonly number[], y: readonly number[]): number {
  const xa = toNums(x);
  const ya = toNums(y);
  const xMin = min(xa);
  const xMax = max(xa);
  const yMin = min(ya);
  const yMax = max(ya);
  const scope = (xMax - xMin) * (yMax - yMin);
  if (scope < EPS) return 0.0;
  const { ux, cy } = computeCeFdh(xa, ya);
  if (ux.length < 2) return 0.0;
  // Interval [ux[i], ux[i+1]) has ceiling cy[i]; empty area above = width*(yMax-cy[i]).
  let ceilingZone = 0.0;
  for (let i = 0; i < ux.length - 1; i++) {
    ceilingZone += (ux[i + 1]! - ux[i]!) * (yMax - cy[i]!);
  }
  return ceilingZone / scope;
}

/** OLS intercept/slope of `y ~ x` via centered sums (matches R `lm`). */
function olsThroughPeers(px: readonly number[], py: readonly number[]): [number, number] {
  const n = px.length;
  let xBar = 0;
  let yBar = 0;
  for (let i = 0; i < n; i++) {
    xBar += px[i]!;
    yBar += py[i]!;
  }
  xBar /= n;
  yBar /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = px[i]! - xBar;
    sxx += dx * dx;
    sxy += dx * (py[i]! - yBar);
  }
  const b = sxy / sxx;
  const a = yBar - b * xBar;
  return [a, b];
}

/** Empty area above the line `y = a + b*x` within the scope rectangle. */
export function lineCeilingZone(
  a: number,
  b: number,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
): number {
  const scopeHeight = yMax - yMin;
  if (Math.abs(b) < EPS) {
    if (a >= yMax) return 0.0;
    if (a <= yMin) return (xMax - xMin) * scopeHeight;
    return (xMax - xMin) * (yMax - a);
  }
  const xCross = [(yMin - a) / b, (yMax - a) / b];
  const inner = xCross.filter((xc) => xMin < xc && xc < xMax);
  const breaks = [...new Set([xMin, ...inner, xMax])].sort((p, q) => p - q);
  let total = 0.0;
  for (let i = 0; i < breaks.length - 1; i++) {
    const xl = breaks[i]!;
    const xr = breaks[i + 1]!;
    const lineAtMid = a + (b * (xl + xr)) / 2;
    if (lineAtMid >= yMax) continue;
    if (lineAtMid <= yMin) {
      total += (xr - xl) * scopeHeight;
    } else {
      total += (yMax - a) * (xr - xl) - (b / 2) * (xr * xr - xl * xl);
    }
  }
  return total;
}

/** CR-FDH effect size: OLS through the CE-FDH peers, clipped to [0, 1]. */
export function crFdhEffectSize(x: readonly number[], y: readonly number[]): number {
  const xa = toNums(x);
  const ya = toNums(y);
  const xMin = min(xa);
  const xMax = max(xa);
  const yMin = min(ya);
  const yMax = max(ya);
  const scope = (xMax - xMin) * (yMax - yMin);
  if (scope < EPS) return 0.0;
  const { px, py } = getCeFdhPeers(xa, ya);
  if (px.length < 2) return ceFdhEffectSize(xa, ya);
  const [a, b] = olsThroughPeers(px, py);
  const ceilingZone = lineCeilingZone(a, b, xMin, xMax, yMin, yMax);
  return Math.max(0.0, Math.min(ceilingZone / scope, 1.0));
}

/**
 * CR-FDH ceiling line `[a, b]`: OLS through the CE-FDH peers.
 *
 * Returns `null` when fewer than two peers exist (no line is defined). Carries
 * the plot layer's CR-FDH geometry so a future `plotNca` only draws.
 */
export function crFdhLine(x: readonly number[], y: readonly number[]): [number, number] | null {
  const { px, py } = getCeFdhPeers(x, y);
  if (px.length < 2) return null;
  return olsThroughPeers(px, py);
}

/** Dispatch to the requested ceiling technique. */
export function ncaEffectSize(
  x: readonly number[],
  y: readonly number[],
  ceilingType: string,
): number {
  if (ceilingType === "ce_fdh") return ceFdhEffectSize(x, y);
  if (ceilingType === "cr_fdh") return crFdhEffectSize(x, y);
  throw new Error(
    `Supported ceiling techniques: ${INTERNAL_CEILINGS.join(", ")}. Got '${ceilingType}'.`,
  );
}

/** Fisher–Yates permutation of `[0, n)` using `gen` (default RNG path). */
function fisherYates(gen: () => number, n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(gen() * (i + 1));
    const tmp = idx[i]!;
    idx[i] = idx[j]!;
    idx[j] = tmp;
  }
  return idx;
}

/** Options for {@link ncaPermutationTest}: injected `perms`, else a default `rng`. */
export interface PermTestOptions {
  perms?: number[][];
  rng?: () => number;
}

/**
 * Permutation p-value: fraction of permuted-y effect sizes `>= observedD`.
 *
 * `perms` (shape `(nPerm, n)`, 0-based) injects R's permutation streams;
 * otherwise `rng` draws one permutation per rep. Counts are exact (a boolean
 * mean), so injected runs reproduce R bit-for-bit.
 */
export function ncaPermutationTest(
  x: readonly number[],
  y: readonly number[],
  ceilingType: string,
  observedD: number,
  nPerm: number,
  opts: PermTestOptions = {},
): number {
  const ya = toNums(y);
  const n = ya.length;
  let count = 0;
  for (let i = 0; i < nPerm; i++) {
    const perm = opts.perms ? opts.perms[i]! : fisherYates(opts.rng!, n);
    const permY = perm.map((k) => ya[k]!);
    if (ncaEffectSize(x, permY, ceilingType) >= observedD) count++;
  }
  return count / nPerm;
}

/**
 * Required predictor percentage at each of `steps + 1` outcome levels.
 *
 * NaN marks an outcome level the predictor cannot reach within its observed
 * range. Percentages are rounded to one decimal (R half-even).
 */
export function computeBottleneckColumn(
  x: readonly number[],
  y: readonly number[],
  ceilingType: string,
  steps: number,
): number[] {
  const xa = toNums(x);
  const ya = toNums(y);
  const xMin = min(xa);
  const xMax = max(xa);
  const yMin = min(ya);
  const yMax = max(ya);
  const yLevels = outcomeLevels(steps);
  const yTargets = yLevels.map((lvl) => yMin + (lvl / 100.0) * (yMax - yMin));

  if (xMax - xMin < EPS) return new Array(steps + 1).fill(NaN);

  let xNeeded: number[];
  if (ceilingType === "ce_fdh") {
    const { ux, cy } = computeCeFdh(xa, ya);
    xNeeded = yTargets.map((yt) => {
      for (let i = 0; i < cy.length; i++) if (cy[i]! >= yt) return ux[i]!;
      return NaN;
    });
  } else if (ceilingType === "cr_fdh") {
    const { px, py } = getCeFdhPeers(xa, ya);
    if (px.length < 2) return computeBottleneckColumn(xa, ya, "ce_fdh", steps);
    const [a, b] = olsThroughPeers(px, py);
    if (Math.abs(b) < EPS) return new Array(steps + 1).fill(NaN);
    xNeeded = yTargets.map((yt) => (yt - a) / b);
  } else {
    return new Array(steps + 1).fill(NaN);
  }

  return xNeeded.map((xn) => {
    const pct = ((xn - xMin) / (xMax - xMin)) * 100.0;
    if (pct < 0 || pct > 100) return NaN;
    return round1HalfEven(pct);
  });
}

// =============================================================================
// NCA-ESSE internals
// =============================================================================

/** Joint NCA ECDF: per obs `i`, share of obs with `x <= x_i` and `y >= y_i`. */
export function computeEcdfNca(x: readonly number[], y: readonly number[]): number[] {
  const xa = toNums(x);
  const ya = toNums(y);
  const n = xa.length;
  return xa.map((_, i) => {
    let count = 0;
    for (let k = 0; k < n; k++) if (xa[k]! <= xa[i]! && ya[k]! >= ya[i]!) count++;
    return count / n;
  });
}

/** Joint-uniform CE-FDH benchmark `d = t(1 - ln t)` (0 at `t = 0`). */
export function benchmarkEffectSize(t: number): number {
  if (t === 0) return 0.0;
  return t * (1 - Math.log(t));
}

// =============================================================================
// Result records
// =============================================================================

/** Result of {@link assessNca} (R class `nca_analysis`). */
export interface NcaAnalysis {
  readonly kind: "nca_analysis";
  /** `predictors x ceilings` ceiling effect sizes `d`. */
  readonly effectSizes: NamedMatrix;
  /** `predictors x ceilings` permutation p-values, or null when `testRep === 0`. */
  readonly significance: NamedMatrix | null;
  /** Per-ceiling bottleneck table `outcome-levels x [target, ...predictors]`. */
  readonly bottleneck: Readonly<Record<string, NamedMatrix>>;
  readonly necessaryPredictors: readonly string[];
  readonly plsModel: PlsModel;
  readonly target: string;
  readonly predictors: readonly string[];
  readonly ceilings: readonly string[];
  toString(): string;
  summarize(): string;
}

/** Result of {@link assessNcaEsse} (R class `nca_esse`). */
export interface NcaEsse {
  readonly kind: "nca_esse";
  /** `thresholds x predictors` empirical effect sizes (rows `"0%"`, `"0.5%"`, ...). */
  readonly effectSizes: NamedMatrix;
  readonly benchmark: NamedMatrix;
  readonly delta: NamedMatrix;
  readonly significance: NamedMatrix | null;
  readonly plsModel: PlsModel;
  readonly target: string;
  readonly predictors: readonly string[];
  readonly thresholds: readonly number[];
  readonly ceiling: string;
  readonly nObs: number;
  toString(): string;
  summarize(): string;
}

function ncaHeader(rec: NcaAnalysis, title: string): string[] {
  const nObs = rec.plsModel.constructScores.values.length;
  return [
    title,
    "=".repeat(title.length),
    `Target: ${rec.target}`,
    `Predictors: ${rec.predictors.join(", ")}`,
    `Ceilings: ${rec.ceilings.join(", ")}`,
    `Observations: ${nObs}`,
    "",
  ];
}

function necessaryLine(rec: NcaAnalysis): string {
  if (rec.necessaryPredictors.length > 0) {
    return `Necessary conditions (d >= 0.1, p < 0.05): ${rec.necessaryPredictors.join(", ")}`;
  }
  return "No necessary conditions identified (d >= 0.1, p < 0.05)";
}

function makeNcaAnalysis(
  fields: Omit<NcaAnalysis, "kind" | "toString" | "summarize">,
): NcaAnalysis {
  const rec = {
    kind: "nca_analysis" as const,
    ...fields,
    toString(): string {
      const lines = ncaHeader(this, "Necessary Condition Analysis (NCA)");
      lines.push("Effect Sizes (d):", formatTable(this.effectSizes, 4), "");
      if (this.significance) {
        lines.push("Permutation p-values:", formatTable(this.significance, 4), "");
      }
      lines.push(necessaryLine(this));
      return lines.join("\n");
    },
    summarize(): string {
      const lines = ncaHeader(this, "Necessary Condition Analysis (NCA) Summary");
      lines.push("Effect Sizes (d):", formatTable(this.effectSizes, 4), "");
      if (this.significance) {
        lines.push("Permutation p-values:", formatTable(this.significance, 4), "");
      }
      lines.push(necessaryLine(this), "");
      for (const ceil of this.ceilings) {
        const table = this.bottleneck[ceil];
        if (!table) continue;
        lines.push(`Bottleneck table (${ceil}):`, formatTable(table, 1), "");
      }
      return lines.join("\n");
    },
  };
  return Object.freeze(rec);
}

function makeNcaEsse(fields: Omit<NcaEsse, "kind" | "toString" | "summarize">): NcaEsse {
  const rec = {
    kind: "nca_esse" as const,
    ...fields,
    toString(): string {
      const lo = min(this.thresholds as number[]) * 100;
      const hi = max(this.thresholds as number[]) * 100;
      const title = "NCA-ESSE: Effect Size Sensitivity Extension";
      const lines = [
        title,
        "=".repeat(title.length),
        `Target: ${this.target}`,
        `Predictors: ${this.predictors.join(", ")}`,
        `Ceiling: ${this.ceiling}`,
        `Observations: ${this.nObs}`,
        `Thresholds: ${gFormat(lo)}% to ${gFormat(hi)}%`,
        "",
        "Empirical effect sizes by ECDF threshold:",
        formatTable(this.effectSizes, 4),
        "",
        "Benchmark (uniform) effect sizes:",
        formatTable(this.benchmark, 4),
        "",
        "Sensitivity (empirical - benchmark):",
        formatTable(this.delta, 4),
      ];
      if (this.significance) {
        lines.push("", "Permutation p-values:", formatTable(this.significance, 4));
      }
      return lines.join("\n");
    },
    summarize(): string {
      const title = "NCA-ESSE Summary (Becker et al., 2026)";
      const lines = [
        title,
        "=".repeat(title.length),
        `Target: ${this.target}`,
        `Ceiling: ${this.ceiling}`,
        `Observations: ${this.nObs}`,
        "",
      ];
      const head =
        `  ${"ECDF_threshold".padStart(14)} ${"Empirical_d".padStart(12)} ` +
        `${"Benchmark_d".padStart(12)} ${"Difference".padStart(12)}`;
      for (let pj = 0; pj < this.predictors.length; pj++) {
        lines.push(`Predictor: ${this.predictors[pj]}`, "-".repeat(60), head);
        for (let ti = 0; ti < this.thresholds.length; ti++) {
          const emp = this.effectSizes.values[ti]![pj]!;
          const bench = this.benchmark.values[ti]![pj]!;
          const diff = this.delta.values[ti]![pj]!;
          lines.push(
            `  ${this.thresholds[ti]!.toFixed(4).padStart(14)} ${emp.toFixed(4).padStart(12)} ` +
              `${bench.toFixed(4).padStart(12)} ${diff.toFixed(4).padStart(12)}`,
          );
        }
        lines.push("");
      }
      return lines.join("\n");
    },
  };
  return Object.freeze(rec);
}

// =============================================================================
// Input validation
// =============================================================================

function validateTestRep(testRep: number | undefined): number {
  const v = testRep ?? 1000;
  if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || !Number.isInteger(v)) {
    throw new Error("test.rep must be a non-negative integer.");
  }
  return v;
}

function resolvePredictors(
  model: PlsModel,
  target: string,
  predictors: readonly string[] | undefined,
): string[] {
  const constructNames = model.constructScores.cols;
  if (!constructNames.includes(target)) {
    throw new Error(
      `target '${target}' not found in model constructs: ${constructNames.join(", ")}`,
    );
  }
  if (predictors === undefined) {
    const auto = model.smMatrix.constructAntecedents(target);
    if (auto.length === 0) {
      throw new Error(
        `No direct predictors found for target '${target}' in the structural model.`,
      );
    }
    return auto;
  }
  const invalid = predictors.filter((p) => !constructNames.includes(p));
  if (invalid.length > 0) {
    throw new Error(`Predictor(s) not found in model constructs: ${invalid.join(", ")}`);
  }
  return [...predictors];
}

/** Column vector of a construct from the model's construct scores. */
function scoreColumn(model: PlsModel, construct: string): number[] {
  const scores = model.constructScores;
  const j = scores.cols.indexOf(construct);
  return scores.values.map((row) => row[j]!);
}

// =============================================================================
// Public API
// =============================================================================

/** Options for {@link assessNca}. */
export interface NcaOptions {
  target: string;
  /** Defaults to the target's direct structural predictors (interactions included). */
  predictors?: string[];
  ceilings?: string[];
  /** R `test.rep`; 0 disables the permutation test (default 1000). */
  testRep?: number;
  steps?: number;
  /** Seeds the default permutation RNG (default 123); ignored when `perms` is given. */
  seed?: number;
  /**
   * Injected 0-based permutation streams, shape
   * `(ceilings * predictors * testRep, n)`, consumed flat in loop order
   * (ceiling outer, predictor, rep) for exact R parity.
   */
  perms?: number[][];
}

/** Named-args form of {@link assessNca}. */
export interface AssessNcaArgs extends NcaOptions {
  model: unknown;
}

/**
 * Necessary Condition Analysis on an estimated PLS-SEM model's construct scores.
 *
 * Returns null (with a warning) for non-seminr input. Higher-order models are
 * supported (NCA works on construct scores; there is no PLSpredict guard).
 * `predictors` auto-detects the target's direct structural predictors when
 * omitted (interaction terms included). Callable as `assessNca(model, options)`
 * or `assessNca({ model, ...options })`.
 */
export function assessNca(args: AssessNcaArgs): NcaAnalysis | null;
export function assessNca(model: unknown, options: NcaOptions): NcaAnalysis | null;
export function assessNca(
  modelOrArgs: unknown,
  positionalOptions?: NcaOptions,
): NcaAnalysis | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessNcaArgs).model : modelOrArgs;
  const options = (named ? (modelOrArgs as AssessNcaArgs) : positionalOptions) as NcaOptions;

  if (!validateSeminrModel(seminrModel, "assessNca")) return null;
  const model = seminrModel;

  const testRep = validateTestRep(options.testRep);
  const predictors = resolvePredictors(model, options.target, options.predictors);
  const ceilings = options.ceilings ? [...options.ceilings] : [...INTERNAL_CEILINGS];
  const steps = options.steps ?? 10;
  const seed = options.seed ?? 123;
  const { target } = options;

  const y = scoreColumn(model, target);

  if (options.perms !== undefined) {
    const expected = ceilings.length * predictors.length * testRep;
    if (options.perms.length !== expected) {
      throw new Error(
        `perms must supply ${expected} permutation rows (${ceilings.length} ceilings x ` +
          `${predictors.length} predictors x ${testRep} reps), got ${options.perms.length}.`,
      );
    }
  }
  const gen = options.perms !== undefined ? undefined : mulberry32(seed);

  const yLevels = outcomeLevels(steps);
  const rowLabels = yLevels.map((lvl) => gFormat(lvl));

  const effect = predictors.map(() => ceilings.map(() => NaN));
  const signif = testRep > 0 ? predictors.map(() => ceilings.map(() => NaN)) : null;
  const bottleneck: Record<string, NamedMatrix> = {};
  let cursor = 0;

  for (let cj = 0; cj < ceilings.length; cj++) {
    const ceil = ceilings[cj]!;
    const bnValues = yLevels.map((lvl) => [lvl, ...predictors.map(() => NaN)]);
    for (let pi = 0; pi < predictors.length; pi++) {
      const x = scoreColumn(model, predictors[pi]!);
      const d = ncaEffectSize(x, y, ceil);
      effect[pi]![cj] = d;
      if (testRep > 0) {
        let p: number;
        if (options.perms !== undefined) {
          const callPerms = options.perms.slice(cursor, cursor + testRep);
          cursor += testRep;
          p = ncaPermutationTest(x, y, ceil, d, testRep, { perms: callPerms });
        } else {
          p = ncaPermutationTest(x, y, ceil, d, testRep, { rng: gen });
        }
        signif![pi]![cj] = p;
      }
      const col = computeBottleneckColumn(x, y, ceil, steps);
      for (let r = 0; r <= steps; r++) bnValues[r]![pi + 1] = col[r]!;
    }
    bottleneck[ceil] = namedMatrix(rowLabels, [target, ...predictors], bnValues);
  }

  const effectSizes = namedMatrix(predictors, ceilings, effect);
  const significance = signif ? namedMatrix(predictors, ceilings, signif) : null;

  const necessaryPredictors = predictors.filter((_, pi) =>
    ceilings.some(
      (_c, cj) => effect[pi]![cj]! >= 0.1 && (signif === null || signif[pi]![cj]! < 0.05),
    ),
  );

  return makeNcaAnalysis({
    effectSizes,
    significance,
    bottleneck: Object.freeze(bottleneck),
    necessaryPredictors,
    plsModel: model,
    target,
    predictors,
    ceilings,
  });
}

/** R default `seq(0, 0.05, by = 0.005)` (11 values), rounded for clean labels. */
function defaultEsseThresholds(): number[] {
  return Array.from({ length: 11 }, (_, i) => Number((0.005 * i).toFixed(3)));
}

/** Options for {@link assessNcaEsse}. */
export interface NcaEsseOptions {
  target: string;
  predictors?: string[];
  /** Defaults to `seq(0, 0.05, by = 0.005)`. */
  thresholds?: number[];
  ceiling?: string;
  /** R `test.rep`; 0 disables the permutation test (default 0). */
  testRep?: number;
  /** Accepted for R signature compatibility but unused. */
  steps?: number;
  seed?: number;
  /**
   * Injected permutation streams (an array of per-rep permutation arrays, each
   * matching the filtered length), consumed `testRep` at a time per kept cell.
   */
  perms?: number[][];
}

/** Named-args form of {@link assessNcaEsse}. */
export interface AssessNcaEsseArgs extends NcaEsseOptions {
  model: unknown;
}

/**
 * NCA-ESSE effect-size sensitivity analysis (Becker et al. 2026).
 *
 * Returns null (with a warning) for non-seminr input. At each ECDF `threshold`
 * the most extreme observations are dropped before the CE-FDH effect size is
 * recomputed and compared to the joint-uniform benchmark `d = t(1 - ln t)`. A
 * `ceiling` other than `"ce_fdh"` warns (the benchmark is CE-FDH-specific).
 * `steps` is accepted for signature compatibility but unused. Callable as
 * `assessNcaEsse(model, options)` or `assessNcaEsse({ model, ...options })`.
 */
export function assessNcaEsse(args: AssessNcaEsseArgs): NcaEsse | null;
export function assessNcaEsse(model: unknown, options: NcaEsseOptions): NcaEsse | null;
export function assessNcaEsse(
  modelOrArgs: unknown,
  positionalOptions?: NcaEsseOptions,
): NcaEsse | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessNcaEsseArgs).model : modelOrArgs;
  const options = (named ? (modelOrArgs as AssessNcaEsseArgs) : positionalOptions) as NcaEsseOptions;

  if (!validateSeminrModel(seminrModel, "assessNcaEsse")) return null;
  const model = seminrModel;

  const thresholds = options.thresholds ? [...options.thresholds] : defaultEsseThresholds();
  if (thresholds.some((t) => t < 0 || t > 1)) {
    throw new Error("thresholds must be between 0 and 1.");
  }
  const testRep = validateTestRep(options.testRep ?? 0);
  const predictors = resolvePredictors(model, options.target, options.predictors);
  const ceiling = options.ceiling ?? "ce_fdh";
  const { target } = options;

  if (ceiling !== "ce_fdh") {
    console.warn(
      "NCA-ESSE benchmark is derived for CE-FDH (Becker et al., 2026). " +
        `Benchmark may not be directly comparable with '${ceiling}'.`,
    );
  }
  if (!INTERNAL_CEILINGS.includes(ceiling)) {
    throw new Error(
      `Supported ceiling techniques: ${INTERNAL_CEILINGS.join(", ")}. Got '${ceiling}'.`,
    );
  }

  const scores = model.constructScores;
  const nObs = scores.values.length;
  const yAll = scoreColumn(model, target);
  const labels = thresholds.map((t) => `${gFormat(t * 100)}%`);

  const empirical = thresholds.map(() => predictors.map(() => NaN));
  const signif = testRep > 0 ? thresholds.map(() => predictors.map(() => NaN)) : null;
  const gen = options.perms !== undefined ? undefined : mulberry32(options.seed ?? 123);
  let permCursor = 0;

  for (let pj = 0; pj < predictors.length; pj++) {
    const x = scoreColumn(model, predictors[pj]!);
    const ecdf = computeEcdfNca(x, yAll);
    for (let ti = 0; ti < thresholds.length; ti++) {
      const t = thresholds[ti]!;
      let fx: number[];
      let fy: number[];
      if (t === 0) {
        fx = x;
        fy = yAll;
      } else {
        fx = [];
        fy = [];
        for (let k = 0; k < x.length; k++) {
          if (ecdf[k]! > t) {
            fx.push(x[k]!);
            fy.push(yAll[k]!);
          }
        }
      }
      if (fx.length < 10) {
        console.warn(
          `Fewer than 10 observations at threshold ${labels[ti]} for ${predictors[pj]}; skipping.`,
        );
        continue;
      }
      const d = ncaEffectSize(fx, fy, ceiling);
      empirical[ti]![pj] = d;
      if (testRep > 0) {
        if (options.perms !== undefined) {
          const callPerms = options.perms.slice(permCursor, permCursor + testRep);
          permCursor += testRep;
          signif![ti]![pj] = ncaPermutationTest(fx, fy, ceiling, d, testRep, { perms: callPerms });
        } else {
          signif![ti]![pj] = ncaPermutationTest(fx, fy, ceiling, d, testRep, { rng: gen });
        }
      }
    }
  }

  const benchVec = thresholds.map((t) => benchmarkEffectSize(t));
  const bench = thresholds.map((_, ti) => predictors.map(() => benchVec[ti]!));
  const delta = empirical.map((row, ti) => row.map((v, pj) => v - bench[ti]![pj]!));

  return makeNcaEsse({
    effectSizes: namedMatrix(labels, predictors, empirical),
    benchmark: namedMatrix(labels, predictors, bench),
    delta: namedMatrix(labels, predictors, delta),
    significance: signif ? namedMatrix(labels, predictors, signif) : null,
    plsModel: model,
    target,
    predictors,
    thresholds,
    ceiling,
    nObs,
  });
}
