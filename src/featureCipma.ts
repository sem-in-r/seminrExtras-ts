/**
 * Importance-Performance Map Analysis (IPMA) and combined IPMA (cIPMA),
 * ported from feature_cipma.R (via the Python port's cipma.py).
 *
 * IPMA (Ringle & Sarstedt 2016) plots each predecessor of a target construct
 * on two axes: *importance* — the unstandardized total effect on the target —
 * and *performance* — the weighted average of the construct's 0-100 rescaled
 * indicator means. cIPMA (Sarstedt et al. 2024) overlays Necessary Condition
 * Analysis: a construct is classified by crossing high/low importance
 * (above/below the median unstandardized total effect) with necessity (NCA
 * `d >= 0.1` and, when permutations are run, `p < 0.05`).
 *
 * Unstandardized total effects come from `T = (I - B)^{-1} - I` after rescaling
 * each standardized path coefficient by the ratio of rescaled-score SDs.
 * Higher-order constructs chain through their lower-order components
 * (HOC -> LOC -> indicators) for both performance and observation-level
 * rescaling. Interaction constructs (name contains `*`) are excluded — their
 * 0-100 performance is not meaningful.
 *
 * Parity notes (py plan F14): with `ncaTestRep === 0` the whole analysis is
 * deterministic (no RNG), so goldens match R at rel 1e-9 / abs 1e-12 on the
 * importance/performance vectors and exactly on the rounded classification
 * table. The R parameter `nca_test.rep` is `ncaTestRep` here.
 */

import { namedMatrix, type NamedMatrix, type PlsModel } from "@seminr/core";
import { fromRows, solve, toRows } from "@compstats/core/linalg";
import { sd } from "@compstats/core/stats";
import { isNamedArgs, validateSeminrModel } from "./helpers.ts";
import { assessNca, type NcaAnalysis } from "./featureNca.ts";
import { formatTable, gFormat } from "./records.ts";

const EPS = Number.EPSILON;

/** Default NCA ceiling techniques for cIPMA (R `c("ce_fdh", "cr_fdh")`). */
export const DEFAULT_NCA_CEILINGS: readonly string[] = ["ce_fdh", "cr_fdh"];

// =============================================================================
// Internal helpers
// =============================================================================

/** Whether `name` denotes an interaction construct (contains `*`). */
export function isInteractionConstruct(name: string): boolean {
  return name.includes("*");
}

/** The minimal model surface {@link checkPositiveWeights} traverses (testable with fakes). */
export interface WeightsModelLike {
  mmMatrix: { constructItems(construct: string): string[] };
  outerWeights: { rows: string[]; cols: string[]; values: number[][] };
}

function outerWeight(model: WeightsModelLike, item: string, construct: string): number {
  const ow = model.outerWeights;
  return ow.values[ow.rows.indexOf(item)]![ow.cols.indexOf(construct)]!;
}

/**
 * Constructs whose IPMA outer weights are not all positive.
 *
 * A construct is flagged if any of its own indicator weights is negative; for
 * a higher-order construct (whose items are themselves outer-weight columns)
 * the lower-order components are checked recursively and the HOC itself is
 * flagged when any component is.
 */
export function checkPositiveWeights(
  model: WeightsModelLike,
  constructs: readonly string[],
): string[] {
  const neg: string[] = [];
  const constructCols = model.outerWeights.cols;
  for (const construct of constructs) {
    const items = model.mmMatrix.constructItems(construct);
    const weights = items.map((it) => outerWeight(model, it, construct));
    if (weights.some((w) => w < 0)) {
      neg.push(construct);
      continue;
    }
    const locNames = items.filter((it) => constructCols.includes(it));
    if (locNames.length > 0 && checkPositiveWeights(model, locNames).length > 0) {
      neg.push(construct);
    }
  }
  return neg;
}

/**
 * Weighted 0-100 rescaled performance per construct (R `compute_ipma_performance`).
 *
 * Regular construct: weighted average of `(mean - scaleMin) / range * 100` over
 * its indicators, weights = outer weights (the ratio normalizes them). HOC:
 * aggregate lower-order-component performances by HOC-to-LOC weights. Missing
 * indicators yield `NaN` with a warning.
 */
export function computeIpmaPerformance(
  model: PlsModel,
  constructs: readonly string[],
  scaleMin: number,
  scaleMax: number,
): Record<string, number> {
  const scaleRange = scaleMax - scaleMin;
  const constructCols = model.outerWeights.cols;
  const data = model.data;
  const performance: Record<string, number> = {};
  for (const construct of constructs) {
    const items = model.mmMatrix.constructItems(construct);
    const locNames = items.filter((it) => constructCols.includes(it));
    if (locNames.length > 0) {
      const locPerf = computeIpmaPerformance(model, locNames, scaleMin, scaleMax);
      const hocW = locNames.map((loc) => outerWeight(model, loc, construct));
      const num = hocW.reduce((s, w, k) => s + w * locPerf[locNames[k]!]!, 0);
      performance[construct] = num / hocW.reduce((s, w) => s + w, 0);
      continue;
    }
    if (!items.every((it) => data.columns.includes(it))) {
      console.warn(
        `Construct '${construct}': some items not found in data. Performance set to NA.`,
      );
      performance[construct] = NaN;
      continue;
    }
    const weights = items.map((it) => outerWeight(model, it, construct));
    const idx = items.map((it) => data.columns.indexOf(it));
    const n = data.values.length;
    const num = weights.reduce((s, w, k) => {
      let colSum = 0;
      for (const row of data.values) colSum += row[idx[k]!]!;
      return s + w * (((colSum / n - scaleMin) / scaleRange) * 100);
    }, 0);
    performance[construct] = num / weights.reduce((s, w) => s + w, 0);
  }
  return performance;
}

/**
 * Per-observation 0-100 performance, `N x constructs.length` (construct order).
 *
 * Same rescaling as {@link computeIpmaPerformance} per observation; HOC columns
 * chain through the lower-order components. Missing indicators leave that
 * column all-`NaN` silently.
 */
export function computeObservationPerformance(
  model: PlsModel,
  constructs: readonly string[],
  scaleMin: number,
  scaleMax: number,
): number[][] {
  const scaleRange = scaleMax - scaleMin;
  const constructCols = model.outerWeights.cols;
  const data = model.data;
  const n = data.values.length;
  const out: number[][] = Array.from({ length: n }, () => new Array(constructs.length).fill(NaN));
  for (let k = 0; k < constructs.length; k++) {
    const construct = constructs[k]!;
    const items = model.mmMatrix.constructItems(construct);
    const locNames = items.filter((it) => constructCols.includes(it));
    if (locNames.length > 0) {
      const locObs = computeObservationPerformance(model, locNames, scaleMin, scaleMax);
      const hocW = locNames.map((loc) => outerWeight(model, loc, construct));
      const wSum = hocW.reduce((s, w) => s + w, 0);
      for (let i = 0; i < n; i++) {
        let acc = 0;
        for (let m = 0; m < hocW.length; m++) acc += locObs[i]![m]! * hocW[m]!;
        out[i]![k] = acc / wSum;
      }
      continue;
    }
    if (!items.every((it) => data.columns.includes(it))) continue;
    const weights = items.map((it) => outerWeight(model, it, construct));
    const idx = items.map((it) => data.columns.indexOf(it));
    const wSum = weights.reduce((s, w) => s + w, 0);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let m = 0; m < weights.length; m++) {
        acc += weights[m]! * (((data.values[i]![idx[m]!]! - scaleMin) / scaleRange) * 100);
      }
      out[i]![k] = acc / wSum;
    }
  }
  return out;
}

/** Total effects `T = (I - B)^{-1} - I` for `B[from, to]` (R `compute_total_effects`). */
export function computeTotalEffects(pathCoefMatrix: readonly (readonly number[])[]): number[][] {
  const k = pathCoefMatrix.length;
  const iMinusB = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => (i === j ? 1 : 0) - pathCoefMatrix[i]![j]!),
  );
  // `solve(a)` with no right-hand side is R's `solve(a)`: the inverse. This
  // was a column-by-column solve against the identity, which is the same
  // computation spelled out k times.
  const inverse = toRows(solve(fromRows(iMinusB)));
  return inverse.map((row, i) => row.map((v, j) => v - (i === j ? 1 : 0)));
}

function pathCoefBlock(model: PlsModel, constructs: readonly string[]): number[][] {
  const pc = model.pathCoef;
  return constructs.map((a) =>
    constructs.map((b) => pc.values[pc.rows.indexOf(a)]![pc.cols.indexOf(b)]!),
  );
}

/** R `sd(x, na.rm = TRUE)`: NaN-dropping ddof-1 SD (NaN if < 2 values). */
function sdNoNa(col: readonly number[]): number {
  const vals = col.filter((v) => !Number.isNaN(v));
  if (vals.length < 2) return NaN;
  return sd(vals);
}

function stdTotalEffects(model: PlsModel, constructs: readonly string[]): NamedMatrix {
  return namedMatrix([...constructs], [...constructs], computeTotalEffects(pathCoefBlock(model, constructs)));
}

/**
 * Unstandardized total-effects matrix (R `compute_unstd_total_effects`).
 *
 * Each nonzero standardized path is scaled by `SD(to) / SD(from)` of the
 * rescaled construct scores; a cell becomes `NaN` when either SD is `NaN` or
 * the source SD is below machine epsilon. Zero cells stay exactly zero.
 */
export function computeUnstdTotalEffects(
  model: PlsModel,
  constructs: readonly string[],
  scaleMin: number,
  scaleMax: number,
): NamedMatrix {
  const obsPerf = computeObservationPerformance(model, constructs, scaleMin, scaleMax);
  const sdPerf = constructs.map((_, k) => sdNoNa(obsPerf.map((row) => row[k]!)));
  const bStd = pathCoefBlock(model, constructs);
  const bUnstd = bStd.map((row) => [...row]);
  for (let i = 0; i < constructs.length; i++) {
    for (let j = 0; j < constructs.length; j++) {
      if (bStd[i]![j] !== 0) {
        const sdFrom = sdPerf[i]!;
        const sdTo = sdPerf[j]!;
        bUnstd[i]![j] =
          Number.isNaN(sdFrom) || Number.isNaN(sdTo) || sdFrom < EPS
            ? NaN
            : (bStd[i]![j]! * sdTo) / sdFrom;
      }
    }
  }
  return namedMatrix([...constructs], [...constructs], computeTotalEffects(bUnstd));
}

function namedGet(matrix: NamedMatrix, row: string, col: string): number {
  return matrix.values[matrix.rows.indexOf(row)]![matrix.cols.indexOf(col)]!;
}

/** Round to `digits` decimals, half-to-even (matches R `round` / py `round`). */
function roundHalfEven(x: number, digits: number): number {
  if (!Number.isFinite(x)) return x;
  const scale = 10 ** digits;
  const scaled = x * scale;
  const floor = Math.floor(scaled);
  const diff = scaled - floor;
  let r: number;
  if (diff < 0.5) r = floor;
  else if (diff > 0.5) r = floor + 1;
  else r = floor % 2 === 0 ? floor : floor + 1;
  return r / scale;
}

function nanMedian(values: readonly number[]): number {
  const vals = values.filter((v) => !Number.isNaN(v)).sort((a, b) => a - b);
  const n = vals.length;
  if (n === 0) return NaN;
  return n % 2 === 1 ? vals[(n - 1) / 2]! : (vals[n / 2 - 1]! + vals[n / 2]!) / 2;
}

// =============================================================================
// Result records
// =============================================================================

/** One construct's cIPMA classification (importance/performance pre-rounded). */
export interface CipmaClassificationRow {
  readonly construct: string;
  /** Unstandardized total effect, rounded to 4 dp (half-even). */
  readonly importance: number;
  /** 0-100 performance, rounded to 2 dp (half-even). */
  readonly performance: number;
  readonly highImportance: boolean;
  readonly necessary: boolean;
  readonly priority: string;
}

/**
 * Result of {@link assessCipma} / {@link assessIpma} (R class `cipma_analysis`).
 *
 * `importanceUnstd`, `importanceStd` and `performance` are name-keyed vectors
 * in construct order. `nca` is the embedded {@link NcaAnalysis} (null for a
 * plain IPMA). `classification` is the four-way priority table; `scaleRange`
 * is `[scaleMin, scaleMax]`.
 */
export interface CipmaAnalysis {
  readonly kind: "cipma_analysis";
  readonly importanceUnstd: Record<string, number>;
  readonly importanceStd: Record<string, number>;
  readonly performance: Record<string, number>;
  readonly nca: NcaAnalysis | null;
  readonly classification: readonly CipmaClassificationRow[];
  readonly target: string;
  readonly constructs: readonly string[];
  readonly scaleRange: readonly [number, number];
  readonly negativeWeightConstructs: readonly string[];
  readonly excludedInteractions: readonly string[];
  readonly plsModel: PlsModel;
  toString(): string;
  summarize(): string;
}

/** Render a name-keyed vector as aligned `name  value` rows. */
function formatNamedVector(vec: Record<string, number>, digits: number): string {
  const names = Object.keys(vec);
  if (names.length === 0) return "";
  const width = Math.max(...names.map((n) => n.length));
  return names.map((n) => `  ${n.padEnd(width)}  ${vec[n]!.toFixed(digits)}`).join("\n");
}

/** Aligned text table from pre-formatted string cells (mixed-precision safe). */
function formatStrTable(
  rowLabels: readonly string[],
  colLabels: readonly string[],
  cells: readonly (readonly string[])[],
): string {
  const colWidths = colLabels.map((col, j) =>
    Math.max(col.length, ...cells.map((row) => row[j]!.length)),
  );
  const rowWidth = Math.max(0, ...rowLabels.map((r) => r.length));
  const header = [
    " ".repeat(rowWidth),
    ...colLabels.map((col, j) => col.padStart(colWidths[j]!)),
  ].join(" ");
  const lines = [header];
  rowLabels.forEach((label, i) => {
    const body = colLabels.map((_, j) => cells[i]![j]!.padStart(colWidths[j]!));
    lines.push([label.padEnd(rowWidth), ...body].join(" "));
  });
  return lines.join("\n");
}


// =============================================================================
// Classification
// =============================================================================

/**
 * Four-way cIPMA classification (R `classify_cipma_constructs`).
 *
 * High importance = unstandardized total effect strictly above the median;
 * necessity comes from `ncaResult.necessaryPredictors` (all false without
 * NCA). Importance/performance are rounded (4/2 dp, half-even).
 */
export function classifyCipmaConstructs(
  importanceUnstd: Record<string, number>,
  performance: Record<string, number>,
  ncaResult: NcaAnalysis | null,
): CipmaClassificationRow[] {
  const constructs = Object.keys(importanceUnstd);
  const median = nanMedian(constructs.map((c) => importanceUnstd[c]!));
  const necessarySet = new Set(
    ncaResult ? constructs.filter((c) => ncaResult.necessaryPredictors.includes(c)) : [],
  );

  return constructs.map((c) => {
    const imp = importanceUnstd[c]!;
    const high = imp > median;
    const nec = necessarySet.has(c);
    const priority =
      high && nec
        ? "Top priority"
        : high
          ? "Important driver"
          : nec
            ? "Bottleneck risk"
            : "Low priority";
    return Object.freeze({
      construct: c,
      importance: roundHalfEven(imp, 4),
      performance: roundHalfEven(performance[c]!, 2),
      highImportance: high,
      necessary: nec,
      priority,
    });
  });
}

// =============================================================================
// Public API
// =============================================================================

/** Options for {@link assessCipma}. */
export interface CipmaOptions {
  target: string;
  scaleMin?: number;
  scaleMax?: number;
  /** Run the NCA overlay (default true; false = plain IPMA). */
  nca?: boolean;
  ncaCeilings?: readonly string[];
  /** R `nca_test.rep`; with the default 0 the analysis is deterministic. */
  ncaTestRep?: number;
  ncaSteps?: number;
  seed?: number;
}

/** Named-args form of {@link assessCipma}. */
export interface AssessCipmaArgs extends CipmaOptions {
  model: unknown;
}

/** Options for {@link assessIpma}. */
export interface IpmaOptions {
  target: string;
  scaleMin?: number;
  scaleMax?: number;
  seed?: number;
}

/** Named-args form of {@link assessIpma}. */
export interface AssessIpmaArgs extends IpmaOptions {
  model: unknown;
}

function isScalarNumber(x: unknown): x is number {
  return typeof x === "number" && !Number.isNaN(x);
}

/**
 * Combined Importance-Performance Map Analysis for an estimated PLS-SEM model.
 *
 * Returns null (with a warning) for non-seminr input. Computes each
 * predecessor's importance (unstandardized total effect on `target`) and
 * performance (0-100 rescaled), and — when `nca` is true — runs NCA to
 * classify constructs by importance x necessity. Interaction constructs are
 * excluded. Callable as `assessCipma(model, options)` or
 * `assessCipma({ model, ...options })`.
 */
export function assessCipma(args: AssessCipmaArgs): CipmaAnalysis | null;
export function assessCipma(model: unknown, options: CipmaOptions): CipmaAnalysis | null;
export function assessCipma(
  modelOrArgs: unknown,
  positionalOptions?: CipmaOptions,
): CipmaAnalysis | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessCipmaArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessCipmaArgs) : positionalOptions!;

  if (!validateSeminrModel(seminrModel, "assessCipma")) return null;
  const model = seminrModel;
  const target = options.target;
  const runNca = options.nca ?? true;

  const constructNames = model.constructScores.cols;
  if (!constructNames.includes(target)) {
    throw new Error(
      `target '${target}' not found in model constructs: ${constructNames.join(", ")}`,
    );
  }
  const rawMin = options.scaleMin ?? 1;
  const rawMax = options.scaleMax ?? 7;
  if (!isScalarNumber(rawMin) || !isScalarNumber(rawMax)) {
    throw new Error("scale_min and scale_max must be single numeric values.");
  }
  if (rawMin >= rawMax) {
    throw new Error("scale_min must be less than scale_max.");
  }
  const scaleMin = rawMin;
  const scaleMax = rawMax;

  const excludedInteractions = constructNames.filter(isInteractionConstruct);
  const pcNames = model.pathCoef.cols;
  const allConstructs = constructNames.filter(
    (c) => !isInteractionConstruct(c) && pcNames.includes(c),
  );
  let ipmaConstructs = allConstructs.filter((c) => c !== target);
  if (ipmaConstructs.length === 0) {
    throw new Error("No constructs available for IPMA (all excluded or only target exists).");
  }

  const negWeightConstructs = checkPositiveWeights(model, ipmaConstructs);
  if (negWeightConstructs.length > 0) {
    console.warn(
      "Negative outer weights detected for: " +
        negWeightConstructs.join(", ") +
        ". IPMA performance rescaling may be unreliable for these constructs. " +
        "Consider reverse-coding indicators or checking the model specification " +
        "(Ringle & Sarstedt, 2016).",
    );
  }

  let performance = computeIpmaPerformance(model, ipmaConstructs, scaleMin, scaleMax);

  const tStd = stdTotalEffects(model, allConstructs);
  let importanceStd = Object.fromEntries(
    ipmaConstructs.map((c) => [c, namedGet(tStd, c, target)]),
  );
  const tUnstd = computeUnstdTotalEffects(model, allConstructs, scaleMin, scaleMax);
  let importanceUnstd = Object.fromEntries(
    ipmaConstructs.map((c) => [c, namedGet(tUnstd, c, target)]),
  );

  const hasEffect = ipmaConstructs.filter((c) => Math.abs(importanceStd[c]!) > EPS);
  if (hasEffect.length === 0) {
    throw new Error(`No constructs have a non-zero total effect on '${target}'.`);
  }
  ipmaConstructs = hasEffect;
  performance = Object.fromEntries(ipmaConstructs.map((c) => [c, performance[c]!]));
  importanceStd = Object.fromEntries(ipmaConstructs.map((c) => [c, importanceStd[c]!]));
  importanceUnstd = Object.fromEntries(ipmaConstructs.map((c) => [c, importanceUnstd[c]!]));

  let ncaResult: NcaAnalysis | null = null;
  if (runNca) {
    ncaResult = assessNca(model, {
      target,
      predictors: ipmaConstructs,
      ceilings: [...(options.ncaCeilings ?? DEFAULT_NCA_CEILINGS)],
      testRep: options.ncaTestRep ?? 0,
      steps: options.ncaSteps ?? 10,
      seed: options.seed ?? 123,
    });
  }

  const classification = classifyCipmaConstructs(importanceUnstd, performance, ncaResult);

  const constructs = [...ipmaConstructs];
  const scaleRange: readonly [number, number] = [scaleMin, scaleMax];
  const nObs = model.constructScores.values.length;

  const ipTable = () =>
    formatStrTable(
      constructs,
      ["Unstd. Total Effect", "Std. Total Effect", "Performance"],
      constructs.map((c) => [
        importanceUnstd[c]!.toFixed(4),
        importanceStd[c]!.toFixed(4),
        performance[c]!.toFixed(2),
      ]),
    );

  const ncaTable = () => {
    const eff = ncaResult!.effectSizes;
    const nec = Object.fromEntries(classification.map((row) => [row.construct, row.necessary]));
    return formatStrTable(
      constructs,
      [...eff.cols.map((ceil) => `d (${ceil})`), "Necessary"],
      constructs.map((c) => [
        ...eff.cols.map((ceil) => namedGet(eff, c, ceil).toFixed(4)),
        nec[c] ? "Yes" : "No",
      ]),
    );
  };

  const classificationTable = () =>
    formatStrTable(
      constructs,
      ["Priority"],
      classification.map((row) => [row.priority]),
    );

  return Object.freeze({
    kind: "cipma_analysis" as const,
    importanceUnstd,
    importanceStd,
    performance,
    nca: ncaResult,
    classification,
    target,
    constructs,
    scaleRange,
    negativeWeightConstructs: negWeightConstructs,
    excludedInteractions,
    plsModel: model,
    toString(): string {
      const hasNca = ncaResult !== null;
      const title = hasNca
        ? "Combined Importance-Performance Map Analysis (cIPMA)"
        : "Importance-Performance Map Analysis (IPMA)";
      const lines = [
        title,
        "=".repeat(title.length),
        `Target: ${target}`,
        `Constructs: ${constructs.join(", ")}`,
        `Scale range: ${gFormat(scaleMin)} - ${gFormat(scaleMax)}`,
        `Observations: ${nObs}`,
      ];
      if (excludedInteractions.length > 0) {
        lines.push(`Excluded (interaction): ${excludedInteractions.join(", ")}`);
      }
      if (negWeightConstructs.length > 0) {
        lines.push(`Warning - negative weights: ${negWeightConstructs.join(", ")}`);
      }
      lines.push("", "Importance-Performance Results:", ipTable());
      if (hasNca) lines.push("", "Necessary Conditions (NCA):", ncaTable());
      lines.push("", hasNca ? "cIPMA Classification:" : "IPMA Classification:", classificationTable());
      return lines.join("\n");
    },
    summarize(): string {
      const hasNca = ncaResult !== null;
      const title = hasNca
        ? "Combined Importance-Performance Map Analysis (cIPMA) Summary"
        : "Importance-Performance Map Analysis (IPMA) Summary";
      const lines = [
        title,
        "=".repeat(title.length),
        `Target: ${target}`,
        `Scale range: ${gFormat(scaleMin)} - ${gFormat(scaleMax)}`,
        `Observations: ${nObs}`,
      ];
      if (excludedInteractions.length > 0) {
        lines.push(`Excluded interactions: ${excludedInteractions.join(", ")}`);
      }
      if (negWeightConstructs.length > 0) {
        lines.push(`Negative weight constructs: ${negWeightConstructs.join(", ")}`);
      }
      lines.push(
        "",
        `Importance (Unstandardized Total Effects on ${target} ):`,
        formatNamedVector(importanceUnstd, 4),
        "",
        `Importance (Standardized Total Effects on ${target} ):`,
        formatNamedVector(importanceStd, 4),
        "",
        "Performance (0-100 rescaled):",
        formatNamedVector(performance, 2),
      );
      if (hasNca) {
        lines.push("", "NCA Effect Sizes:", formatTable(ncaResult!.effectSizes, 4));
        if (ncaResult!.significance !== null) {
          lines.push("", "NCA Permutation p-values:", formatTable(ncaResult!.significance, 4));
        }
        if (ncaResult!.necessaryPredictors.length > 0) {
          lines.push("", `Necessary conditions: ${ncaResult!.necessaryPredictors.join(", ")}`);
          for (const [ceil, table] of Object.entries(ncaResult!.bottleneck)) {
            lines.push("", `Bottleneck table (${ceil}):`, formatTable(table, 1));
          }
        } else {
          lines.push("", "No necessary conditions identified.");
        }
      }
      lines.push("", "Construct Classification:", classificationTable());
      return lines.join("\n");
    },
  });
}

/**
 * Importance-Performance Map Analysis (IPMA) — cIPMA without the NCA overlay.
 *
 * Convenience wrapper for `assessCipma({..., nca: false})`; the returned
 * record's `nca` field is null.
 */
export function assessIpma(args: AssessIpmaArgs): CipmaAnalysis | null;
export function assessIpma(model: unknown, options: IpmaOptions): CipmaAnalysis | null;
export function assessIpma(
  modelOrArgs: unknown,
  positionalOptions?: IpmaOptions,
): CipmaAnalysis | null {
  const named = isNamedArgs(modelOrArgs);
  const model = named ? (modelOrArgs as AssessIpmaArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessIpmaArgs) : positionalOptions!;
  return assessCipma(model, {
    target: options.target,
    scaleMin: options.scaleMin,
    scaleMax: options.scaleMax,
    nca: false,
    seed: options.seed,
  });
}
