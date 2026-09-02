/**
 * FIMIX-PLS (finite mixture PLS): EM-based latent-class segmentation.
 *
 * Port of `feature_fimix.R` from the seminrExtras R package (via the Python
 * port's fimix.py). Observations are probabilistically assigned to K segments,
 * each with segment-specific structural path coefficients, by fitting a
 * mixture of regressions to every structural equation simultaneously
 * (Hahn et al. 2002; Sarstedt et al. 2011).
 *
 * RNG surface: the only randomness is the per-start initial posterior matrix
 * (R: `init_random_posteriors` — N*K exponential draws, row-normalized). The
 * EM loop itself is deterministic, so exact R parity is achieved by injecting
 * the R-generated initial matrices via `inits`; the default path draws
 * exponentials from `mulberry32(seed)` (statistically equivalent, not
 * bit-identical to R).
 */

import { mulberry32, namedMatrix, type NamedMatrix, type PlsModel } from "@seminr/core";
import { solve } from "@compstats/core/linalg";
import { asMatrix, isNamedArgs, validatePositiveInteger, validateSeminrModel } from "./helpers.ts";
import { nonzeroPathLines } from "./records.ts";

const EPS = Number.EPSILON;
const IC_KEYS = ["lnL", "AIC", "AIC3", "AIC4", "BIC", "CAIC", "HQ", "MDL5", "EN"] as const;
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

// ---------------------------------------------------------------------------
// Internal machinery (R ``:::`` analog — importable, not in the barrel)
// ---------------------------------------------------------------------------

/** R `log_sum_exp`: max-shifted log of a sum of exponentials. */
export function logSumExp(x: readonly number[]): number {
  const m = Math.max(...x);
  if (!Number.isFinite(m)) return m;
  let s = 0;
  for (const v of x) s += Math.exp(v - m);
  return m + Math.log(s);
}

/** One endogenous construct's regression: scores on antecedent scores. */
export interface StructuralEquation {
  readonly target: string;
  readonly predictors: readonly string[];
  readonly y: readonly number[];
  /** Intercept column + predictor score columns. */
  readonly x: readonly (readonly number[])[];
}

/**
 * R `extract_structural_equations`: one equation per endogenous construct.
 *
 * Keys follow R's `unique(sm[, "target"])` first-appearance order, and each
 * equation's predictors keep structural-model row order — both orders drive
 * downstream loops and result labels.
 */
export function extractStructuralEquations(model: PlsModel): Map<string, StructuralEquation> {
  const scores = model.constructScores;
  const col = (name: string) => {
    const j = scores.cols.indexOf(name);
    return scores.values.map((row) => row[j]!);
  };
  const equations = new Map<string, StructuralEquation>();
  for (const target of model.smMatrix.allEndogenous()) {
    const predictors = model.smMatrix.constructAntecedents(target);
    const predCols = predictors.map(col);
    const x = scores.values.map((_, i) => [1, ...predCols.map((pc) => pc[i]!)]);
    equations.set(target, { target, predictors, y: col(target), x });
  }
  return equations;
}

/**
 * R `weighted_ols`: WLS coefficients + weighted residual variance.
 *
 * A singular system returns `[NaN coefficients, NaN]` — the EM caller treats
 * that start as failed, mirroring R's tryCatch.
 */
function weightedOls(
  y: readonly number[],
  x: readonly (readonly number[])[],
  weights: readonly number[],
): [number[], number] {
  const p = x[0]!.length;
  const n = y.length;
  const xtwx: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const xtwy: number[] = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    const w = weights[i]!;
    const row = x[i]!;
    for (let a = 0; a < p; a++) {
      const wa = w * row[a]!;
      xtwy[a]! += wa * y[i]!;
      const xa = xtwx[a]!;
      for (let b = 0; b < p; b++) xa[b]! += wa * row[b]!;
    }
  }
  let beta: number[];
  try {
    beta = solve(asMatrix(xtwx), xtwy);
  } catch {
    return [new Array(p).fill(NaN), NaN];
  }
  if (!beta.every(Number.isFinite)) return [new Array(p).fill(NaN), NaN];
  let ssw = 0;
  let sw = 0;
  for (let i = 0; i < n; i++) {
    let mu = 0;
    const row = x[i]!;
    for (let a = 0; a < p; a++) mu += row[a]! * beta[a]!;
    const r = y[i]! - mu;
    ssw += weights[i]! * r * r;
    sw += weights[i]!;
  }
  // Floor variance to prevent degenerate densities (R: .Machine$double.eps * 100).
  const sigma2 = Math.max(ssw / sw, EPS * 100);
  return [beta, sigma2];
}

/** R `count_fimix_parameters`: Q = K * sum_j(p_j + 2) + (K - 1). */
export function countFimixParameters(
  equations: ReadonlyMap<string, StructuralEquation>,
  k: number,
): number {
  let paramsPerEq = 0;
  for (const eq of equations.values()) paramsPerEq += eq.predictors.length + 2;
  return k * paramsPerEq + (k - 1);
}

/** R `compute_entropy`: normed entropy EN = 1 - (-sum P log P)/(N log K). */
export function computeEntropy(posteriors: readonly (readonly number[])[], k: number): number {
  if (k <= 1) return NaN;
  const n = posteriors.length;
  let raw = 0;
  for (const row of posteriors) {
    for (const v of row) {
      const p = Math.max(v, EPS);
      raw -= p * Math.log(p);
    }
  }
  return 1 - raw / (n * Math.log(k));
}

/** R `compute_fimix_criteria`: information criteria in R's key order. */
export function computeFimixCriteria(
  lnl: number,
  q: number,
  n: number,
  k: number,
  posteriors: readonly (readonly number[])[],
): Record<string, number> {
  return {
    lnL: lnl,
    AIC: -2 * lnl + 2 * q,
    AIC3: -2 * lnl + 3 * q,
    AIC4: -2 * lnl + 4 * q,
    BIC: -2 * lnl + q * Math.log(n),
    CAIC: -2 * lnl + q * (Math.log(n) + 1),
    HQ: -2 * lnl + 2 * q * Math.log(Math.log(n)),
    MDL5: -2 * lnl + (q / 2) * Math.log(n),
    EN: computeEntropy(posteriors, k),
  };
}

interface EmFit {
  converged: boolean;
  lnl: number;
  iterations: number;
  posteriors?: number[][];
  piK?: number[];
  /** segmentParams.get(eqName)![k] = [coefficients, variance] */
  segmentParams?: Map<string, [number[], number][]>;
}

/** R `run_fimix_em`: M-step-first EM with log-sum-exp posteriors. */
function runFimixEm(
  equations: ReadonlyMap<string, StructuralEquation>,
  k: number,
  maxIter: number,
  stopCriterion: number,
  posteriorsInit: readonly (readonly number[])[],
): EmFit {
  const n = posteriorsInit.length;
  let posteriors = posteriorsInit.map((row) => [...row]);
  let lnlOld = -Infinity;
  let converged = false;
  let iteration = 0;
  let piK: number[] = new Array(k).fill(NaN);
  const segmentParams = new Map<string, [number[], number][]>();

  for (let it = 1; it <= maxIter; it++) {
    iteration = it;

    // --- M-step ---
    const nK = new Array(k).fill(0);
    for (const row of posteriors) for (let s = 0; s < k; s++) nK[s] += row[s]!;
    piK = nK.map((v) => v / n);
    if (Math.min(...nK) < 1) {
      return { converged: false, lnl: -Infinity, iterations: iteration }; // degenerate segment
    }

    for (const [eqName, eq] of equations) {
      const params: [number[], number][] = [];
      for (let seg = 0; seg < k; seg++) {
        const [beta, sigma2] = weightedOls(
          eq.y,
          eq.x,
          posteriors.map((row) => row[seg]!),
        );
        if (!beta.every(Number.isFinite)) {
          return { converged: false, lnl: -Infinity, iterations: iteration };
        }
        params.push([beta, sigma2]);
      }
      segmentParams.set(eqName, params);
    }

    // --- E-step ---
    const logDens: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
    for (let seg = 0; seg < k; seg++) {
      const logPi = Math.log(piK[seg]!);
      for (let i = 0; i < n; i++) logDens[i]![seg] = logPi;
      for (const eq of equations.values()) {
        const [beta, sigma2] = segmentParams.get(eq.target)![seg]!;
        const sigma = Math.sqrt(sigma2);
        const logSigma = Math.log(sigma);
        for (let i = 0; i < n; i++) {
          let mu = 0;
          const row = eq.x[i]!;
          for (let a = 0; a < beta.length; a++) mu += row[a]! * beta[a]!;
          const z = (eq.y[i]! - mu) / sigma;
          logDens[i]![seg]! += -LOG_SQRT_2PI - logSigma - 0.5 * z * z;
        }
      }
    }

    let lnlNew = 0;
    posteriors = logDens.map((row) => {
      const denom = logSumExp(row);
      lnlNew += denom;
      // Clamp to avoid exact 0/1, then renormalize (R pmax + rowSums).
      const post = row.map((v) => Math.max(Math.exp(v - denom), EPS));
      const s = post.reduce((a, b) => a + b, 0);
      return post.map((v) => v / s);
    });

    // --- Convergence check ---
    if (Math.abs(lnlNew - lnlOld) < stopCriterion) {
      converged = true;
      lnlOld = lnlNew;
      break;
    }
    lnlOld = lnlNew;
  }

  return { converged, lnl: lnlOld, iterations: iteration, posteriors, piK, segmentParams };
}

function buildSegmentPaths(
  equations: ReadonlyMap<string, StructuralEquation>,
  segmentParams: ReadonlyMap<string, [number[], number][]>,
  model: PlsModel,
  k: number,
): [NamedMatrix[], Record<string, number>[]] {
  const template = model.pathCoef;
  const { rows, cols } = template;
  const paths: NamedMatrix[] = [];
  const intercepts: Record<string, number>[] = [];
  for (let seg = 0; seg < k; seg++) {
    const values = template.values.map((row) => row.map(() => 0));
    const segIntercepts: Record<string, number> = {};
    for (const [eqName, eq] of equations) {
      const [beta] = segmentParams.get(eqName)![seg]!;
      segIntercepts[eqName] = beta[0]!;
      eq.predictors.forEach((pred, pIdx) => {
        const ri = rows.indexOf(pred);
        const cj = cols.indexOf(eq.target);
        if (ri >= 0 && cj >= 0) values[ri]![cj] = beta[pIdx + 1]!;
      });
    }
    paths.push(namedMatrix(rows, cols, values));
    intercepts.push(segIntercepts);
  }
  return [paths, intercepts];
}

// ---------------------------------------------------------------------------
// Result records
// ---------------------------------------------------------------------------

function segmentNames(k: number): string[] {
  return Array.from({ length: k }, (_, seg) => `Segment_${seg + 1}`);
}

/** Result of {@link assessFimix} (R class `fimix_analysis`). */
export interface FimixAnalysis {
  readonly kind: "fimix_analysis";
  readonly k: number;
  readonly segmentProportions: Readonly<Record<string, number>>;
  readonly segmentSizes: Readonly<Record<string, number>>;
  readonly posterior: NamedMatrix;
  /** 1-based hard assignments (argmax, ties -> first). */
  readonly segmentAssignment: readonly number[];
  readonly segmentPaths: readonly NamedMatrix[];
  readonly segmentIntercepts: readonly Readonly<Record<string, number>>[];
  readonly segmentVariances: NamedMatrix;
  readonly logLikelihood: number;
  readonly nParameters: number;
  readonly infoCriteria: Readonly<Record<string, number>>;
  readonly converged: boolean;
  readonly iterations: number;
  readonly nStartsCompleted: number;
  readonly plsModel: PlsModel;
  readonly nObs: number;
  toString(): string;
  summarize(): string;
}

/** Result of {@link assessFimixCompare} (R class `fimix_comparison`). */
export interface FimixComparison {
  readonly kind: "fimix_comparison";
  readonly solutions: Readonly<Record<string, FimixAnalysis | null>>;
  /** Rows "K<k>", cols K + information criteria. */
  readonly fitTable: NamedMatrix;
  readonly kRange: readonly number[];
  readonly plsModel: PlsModel;
  toString(): string;
  summarize(): string;
}

// ---------------------------------------------------------------------------
// Main entry points
// ---------------------------------------------------------------------------

/** Default-path initial posteriors: mulberry32 analog of R's rexp draws. */
function defaultInits(n: number, k: number, nstart: number, seed: number | undefined): number[][][] {
  const rng = mulberry32(seed ?? Math.floor(Math.random() * 0x100000000));
  const inits: number[][][] = [];
  for (let r = 0; r < nstart; r++) {
    const mat: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row = Array.from({ length: k }, () => -Math.log(1 - rng()));
      const s = row.reduce((a, b) => a + b, 0);
      mat.push(row.map((v) => v / s));
    }
    inits.push(mat);
  }
  return inits;
}

/** Options of {@link assessFimix}. */
export interface FimixOptions {
  K?: number;
  nstart?: number;
  maxIter?: number;
  stopCriterion?: number;
  seed?: number;
  /** Parity injection: nstart N x K initial posterior matrices (rows sum to 1). */
  inits?: readonly (readonly (readonly number[])[])[];
}

/** Named-args form of {@link assessFimix}. */
export interface AssessFimixArgs extends FimixOptions {
  model: unknown;
}

/**
 * FIMIX-PLS latent-class segmentation (R `assess_fimix`).
 *
 * Callable as `assessFimix(model, options?)` or `assessFimix({ model, ... })`.
 */
export function assessFimix(args: AssessFimixArgs): FimixAnalysis | null;
export function assessFimix(model: unknown, options?: FimixOptions): FimixAnalysis | null;
export function assessFimix(
  modelOrArgs: unknown,
  positionalOptions: FimixOptions = {},
): FimixAnalysis | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessFimixArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessFimixArgs) : positionalOptions;

  if (!validateSeminrModel(seminrModel, "assessFimix")) return null;
  const model = seminrModel;

  const k = validatePositiveInteger(options.K ?? 2, 2, "K");
  const nstartRaw = options.nstart ?? 10;
  if (typeof nstartRaw !== "number" || Number.isNaN(nstartRaw) || nstartRaw < 1) {
    throw new Error("nstart must be an integer >= 1.");
  }
  const nstart = Math.trunc(nstartRaw);
  const maxIter = options.maxIter ?? 5000;
  const stopCriterion = options.stopCriterion ?? 1e-6;

  const equations = extractStructuralEquations(model);
  if (equations.size === 0) {
    throw new Error("No endogenous constructs found in the structural model.");
  }

  const n = model.constructScores.values.length;
  const q = countFimixParameters(equations, k);

  let maxPredictors = 0;
  for (const eq of equations.values()) maxPredictors = Math.max(maxPredictors, eq.predictors.length);
  const minSegmentObs = Math.max(maxPredictors + 2, 10);
  if (n / k < minSegmentObs) {
    console.warn(
      `Sample size (${n}) may be too small for K = ${k} segments. ` +
        `Minimum ~${minSegmentObs} observations per segment recommended.`,
    );
  }

  let startInits: readonly (readonly (readonly number[])[])[];
  if (options.inits === undefined) {
    startInits = defaultInits(n, k, nstart, options.seed ?? 123);
  } else {
    if (options.inits.length !== nstart) {
      throw new Error(`inits must supply exactly nstart (${nstart}) matrices.`);
    }
    startInits = options.inits;
  }

  let best: EmFit | null = null;
  let bestLnl = -Infinity;
  let nCompleted = 0;
  for (const posteriorsInit of startInits) {
    const fit = runFimixEm(equations, k, maxIter, stopCriterion, posteriorsInit);
    nCompleted += 1;
    if (fit.converged && fit.lnl > bestLnl) {
      best = fit;
      bestLnl = fit.lnl;
    }
  }

  if (best === null) {
    console.warn(
      `FIMIX-PLS did not converge for K = ${k} in any of ${nstart} random ` +
        "starts. Consider reducing K or increasing max_iter.",
    );
    return null;
  }

  const [paths, intercepts] = buildSegmentPaths(equations, best.segmentParams!, model, k);

  const eqNames = [...equations.keys()];
  const segNames = segmentNames(k);
  const varValues = eqNames.map((eq) =>
    Array.from({ length: k }, (_, seg) => best.segmentParams!.get(eq)![seg]![1]),
  );
  const segmentVariances = namedMatrix(eqNames, segNames, varValues);

  const assignment = best.posteriors!.map((row) => {
    let arg = 0;
    for (let s = 1; s < k; s++) if (row[s]! > row[arg]!) arg = s;
    return arg + 1;
  });
  const sizes = new Array(k).fill(0);
  for (const a of assignment) sizes[a - 1] += 1;

  const posterior = namedMatrix(
    Array.from({ length: n }, (_, i) => String(i + 1)),
    segNames,
    best.posteriors!,
  );

  const segmentProportions = Object.fromEntries(segNames.map((nm, s) => [nm, best.piK![s]!]));
  const segmentSizes = Object.fromEntries(segNames.map((nm, s) => [nm, sizes[s]!]));
  const infoCriteria = computeFimixCriteria(best.lnl, q, n, k, best.posteriors!);
  const { lnl, converged, iterations } = best;

  return Object.freeze({
    kind: "fimix_analysis" as const,
    k,
    segmentProportions,
    segmentSizes,
    posterior,
    segmentAssignment: assignment,
    segmentPaths: paths,
    segmentIntercepts: intercepts,
    segmentVariances,
    logLikelihood: lnl,
    nParameters: q,
    infoCriteria,
    converged,
    iterations,
    nStartsCompleted: nCompleted,
    plsModel: model,
    nObs: n,
    toString(): string {
      const lines = [
        "FIMIX-PLS Analysis",
        "==================",
        `Segments: ${k}`,
        `Observations: ${n}`,
        `Converged: ${converged ? "Yes" : "No"} ( ${iterations} iterations )`,
        `Random starts: ${nCompleted}`,
        "",
        "Segment Proportions:",
      ];
      for (const segName of segNames) {
        const label = segName.replace("_", " ");
        lines.push(
          `  ${label}: ${segmentProportions[segName]!.toFixed(4)} (n = ${segmentSizes[segName]})`,
        );
      }
      lines.push(
        "",
        "Fit Criteria:",
        "  " +
          ["lnL", "AIC", "AIC3", "AIC4", "BIC", "CAIC"]
            .map((key) => `${key} = ${infoCriteria[key]!.toFixed(2)}`)
            .join("  ") +
          `  EN = ${infoCriteria["EN"]!.toFixed(4)}`,
        "",
        "Segment Path Coefficients:",
      );
      for (let seg = 0; seg < k; seg++) {
        lines.push(`\n  Segment ${seg + 1} :`);
        lines.push(...nonzeroPathLines(paths[seg]!, "    "));
      }
      return lines.join("\n");
    },
    summarize(): string {
      const lines = [
        "FIMIX-PLS Analysis Summary",
        "==========================",
        `Segments: ${k}`,
        `Observations: ${n}`,
        `Free parameters: ${q}`,
        `Converged: ${converged ? "Yes" : "No"} ( ${iterations} iterations )`,
        `Random starts: ${nCompleted}`,
        "",
        "Segment Proportions:",
        "  " +
          segNames.map((nm) => `${nm} = ${segmentProportions[nm]!.toFixed(4)}`).join("  "),
        "",
        "Fit Criteria:",
        "  " +
          IC_KEYS.map((key) => `${key} = ${Number(infoCriteria[key]!.toFixed(4))}`).join("  "),
      ];
      for (let seg = 0; seg < k; seg++) {
        lines.push(
          "",
          `--- Segment ${seg + 1} ---`,
          "Intercepts:",
          "  " +
            Object.entries(intercepts[seg]!)
              .map(([nm, v]) => `${nm} = ${v.toFixed(4)}`)
              .join("  "),
          "Path Coefficients:",
        );
        lines.push(...nonzeroPathLines(paths[seg]!, "  "));
        lines.push(
          "Residual Variances:",
          "  " +
            eqNames
              .map((nm, ei) => `${nm} = ${segmentVariances.values[ei]![seg]!.toFixed(4)}`)
              .join("  "),
        );
      }
      return lines.join("\n");
    },
  });
}

/** Options of {@link assessFimixCompare}. */
export interface FimixCompareOptions {
  KRange?: readonly number[];
  nstart?: number;
  maxIter?: number;
  stopCriterion?: number;
  seed?: number;
  /** Parity injection: K -> per-start initial posterior matrices. */
  inits?: Readonly<Record<number, readonly (readonly (readonly number[])[])[]>>;
}

/** Named-args form of {@link assessFimixCompare}. */
export interface AssessFimixCompareArgs extends FimixCompareOptions {
  model: unknown;
}

function fitLines(fitTable: NamedMatrix): string[] {
  const header = ["K", ...IC_KEYS];
  const widths = header.map((h) => Math.max(8, h.length + 2));
  const lines = [header.map((h, j) => h.padStart(widths[j]!)).join("")];
  for (const row of fitTable.values) {
    const cells = [String(Math.trunc(row[0]!))];
    IC_KEYS.forEach((key, j0) => {
      const value = row[j0 + 1]!;
      if (Number.isNaN(value)) cells.push("NA");
      else if (key === "EN") cells.push(value.toFixed(4));
      else cells.push(value.toFixed(2));
    });
    lines.push(cells.map((c, j) => c.padStart(widths[j]!)).join(""));
  }
  return lines;
}

/**
 * Compare FIMIX-PLS solutions across K values (R `assess_fimix_compare`).
 *
 * Each K re-runs {@link assessFimix} with the SAME seed (R re-seeds inside),
 * so solutions are independent replays.
 */
export function assessFimixCompare(args: AssessFimixCompareArgs): FimixComparison | null;
export function assessFimixCompare(
  model: unknown,
  options?: FimixCompareOptions,
): FimixComparison | null;
export function assessFimixCompare(
  modelOrArgs: unknown,
  positionalOptions: FimixCompareOptions = {},
): FimixComparison | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessFimixCompareArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessFimixCompareArgs) : positionalOptions;

  if (!validateSeminrModel(seminrModel, "assessFimixCompare")) return null;
  const model = seminrModel;

  const kValues = [...(options.KRange ?? [2, 3, 4, 5])];
  if (
    kValues.length < 1 ||
    kValues.some((k) => typeof k !== "number" || Number.isNaN(k) || k < 2)
  ) {
    throw new Error("K_range must be a vector of integers >= 2.");
  }

  const solutions: Record<string, FimixAnalysis | null> = {};
  const fitRows: number[][] = [];
  for (const k of kValues) {
    const sol = assessFimix(model, {
      K: k,
      nstart: options.nstart,
      maxIter: options.maxIter,
      stopCriterion: options.stopCriterion,
      seed: options.seed,
      inits: options.inits?.[k],
    });
    solutions[`K${Math.trunc(k)}`] = sol;
    fitRows.push([
      k,
      ...IC_KEYS.map((key) => (sol !== null ? sol.infoCriteria[key]! : NaN)),
    ]);
  }

  const fitTable = namedMatrix(
    kValues.map((k) => `K${Math.trunc(k)}`),
    ["K", ...IC_KEYS],
    fitRows,
  );
  const kRange = kValues.map((k) => Math.trunc(k));

  return Object.freeze({
    kind: "fimix_comparison" as const,
    solutions,
    fitTable,
    kRange,
    plsModel: model,
    toString(): string {
      const lines = [
        "FIMIX-PLS Segment Selection",
        "===========================",
        `K range: ${Math.min(...kRange)} to ${Math.max(...kRange)}`,
        "",
        "Fit Criteria:",
        ...fitLines(fitTable),
        "",
        "Best K by criterion:",
      ];
      const kCol = fitTable.values.map((row) => row[0]!);
      for (const crit of ["AIC", "AIC3", "AIC4", "BIC", "CAIC"]) {
        const j = fitTable.cols.indexOf(crit);
        const vals = fitTable.values.map((row) => row[j]!);
        if (vals.every(Number.isNaN)) continue;
        let bestI = -1;
        vals.forEach((v, i) => {
          if (!Number.isNaN(v) && (bestI < 0 || v < vals[bestI]!)) bestI = i;
        });
        lines.push(`   ${crit} : ${Math.trunc(kCol[bestI]!)}`);
      }
      const enJ = fitTable.cols.indexOf("EN");
      const enVals = fitTable.values.map((row) => row[enJ]!);
      if (!enVals.every(Number.isNaN)) {
        let bestI = -1;
        enVals.forEach((v, i) => {
          if (!Number.isNaN(v) && (bestI < 0 || v > enVals[bestI]!)) bestI = i;
        });
        lines.push(`   EN (best): ${Math.trunc(kCol[bestI]!)}`);
      }
      return lines.join("\n");
    },
    summarize(): string {
      const lines = [
        "FIMIX-PLS Comparison Summary",
        "============================",
        "",
        "Fit Table:",
        ...fitLines(fitTable),
      ];
      for (const sol of Object.values(solutions)) {
        if (sol === null) continue;
        const sizes = Object.values(sol.segmentSizes).join(", ");
        lines.push(
          "",
          `--- K = ${sol.k} ---`,
          `Converged: ${sol.converged ? "Yes" : "No"}`,
          `Segment sizes: ${sizes}`,
          `EN: ${sol.infoCriteria["EN"]!.toFixed(4)}`,
        );
      }
      return lines.join("\n");
    },
  });
}
