/**
 * PLS-POS: prediction-oriented segmentation via deterministic hill-climbing.
 *
 * Port of `feature_pos.R` from the seminrExtras R package (via the Python
 * port's pos.py). PLS-POS uncovers unobserved heterogeneity by maximizing the
 * sum of R-squared across all endogenous constructs across K segments (Becker
 * et al. 2013, MISQ 37(3)), using hard assignments, a distance measure built
 * from segment-specific structural residuals, and one-observation-at-a-time
 * reassignment.
 *
 * RNG surface: the only randomness is the initial partition drawn once per
 * start (R `random_partition`); rerun-based segment estimation and the
 * hill-climb are deterministic. Exact R parity is achieved by injecting the
 * R-generated initial assignments via `partitions`; the default path draws
 * from `mulberry32(seed)` (statistically equivalent, not bit-identical to R).
 */

import { namedMatrix, rerun, type NamedMatrix, type PlsModel } from "@seminr/core";
import {
  getEndogenousConstructs,
  isNamedArgs,
  seqSum,
  validatePositiveInteger,
  validateSeminrModel,
} from "./helpers.ts";
import { mulberry32 } from "@seminr/core";
import { nonzeroPathLines } from "./records.ts";

const EPS = Number.EPSILON;

// ---------------------------------------------------------------------------
// Internal machinery (R ``:::`` analog — importable, not in the barrel)
// ---------------------------------------------------------------------------

function nmGet(m: NamedMatrix, row: string, col: string): number {
  return m.values[m.rows.indexOf(row)]![m.cols.indexOf(col)]!;
}

/** R `pos_endogenous`: endogenous constructs minus interaction terms. */
export function posEndogenous(model: PlsModel): string[] {
  return getEndogenousConstructs(model).filter((c) => !c.includes("*"));
}

/**
 * R `estimate_segment_models`: rerun the model on each segment's rows.
 *
 * Returns null (whole list) if any segment has fewer than 2 observations;
 * individual entries are null when that segment's re-estimation fails.
 */
export function estimateSegmentModels(
  model: PlsModel,
  assignment: readonly number[],
  k: number,
): (PlsModel | null)[] | null {
  const segmentModels: (PlsModel | null)[] = [];
  for (let seg = 1; seg <= k; seg++) {
    const values = model.data.values.filter((_, i) => assignment[i] === seg);
    if (values.length < 2) return null;
    try {
      segmentModels.push(rerun(model, { data: { columns: model.data.columns, values } }));
    } catch {
      segmentModels.push(null);
    }
  }
  return segmentModels;
}

/** R `compute_pos_objective`: sum of R-squared across segments; -Inf on failure. */
export function computePosObjective(
  segmentModels: readonly (PlsModel | null)[],
  endogenous: readonly string[],
): number {
  let total = 0;
  for (const segModel of segmentModels) {
    if (segModel === null) return -Infinity;
    const vals = endogenous
      .map((b) => nmGet(segModel.rSquared, "Rsq", b))
      .filter((v) => !Number.isNaN(v));
    total += seqSum(vals);
  }
  return total;
}

/**
 * R `compute_structural_residuals`: N x B x K squared structural residuals.
 *
 * Predictions use GLOBAL construct scores with segment-specific path
 * coefficients (Becker et al. 2013, Appendix B).
 */
export function computeStructuralResiduals(
  model: PlsModel,
  segmentModels: readonly (PlsModel | null)[],
  endogenous: readonly string[],
): number[][][] {
  const scores = model.constructScores;
  const n = scores.values.length;
  const k = segmentModels.length;
  const bCount = endogenous.length;
  const residuals: number[][][] = Array.from({ length: n }, () =>
    Array.from({ length: bCount }, () => new Array(k).fill(NaN)),
  );

  segmentModels.forEach((segModel, seg) => {
    if (segModel === null) return;
    endogenous.forEach((target, bIdx) => {
      const predictors = model.smMatrix.constructAntecedents(target);
      if (predictors.length === 0) return;
      const beta = predictors.map((p) => nmGet(segModel.pathCoef, p, target));
      const predIdx = predictors.map((p) => scores.cols.indexOf(p));
      const yIdx = scores.cols.indexOf(target);
      for (let i = 0; i < n; i++) {
        const row = scores.values[i]!;
        let yhat = 0;
        for (let p = 0; p < beta.length; p++) yhat += row[predIdx[p]!]! * beta[p]!;
        const r = row[yIdx]! - yhat;
        residuals[i]![bIdx]![seg] = r * r;
      }
    });
  });
  return residuals;
}

/** R `compute_pos_distances`: D[i, k] = sum_b sqrt(e2_i / sum_i e2). */
export function computePosDistances(sqResiduals: readonly number[][][]): number[][] {
  const n = sqResiduals.length;
  const bCount = sqResiduals[0]?.length ?? 0;
  const k = sqResiduals[0]?.[0]?.length ?? 0;
  const distances: number[][] = Array.from({ length: n }, () => new Array(k).fill(0));
  for (let seg = 0; seg < k; seg++) {
    for (let bIdx = 0; bIdx < bCount; bIdx++) {
      const e2 = sqResiduals.map((row) => row[bIdx]![seg]!);
      const sumE2 = seqSum(e2.filter((v) => !Number.isNaN(v)));
      if (sumE2 < EPS) continue; // perfect fit for all observations
      for (let i = 0; i < n; i++) distances[i]![seg]! += Math.sqrt(e2[i]! / sumE2);
    }
  }
  return distances;
}

/** One improving move of the hill climb. */
export interface PosCandidate {
  /** 0-based observation index. */
  readonly obs: number;
  /** 1-based segment labels. */
  readonly fromK: number;
  readonly toK: number;
  readonly diff: number;
}

/**
 * R `build_candidate_list`: improving moves sorted by potential, descending.
 *
 * Ties keep ascending observation order (R's stable radix sort); returns null
 * when no observation has a positive improvement (convergence).
 */
export function buildCandidateList(
  distances: readonly (readonly number[])[],
  assignment: readonly number[],
  k: number,
): PosCandidate[] | null {
  const candidates: PosCandidate[] = [];
  for (let i = 0; i < distances.length; i++) {
    const current = assignment[i]! - 1;
    const dCurrent = distances[i]![current]!;
    let bestAlt = -1;
    for (let s = 0; s < k; s++) {
      if (s === current) continue;
      if (bestAlt < 0 || distances[i]![s]! < distances[i]![bestAlt]!) bestAlt = s; // ties -> lowest
    }
    const diff = dCurrent - distances[i]![bestAlt]!;
    if (diff > 0) candidates.push({ obs: i, fromK: current + 1, toK: bestAlt + 1, diff });
  }
  if (candidates.length === 0) return null;
  return candidates.sort((a, b) => b.diff - a.diff); // stable: ties keep obs order
}

/** Default-path initial partitions: mulberry32 analog of R `random_partition`. */
function defaultPartitions(
  n: number,
  k: number,
  minSize: number,
  nstart: number,
  seed: number | undefined,
): number[][] {
  const rng = mulberry32(seed ?? Math.floor(Math.random() * 0x100000000));
  const guaranteed: number[] = [];
  for (let seg = 1; seg <= k; seg++) for (let j = 0; j < minSize; j++) guaranteed.push(seg);
  const remaining = n - k * minSize;
  const partitions: number[][] = [];
  for (let r = 0; r < nstart; r++) {
    const pool = [...guaranteed];
    for (let j = 0; j < remaining; j++) pool.push(1 + Math.floor(rng() * k));
    // Fisher-Yates permutation
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = pool[i]!;
      pool[i] = pool[j]!;
      pool[j] = tmp;
    }
    partitions.push(pool);
  }
  return partitions;
}

// ---------------------------------------------------------------------------
// Result records
// ---------------------------------------------------------------------------

/** Result of {@link assessPos} (R class `pos_analysis`). */
export interface PosAnalysis {
  readonly kind: "pos_analysis";
  readonly k: number;
  /** 1-based hard assignments. */
  readonly segmentAssignment: readonly number[];
  /** Keys "Segment 1".."Segment K". */
  readonly segmentSizes: Readonly<Record<string, number>>;
  readonly segmentModels: readonly PlsModel[];
  /** endogenous x "Segment k". */
  readonly segmentRsquared: NamedMatrix;
  readonly segmentPaths: readonly NamedMatrix[];
  readonly objective: number;
  readonly converged: boolean;
  readonly iterations: number;
  readonly nstart: number;
  readonly allObjectives: readonly number[];
  readonly endogenous: readonly string[];
  readonly plsModel: PlsModel;
  readonly nObs: number;
  toString(): string;
  summarize(): string;
}

/** One row of the {@link PosComparison} fit table (mixed types). */
export interface PosFitRow {
  readonly k: number;
  readonly sumR2: number;
  readonly avgR2PerSegment: number;
  readonly converged: boolean | null;
  readonly iterations: number | null;
}

/** Result of {@link assessPosCompare} (R class `pos_comparison`). */
export interface PosComparison {
  readonly kind: "pos_comparison";
  readonly solutions: Readonly<Record<string, PosAnalysis | null>>;
  readonly fitTable: readonly PosFitRow[];
  readonly kRange: readonly number[];
  readonly plsModel: PlsModel;
  toString(): string;
  summarize(): string;
}

// ---------------------------------------------------------------------------
// Main entry points
// ---------------------------------------------------------------------------

/** Steps 2-6 of Becker et al. Appendix B: one-observation reassignment loop. */
function hillClimb(
  model: PlsModel,
  segModelsIn: (PlsModel | null)[],
  assignmentIn: number[],
  endogenous: readonly string[],
  k: number,
  maxIter: number,
  searchDepth: number,
  minSegmentSize: number,
): [number[], (PlsModel | null)[], number, boolean, number] {
  let assignment = assignmentIn;
  let segModels = segModelsIn;
  let obj = computePosObjective(segModels, endogenous);
  let converged = false;
  let iteration = 0;

  for (let iterIdx = 1; iterIdx <= maxIter; iterIdx++) {
    iteration = iterIdx;

    const sqResid = computeStructuralResiduals(model, segModels, endogenous);
    const distances = computePosDistances(sqResid);
    const candidates = buildCandidateList(distances, assignment, k);
    if (candidates === null) {
      converged = true;
      break;
    }

    let improved = false;
    let nTried = 0;
    for (const cand of candidates) {
      if (nTried >= searchDepth) break;
      nTried += 1;

      const fromSize = assignment.filter((a) => a === cand.fromK).length;
      if (fromSize - 1 < minSegmentSize) continue;

      const newAssignment = [...assignment];
      newAssignment[cand.obs] = cand.toK;

      // Re-estimate only the two affected segments (R-faithful, including the
      // quirk that a rerun failure leaves the stale model in place — the
      // unchanged objective then simply fails the strict > test).
      const newSegModels = [...segModels];
      for (const affectedK of [cand.fromK, cand.toK]) {
        const values = model.data.values.filter((_, i) => newAssignment[i] === affectedK);
        let newModel: PlsModel | null;
        try {
          newModel = rerun(model, { data: { columns: model.data.columns, values } });
        } catch {
          newModel = null;
        }
        if (newModel === null) break;
        newSegModels[affectedK - 1] = newModel;
      }
      if (newSegModels[cand.fromK - 1] === null || newSegModels[cand.toK - 1] === null) continue;

      const newObj = computePosObjective(newSegModels, endogenous);
      if (newObj > obj) {
        assignment = newAssignment;
        segModels = newSegModels;
        obj = newObj;
        improved = true;
        break;
      }
    }

    if (!improved) {
      converged = true;
      break;
    }
  }

  return [assignment, segModels, obj, converged, iteration];
}

function coercePartition(raw: readonly number[], n: number, k: number): number[] {
  if (raw.length !== n || Math.min(...raw) < 1 || Math.max(...raw) > k) {
    throw new Error(
      `each injected partition must be a length-${n} vector of segment labels in 1..${k}.`,
    );
  }
  return [...raw];
}

/** Options of {@link assessPos}. */
export interface PosOptions {
  K?: number;
  nstart?: number;
  maxIter?: number;
  /** Candidates tried per iteration (default: all N). */
  searchDepth?: number;
  /** Default: max(10, max predictors + 2). */
  minSegmentSize?: number;
  seed?: number;
  /** Parity injection: nstart length-N 1-based initial assignment vectors. */
  partitions?: readonly (readonly number[])[];
}

/** Named-args form of {@link assessPos}. */
export interface AssessPosArgs extends PosOptions {
  model: unknown;
}

/**
 * PLS-POS prediction-oriented segmentation (R `assess_pos`).
 *
 * Callable as `assessPos(model, options?)` or `assessPos({ model, ... })`.
 */
export function assessPos(args: AssessPosArgs): PosAnalysis | null;
export function assessPos(model: unknown, options?: PosOptions): PosAnalysis | null;
export function assessPos(
  modelOrArgs: unknown,
  positionalOptions: PosOptions = {},
): PosAnalysis | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessPosArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessPosArgs) : positionalOptions;

  if (!validateSeminrModel(seminrModel, "assessPos")) return null;
  const model = seminrModel;

  const k = validatePositiveInteger(options.K ?? 2, 2, "K");
  const nstart = options.nstart ?? 10;
  const maxIter = options.maxIter ?? 100;

  const n = model.constructScores.values.length;
  const endogenous = posEndogenous(model);
  if (endogenous.length === 0) {
    throw new Error("No endogenous constructs found in the model.");
  }

  let maxPreds = 0;
  for (const target of endogenous) {
    maxPreds = Math.max(maxPreds, model.smMatrix.constructAntecedents(target).length);
  }
  const minSegmentSize = options.minSegmentSize ?? Math.max(10, maxPreds + 2);

  if (n < k * minSegmentSize) {
    throw new Error(
      `Sample size (${n}) is too small for K=${k} segments ` +
        `with min_segment_size=${minSegmentSize}.`,
    );
  }

  const searchDepth = options.searchDepth ?? n;

  let startPartitions: number[][];
  if (options.partitions === undefined) {
    startPartitions = defaultPartitions(n, k, minSegmentSize, nstart, options.seed ?? 123);
  } else {
    if (options.partitions.length !== nstart) {
      throw new Error(`partitions must supply exactly nstart (${nstart}) assignments.`);
    }
    startPartitions = options.partitions.map((p) => coercePartition(p, n, k));
  }

  let bestObjective = -Infinity;
  let best: [number[], (PlsModel | null)[], number, boolean, number] | null = null;
  const allObjectives: number[] = [];

  for (let start = 0; start < startPartitions.length; start++) {
    let assignment = startPartitions[start]!;
    let segModels = estimateSegmentModels(model, assignment, k);
    if (segModels === null || segModels.some((m) => m === null)) {
      allObjectives.push(-Infinity);
      continue;
    }

    // Distance-based initial reassignment (step 1.3).
    const sqResid = computeStructuralResiduals(model, segModels, endogenous);
    const distances = computePosDistances(sqResid);
    assignment = distances.map((row) => {
      let arg = 0;
      for (let s = 1; s < k; s++) if (row[s]! < row[arg]!) arg = s;
      return arg + 1;
    });

    const sizes = new Array(k).fill(0);
    for (const a of assignment) sizes[a - 1] += 1;
    if (sizes.some((s) => s < minSegmentSize)) {
      allObjectives.push(-Infinity);
      continue;
    }

    segModels = estimateSegmentModels(model, assignment, k);
    if (segModels === null || segModels.some((m) => m === null)) {
      allObjectives.push(-Infinity);
      continue;
    }

    const [finalAssignment, finalModels, obj, converged, iteration] = hillClimb(
      model,
      segModels,
      assignment,
      endogenous,
      k,
      maxIter,
      searchDepth,
      minSegmentSize,
    );
    allObjectives.push(obj);

    if (obj > bestObjective) {
      bestObjective = obj;
      best = [finalAssignment, finalModels, obj, converged, iteration];
    }
  }

  if (best === null) {
    throw new Error(
      `PLS-POS failed to find a valid segmentation across all ${nstart} random ` +
        "starts. Consider increasing min_segment_size or decreasing K.",
    );
  }

  const [assignment, segModelsBest, objective, converged, iterations] = best;
  const finalModels = segModelsBest.filter((m): m is PlsModel => m !== null);

  const segNames = Array.from({ length: k }, (_, seg) => `Segment ${seg + 1}`);
  const sizes = new Array(k).fill(0);
  for (const a of assignment) sizes[a - 1] += 1;

  const rsqValues = endogenous.map((target) =>
    finalModels.map((m) => nmGet(m.rSquared, "Rsq", target)),
  );
  const segmentRsquared = namedMatrix(endogenous, segNames, rsqValues);
  const segmentSizes = Object.fromEntries(segNames.map((nm, s) => [nm, sizes[s]!]));
  const segmentPaths = finalModels.map((m) => m.pathCoef);
  const globalRsq = model.rSquared;
  const globalPaths = model.pathCoef;

  return Object.freeze({
    kind: "pos_analysis" as const,
    k,
    segmentAssignment: assignment,
    segmentSizes,
    segmentModels: finalModels,
    segmentRsquared,
    segmentPaths,
    objective,
    converged,
    iterations,
    nstart,
    allObjectives,
    endogenous,
    plsModel: model,
    nObs: n,
    toString(): string {
      const lines = [
        "PLS-POS Analysis",
        "================",
        `Segments: ${k}`,
        `Observations: ${n}`,
        `Converged: ${converged ? "Yes" : "No"} ( ${iterations} iterations )`,
        `Random starts: ${nstart}`,
        `Objective (Sum R²): ${objective.toFixed(4)}`,
        "",
        "Segment Sizes:",
      ];
      for (const [nm, size] of Object.entries(segmentSizes)) {
        lines.push(`  ${nm}: ${size} (proportion ${(size / n).toFixed(4)})`);
      }
      lines.push("", "R² per Endogenous Construct:");
      segmentRsquared.rows.forEach((construct, i) => {
        const cells = segmentRsquared.cols
          .map((nm, j) => `${nm} = ${segmentRsquared.values[i]![j]!.toFixed(4)}`)
          .join("  ");
        const globalValue = nmGet(globalRsq, "Rsq", construct);
        lines.push(`  ${construct}: ${cells}  Global = ${globalValue.toFixed(4)}`);
      });
      lines.push("", "Segment Path Coefficients:");
      for (let seg = 0; seg < k; seg++) {
        lines.push(`\n  Segment ${seg + 1} :`);
        lines.push(...nonzeroPathLines(segmentPaths[seg]!, "    "));
      }
      return lines.join("\n");
    },
    summarize(): string {
      const lines = [
        "PLS-POS Analysis — Detailed Summary",
        "======================================",
        `Segments: ${k} | Observations: ${n}`,
        `Objective (Sum R²): ${objective.toFixed(4)}`,
        `Converged: ${converged ? "Yes" : "No"} ( ${iterations} iterations )`,
        `Random starts: ${nstart}`,
        "Start objectives: " +
          allObjectives.map((o) => (Number.isFinite(o) ? o.toFixed(4) : "-Inf")).join(", "),
        "",
        "Segment Sizes:",
      ];
      Object.values(segmentSizes).forEach((size, idx) => {
        lines.push(`  Segment ${idx + 1}: ${size} (${Math.round((1000 * size) / n) / 10}%)`);
      });
      lines.push("", "R² Comparison (Segment vs Global):");
      segmentRsquared.rows.forEach((construct, i) => {
        const cells = segmentRsquared.cols
          .map((nm, j) => `${nm} = ${segmentRsquared.values[i]![j]!.toFixed(4)}`)
          .join("  ");
        const globalValue = nmGet(globalRsq, "Rsq", construct);
        lines.push(`  ${construct}: ${cells}  Global = ${globalValue.toFixed(4)}`);
      });
      lines.push("", "Path Coefficients per Segment:");
      globalPaths.cols.forEach((to, j) => {
        globalPaths.rows.forEach((from, i) => {
          if (globalPaths.values[i]![j] === 0) return;
          const cells = [`Global = ${globalPaths.values[i]![j]!.toFixed(4)}`];
          for (let seg = 0; seg < k; seg++) {
            cells.push(`Seg.${seg + 1} = ${segmentPaths[seg]!.values[i]![j]!.toFixed(4)}`);
          }
          lines.push(`  ${from} -> ${to}: ` + cells.join("  "));
        });
      });
      return lines.join("\n");
    },
  });
}

/** Options of {@link assessPosCompare}. */
export interface PosCompareOptions {
  KRange?: readonly number[];
  nstart?: number;
  maxIter?: number;
  searchDepth?: number;
  minSegmentSize?: number;
  seed?: number;
  /** Parity injection: K -> per-start initial assignments. */
  partitions?: Readonly<Record<number, readonly (readonly number[])[]>>;
}

/** Named-args form of {@link assessPosCompare}. */
export interface AssessPosCompareArgs extends PosCompareOptions {
  model: unknown;
}

/**
 * Compare PLS-POS solutions across K values (R `assess_pos_compare`).
 *
 * Each K re-runs {@link assessPos} with the SAME seed (R re-seeds inside). A K
 * whose run errors contributes a null solution and an NA fit row (R catches
 * the error with tryCatch).
 */
export function assessPosCompare(args: AssessPosCompareArgs): PosComparison | null;
export function assessPosCompare(model: unknown, options?: PosCompareOptions): PosComparison | null;
export function assessPosCompare(
  modelOrArgs: unknown,
  positionalOptions: PosCompareOptions = {},
): PosComparison | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessPosCompareArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessPosCompareArgs) : positionalOptions;

  if (!validateSeminrModel(seminrModel, "assessPosCompare")) return null;
  const model = seminrModel;

  const kRange = [...(options.KRange ?? [2, 3, 4, 5])].map((k) => Math.trunc(k));
  const solutions: Record<string, PosAnalysis | null> = {};
  const fitRows: PosFitRow[] = [];
  for (const k of kRange) {
    let sol: PosAnalysis | null;
    try {
      sol = assessPos(model, {
        K: k,
        nstart: options.nstart,
        maxIter: options.maxIter,
        searchDepth: options.searchDepth,
        minSegmentSize: options.minSegmentSize,
        seed: options.seed,
        partitions: options.partitions?.[k],
      });
    } catch {
      sol = null;
    }

    solutions[`K${k}`] = sol;
    if (sol !== null) {
      const colSums = sol.segmentRsquared.cols.map((_, j) =>
        sol.segmentRsquared.values.reduce(
          (s, row) => s + (Number.isNaN(row[j]!) ? 0 : row[j]!),
          0,
        ),
      );
      fitRows.push({
        k,
        sumR2: sol.objective,
        avgR2PerSegment: colSums.reduce((a, b) => a + b, 0) / colSums.length,
        converged: sol.converged,
        iterations: sol.iterations,
      });
    } else {
      fitRows.push({ k, sumR2: NaN, avgR2PerSegment: NaN, converged: null, iterations: null });
    }
  }

  return Object.freeze({
    kind: "pos_comparison" as const,
    solutions,
    fitTable: fitRows,
    kRange,
    plsModel: model,
    toString(): string {
      const lines = [
        "PLS-POS Comparison",
        "==================",
        "K range: " + kRange.join(", "),
        "",
        `${"K".padStart(4)}${"Sum_R2".padStart(12)}${"Avg_R2_per_segment".padStart(22)}` +
          `${"Converged".padStart(12)}${"Iterations".padStart(12)}`,
      ];
      for (const row of fitRows) {
        const sumR2 = Number.isNaN(row.sumR2) ? "NA" : row.sumR2.toFixed(4);
        const avg = Number.isNaN(row.avgR2PerSegment) ? "NA" : row.avgR2PerSegment.toFixed(4);
        const conv = row.converged === null ? "NA" : row.converged ? "TRUE" : "FALSE";
        const iters = row.iterations === null ? "NA" : String(row.iterations);
        lines.push(
          `${String(row.k).padStart(4)}${sumR2.padStart(12)}${avg.padStart(22)}` +
            `${conv.padStart(12)}${iters.padStart(12)}`,
        );
      }
      return lines.join("\n");
    },
    summarize(): string {
      // R `summary.pos_comparison` returns the object unchanged.
      return this.toString();
    },
  });
}

/** Extract the per-segment re-estimated PLS models (R `pos_segments`). */
export function posSegments(posResult: PosAnalysis): readonly PlsModel[] {
  if (
    typeof posResult !== "object" ||
    posResult === null ||
    (posResult as { kind?: string }).kind !== "pos_analysis"
  ) {
    throw new Error("pos_result must be a pos_analysis object from assess_pos().");
  }
  return posResult.segmentModels;
}
