/**
 * Predictive Contribution of the Mediator (PCM), ported from feature_pcm.R
 * (via the Python port's pcm.py).
 *
 * PCM (Danks 2021) quantifies how much a mediator improves out-of-sample
 * prediction of an outcome. For each simple mediation triple `X -> M -> Y` to a
 * target, an isolated partial-mediation sub-model (`X -> M`, `M -> Y`, `X -> Y`)
 * is estimated, then predicted twice via `predictPls`: Direct Antecedents (DA, Y
 * from M and X) and Earliest Antecedents (EA, Y from X only). Per target
 * indicator, `PCM = (METRIC_EA - METRIC_DA) / METRIC_EA` for RMSE and MAE.
 *
 * Parity notes (plan F10): R's `reps` re-runs prediction on the same shuffled
 * folds and averages identical matrices — a numeric no-op, so it is validated
 * and stored only (never passed to predict). `orderings` injects the R
 * cross-validation fold orders (one per predict call, in call order path1-DA,
 * path1-EA, path2-DA, ...) for exact fixture parity; otherwise `seed` seeds a
 * `mulberry32` stream that draws one child seed per predict call, so same-seed
 * runs are reproducible (documented as non-bit-identical to R, plan Q4).
 */

import {
  composite,
  constructs,
  estimatePls,
  modeB,
  mulberry32,
  namedMatrix,
  paths,
  predictDA,
  predictEA,
  predictPls,
  reflective,
  relationships,
  type NamedMatrix,
  type PlsModel,
} from "@seminr/core";
import {
  isNamedArgs,
  itemsOfConstruct,
  rMean,
  validateForPrediction,
  validatePositiveInteger,
  validateSeminrModel,
} from "./helpers.ts";

const RESULT_COLS = ["RMSE_DA", "RMSE_EA", "PCM_RMSE", "MAE_DA", "MAE_EA", "PCM_MAE"];

/** A single mediation triple `antecedent -> mediator -> target`. */
export interface PcmPath {
  readonly antecedent: string;
  readonly mediator: string;
  readonly target: string;
}

/**
 * Per-path PCM metrics over the target construct's indicators.
 *
 * `results` is a NamedMatrix with the target indicators as rows and
 * {@link RESULT_COLS} as columns. `pcmRmse`/`pcmMae` are the per-indicator PCM
 * values (the `PCM_RMSE`/`PCM_MAE` columns) for convenient averaging.
 */
export interface PcmPathResult {
  readonly antecedent: string;
  readonly mediator: string;
  readonly target: string;
  readonly results: NamedMatrix;
  readonly pcmRmse: readonly number[];
  readonly pcmMae: readonly number[];
}

/** Result of {@link assessPcm} (R class `pcm_analysis`). */
export interface PcmAnalysis {
  readonly kind: "pcm_analysis";
  readonly target: string;
  readonly mediationPaths: readonly PcmPath[];
  readonly pcmResults: readonly PcmPathResult[];
  readonly noFolds: number;
  readonly reps: number;
  toString(): string;
  summarize(): string;
}

/** Options for {@link assessPcm}. */
export interface PcmFeatureOptions {
  /** Terminal outcome to explain; auto-detected when omitted. */
  target?: string;
  /** Number of cross-validation folds (default 10, integer >= 2). */
  noFolds?: number;
  /** Accepted for R API compatibility and stored only; a prediction no-op. */
  reps?: number;
  /** Seeds the default per-predict-call child-seed stream (reproducibility). */
  seed?: number;
  /**
   * Injected 0-based fold orderings for exact R parity, one per predict call in
   * call order (path1-DA, path1-EA, path2-DA, ...); length must be
   * `2 * mediationPaths.length`.
   */
  orderings?: readonly (readonly number[])[];
}

/** Named-args form of {@link assessPcm}. */
export interface AssessPcmArgs extends PcmFeatureOptions {
  model: unknown;
}

/** Classify a PCM value using Danks (2021) rules of thumb. */
export function classifyPcm(value: number): string {
  if (Number.isNaN(value)) return "NA";
  if (value < 0) return "Negative";
  if (value < 0.05) return "Weak";
  if (value < 0.1) return "Moderate";
  return "Strong";
}

/** Detect the single final endogenous construct (a target but never a source). */
export function detectFinalEndogenous(model: PlsModel): string {
  const endogenous = model.smMatrix.allEndogenous();
  // Sources = every construct that appears as an antecedent of any target.
  const sourceSet = new Set<string>();
  for (const c of endogenous) {
    for (const s of model.smMatrix.constructAntecedents(c)) sourceSet.add(s);
  }
  const final = endogenous.filter((c) => !sourceSet.has(c));
  if (final.length === 1) return final[0]!;
  if (final.length === 0) {
    throw new Error(
      "No final endogenous construct found (all endogenous constructs are also " +
        "sources). Please specify 'target'.",
    );
  }
  throw new Error(
    `Multiple final endogenous constructs found: ${final.join(", ")}. Please specify 'target'.`,
  );
}

/**
 * Whether `construct` is higher-order: its items are LOC construct names.
 *
 * HOC items appear as column names of `outerWeights` (they are LOC scores, not
 * raw indicators), so a sub-model cannot be rebuilt from raw items.
 */
function isHocConstruct(construct: string, model: PlsModel): boolean {
  const items = itemsOfConstruct(construct, model);
  const weightCols = model.outerWeights.cols;
  return items.some((item) => weightCols.includes(item));
}

/**
 * Find all simple mediation triples `X -> M -> Y` for `target` (Y).
 *
 * A mediation path exists when a direct predictor `M` of the target is itself
 * endogenous (has an antecedent `X`). Interaction constructs (`*` in the name)
 * are excluded; triples touching a higher-order construct are skipped with a
 * warning (the sub-model cannot be rebuilt from raw indicators).
 */
export function findMediationPaths(model: PlsModel, target: string): PcmPath[] {
  const directPreds = model.smMatrix
    .constructAntecedents(target)
    .filter((source) => !source.includes("*"));

  const found: PcmPath[] = [];
  for (const mediator of directPreds) {
    const antecedents = model.smMatrix
      .constructAntecedents(mediator)
      .filter((source) => !source.includes("*"));
    for (const antecedent of antecedents) {
      const triple = [antecedent, mediator, target];
      const hoc = triple.filter((c) => isHocConstruct(c, model));
      if (hoc.length > 0) {
        console.warn(
          `Skipping ${antecedent} -> ${mediator} -> ${target}: higher-order ` +
            `construct(s) ${hoc.join(", ")} cannot be used in PCM sub-models.`,
        );
        continue;
      }
      found.push({ antecedent, mediator, target });
    }
  }
  return found;
}

/**
 * Estimate an isolated partial-mediation sub-model for one triple.
 *
 * Rebuilds the measurement model of the three constructs from the parent
 * (preserving each construct's mode), with structural paths `X -> M`,
 * `{X, M} -> Y`, then re-estimates on the parent's raw data.
 */
export function buildIsolatedSubModel(model: PlsModel, path: PcmPath): PlsModel {
  const specs = [path.antecedent, path.mediator, path.target].map((name) => {
    const items = itemsOfConstruct(name, model);
    const mode = model.mmMatrix.constructMode(name);
    if (mode === "C") return reflective(name, items);
    if (mode === "B") return composite(name, items, modeB);
    return composite(name, items);
  });
  const measurementModel = constructs(...specs);
  const structuralModel = relationships(
    paths(path.antecedent, path.mediator),
    paths([path.antecedent, path.mediator], path.target),
  );
  return estimatePls(model.rawdata, measurementModel, structuralModel);
}

/** Per-indicator RMSE and MAE over out-of-sample residual columns. */
function indicatorMetrics(residuals: NamedMatrix, items: string[]): { rmse: number[]; mae: number[] } {
  const rmse: number[] = [];
  const mae: number[] = [];
  for (const item of items) {
    const j = residuals.cols.indexOf(item);
    const column = residuals.values.map((row) => row[j]!);
    const n = column.length;
    let sumSq = 0;
    let sumAbs = 0;
    for (const e of column) {
      sumSq += e * e;
      sumAbs += Math.abs(e);
    }
    rmse.push(Math.sqrt(sumSq / n));
    mae.push(sumAbs / n);
  }
  return { rmse, mae };
}

/** Build the isolated sub-model and compute PCM per target indicator. */
function computePcmForPath(
  model: PlsModel,
  path: PcmPath,
  noFolds: number,
  orderingDa: readonly number[] | undefined,
  orderingEa: readonly number[] | undefined,
  seedDa: number | undefined,
  seedEa: number | undefined,
): PcmPathResult {
  const subModel = buildIsolatedSubModel(model, path);
  const predDa = predictPls(subModel, {
    technique: predictDA,
    noFolds,
    ordering: orderingDa,
    seed: seedDa,
  });
  const predEa = predictPls(subModel, {
    technique: predictEA,
    noFolds,
    ordering: orderingEa,
    seed: seedEa,
  });

  const targetItems = itemsOfConstruct(path.target, subModel);
  const da = indicatorMetrics(predDa.items.plsOutOfSampleResiduals, targetItems);
  const ea = indicatorMetrics(predEa.items.plsOutOfSampleResiduals, targetItems);

  // PCM = (METRIC_EA - METRIC_DA) / METRIC_EA (Danks 2021, eq. 2).
  const pcmRmse = targetItems.map((_, k) => (ea.rmse[k]! - da.rmse[k]!) / ea.rmse[k]!);
  const pcmMae = targetItems.map((_, k) => (ea.mae[k]! - da.mae[k]!) / ea.mae[k]!);

  const values = targetItems.map((_, k) => [
    da.rmse[k]!,
    ea.rmse[k]!,
    pcmRmse[k]!,
    da.mae[k]!,
    ea.mae[k]!,
    pcmMae[k]!,
  ]);
  return Object.freeze({
    antecedent: path.antecedent,
    mediator: path.mediator,
    target: path.target,
    results: namedMatrix(targetItems, RESULT_COLS, values),
    pcmRmse,
    pcmMae,
  });
}

/** `value.toFixed(decimals)` right-padded to `width` (Python `{v:width.df}`). */
function fmtNum(value: number, width: number, decimals: number): string {
  return value.toFixed(decimals).padStart(width);
}

function renderPrint(analysis: PcmAnalysis): string {
  const lines = [
    "Predictive Contribution of the Mediator (PCM)",
    "=".repeat(46),
    `Target: ${analysis.target}`,
    `Cross-validation: ${analysis.noFolds} folds, ${analysis.reps} reps`,
    `Mediation paths: ${analysis.mediationPaths.length}`,
    "",
  ];
  for (const res of analysis.pcmResults) {
    // R `mean()` (`mean.default`, two-pass) — feature_pcm.R:361,364.
    const avgRmse = rMean(res.pcmRmse);
    const avgMae = rMean(res.pcmMae);
    lines.push(`  ${res.antecedent} -> ${res.mediator} -> ${res.target}`);
    lines.push(`    Avg PCM (RMSE): ${fmtNum(avgRmse, 7, 4)}  [${classifyPcm(avgRmse)}]`);
    lines.push(`    Avg PCM (MAE):  ${fmtNum(avgMae, 7, 4)}  [${classifyPcm(avgMae)}]`);
    lines.push("");
  }
  return lines.join("\n");
}

function renderSummary(analysis: PcmAnalysis): string {
  const lines = [
    "Predictive Contribution of the Mediator (PCM)",
    "=".repeat(46),
    `Target: ${analysis.target}`,
    `Cross-validation: ${analysis.noFolds} folds, ${analysis.reps} reps`,
    "",
  ];
  const header =
    `  ${"Indicator".padEnd(12)} ${"RMSE_DA".padStart(8)} ${"RMSE_EA".padStart(8)} ` +
    `${"PCM_RMSE".padStart(9)} ${"MAE_DA".padStart(8)} ${"MAE_EA".padStart(8)} ` +
    `${"PCM_MAE".padStart(9)}  Conclusion`;
  for (const res of analysis.pcmResults) {
    lines.push(`Mediation: ${res.antecedent} -> ${res.mediator} -> ${res.target}`);
    lines.push("-".repeat(60));
    lines.push(header);
    for (let i = 0; i < res.results.rows.length; i++) {
      const v = res.results.values[i]!;
      lines.push(
        `  ${res.results.rows[i]!.padEnd(12)} ${fmtNum(v[0]!, 8, 4)} ${fmtNum(v[1]!, 8, 4)} ` +
          `${fmtNum(v[2]!, 9, 4)} ${fmtNum(v[3]!, 8, 4)} ${fmtNum(v[4]!, 8, 4)} ` +
          `${fmtNum(v[5]!, 9, 4)}  ${classifyPcm(v[2]!)}`,
      );
    }
    lines.push("");
  }
  lines.push("PCM thresholds: < 0 Negative | 0-0.05 Weak | 0.05-0.10 Moderate | > 0.10 Strong");
  lines.push("Reference: Danks (2021), The DATA BASE for Advances in IS, 52(SI), 24-42.");
  return lines.join("\n");
}

/**
 * Assess the predictive contribution of mediators to a target construct.
 *
 * Returns null (with a warning) for non-seminr or higher-order models,
 * mirroring R. `target` auto-detects the single final endogenous construct when
 * omitted. `seed` and `orderings` are TS-only conveniences for reproducibility /
 * exact R fold-order parity (see module docstring); `reps` is accepted for R API
 * compatibility but does not affect prediction. Callable as
 * `assessPcm(model, options?)` or `assessPcm({ model, ...options })`.
 */
export function assessPcm(args: AssessPcmArgs): PcmAnalysis | null;
export function assessPcm(model: unknown, options?: PcmFeatureOptions): PcmAnalysis | null;
export function assessPcm(
  modelOrArgs: unknown,
  positionalOptions: PcmFeatureOptions = {},
): PcmAnalysis | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessPcmArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessPcmArgs) : positionalOptions;

  if (!validateSeminrModel(seminrModel, "assessPcm")) return null;
  if (!validateForPrediction(seminrModel, "assessPcm")) return null;
  const model = seminrModel;

  const noFolds = validatePositiveInteger(options.noFolds ?? 10, 2, "noFolds");
  const reps = validatePositiveInteger(options.reps ?? 10, 1, "reps");

  let target = options.target;
  if (target === undefined) target = detectFinalEndogenous(model);
  if (!model.pathCoef.cols.includes(target)) {
    throw new Error(`Target '${target}' not found in model constructs.`);
  }

  const mediationPaths = findMediationPaths(model, target);
  if (mediationPaths.length === 0) {
    throw new Error(
      `No mediation paths found for target '${target}'. The target must have at ` +
        "least one predictor that is itself endogenous.",
    );
  }

  const { orderings } = options;
  if (orderings !== undefined && orderings.length !== 2 * mediationPaths.length) {
    throw new Error(
      `orderings must supply one permutation per predict call ` +
        `(${2 * mediationPaths.length} for ${mediationPaths.length} paths), got ${orderings.length}.`,
    );
  }

  // Default path derives one child seed per predict call so DA/EA folds differ
  // yet a given `seed` reproduces exactly (non-bit-identical to R, plan Q4).
  const childSeeds =
    orderings !== undefined
      ? undefined
      : mulberry32(options.seed ?? Math.floor(Math.random() * 0x100000000));
  const nextSeed = () => Math.floor(childSeeds!() * 0x100000000);

  const pcmResults: PcmPathResult[] = [];
  for (let i = 0; i < mediationPaths.length; i++) {
    const path = mediationPaths[i]!;
    const orderingDa = orderings ? orderings[2 * i] : undefined;
    const orderingEa = orderings ? orderings[2 * i + 1] : undefined;
    const seedDa = orderings ? undefined : nextSeed();
    const seedEa = orderings ? undefined : nextSeed();
    pcmResults.push(
      computePcmForPath(model, path, noFolds, orderingDa, orderingEa, seedDa, seedEa),
    );
  }

  const analysis: PcmAnalysis = Object.freeze({
    kind: "pcm_analysis" as const,
    target,
    mediationPaths: Object.freeze(mediationPaths),
    pcmResults: Object.freeze(pcmResults),
    noFolds,
    reps,
    toString(): string {
      return renderPrint(this);
    },
    summarize(): string {
      return renderSummary(this);
    },
  });
  return analysis;
}
