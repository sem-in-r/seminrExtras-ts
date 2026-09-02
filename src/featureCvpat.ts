/**
 * Cross-Validated Predictive Ability Test (CVPAT), ported from feature_cvpat.R
 * (via the Python port's cvpat.py).
 *
 * CVPAT compares the out-of-sample predictive loss of PLS models against
 * benchmarks (LM, IA) or against each other, with bootstrap significance tests
 * (Liengaard et al. 2021; Sharma et al. 2023).
 *
 * Parity notes (py plan F9): R's `assess_cvpat` truncates its LOSS columns to 7
 * significant digits (an `as.matrix`-on-data.frame artifact); this port keeps
 * full precision — a documented divergence. `draws` injects the flat bootstrap
 * index streams in R call order for exact fixture parity; the default path uses
 * `mulberry32(seed)`.
 */

import {
  mulberry32,
  namedMatrix,
  predictDA,
  predictPls,
  type Dataset,
  type NamedMatrix,
  type PlsModel,
} from "@seminr/core";
import {
  bootstrapCvpat,
  calculateLvLosses,
  cvpatPerConstruct,
  getEndogenousConstructs,
  getEndogenousItems,
  isNamedArgs,
  overallLoss,
  validateForPrediction,
  type CvpatBoot,
  type CvpatDraws,
} from "./helpers.ts";
import { mean } from "@compstats/core/stats";
import { formatTable } from "./records.ts";

const ASSESS_DESCRIPTION = "CVPAT as per Sharma et al. (2023).";
const COMPARE_DESCRIPTION =
  "CVPAT as per Sharma, Liengaard, Hair, Sarstedt, & Ringle, (2023).\n" +
  "  Both models under comparison have identical endogenous constructs with identical " +
  "measurement models.\n  Purely exogenous constructs can differ in regards to their " +
  "relationships with both nomological\n  partners and measurement indicators.";

/** Single-model CVPAT against the LM and IA benchmarks (R: list of two tables). */
export interface CvpatAssessment {
  readonly kind: "cvpat_assessment";
  readonly cvpatCompareLm: NamedMatrix;
  readonly cvpatCompareIa: NamedMatrix;
  readonly description: string;
  toString(): string;
  summarize(): string;
}

/** CVPAT loss comparison between two PLS models (R: table_output matrix). */
export interface CvpatComparison {
  readonly kind: "cvpat_comparison";
  readonly table: NamedMatrix;
  readonly description: string;
  toString(): string;
  summarize(): string;
}

/** The technique signature of `@seminr/core`'s predictDA/predictEA. */
type PredictTechnique = typeof predictDA;

/** Options shared by {@link assessCvpat} and {@link assessCvpatCompare}. */
export interface CvpatFeatureOptions {
  testtype?: string;
  nboot?: number;
  /** Seeds both the predict shuffle and the default bootstrap RNG (default 123). */
  seed?: number;
  technique?: PredictTechnique;
  /** Number of cross-validation folds; omit for LOOCV (R noFolds=NULL). */
  noFolds?: number;
  /**
   * Accepted for R API compatibility and ignored: R's reps re-runs
   * prediction_matrices on the SAME shuffled order and averages the identical
   * results — a numeric no-op (verified empirically in the py port).
   */
  reps?: number;
  /** Accepted for R API compatibility and ignored (predictPls is single-threaded). */
  cores?: number;
  /** Injected flat 0-based bootstrap index streams for exact R parity. */
  draws?: CvpatDraws;
}

/** Named-args form of {@link assessCvpat}. */
export interface AssessCvpatArgs extends CvpatFeatureOptions {
  model: unknown;
}

/** Named-args form of {@link assessCvpatCompare}. */
export interface AssessCvpatCompareArgs extends CvpatFeatureOptions {
  establishedModel: unknown;
  alternativeModel: unknown;
}

/** Slice consecutive nboot-row blocks off flat injected draw streams. */
class DrawCursor {
  private offset = 0;

  constructor(
    private readonly draws: CvpatDraws | undefined,
    private readonly nboot: number,
  ) {}

  take(blocks: number): CvpatDraws | undefined {
    if (!this.draws) return undefined;
    const rows = blocks * this.nboot;
    const block: CvpatDraws = {
      pairIndices: this.draws.pairIndices.slice(this.offset, this.offset + rows),
      dnullIndices: this.draws.dnullIndices.slice(this.offset, this.offset + rows),
    };
    this.offset += rows;
    return block;
  }
}

/** Re-select residual columns in endogenous-item order (R relabels positionally). */
function residualsDataset(residuals: NamedMatrix, endoMvs: string[]): Dataset {
  const idx = endoMvs.map((item) => residuals.cols.indexOf(item));
  return { columns: endoMvs, values: residuals.values.map((row) => idx.map((j) => row[j]!)) };
}

/**
 * R `colMeans` (`array.c` `do_colsum`): ONE pass, no correction.
 *
 * Deliberately not `@compstats/core`'s `mean`. R has three means and this is
 * the only site
 * in the package that wants the uncorrected one — `feature_cvpat.R:204` calls
 * `colMeans()`, where `:231` calls `mean()` a few lines later. Routing this
 * through `mean` would break parity in the opposite direction; the rule is
 * pinned by "R's colMeans is the uncorrected pass" in `tests/helpers.test.ts`.
 */
function colMeans(losses: Dataset): number[] {
  const n = losses.values.length;
  return losses.columns.map((_, j) => {
    let sum = 0;
    for (const row of losses.values) sum += row[j]!;
    return sum / n;
  });
}

/** Build the 5-column CVPAT table: losses, diff, Boot T, Boot P (+ Overall row). */
function withOverall(
  constructRows: string[],
  lossOneMeans: number[],
  lossTwoMeans: number[],
  perConstruct: Record<string, CvpatBoot>,
  overallOne: number,
  overallTwo: number,
  overallBoot: CvpatBoot,
  colNames: [string, string],
): NamedMatrix {
  const rows = [...constructRows, "Overall"];
  const values = constructRows.map((construct, i) => {
    const boot = perConstruct[construct]!;
    return [
      lossOneMeans[i]!,
      lossTwoMeans[i]!,
      lossOneMeans[i]! - lossTwoMeans[i]!,
      boot.bootTValue,
      boot.bootPValue,
    ];
  });
  values.push([
    overallOne,
    overallTwo,
    overallOne - overallTwo,
    overallBoot.bootTValue,
    overallBoot.bootPValue,
  ]);
  return namedMatrix(rows, [colNames[0], colNames[1], "Diff", "Boot T value", "Boot P Value"], values);
}

function checkReps(reps: number | undefined): void {
  if (reps !== undefined && (!Number.isInteger(reps) || reps < 1)) {
    throw new Error(`reps must be a positive integer or undefined, got ${reps}`);
  }
}

function selectColumns(losses: Dataset, columns: string[]): Dataset {
  const idx = columns.map((c) => losses.columns.indexOf(c));
  return { columns, values: losses.values.map((row) => idx.map((j) => row[j]!)) };
}

function predictErrors(
  model: PlsModel,
  endoMvs: string[],
  options: CvpatFeatureOptions,
  seed: number,
): Dataset {
  const prediction = predictPls(model, {
    technique: options.technique ?? predictDA,
    noFolds: options.noFolds,
    seed,
  });
  return residualsDataset(prediction.items.plsOutOfSampleResiduals, endoMvs);
}

/**
 * Assess one PLS model's predictive loss against the LM and IA benchmarks.
 *
 * Returns null (with a warning) for non-seminr or higher-order models,
 * mirroring R. Callable as `assessCvpat(model, options?)` or
 * `assessCvpat({ model, ...options })`.
 */
export function assessCvpat(args: AssessCvpatArgs): CvpatAssessment | null;
export function assessCvpat(model: unknown, options?: CvpatFeatureOptions): CvpatAssessment | null;
export function assessCvpat(
  modelOrArgs: unknown,
  positionalOptions: CvpatFeatureOptions = {},
): CvpatAssessment | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessCvpatArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessCvpatArgs) : positionalOptions;

  checkReps(options.reps);
  const testtype = options.testtype ?? "two.sided";
  const nboot = options.nboot ?? 2000;
  const seed = options.seed ?? 123;
  const gen = mulberry32(seed);
  if (!validateForPrediction(seminrModel, "assessCvpat")) return null;
  const model = seminrModel;

  const endoLvs = getEndogenousConstructs(model);
  const endoMvs = getEndogenousItems(model, endoLvs);

  // IA benchmark: item error against the training means (model.meanData).
  const dataIdx = endoMvs.map((item) => model.data.columns.indexOf(item));
  const iaMeans = endoMvs.map((item) => model.meanData[item]!);
  const iaError: Dataset = {
    columns: endoMvs,
    values: model.data.values.map((row) => dataIdx.map((j, k) => row[j]! - iaMeans[k]!)),
  };

  // seed also fixes predictPls' row shuffle (mathematically irrelevant under
  // LOOCV, but keeps same-seed runs bitwise reproducible under k-fold).
  const prediction = predictPls(model, {
    technique: options.technique ?? predictDA,
    noFolds: options.noFolds,
    seed,
  });
  const plsError = residualsDataset(prediction.items.plsOutOfSampleResiduals, endoMvs);
  const lmError = residualsDataset(prediction.items.lmOutOfSampleResiduals, endoMvs);

  const lvLossesIa = calculateLvLosses(endoLvs, model, iaError);
  const lvLossesLm = calculateLvLosses(endoLvs, model, lmError);
  const lvLossesPls = calculateLvLosses(endoLvs, model, plsError);
  const iaOverall = overallLoss(lvLossesIa);
  const lmOverall = overallLoss(lvLossesLm);
  const plsOverall = overallLoss(lvLossesPls);

  // Bootstrap calls in R order over ONE stream: IA overall, LM overall,
  // IA per-construct, LM per-construct.
  const cursor = new DrawCursor(options.draws, nboot);
  const k = endoLvs.length;
  const blockOpts = (blocks: number) => {
    const draws = cursor.take(blocks);
    return draws ? { draws } : { rng: gen };
  };
  const plsVIa = bootstrapCvpat(plsOverall, iaOverall, testtype, nboot, blockOpts(1));
  const plsVLm = bootstrapCvpat(plsOverall, lmOverall, testtype, nboot, blockOpts(1));
  const iaCvpat = cvpatPerConstruct(lvLossesPls, lvLossesIa, testtype, nboot, blockOpts(k));
  const lmCvpat = cvpatPerConstruct(lvLossesPls, lvLossesLm, testtype, nboot, blockOpts(k));

  // R `mean()` (`mean.default`, two-pass) for the Overall row, R `colMeans`
  // (one pass) for the per-construct rows — feature_cvpat.R:204 vs :231-233.
  const plsMeans = colMeans(lvLossesPls);
  const matLm = withOverall(
    endoLvs,
    plsMeans,
    colMeans(lvLossesLm),
    lmCvpat,
    mean(plsOverall),
    mean(lmOverall),
    plsVLm,
    ["PLS Loss", "LM Loss"],
  );
  const matIa = withOverall(
    endoLvs,
    plsMeans,
    colMeans(lvLossesIa),
    iaCvpat,
    mean(plsOverall),
    mean(iaOverall),
    plsVIa,
    ["PLS Loss", "IA Loss"],
  );
  return Object.freeze({
    kind: "cvpat_assessment" as const,
    cvpatCompareLm: matLm,
    cvpatCompareIa: matIa,
    description: ASSESS_DESCRIPTION,
    toString(): string {
      return (
        `${ASSESS_DESCRIPTION}\n\nCVPAT_compare_LM:\n${formatTable(matLm)}` +
        `\n\nCVPAT_compare_IA:\n${formatTable(matIa)}`
      );
    },
    summarize(): string {
      return this.toString();
    },
  });
}

/**
 * CVPAT significance test of loss between two competing PLS models.
 *
 * The models must share identical endogenous constructs and measurement items
 * (only paths from purely exogenous constructs may differ); otherwise throws,
 * mirroring R's stop(). Callable as
 * `assessCvpatCompare(established, alternative, options?)` or
 * `assessCvpatCompare({ establishedModel, alternativeModel, ...options })`.
 */
export function assessCvpatCompare(args: AssessCvpatCompareArgs): CvpatComparison | null;
export function assessCvpatCompare(
  establishedModel: unknown,
  alternativeModel: unknown,
  options?: CvpatFeatureOptions,
): CvpatComparison | null;
export function assessCvpatCompare(
  establishedOrArgs: unknown,
  positionalAlternative?: unknown,
  positionalOptions: CvpatFeatureOptions = {},
): CvpatComparison | null {
  const named =
    typeof establishedOrArgs === "object" &&
    establishedOrArgs !== null &&
    "establishedModel" in establishedOrArgs;
  const established = named
    ? (establishedOrArgs as AssessCvpatCompareArgs).establishedModel
    : establishedOrArgs;
  const alternative = named
    ? (establishedOrArgs as AssessCvpatCompareArgs).alternativeModel
    : positionalAlternative;
  const options = named ? (establishedOrArgs as AssessCvpatCompareArgs) : positionalOptions;

  checkReps(options.reps);
  // Validate BOTH models BEFORE seeding — asymmetric vs assessCvpat, R-faithful.
  if (!validateForPrediction(established, "assessCvpatCompare")) return null;
  if (!validateForPrediction(alternative, "assessCvpatCompare")) return null;
  const testtype = options.testtype ?? "two.sided";
  const nboot = options.nboot ?? 2000;
  const seed = options.seed ?? 123;
  const gen = mulberry32(seed);

  const endoLvs1 = getEndogenousConstructs(established);
  const endoLvs2 = getEndogenousConstructs(alternative);
  const endoMvs1 = getEndogenousItems(established, endoLvs1);
  const endoMvs2 = getEndogenousItems(alternative, endoLvs2);

  const sameSet = (a: string[], b: string[]) =>
    a.length === b.length && new Set(a).size === new Set([...a, ...b]).size;
  if (!(sameSet(endoMvs1, endoMvs2) && sameSet(endoLvs1, endoLvs2))) {
    throw new Error(
      "CVPAT can only be applied to models with identical endogenous constructs and measures",
    );
  }

  const lvLossesOne = calculateLvLosses(
    endoLvs1,
    established,
    predictErrors(established, endoMvs1, options, seed),
  );
  const lvLossesTwo = calculateLvLosses(
    endoLvs2,
    alternative,
    predictErrors(alternative, endoMvs2, options, seed),
  );
  const overallOne = overallLoss(lvLossesOne);
  const overallTwo = overallLoss(lvLossesTwo);

  const cursor = new DrawCursor(options.draws, nboot);
  const blockOpts = (blocks: number) => {
    const draws = cursor.take(blocks);
    return draws ? { draws } : { rng: gen };
  };
  const overallBoot = bootstrapCvpat(overallOne, overallTwo, testtype, nboot, blockOpts(1));

  let constructRows: string[];
  let lossOne: Dataset;
  let lossTwo: Dataset;
  if (endoLvs1.length === endoLvs2.length && endoLvs1.every((c, i) => c === endoLvs2[i])) {
    constructRows = endoLvs1;
    lossOne = lvLossesOne;
    lossTwo = lvLossesTwo;
  } else {
    // Sets are equal (checked above), so this fires only on ORDER mismatch and
    // the overlap is all constructs; R's "Cannot compare directly" branch
    // (empty overlap) is unreachable dead code.
    constructRows = endoLvs1.filter((c) => endoLvs2.includes(c));
    console.warn("Not all endogenous vars co-occur in models 1 and 2. Only comparing overlap.");
    lossOne = selectColumns(lvLossesOne, constructRows);
    lossTwo = selectColumns(lvLossesTwo, constructRows);
  }

  const perConstruct = cvpatPerConstruct(
    lossOne,
    lossTwo,
    testtype,
    nboot,
    blockOpts(constructRows.length),
  );
  const table = withOverall(
    constructRows,
    colMeans(lossOne),
    colMeans(lossTwo),
    perConstruct,
    mean(overallOne),
    mean(overallTwo),
    overallBoot,
    ["Base Model Loss", "Alt Model Loss"],
  );
  return Object.freeze({
    kind: "cvpat_comparison" as const,
    table,
    description: COMPARE_DESCRIPTION,
    toString(): string {
      return `${COMPARE_DESCRIPTION}\n\n${formatTable(table)}`;
    },
    summarize(): string {
      return this.toString();
    },
  });
}
