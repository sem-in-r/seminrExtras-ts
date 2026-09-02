/**
 * Shared helper kernel ported from seminrExtras' `helpers.R` (via the Python
 * port's `helpers.py`).
 *
 * Validation, endogenous-construct extraction, CVPAT loss calculation, and the
 * CVPAT bootstrap significance test. Numeric semantics follow R exactly (plan
 * F8): R's `mean` and `sd` from `@compstats/core`, plain sums via {@link seqSum}, and
 * type-7 quantiles via `@compstats/core`'s `quantile`.
 *
 * Stochastic entry points accept injectable `draws` (the exact resampling index
 * streams) for bit-level parity with R fixtures; the default path uses
 * `@seminr/core`'s `mulberry32` via `rng` and is statistically equivalent but
 * not bit-identical to R (plan Q4/RNG contract).
 */

import { mulberry32, type Dataset, type PlsModel } from "@seminr/core";
import { mean, pt, quantile, sd } from "@compstats/core/stats";

// =============================================================================
// Validation helpers
// =============================================================================

/**
 * Detect the named-args call style `fn({ model, ...options })` vs positional
 * `fn(model, options)`: an estimated model never carries a `model` key but
 * always carries `pathCoef`. Shared by every dual-signature entry point.
 */
export function isNamedArgs(x: unknown): x is { model: unknown } {
  return typeof x === "object" && x !== null && "model" in x && !("pathCoef" in x);
}

/**
 * Warn and return false unless `model` looks like an estimated seminr PLS model.
 *
 * `@seminr/core`'s `PlsModel` is an interface (not a class), so this duck-types
 * on the accessor fields every estimated model carries.
 */
export function validateSeminrModel(model: unknown, funcName = "This function"): model is PlsModel {
  const m = model as Partial<PlsModel> | null | undefined;
  const ok =
    m != null &&
    typeof m === "object" &&
    m.mmMatrix != null &&
    m.smMatrix != null &&
    m.pathCoef != null;
  if (!ok) {
    console.warn(`${funcName} only works with SEMinR models.`);
    return false;
  }
  return true;
}

/** Whether the model contains higher-order constructs (R: `!is.null(model$hoc)`). */
export function hasHigherOrder(model: PlsModel): boolean {
  return Boolean(model.hoc);
}

/** Validate a model for PLSpredict-based features (rejects HOC models). */
export function validateForPrediction(model: unknown, funcName = "This function"): model is PlsModel {
  if (!validateSeminrModel(model, funcName)) return false;
  if (hasHigherOrder(model)) {
    console.warn("There is no published solution for applying PLSpredict to higher-order models.");
    return false;
  }
  return true;
}

/** Validate an integer-valued scalar >= `minimum` (shared R-style guard). */
export function validatePositiveInteger(value: unknown, minimum: number, label: string): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < minimum ||
    !Number.isInteger(value)
  ) {
    throw new Error(`${label} must be an integer >= ${minimum}.`);
  }
  return value;
}

// =============================================================================
// Endogenous construct helpers
// =============================================================================

/** Measurement indicator names of a construct, in measurement-model order. */
export function itemsOfConstruct(construct: string, model: PlsModel): string[] {
  return model.mmMatrix.constructItems(construct);
}

/** Unique structural-model targets, in first-appearance order. */
export function getEndogenousConstructs(model: PlsModel): string[] {
  return model.smMatrix.allEndogenous();
}

/**
 * Indicators of the (given or endogenous) constructs, interactions excluded.
 *
 * Interaction constructs (name contains `*`) have no measurement indicators.
 */
export function getEndogenousItems(model: PlsModel, constructs?: string[]): string[] {
  const targets = constructs ?? getEndogenousConstructs(model);
  const items: string[] = [];
  for (const construct of targets) {
    if (construct.includes("*")) continue;
    items.push(...itemsOfConstruct(construct, model));
  }
  return items;
}

// =============================================================================
// Loss calculation helpers
// =============================================================================

/** Per-observation loss of a construct: mean squared error over its items. */
export function lvLoss(construct: string, model: PlsModel, error: Dataset): number[] {
  const items = itemsOfConstruct(construct, model);
  const idx = items.map((item) => error.columns.indexOf(item));
  return error.values.map((row) => {
    let sum = 0;
    for (const j of idx) {
      const e = row[j]!;
      sum += e * e;
    }
    return sum / idx.length;
  });
}

/** Row means of a loss matrix (a 1-D loss vector passes through unchanged). */
export function overallLoss(error: Dataset | number[][] | number[]): number[] {
  const values: number[][] | number[] = Array.isArray(error) ? error : error.values;
  if (values.length > 0 && Array.isArray(values[0])) {
    return (values as number[][]).map((row) => {
      let sum = 0;
      for (const v of row) sum += v;
      return sum / row.length;
    });
  }
  return values as number[];
}

/** Column-bind per-construct losses; columns are the construct names. */
export function calculateLvLosses(
  constructs: string[],
  model: PlsModel,
  errorMatrix: Dataset,
): Dataset {
  const columns = [...constructs];
  const perConstruct = columns.map((c) => lvLoss(c, model, errorMatrix));
  const n = perConstruct.length > 0 ? perConstruct[0]!.length : 0;
  const values: number[][] = Array.from({ length: n }, (_, i) => perConstruct.map((col) => col[i]!));
  return { columns, values };
}

// =============================================================================
// Numeric core — R-faithful accumulation (plan F8)
// =============================================================================

/**
 * Strict left-to-right float64 summation (R's LDOUBLE loop on arm64).
 *
 * A plain accumulating loop — NOT pairwise/`reduce` tricks — is what keeps the
 * strict-inequality bootstrap counts bit-compatible with the R goldens.
 */
export function seqSum(x: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i]!;
  return s;
}

/**
 * R's `mean.default` and `var`/`sd`, delegated to `@compstats/core`.
 *
 * Both were implemented here through v0.1.1 and both are gone: upstream's
 * `mean` is R's `do_mean` including the correcting second pass, and its `sd`
 * centres on that same mean, as `cov.c`'s `MEAN` macro does. The conformance
 * fixture `tests/fixtures/helpers/arith.json` pins both against R as exact
 * doubles, so the delegation is verified rather than assumed.
 *
 * `seqSum` stays. It is not a mean and has callers that want a plain
 * left-to-right sum in its own right — `cart.ts`'s node statistics and
 * `featurePos.ts`'s residual sums — and R's third mean, `colMeans`
 * (`array.c` `do_colsum`), is a single uncorrected pass built on exactly this.
 * Do not route a `colMeans` site through `mean`.
 */

function pairedTStat(diff: readonly number[], mu = 0): number {
  const n = diff.length;
  return (mean(diff) - mu) / (sd(diff) / Math.sqrt(n));
}

// =============================================================================
// Bootstrap CVPAT
// =============================================================================

/**
 * The two per-iteration resampling index streams of {@link bootstrapCvpat}
 * (paired-rows resample, d_null resample), each of shape `(nboot, n)`, 0-based.
 * Mirrors the exact RNG call order of the R implementation.
 */
export interface CvpatDraws {
  pairIndices: number[][];
  dnullIndices: number[][];
}

/** Options for the stochastic CVPAT entry points. */
export interface CvpatOptions {
  /** Injected 0-based resampling streams for exact R parity. */
  draws?: CvpatDraws;
  /** Default-path RNG: a number seeds `mulberry32`, a function is used directly. */
  rng?: (() => number) | number;
}

/**
 * One CVPAT bootstrap comparison (R `bootstrap_cvpat` result row).
 *
 * Fields map to the R columns: `Std. T value`, `Std. P value` (a percentile-t
 * p-value, not the classic t-test p), `Boot T value`, `Boot P Value`,
 * `Perc. P Value`.
 */
export interface CvpatBoot {
  readonly kind: "cvpat_boot";
  readonly stdTValue: number;
  readonly stdPValue: number;
  readonly bootTValue: number;
  readonly bootPValue: number;
  readonly percPValue: number;
}

function makeCvpatBoot(
  stdTValue: number,
  stdPValue: number,
  bootTValue: number,
  bootPValue: number,
  percPValue: number,
): CvpatBoot {
  return Object.freeze({
    kind: "cvpat_boot",
    stdTValue,
    stdPValue,
    bootTValue,
    bootPValue,
    percPValue,
  });
}

function resolveRng(rng: CvpatOptions["rng"]): () => number {
  if (typeof rng === "function") return rng;
  const seed = typeof rng === "number" ? rng : Math.floor(Math.random() * 0x100000000);
  return mulberry32(seed);
}

/** R's rank-based one-sided percentile p: 0 when no bootstrap stat exceeds. */
export function greaterPercentileP(bootStats: readonly number[], original: number, nboot: number): number {
  let countLe = 0;
  for (const v of bootStats) if (v <= original) countLe++;
  if (countLe === nboot) return 0;
  return 1 - countLe / (nboot + 1);
}

/**
 * Bootstrap significance test of the average loss difference (helpers.R:157).
 *
 * `lossM2 - lossM1` is the tested difference (m2 = benchmark/alternative loss,
 * m1 = established model's loss). `opts.draws` injects the two resampling index
 * streams for exact R parity; otherwise `opts.rng` seeds the default generator,
 * drawing in the same two-per-iteration order (pair row then d_null row).
 */
export function bootstrapCvpat(
  lossM1: readonly number[],
  lossM2: readonly number[],
  testtype = "two.sided",
  nboot = 2000,
  opts: CvpatOptions = {},
): CvpatBoot {
  if (testtype !== "two.sided" && testtype !== "greater") {
    // R only implements these two branches ("less" would hit undefined
    // variables); fail loudly instead.
    throw new Error(`testtype must be 'two.sided' or 'greater', got '${testtype}'`);
  }

  const n = lossM1.length;
  const d = lossM1.map((v, i) => lossM2[i]! - v);
  const orgTTest = pairedTStat(d);
  const orgDBar = mean(d);
  const dNull = d.map((v) => v - orgDBar);

  let pairIdx: number[][];
  let dnullIdx: number[][];
  if (opts.draws) {
    pairIdx = opts.draws.pairIndices;
    dnullIdx = opts.draws.dnullIndices;
  } else {
    const gen = resolveRng(opts.rng);
    pairIdx = [];
    dnullIdx = [];
    for (let b = 0; b < nboot; b++) {
      // Two draws per iteration, mirroring R's RNG order.
      pairIdx.push(Array.from({ length: n }, () => Math.floor(gen() * n)));
      dnullIdx.push(Array.from({ length: n }, () => Math.floor(gen() * n)));
    }
  }

  const tStat: number[] = new Array(nboot);
  const bootDBar: number[] = new Array(nboot);
  for (let b = 0; b < nboot; b++) {
    const bootDiff = pairIdx[b]!.map((i) => d[i]!);
    tStat[b] = pairedTStat(bootDiff, orgDBar);
    bootDBar[b] = mean(dnullIdx[b]!.map((i) => dNull[i]!));
  }

  const std = sd(bootDBar);
  let tStatBootVar: number;
  if (Number.isNaN(std) || std < Number.EPSILON) {
    console.warn("Bootstrap variance near zero; t-statistic set to NA");
    tStatBootVar = NaN;
  } else {
    tStatBootVar = orgDBar / std;
  }

  const absT = Math.abs(orgTTest);
  const absD = Math.abs(orgDBar);
  const countGt = (arr: readonly number[], thresh: number) =>
    arr.reduce((acc, v) => acc + (v > thresh ? 1 : 0), 0);
  const countLe = (arr: readonly number[], thresh: number) =>
    arr.reduce((acc, v) => acc + (v <= thresh ? 1 : 0), 0);

  let pPercT: number;
  let pPercD: number;
  let pVarT: number;
  if (testtype === "two.sided") {
    pPercT = (countGt(tStat, absT) + countLe(tStat, -absT)) / nboot;
    pPercD = (countGt(bootDBar, absD) + countLe(bootDBar, -absD)) / nboot;
    // R: `2 * pt(-abs(t), n - 1, lower.tail = TRUE)` (helpers.R:202).
    pVarT = Number.isNaN(tStatBootVar) ? NaN : 2 * pt(-Math.abs(tStatBootVar), n - 1);
  } else {
    pPercT = greaterPercentileP(tStat, orgTTest, nboot);
    pPercD = greaterPercentileP(bootDBar, orgDBar, nboot);
    // R: `pt(t, n - 1, lower.tail = FALSE)` (helpers.R:220) -- the upper tail as
    // a real argument, not one minus the lower. The difference is the far tail:
    // at df 249 and t = 10 the subtraction gives exactly 0 where this gives
    // 2.6e-20, so a strongly one-sided CVPAT would have printed p = 0.
    pVarT = Number.isNaN(tStatBootVar)
      ? NaN
      : pt(tStatBootVar, n - 1, undefined, { lowerTail: false });
  }

  return makeCvpatBoot(orgTTest, pPercT, tStatBootVar, pVarT, pPercD);
}

/**
 * Run {@link bootstrapCvpat} per loss-matrix column (helpers.R:236).
 *
 * Constructs are processed in column order over ONE continuing random stream,
 * as in R. Injected `draws` arrays therefore carry `nboot * k` rows (the
 * per-construct blocks in sequence).
 */
export function cvpatPerConstruct(
  lossOne: Dataset,
  lossTwo: Dataset,
  testtype = "two.sided",
  nboot = 2000,
  opts: CvpatOptions = {},
): Record<string, CvpatBoot> {
  const constructs = lossOne.columns;
  const gen = opts.draws ? undefined : resolveRng(opts.rng);
  const results: Record<string, CvpatBoot> = {};
  for (let i = 0; i < constructs.length; i++) {
    const construct = constructs[i]!;
    const col1 = lossOne.columns.indexOf(construct);
    const col2 = lossTwo.columns.indexOf(construct);
    const m1 = lossOne.values.map((row) => row[col1]!);
    const m2 = lossTwo.values.map((row) => row[col2]!);
    const blockOpts: CvpatOptions = opts.draws
      ? {
          draws: {
            pairIndices: opts.draws.pairIndices.slice(i * nboot, (i + 1) * nboot),
            dnullIndices: opts.draws.dnullIndices.slice(i * nboot, (i + 1) * nboot),
          },
        }
      : { rng: gen };
    results[construct] = bootstrapCvpat(m1, m2, testtype, nboot, blockOpts);
  }
  return results;
}

// =============================================================================
// Bootstrap confidence intervals
// =============================================================================

/**
 * Percentile CI of a (possibly indirect) path from a 3-D bootstrap array.
 *
 * `bootValues` is indexed `[rowIdx][colIdx][b]`. The indirect path is the
 * elementwise product `[from, through, :] * [through, to, :]`. Quantiles are
 * R type-7 (via `@compstats/core`'s `quantile`).
 */
export function confInt(
  bootValues: number[][][],
  rows: readonly string[],
  cols: readonly string[],
  from: string,
  to: string,
  through: string | null = null,
  alpha = 0.05,
): [number, number] {
  let coefficient: number[];
  if (through === null) {
    coefficient = bootValues[rows.indexOf(from)]![cols.indexOf(to)]!;
  } else {
    const a = bootValues[rows.indexOf(from)]![cols.indexOf(through)]!;
    const b = bootValues[rows.indexOf(through)]![cols.indexOf(to)]!;
    coefficient = a.map((v, i) => v * b[i]!);
  }
  return [quantile(coefficient, alpha / 2), quantile(coefficient, 1 - alpha / 2)];
}
