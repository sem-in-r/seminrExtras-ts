/**
 * Bootstrap congruence coefficient testing, ported from feature_congruence.R
 * (via the Python port's congruence.py).
 *
 * The congruence coefficient `rc` (Franke, Sarstedt & Danks 2021) measures how
 * similarly two constructs relate to the rest of a PLS-SEM model's nomological
 * network — a cosine similarity over correlation patterns, with composite
 * reliability (rhoC) on the diagonal. `congruenceTest` bootstraps `rc` for every
 * construct pair and tests `H0: rc < threshold`.
 *
 * Parity notes (py plan F11): resampling matches R's `rerun` re-estimation to
 * machine precision, so `draws` injects the R index streams for exact fixture
 * parity; the default path uses `mulberry32(seed)`, drawing one resample vector
 * per iteration. Correlation via `@compstats/core`'s one-matrix `cor(x)` (R's
 * `stats::cor(construct_scores)`, feature_congruence.R:128,151), R-ddof SD via
 * R's `sd`, type-7 quantile CIs via `confInt` — the first two from
 * `@compstats/core`, the last from the shared kernel.
 */

import {
  mulberry32,
  namedMatrix,
  rerun,
  rhoCAve,
  constructsInModel,
  type Dataset,
  type NamedMatrix,
  type PlsModel,
} from "@seminr/core";
import { cor, fromRows, toRows } from "@compstats/core/linalg";
import { sd } from "@compstats/core/stats";
import { confInt, isNamedArgs, validateSeminrModel } from "./helpers.ts";
import { ciColumnLabels, formatTable } from "./records.ts";

const HEADING = "Congruence coefficient test (Franke, Sarstedt & Danks, 2021)";

/** Bootstrap congruence-coefficient test over all construct pairs (R table_output). */
export interface CongruenceTest {
  readonly kind: "congruence_test";
  /**
   * One row per construct pair (`"X  ->  Y"`), columns `Original Est.`, `Diff`,
   * `Bootstrap SD`, `T Stat.` and the two percentile-CI bounds.
   */
  readonly results: NamedMatrix;
  toString(): string;
  summarize(): string;
}

/** Options for {@link congruenceTest}. */
export interface CongruenceOptions {
  nboot?: number;
  /** Seeds the default bootstrap RNG (default 123). */
  seed?: number;
  alpha?: number;
  threshold?: number;
  /**
   * Injected bootstrap row-index stream, shape `(nboot, n)`, 0-based — one
   * row-resample vector per iteration for exact R parity. Mirrors py's `draws`.
   */
  draws?: number[][];
}

/** Named-args form of {@link congruenceTest}. */
export interface CongruenceTestArgs extends CongruenceOptions {
  model: unknown;
}

/** rhoC (composite reliability) per construct, aligned to `names`. */
function rhoDiagonal(model: PlsModel, names: string[]): number[] {
  // rhoC_AVE may carry extra rows (e.g. lower-order components of a HOC), so
  // index by construct name rather than position (R: `rhoC_AVE(x)[names, 1]`).
  const rca = rhoCAve(model.outerLoadings, constructsInModel(model).names);
  return names.map((c) => rca.values[rca.rows.indexOf(c)]![0]!);
}

/** Construct-score correlation matrix with rhoC written onto the diagonal. */
function correlationWithRho(model: PlsModel): { names: string[]; mat: number[][] } {
  const scores = model.constructScores;
  const names = [...scores.cols];
  const mat = toRows(cor(fromRows(scores.values)));
  const diag = rhoDiagonal(model, names);
  for (let i = 0; i < names.length; i++) mat[i]![i] = diag[i]!;
  return { names, mat };
}

/** Congruence coefficient of columns `x` and `y` (Franke et al. 2021, Eq. 2). */
function congruence(mat: number[][], x: number, y: number): number {
  const dot = (i: number, j: number) => {
    let s = 0;
    for (const row of mat) s += row[i]! * row[j]!;
    return s;
  };
  return dot(x, y) / Math.sqrt(dot(x, x) * dot(y, y));
}

/**
 * Bootstrap significance test of congruence coefficients for all construct pairs.
 *
 * Returns null (with a warning) for non-seminr input; higher-order models are
 * supported (no prediction guard). Callable as `congruenceTest(model, options?)`
 * or `congruenceTest({ model, ...options })`. `draws` injects the R bootstrap
 * row-index stream (`(nboot, n)`, 0-based) for exact fixture parity; otherwise
 * `seed` seeds `mulberry32`, drawing one resample vector per iteration.
 */
export function congruenceTest(args: CongruenceTestArgs): CongruenceTest | null;
export function congruenceTest(model: unknown, options?: CongruenceOptions): CongruenceTest | null;
export function congruenceTest(
  modelOrArgs: unknown,
  positionalOptions: CongruenceOptions = {},
): CongruenceTest | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as CongruenceTestArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as CongruenceTestArgs) : positionalOptions;

  if (!validateSeminrModel(seminrModel, "congruenceTest")) return null;
  const model = seminrModel;
  const nboot = options.nboot ?? 2000;
  const seed = options.seed ?? 123;
  const alpha = options.alpha ?? 0.05;
  const threshold = options.threshold ?? 1;

  const { names, mat: origMat } = correlationWithRho(model);
  const k = names.length;
  const pairs: [number, number][] = [];
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) pairs.push([i, j]);

  const n = model.rawdata.values.length;
  let indexStream: number[][];
  if (options.draws) {
    indexStream = options.draws;
  } else {
    const gen = mulberry32(seed);
    indexStream = [];
    for (let b = 0; b < nboot; b++) {
      indexStream.push(Array.from({ length: n }, () => Math.floor(gen() * n)));
    }
  }

  // boot[i][j][b] — filled on the upper-triangle pairs only.
  const boot: number[][][] = Array.from({ length: k }, () =>
    Array.from({ length: k }, () => new Array<number>(nboot).fill(NaN)),
  );
  for (let b = 0; b < nboot; b++) {
    const rows = indexStream[b]!;
    const resampled: Dataset = {
      columns: model.rawdata.columns,
      values: rows.map((r) => model.rawdata.values[r]!),
    };
    const itModel = rerun(model, { data: resampled });
    const { mat: bootMat } = correlationWithRho(itModel);
    for (const [i, j] of pairs) boot[i]![j]![b] = congruence(bootMat, i, j);
  }

  const rowLabels: string[] = [];
  const values: number[][] = [];
  for (const [i, j] of pairs) {
    rowLabels.push(`${names[i]}  ->  ${names[j]}`);
    const original = congruence(origMat, i, j);
    const diff = threshold - Math.abs(original);
    const bootSd = sd(boot[i]![j]!);
    const tStat = bootSd < Number.EPSILON ? NaN : diff / bootSd;
    const [lower, upper] = confInt(boot, names, names, names[i]!, names[j]!, null, alpha);
    values.push([original, diff, bootSd, tStat, lower, upper]);
  }

  const [ciLower, ciUpper] = ciColumnLabels(alpha);
  const results = namedMatrix(
    rowLabels,
    ["Original Est.", "Diff", "Bootstrap SD", "T Stat.", ciLower, ciUpper],
    values,
  );
  return Object.freeze({
    kind: "congruence_test" as const,
    results,
    toString(): string {
      return `${HEADING}\n\n${formatTable(results)}`;
    },
    summarize(): string {
      return this.toString();
    },
  });
}
