/**
 * Confirmatory Tetrad Analysis for PLS-SEM (CTA-PLS), ported from feature_cta.R
 * (via the Python port's cta.py).
 *
 * CTA-PLS (Gudergan et al. 2008; Cefis et al. 2025) tests whether a construct's
 * measurement model is consistent with a reflective (common factor)
 * specification: under a reflective model all non-redundant vanishing tetrads of
 * the indicator covariance matrix equal zero. Constructs with 2-3 indicators can
 * borrow indicators from structurally adjacent constructs (Gudergan et al. 2008,
 * Table 1); higher-order constructs are tested over their LOC scores (with model
 * re-estimation inside the bootstrap).
 *
 * Parity notes (plan F12): after seeding, R consumes exactly ONE
 * `sample(n, n, TRUE)` per bootstrap iteration (`rerun` is deterministic), so
 * `draws` injects the R index stream (shape `(nboot, n)`, 0-based) for exact
 * fixture parity; the default path uses `mulberry32(seed)`. `pAdjust` reproduces
 * R `p.adjust` semantics including the lazy-`n` NA behavior (n = count of non-NA
 * p-values). R's progress/skip `message()`s are not ported; mode labels
 * reproduce R verbatim (a non-"A" raw mode — including reflective "C" and HOC
 * codes — renders as "Mode B (formative)", an R quirk). Boot_SD uses the shared
 * R-faithful `rSd`; T = Estimate/Boot_SD (NaN when SD < eps).
 */

import {
  mulberry32,
  namedMatrix,
  rerun,
  type Dataset,
  type NamedMatrix,
  type PlsModel,
} from "@seminr/core";
import { quantile } from "@compstats/core/stats";
import { cov as csCov, toRows } from "@compstats/core/linalg";
import { asMatrix, isNamedArgs, rSd, validateSeminrModel } from "./helpers.ts";
import { ciColumnLabels } from "./records.ts";

const MIN_VALID_BOOTS = 10;
const DETAIL_COLS_HEAD = ["Estimate", "T_Value", "Boot_Mean", "Boot_SD"] as const;
const DETAIL_COLS_TAIL = ["P_Value", "Adj_P"] as const;

/** One tetrad over indicators (i, j, k, l); `tetradId` selects the form. */
export interface TetradSpec {
  readonly i: string;
  readonly j: string;
  readonly k: string;
  readonly l: string;
  readonly tetradId: number;
}

/** A construct's testable indicators and their data (R resolve_indicators). */
export interface ResolvedIndicators {
  readonly indicators: string[];
  readonly data: number[][];
  readonly isHoc: boolean;
  readonly needsReestimation: boolean;
}

/** Best borrowing donor for a 2-3 indicator focal construct (R find_donor). */
interface Donor {
  readonly construct: string;
  readonly mode: string;
  readonly donorIndicators: string[];
  readonly donorData: number[][];
  readonly isHoc: boolean;
  readonly needsReestimation: boolean;
  readonly borrowed: string[];
  readonly vanishingPattern: string;
  readonly nVanishing: number;
  readonly nDonorIndicators: number;
}

/** Borrowing details for one construct (donor and vanishing pattern). */
export interface CtaBorrowing {
  readonly donor: string;
  readonly donorMode: string;
  readonly vanishingPattern: string;
  readonly nVanishing: number;
}

/** One row of the CTA summary table (R construct_results). */
export interface CtaConstructResult {
  readonly construct: string;
  readonly mode: string;
  readonly indicators: number;
  readonly tetrads: number;
  readonly significant: number;
  readonly verdict: string;
}

/** Per-construct tetrad statistics: numeric table + significance flags. */
export interface CtaTetradDetails {
  readonly table: NamedMatrix;
  readonly significant: boolean[];
}

/** Result of {@link assessCta} (R class `cta_analysis`). */
export interface CtaAnalysis {
  readonly kind: "cta_analysis";
  readonly constructResults: readonly CtaConstructResult[];
  readonly tetradDetails: ReadonlyMap<string, CtaTetradDetails>;
  readonly nboot: number;
  readonly alpha: number;
  readonly correction: string;
  readonly skipped: readonly string[];
  readonly borrowing: ReadonlyMap<string, CtaBorrowing>;
  toString(): string;
  summarize(digits?: number): string;
}

// --- Tetrad machinery -------------------------------------------------------------

/** All 4-element combinations of `items`, in R's lexicographic-by-position order. */
function combinations<T>(items: readonly T[], k: number): T[][] {
  const out: T[][] = [];
  const combo: T[] = [];
  const recurse = (start: number) => {
    if (combo.length === k) {
      out.push([...combo]);
      return;
    }
    for (let idx = start; idx <= items.length - (k - combo.length); idx++) {
      combo.push(items[idx]!);
      recurse(idx + 1);
      combo.pop();
    }
  };
  recurse(0);
  return out;
}

/** Enumerate the 2 non-redundant vanishing tetrads per 4-indicator combination. */
export function enumerateTetrads(indicators: readonly string[]): TetradSpec[] {
  const specs: TetradSpec[] = [];
  for (const [i, j, k, l] of combinations(indicators, 4)) {
    specs.push({ i: i!, j: j!, k: k!, l: l!, tetradId: 1 });
    specs.push({ i: i!, j: j!, k: k!, l: l!, tetradId: 2 });
  }
  return specs;
}

/** Enumerate vanishing tetrads for the borrowing case (R Table 1 patterns). */
export function enumerateBorrowedTetrads(
  ownIndicators: readonly string[],
  borrowedIndicators: readonly string[],
  vanishingPattern: string,
): TetradSpec[] {
  if (vanishingPattern === "all") {
    return enumerateTetrads([...ownIndicators, ...borrowedIndicators]);
  }
  // tau_1342: i, j from own; k, l from borrowed.
  const specs: TetradSpec[] = [];
  for (const [i, j] of combinations(ownIndicators, 2)) {
    for (const [k, l] of combinations(borrowedIndicators, 2)) {
      specs.push({ i: i!, j: j!, k: k!, l: l!, tetradId: 3 });
    }
  }
  return specs;
}

/** Evaluate tetrad values off a covariance matrix with the given column names. */
export function computeTetrads(
  cov: readonly (readonly number[])[],
  names: readonly string[],
  tetrads: readonly TetradSpec[],
): number[] {
  const pos = new Map(names.map((name, idx) => [name, idx]));
  const values = new Array<number>(tetrads.length);
  for (let r = 0; r < tetrads.length; r++) {
    const t = tetrads[r]!;
    const i = pos.get(t.i)!;
    const j = pos.get(t.j)!;
    const k = pos.get(t.k)!;
    const l = pos.get(t.l)!;
    const s = (a: number, b: number) => cov[a]![b]!;
    if (t.tetradId === 1) {
      values[r] = s(i, j) * s(k, l) - s(i, k) * s(j, l);
    } else if (t.tetradId === 2) {
      values[r] = s(i, j) * s(k, l) - s(i, l) * s(j, k);
    } else {
      // tau_1342
      values[r] = s(i, k) * s(j, l) - s(i, l) * s(j, k);
    }
  }
  return values;
}

/** R format_tetrad_label: e.g. `s(i,j)s(k,l) - s(i,k)s(j,l)`. */
export function formatTetradLabel(t: TetradSpec): string {
  if (t.tetradId === 1) {
    return `s(${t.i},${t.j})s(${t.k},${t.l}) - s(${t.i},${t.k})s(${t.j},${t.l})`;
  }
  if (t.tetradId === 2) {
    return `s(${t.i},${t.j})s(${t.k},${t.l}) - s(${t.i},${t.l})s(${t.j},${t.k})`;
  }
  return `s(${t.i},${t.k})s(${t.j},${t.l}) - s(${t.i},${t.l})s(${t.j},${t.k})`;
}

/**
 * R `p.adjust` for methods BH / bonferroni / none, with R's NA semantics.
 *
 * R's lazy default `n = length(p)` is evaluated AFTER NA removal, so the
 * effective n is the count of non-NA p-values; NA entries stay NA.
 */
export function pAdjust(p: readonly number[], method: string): number[] {
  const out = p.slice();
  if (method === "none") return out;
  const validIdx: number[] = [];
  for (let i = 0; i < p.length; i++) if (!Number.isNaN(p[i]!)) validIdx.push(i);
  const n = validIdx.length;
  if (n === 0) return out;
  const pv = validIdx.map((i) => p[i]!);
  if (method === "bonferroni") {
    for (let k = 0; k < n; k++) out[validIdx[k]!] = Math.min(1, n * pv[k]!);
    return out;
  }
  // BH: cummin over descending p of (n / rank) * p, clipped at 1.
  const order = pv
    .map((v, idx) => [v, idx] as const)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1]) // stable descending
    .map(([, idx]) => idx);
  const adjusted = new Array<number>(n);
  let running = Infinity;
  for (let rank = 0; rank < n; rank++) {
    const src = order[rank]!;
    running = Math.min(running, (n / (n - rank)) * pv[src]!);
    adjusted[src] = Math.min(1, running);
  }
  for (let k = 0; k < n; k++) out[validIdx[k]!] = adjusted[k]!;
  return out;
}

// --- Model traversal ---------------------------------------------------------------

function selectColumns(dataset: Dataset, names: readonly string[]): number[][] {
  const idx = names.map((name) => dataset.columns.indexOf(name));
  return dataset.values.map((row) => idx.map((j) => row[j]!));
}

/** Resolve a construct's testable indicators, chaining HOC -> LOC scores. */
export function resolveIndicators(construct: string, model: PlsModel): ResolvedIndicators {
  const items = model.mmMatrix.constructItems(construct);
  const constructCols = model.outerWeights.cols;
  const locNames = items.filter((item) => constructCols.includes(item));
  if (locNames.length > 0 && locNames.length === items.length) {
    // HOC: LOC scores (columns of model.data) serve as "indicators".
    return {
      indicators: locNames,
      data: selectColumns(model.data, locNames),
      isHoc: true,
      needsReestimation: true,
    };
  }
  return {
    indicators: items,
    data: selectColumns(model.data, items),
    isHoc: false,
    needsReestimation: false,
  };
}

/** Constructs adjacent to `construct` in the structural model (targets, then sources). */
export function getStructurallyConnected(construct: string, model: PlsModel): string[] {
  const connected = [
    ...model.smMatrix.constructTargets(construct),
    ...model.smMatrix.constructAntecedents(construct),
  ];
  return [...new Set(connected)];
}

/** Best donor for borrowing (Gudergan 2008 Table 1); null if untestable. */
export function findDonor(
  focal: string,
  focalInfo: ResolvedIndicators,
  model: PlsModel,
): Donor | null {
  const focalMode = model.mmMatrix.constructMode(focal);
  if (focalMode !== "A") return null;
  const nOwn = focalInfo.indicators.length;
  if (nOwn < 2 || nOwn >= 4) return null;
  const nBorrow = 4 - nOwn;

  const neighbors = getStructurallyConnected(focal, model).filter((c) => !c.includes("*"));
  let best: Donor | null = null;
  let bestScore = 0;
  for (const donorName of neighbors) {
    const donorInfo = resolveIndicators(donorName, model);
    if (focalInfo.isHoc !== donorInfo.isHoc) continue;
    if (donorInfo.indicators.length < nBorrow) continue;
    const donorMode = model.mmMatrix.constructMode(donorName);
    let score: number;
    let pattern: string;
    if (nOwn === 3) {
      if (donorMode === "A") {
        score = 2;
        pattern = "all";
      } else {
        score = 0;
        pattern = "none";
      }
    } else {
      score = 1;
      pattern = "tau_1342";
    }
    if (
      score > bestScore ||
      (score === bestScore &&
        best !== null &&
        donorInfo.indicators.length > best.nDonorIndicators)
    ) {
      bestScore = score;
      best = {
        construct: donorName,
        mode: donorMode,
        donorIndicators: donorInfo.indicators,
        donorData: donorInfo.data,
        isHoc: donorInfo.isHoc,
        needsReestimation: donorInfo.needsReestimation,
        borrowed: donorInfo.indicators.slice(0, nBorrow),
        vanishingPattern: pattern,
        nVanishing: score,
        nDonorIndicators: donorInfo.indicators.length,
      };
    }
  }
  if (best === null || best.nVanishing === 0) return null;
  return best;
}

// --- Main entry point ----------------------------------------------------------------

interface TestInfo {
  indicators: string[];
  data: number[][];
  isHoc: boolean;
  needsReestimation: boolean;
  ownIndicators?: string[];
  borrowedIndicators?: string[];
  borrowing?: CtaBorrowing;
}

/**
 * R's one-matrix `cov(x)` (feature_cta.R:552,591), over row-major input.
 *
 * The conversion shim, not the computation: `@compstats/core` holds a matrix
 * column-major, and this package's tetrad machinery is row-major throughout.
 * The one-matrix form matters — the two-argument `cov(x, y)` walks each column
 * pair again and lands on different last bits, which is what this called until
 * `tests/fixtures/helpers/matstats.R` pinned the difference.
 */
function cov(data: number[][]): number[][] {
  return toRows(csCov(asMatrix(data)));
}

/** Column-major slice of pre-selected data by resampled row indices. */
function resampleRows(data: number[][], idx: readonly number[]): number[][] {
  return idx.map((i) => data[i]!);
}

/** Options for {@link assessCta}. */
export interface CtaOptions {
  constructs?: string | readonly string[];
  nboot?: number;
  /** Seeds the default bootstrap RNG (default 123). */
  seed?: number;
  alpha?: number;
  correction?: string;
  borrow?: boolean;
  /** Injected bootstrap row-index stream, shape `(nboot, n)`, 0-based. */
  draws?: number[][];
}

/** Named-args form of {@link assessCta}. */
export interface AssessCtaArgs extends CtaOptions {
  model: unknown;
}

function emptyResult(
  nboot: number,
  alpha: number,
  correction: string,
  skipped: string[],
  borrowing: Map<string, CtaBorrowing>,
): CtaAnalysis {
  return makeAnalysis([], new Map(), nboot, alpha, correction, skipped, borrowing);
}

/**
 * Confirmatory Tetrad Analysis over the model's constructs.
 *
 * Returns null (with a warning) for non-seminr input or when no requested
 * construct exists. Higher-order models are supported (tested over LOC scores).
 * Callable as `assessCta(model, options?)` or `assessCta({ model, ...options })`.
 * `draws` injects the R bootstrap row-index stream (shape `(nboot, n)`, 0-based)
 * for exact fixture parity; otherwise `seed` seeds `mulberry32` (one resample
 * vector per iteration).
 */
export function assessCta(args: AssessCtaArgs): CtaAnalysis | null;
export function assessCta(model: unknown, options?: CtaOptions): CtaAnalysis | null;
export function assessCta(
  modelOrArgs: unknown,
  positionalOptions: CtaOptions = {},
): CtaAnalysis | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessCtaArgs).model : modelOrArgs;
  const options = named ? (modelOrArgs as AssessCtaArgs) : positionalOptions;

  if (!validateSeminrModel(seminrModel, "assessCta")) return null;
  const model = seminrModel;

  const nboot = options.nboot ?? 5000;
  const seed = options.seed ?? 123;
  const alpha = options.alpha ?? 0.05;
  const correction = options.correction ?? "BH";
  const borrow = options.borrow ?? true;

  if (correction !== "BH" && correction !== "bonferroni" && correction !== "none") {
    throw new Error(
      `correction must be one of 'BH', 'bonferroni', 'none'; got '${correction}'`,
    );
  }

  const allConstructs = model.mmMatrix.allConstructs();
  let selected: string[];
  if (options.constructs === undefined) {
    selected = allConstructs;
  } else {
    const requested =
      typeof options.constructs === "string" ? [options.constructs] : [...options.constructs];
    const invalid = requested.filter((c) => !allConstructs.includes(c));
    if (invalid.length > 0) {
      console.warn(`Constructs not found in model: ${invalid.join(", ")}`);
    }
    selected = requested.filter((c) => allConstructs.includes(c));
    if (selected.length === 0) {
      console.warn("No valid constructs to test.");
      return null;
    }
  }

  const excludedInteractions = selected.filter((c) => c.includes("*"));
  selected = selected.filter((c) => !c.includes("*"));

  const skipped: string[] = [];
  const testConstructs: string[] = [];
  const constructInfo = new Map<string, TestInfo>();
  const borrowingDetails = new Map<string, CtaBorrowing>();

  for (const construct of selected) {
    const info = resolveIndicators(construct, model);
    if (info.indicators.length >= 4) {
      testConstructs.push(construct);
      constructInfo.set(construct, {
        indicators: info.indicators,
        data: info.data,
        isHoc: info.isHoc,
        needsReestimation: info.needsReestimation,
      });
    } else if (borrow && info.indicators.length >= 2) {
      const donor = findDonor(construct, info, model);
      if (donor !== null) {
        testConstructs.push(construct);
        const borrowedColIdx = donor.borrowed.map((b) => donor.donorIndicators.indexOf(b));
        const borrowedData = donor.donorData.map((row) => borrowedColIdx.map((j) => row[j]!));
        const borrowing: CtaBorrowing = {
          donor: donor.construct,
          donorMode: donor.mode,
          vanishingPattern: donor.vanishingPattern,
          nVanishing: donor.nVanishing,
        };
        constructInfo.set(construct, {
          indicators: [...info.indicators, ...donor.borrowed],
          data: info.data.map((row, r) => [...row, ...borrowedData[r]!]),
          isHoc: info.isHoc || donor.isHoc,
          needsReestimation: info.needsReestimation || donor.needsReestimation,
          ownIndicators: info.indicators,
          borrowedIndicators: donor.borrowed,
          borrowing,
        });
        borrowingDetails.set(construct, borrowing);
      } else {
        skipped.push(construct);
      }
    } else {
      skipped.push(construct);
    }
  }

  if (testConstructs.length === 0) {
    return emptyResult(
      nboot,
      alpha,
      correction,
      [...skipped, ...excludedInteractions],
      borrowingDetails,
    );
  }

  const nObs = model.data.values.length;
  let indexStream: number[][];
  if (options.draws) {
    indexStream = options.draws;
  } else {
    const gen = mulberry32(seed);
    indexStream = [];
    for (let b = 0; b < nboot; b++) {
      indexStream.push(Array.from({ length: nObs }, () => Math.floor(gen() * nObs)));
    }
  }

  const tetradSpecs = new Map<string, TetradSpec[]>();
  const originalTetrads = new Map<string, number[]>();
  for (const construct of testConstructs) {
    const tinfo = constructInfo.get(construct)!;
    const specs = tinfo.borrowing
      ? enumerateBorrowedTetrads(
          tinfo.ownIndicators!,
          tinfo.borrowedIndicators!,
          tinfo.borrowing.vanishingPattern,
        )
      : enumerateTetrads(tinfo.indicators);
    tetradSpecs.set(construct, specs);
    originalTetrads.set(construct, computeTetrads(cov(tinfo.data), tinfo.indicators, specs));
  }

  const anyReestimation = testConstructs.some((c) => constructInfo.get(c)!.needsReestimation);
  const bootTetrads = new Map<string, number[][]>();
  for (const construct of testConstructs) {
    const nt = tetradSpecs.get(construct)!.length;
    bootTetrads.set(
      construct,
      Array.from({ length: nboot }, () => new Array<number>(nt).fill(NaN)),
    );
  }

  for (let b = 0; b < nboot; b++) {
    const idx = indexStream[b]!;
    let bootModel: PlsModel | null = null;
    if (anyReestimation) {
      const resampled: Dataset = {
        columns: model.rawdata.columns,
        values: resampleRows(model.rawdata.values, idx),
      };
      try {
        bootModel = rerun(model, { data: resampled });
      } catch {
        bootModel = null; // mirror R tryCatch(..., error = NULL)
      }
    }
    for (const construct of testConstructs) {
      const tinfo = constructInfo.get(construct)!;
      let bootData: number[][];
      if (tinfo.needsReestimation) {
        if (bootModel === null) continue; // leave NA row, as in R
        bootData = selectColumns(bootModel.data, tinfo.indicators);
      } else {
        bootData = resampleRows(tinfo.data, idx);
      }
      bootTetrads.get(construct)![b] = computeTetrads(
        cov(bootData),
        tinfo.indicators,
        tetradSpecs.get(construct)!,
      );
    }
  }

  const alphaHalf = alpha / 2;
  const [ciLowerName, ciUpperName] = ciColumnLabels(alpha);
  const detailCols = [...DETAIL_COLS_HEAD, ciLowerName, ciUpperName, ...DETAIL_COLS_TAIL];
  const eps = Number.EPSILON;

  const rows: CtaConstructResult[] = [];
  const details = new Map<string, CtaTetradDetails>();
  for (const construct of testConstructs) {
    const tinfo = constructInfo.get(construct)!;
    const specs = tetradSpecs.get(construct)!;
    const orig = originalTetrads.get(construct)!;
    const bootMat = bootTetrads.get(construct)!;
    const nTetrads = specs.length;

    const rawMode = model.mmMatrix.constructMode(construct);
    let modeLabel = rawMode === "A" ? "Mode A (reflective)" : "Mode B (formative)";
    if (tinfo.isHoc) modeLabel += " [HOC]";
    if (tinfo.borrowing) modeLabel += ` [borrowed from ${tinfo.borrowing.donor}]`;

    const values: number[][] = Array.from({ length: nTetrads }, () =>
      new Array<number>(8).fill(NaN),
    );
    const pValues = new Array<number>(nTetrads).fill(NaN);
    const labels = specs.map(formatTetradLabel);
    for (let tIdx = 0; tIdx < nTetrads; tIdx++) {
      const bootVals = bootMat.map((row) => row[tIdx]!).filter((v) => !Number.isNaN(v));
      if (bootVals.length < MIN_VALID_BOOTS) continue;
      const bootSd = rSd(bootVals);
      const tValue = bootSd < eps ? NaN : orig[tIdx]! / bootSd;
      let countGe = 0;
      let countLe = 0;
      let sum = 0;
      for (const v of bootVals) {
        if (v >= 0) countGe++;
        if (v <= 0) countLe++;
        sum += v;
      }
      const pValue = 2 * Math.min(countGe / bootVals.length, countLe / bootVals.length);
      pValues[tIdx] = pValue;
      values[tIdx] = [
        orig[tIdx]!,
        tValue,
        sum / bootVals.length,
        bootSd,
        quantile(bootVals, alphaHalf),
        quantile(bootVals, 1 - alphaHalf),
        pValue,
        NaN, // Adj_P filled below
      ];
    }

    const allNaN = pValues.every((v) => Number.isNaN(v));
    const adjP = correction === "none" || allNaN ? pValues.slice() : pAdjust(pValues, correction);
    for (let t = 0; t < nTetrads; t++) values[t]![7] = adjP[t]!;
    const significant = adjP.map((v) => !Number.isNaN(v) && v <= alpha);

    details.set(construct, {
      table: namedMatrix(labels, detailCols, values),
      significant,
    });
    const nSig = significant.reduce((acc, s) => acc + (s ? 1 : 0), 0);
    rows.push({
      construct,
      mode: modeLabel,
      indicators: tinfo.indicators.length,
      tetrads: nTetrads,
      significant: nSig,
      verdict: nSig === 0 ? "Reflective supported" : "Reflective rejected",
    });
  }

  return makeAnalysis(
    rows,
    details,
    nboot,
    alpha,
    correction,
    [...skipped, ...excludedInteractions],
    borrowingDetails,
  );
}

// --- Record surface (R print.cta_analysis / summary.cta_analysis) --------------------

function padEnd(s: string, width: number): string {
  return s.length >= width ? s : s + " ".repeat(width - s.length);
}

function renderSummary(a: CtaAnalysis): string {
  const alphaG = String(Number(a.alpha.toPrecision(6)));
  const lines = [
    "Confirmatory Tetrad Analysis (CTA-PLS)",
    "=======================================",
    `Bootstrap samples: ${a.nboot} | Alpha: ${alphaG} | Correction: ${a.correction}`,
    "",
  ];
  if (a.constructResults.length === 0) {
    lines.push("No constructs tested (all had < 4 indicators).");
  } else {
    const header = ["Construct", "Mode", "Indicators", "Tetrads", "Significant", "Verdict"];
    const body = a.constructResults.map((r) => [
      r.construct,
      r.mode,
      String(r.indicators),
      String(r.tetrads),
      String(r.significant),
      r.verdict,
    ]);
    const widths = header.map((h, j) =>
      Math.max(h.length, ...body.map((row) => row[j]!.length)),
    );
    lines.push(header.map((h, j) => padEnd(h, widths[j]!)).join("  "));
    lines.push("-".repeat(widths.reduce((s, w) => s + w, 0) + 2 * (header.length - 1)));
    for (const row of body) lines.push(row.map((c, j) => padEnd(c, widths[j]!)).join("  "));
  }
  if (a.borrowing.size > 0) {
    lines.push("");
    lines.push("Borrowing:");
    for (const [name, b] of a.borrowing) {
      lines.push(
        `  ${name}: borrowed from ${b.donor} (${b.vanishingPattern} pattern, ` +
          `${b.nVanishing} vanishing tetrad(s))`,
      );
    }
  }
  if (a.skipped.length > 0) {
    lines.push("");
    lines.push(`Skipped: ${a.skipped.join(", ")}`);
  }
  return lines.join("\n");
}

function renderDetailed(a: CtaAnalysis, digits: number): string {
  const alphaG = String(Number(a.alpha.toPrecision(6)));
  const lines = [
    "Confirmatory Tetrad Analysis (CTA-PLS) — Detailed Results",
    "==========================================================",
    `Bootstrap samples: ${a.nboot} | Alpha: ${alphaG} | Correction: ${a.correction}`,
    "",
  ];
  const verdicts = new Map(a.constructResults.map((r) => [r.construct, r]));
  for (const [construct, detail] of a.tetradDetails) {
    const row = verdicts.get(construct)!;
    lines.push(`--- ${construct} (${row.mode}) ---`);
    const b = a.borrowing.get(construct);
    if (b) {
      lines.push(
        `Borrowed ${b.nVanishing} vanishing tetrad(s) from ${b.donor} (${b.vanishingPattern})`,
      );
    }
    lines.push(`Verdict: ${row.verdict}`);
    lines.push("");
    const table = detail.table;
    const header = ["Tetrad", ...table.cols, "Significant"];
    const body = table.rows.map((label, i) => [
      label,
      ...table.cols.map((_, j) => table.values[i]![j]!.toFixed(digits)),
      detail.significant[i] ? "*" : "",
    ]);
    const widths = header.map((h, j) =>
      Math.max(h.length, ...body.map((r) => r[j]!.length)),
    );
    lines.push(header.map((h, j) => padEnd(h, widths[j]!)).join(" "));
    for (const r of body) lines.push(r.map((c, j) => padEnd(c, widths[j]!)).join(" "));
    lines.push("");
  }
  if (a.skipped.length > 0) {
    lines.push(`Skipped constructs: ${a.skipped.join(", ")}`);
  }
  return lines.join("\n");
}

function makeAnalysis(
  constructResults: CtaConstructResult[],
  tetradDetails: Map<string, CtaTetradDetails>,
  nboot: number,
  alpha: number,
  correction: string,
  skipped: string[],
  borrowing: Map<string, CtaBorrowing>,
): CtaAnalysis {
  return Object.freeze({
    kind: "cta_analysis" as const,
    constructResults: Object.freeze(constructResults),
    tetradDetails,
    nboot,
    alpha,
    correction,
    skipped: Object.freeze(skipped),
    borrowing,
    toString(): string {
      return renderSummary(this as CtaAnalysis);
    },
    summarize(digits = 4): string {
      return renderDetailed(this as CtaAnalysis, digits);
    },
  });
}
