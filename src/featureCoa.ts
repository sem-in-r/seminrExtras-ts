/**
 * Composite Overfit Analysis (COA), ported from seminrExtras `feature_coa.R`
 * (via the Python port's coa.py).
 *
 * Pipeline (Ray, Danks & Valdez 2022, ISR): (1) per-case Predictive Deviance
 * PD = in-sample fitted - out-of-sample predicted composite score for a focal
 * construct, via k-fold `predictPls`; (2) a full (unpruned) CART tree grown on
 * PD over the construct scores identifies groups of deviant cases (node means
 * outside the `devianceBounds` PD quantiles); (3) each deviant group is removed
 * and the model re-estimated to expose unstable parameters.
 *
 * Parity notes (py plan F16): the only RNG is `predictPls`'s single row
 * permutation — inject `ordering` (0-based) for exact R parity. Observation
 * indices in all records are 1-BASED (R convention, matching `CartTree.where`).
 * `reps` is R's dubious no-op (re-averages identical folds): accepted and
 * ignored. `cores` likewise (predictPls is single-threaded). R's progress
 * `message()` output is not ported.
 */

import {
  namedMatrix,
  predictDA,
  predictPls,
  rerun,
  type NamedMatrix,
  type PlsModel,
  type PlsPrediction,
} from "@seminr/core";
import { quantile } from "@compstats/core/stats";
import { rpartAnova, type CartTree } from "./cart.ts";
import { mean } from "@compstats/core/stats";
import { isNamedArgs, validateForPrediction } from "./helpers.ts";

const VALID_PARAMS = ["path_coef", "outer_weights", "outer_loadings", "rSquared"] as const;
type CoaParam = (typeof VALID_PARAMS)[number];
const PARAM_ATTR: Record<CoaParam, "pathCoef" | "outerWeights" | "outerLoadings" | "rSquared"> = {
  path_coef: "pathCoef",
  outer_weights: "outerWeights",
  outer_loadings: "outerLoadings",
  rSquared: "rSquared",
};

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// -----------------------------------------------------------------------------
// Records
// -----------------------------------------------------------------------------

/** Predictive-deviance results (R class `coa_deviance`). */
export interface CoaDeviance {
  readonly kind: "coa_deviance";
  readonly pd: readonly number[];
  readonly pdData: NamedMatrix;
  readonly isMse: number;
  readonly oosMse: number;
  readonly overfitRatio: number;
  readonly fittedScore: readonly number[];
  readonly predictedScore: readonly number[];
}

/** Deviance tree and deviant case groups (R class `coa_dtree`). */
export interface CoaDtree {
  readonly kind: "coa_dtree";
  readonly tree: CartTree;
  readonly sortedPd: readonly number[];
  /** Group label -> 1-based observation indices. */
  readonly deviantGroups: Readonly<Record<string, readonly number[]>>;
  readonly groupRoots: Readonly<Record<string, number>>;
  readonly uniqueDeviants: readonly number[];
  readonly deviantNodes: readonly number[];
}

/** Per-group re-estimation deltas: removed cases and parameter diffs. */
export interface CoaGroupInstability {
  readonly cases: readonly number[];
  readonly paramDiffs: Readonly<Record<string, NamedMatrix>>;
}

/** Parameter instability per deviant group (R class `coa_unstable`). */
export interface CoaUnstable {
  readonly kind: "coa_unstable";
  readonly groups: Readonly<Record<string, CoaGroupInstability>>;
}

/** Consolidated split rules defining a deviant group (construct ranges). */
export interface CoaRules {
  readonly construct: readonly string[];
  readonly gte: readonly number[];
  readonly lt: readonly number[];
}

/** Competing split criteria at a tree node, ranked by improvement. */
export interface CoaCompetes {
  readonly criterion: readonly string[];
  readonly sign: readonly string[];
  readonly value: readonly number[];
  readonly improve: readonly number[];
}

/** Full COA result (R class `coa_analysis`). */
export interface CoaAnalysis {
  readonly kind: "coa_analysis";
  readonly plsModel: PlsModel;
  readonly focalConstruct: string;
  readonly devianceBounds: readonly [number, number];
  readonly predictiveDeviance: CoaDeviance;
  readonly devianceTree: CoaDtree;
  readonly unstable: CoaUnstable;
  toString(): string;
  summarize(): string;
}

// -----------------------------------------------------------------------------
// Tree traversal helpers (R :::-style internals)
// -----------------------------------------------------------------------------

/** Root-to-node id chain in rpart numbering (parent of k is floor(k / 2)). */
export function pathTo(nodeId: number): number[] {
  const path = [nodeId];
  while (nodeId !== 1) {
    nodeId = Math.floor(nodeId / 2);
    path.push(nodeId);
  }
  return path.reverse();
}

/**
 * Topmost node per ancestor chain: R keeps, for each id, the numerically
 * smallest listed id on its root-path (unique, in processing order).
 */
export function mainAncestors(parentIds: readonly string[]): string[] {
  const ids = parentIds.map((p) => Number(p));
  const idSet = new Set(ids);
  const out: string[] = [];
  for (const pid of ids) {
    const top = String(Math.min(...pathTo(pid).filter((a) => idSet.has(a))));
    if (!out.includes(top)) out.push(top);
  }
  return out;
}

/** 1-based observation indices whose assigned leaf frame-row is masked. */
function casesOf(tree: CartTree, frameMask: readonly boolean[]): number[] {
  const rows = new Set<number>();
  frameMask.forEach((m, i) => {
    if (m) rows.add(i + 1);
  });
  const out: number[] = [];
  tree.where.forEach((w, i) => {
    if (rows.has(w)) out.push(i + 1);
  });
  return out;
}

// -----------------------------------------------------------------------------
// Step 1: predictive deviance
// -----------------------------------------------------------------------------

/** Options of {@link predictiveDeviance} (also embedded in {@link CoaOptions}). */
export interface PredictiveDevianceOptions {
  technique?: typeof predictDA;
  noFolds?: number;
  /** Accepted and ignored (R's re-averaging no-op). */
  reps?: number;
  /** Accepted and ignored (predictPls is single-threaded). */
  cores?: number;
  seed?: number;
  /** Precomputed prediction: skips validation AND prediction (R-faithful). */
  predictModel?: PlsPrediction;
  /** Injected 0-based fold permutation for exact R parity. */
  ordering?: readonly number[];
}

function column(m: NamedMatrix, name: string): number[] {
  const j = m.cols.indexOf(name);
  return m.values.map((row) => row[j]!);
}

/**
 * Per-case predictive deviance for `focalConstruct` (R `predictive_deviance`).
 *
 * With `predictModel` supplied, validation and prediction are both skipped
 * (R-faithful).
 */
export function predictiveDeviance(
  seminrModel: unknown,
  focalConstruct: string,
  options: PredictiveDevianceOptions = {},
): CoaDeviance | null {
  let predictModel = options.predictModel;
  if (predictModel === undefined) {
    if (!validateForPrediction(seminrModel, "predictiveDeviance")) return null;
    predictModel = predictPls(seminrModel, {
      technique: options.technique ?? predictDA,
      noFolds: options.noFolds ?? 10,
      ordering: options.ordering,
      seed: options.seed ?? 123,
    });
  }
  const model = seminrModel as PlsModel;

  const composites = predictModel.composites;
  const fitted = column(composites.compositeInSample, focalConstruct);
  const predicted = column(composites.compositeOutOfSample, focalConstruct);
  const actualStar = column(composites.actualsStar, focalConstruct);

  const isMse = mean(actualStar.map((a, i) => (a - fitted[i]!) ** 2));
  const oosMse = mean(actualStar.map((a, i) => (a - predicted[i]!) ** 2));
  const overfitRatio = (oosMse - isMse) / isMse;

  const pd = fitted.map((f, i) => f - predicted[i]!);
  const scores = model.constructScores;
  const pdData = namedMatrix(
    scores.rows,
    [...scores.cols, "PD"],
    scores.values.map((row, i) => [...row, pd[i]!]),
  );
  return Object.freeze({
    kind: "coa_deviance" as const,
    pd,
    pdData,
    isMse,
    oosMse,
    overfitRatio,
    fittedScore: fitted,
    predictedScore: predicted,
  });
}

// -----------------------------------------------------------------------------
// Step 2: deviance tree
// -----------------------------------------------------------------------------

/** Grow the full CART tree on PD and extract deviant case groups. */
export function devianceTree(
  pdResult: CoaDeviance,
  devianceBounds: readonly number[] = [0.025, 0.975],
): CoaDtree {
  const cols = pdResult.pdData.cols;
  const varNames = cols.filter((c) => c !== "PD");
  const varIdx = varNames.map((c) => cols.indexOf(c));
  const x = pdResult.pdData.values.map((row) => varIdx.map((j) => row[j]!));
  const tree = rpartAnova(x, [...pdResult.pd], varNames, { minsplit: 2, cp: 0 });

  const lo = quantile(pdResult.pd, devianceBounds[0]!);
  const hi = quantile(pdResult.pd, devianceBounds[1]!);

  const isLeaf = tree.var.map((v) => v === "<leaf>");
  const isDeviant = tree.yval.map((y) => y < lo || y > hi);
  const isDeviantLeaf = isDeviant.map((d, i) => d && isLeaf[i]!);
  const isDeviantParent = isDeviant.map((d, i) => d && !isLeaf[i]!);

  const sortedPd = tree.yval.filter((_, i) => isLeaf[i]!).sort((a, b) => b - a);
  const deviants = casesOf(tree, isDeviantLeaf);

  const devParentIds = tree.nodeIds.filter((_, i) => isDeviantParent[i]!);
  const leafIds = tree.nodeIds.filter((_, i) => isLeaf[i]!);

  let grouped: [number, number[]][] = [];
  if (devParentIds.length > 0) {
    for (const ancestor of mainAncestors(devParentIds.map(String))) {
      const anc = Number(ancestor);
      const under = new Set(leafIds.filter((lid) => pathTo(lid).includes(anc)));
      const mask = tree.nodeIds.map((nid) => under.has(nid));
      grouped.push([anc, casesOf(tree, mask)]);
    }
  }

  if (grouped.length > 26) {
    console.warn(
      `More than 26 deviant groups found (${grouped.length}); only the first 26 are labeled A-Z.`,
    );
    grouped = grouped.slice(0, 26);
  }

  const deviantGroups: Record<string, readonly number[]> = {};
  const groupRoots: Record<string, number> = {};
  grouped.forEach(([root, cases], i) => {
    deviantGroups[LETTERS[i]!] = cases;
    groupRoots[LETTERS[i]!] = root;
  });
  const inGroups = new Set(Object.values(deviantGroups).flat());
  const uniqueDeviants = deviants.filter((c) => !inGroups.has(c));
  const deviantNodes = tree.nodeIds.filter((_, i) => isDeviant[i]!);

  return Object.freeze({
    kind: "coa_dtree" as const,
    tree,
    sortedPd,
    deviantGroups,
    groupRoots,
    uniqueDeviants,
    deviantNodes,
  });
}

// -----------------------------------------------------------------------------
// Step 3: unstable parameters
// -----------------------------------------------------------------------------

function normalizeParams(params: string | readonly string[]): CoaParam[] {
  const paramsT = typeof params === "string" ? [params] : [...params];
  const invalid = paramsT.filter((p) => !(VALID_PARAMS as readonly string[]).includes(p));
  if (invalid.length > 0) {
    throw new Error(
      `Invalid params: ${invalid.join(", ")}. Valid options: ${VALID_PARAMS.join(", ")}`,
    );
  }
  return paramsT as CoaParam[];
}

/**
 * Re-estimate without `removeCases` (1-based) and diff the parameters.
 *
 * R uses `pls_model$data` (the PROCESSED data), not rawdata.
 */
function paramDiffs(
  removeCases: readonly number[],
  model: PlsModel,
  params: readonly CoaParam[],
): Record<string, NamedMatrix> {
  const remove = new Set(removeCases.map((c) => c - 1));
  const reducedData = {
    columns: model.data.columns,
    values: model.data.values.filter((_, i) => !remove.has(i)),
  };
  const reduced = rerun(model, { data: reducedData });
  const diffs: Record<string, NamedMatrix> = {};
  for (const param of params) {
    const after = reduced[PARAM_ATTR[param]];
    const before = model[PARAM_ATTR[param]];
    diffs[param] = namedMatrix(
      after.rows,
      after.cols,
      after.values.map((row, i) => row.map((v, j) => v - before.values[i]![j]!)),
    );
  }
  return diffs;
}

/**
 * Parameter deltas from removing each deviant group (R `unstable_params`).
 *
 * Accepts a {@link CoaDtree} (its `deviantGroups` is used) or a mapping of
 * group label to 1-based case indices. No model validation (R-faithful).
 */
export function unstableParams(
  seminrModel: PlsModel,
  deviantGroups: CoaDtree | Readonly<Record<string, readonly number[]>>,
  params: string | readonly string[] = "path_coef",
): CoaUnstable {
  const groupsIn =
    "kind" in deviantGroups && deviantGroups.kind === "coa_dtree"
      ? (deviantGroups as CoaDtree).deviantGroups
      : (deviantGroups as Readonly<Record<string, readonly number[]>>);
  const paramsT = normalizeParams(params);
  const groups: Record<string, CoaGroupInstability> = {};
  for (const [label, cases] of Object.entries(groupsIn)) {
    groups[label] = Object.freeze({
      cases: cases.map((c) => c),
      paramDiffs: paramDiffs(cases, seminrModel, paramsT),
    });
  }
  return Object.freeze({ kind: "coa_unstable" as const, groups });
}

// -----------------------------------------------------------------------------
// Rules extraction
// -----------------------------------------------------------------------------

/**
 * (var, signed ncat, index) of the split that created `nodeId`.
 *
 * Exploits preorder: the frame row before a LEFT child is its parent, whose
 * split block starts at `offsets[parentRow]` (1-based).
 */
function primarySplitAt(
  nodeId: number,
  tree: CartTree,
  offsets: readonly number[],
): [string, number, number] {
  const isOdd = nodeId % 2 === 1;
  const searchNode = isOdd ? nodeId - 1 : nodeId;
  const frameRow = tree.nodeIds.indexOf(searchNode);
  const splitRow = offsets[frameRow - 1]! - 1;
  let ncat = tree.splitNcat[splitRow]!;
  if (isOdd) ncat = -ncat;
  return [tree.splitVar[splitRow]!, ncat, tree.splitIndex[splitRow]!];
}

/** Per construct (appearance order): tightest `>=` (max) and `<` (min) bound. */
function consolidateRules(splits: readonly [string, number, number][]): CoaRules {
  const order: string[] = [];
  for (const [v] of splits) if (!order.includes(v)) order.push(v);
  const gte: number[] = [];
  const lt: number[] = [];
  for (const v of order) {
    const geVals = splits.filter(([sv, ncat]) => sv === v && ncat > 0).map(([, , val]) => val);
    const ltVals = splits.filter(([sv, ncat]) => sv === v && ncat <= 0).map(([, , val]) => val);
    gte.push(geVals.length > 0 ? Math.max(...geVals) : NaN);
    lt.push(ltVals.length > 0 ? Math.min(...ltVals) : NaN);
  }
  return Object.freeze({ construct: order, gte, lt });
}

function rulesForGroup(dtree: CoaDtree, groupName: string): CoaRules {
  const root = dtree.groupRoots[groupName]!;
  const nodePath = pathTo(root).slice(1);
  const offsets = dtree.tree.splitRowOffsets();
  return consolidateRules(nodePath.map((nid) => primarySplitAt(nid, dtree.tree, offsets)));
}

/**
 * Split criteria defining deviant group(s) (R `group_rules`).
 *
 * Pass a {@link CoaAnalysis} or {@link CoaDtree} for all groups, or a group
 * letter plus the {@link CoaAnalysis} for one group.
 */
export function groupRules(
  groupName: string | CoaAnalysis | CoaDtree,
  coaResult?: CoaAnalysis,
): CoaRules | Record<string, CoaRules> {
  if (typeof groupName !== "string") {
    const dtree = groupName.kind === "coa_analysis" ? groupName.devianceTree : groupName;
    return Object.fromEntries(
      Object.keys(dtree.deviantGroups).map((g) => [g, rulesForGroup(dtree, g)]),
    );
  }
  if (coaResult === undefined) {
    throw new Error("coaResult is required when groupName is a group letter");
  }
  return rulesForGroup(coaResult.devianceTree, groupName);
}

function competesAt(nodeId: number, dtree: CoaDtree): CoaCompetes {
  if (nodeId === 1) throw new Error("No splits before root (node 1) of tree");
  const tree = dtree.tree;
  const offsets = tree.splitRowOffsets();
  const isOdd = nodeId % 2 === 1;
  const searchNode = isOdd ? nodeId - 1 : nodeId;
  const parentRow = tree.nodeIds.indexOf(searchNode / 2);
  const start = offsets[parentRow]! - 1;
  const criterion: string[] = [];
  const sign: string[] = [];
  const value: number[] = [];
  const improve: number[] = [];
  for (let r = start; r < start + 1 + tree.ncompete[parentRow]!; r++) {
    const ncat = tree.splitNcat[r]! * (isOdd ? -1 : 1);
    criterion.push(tree.splitVar[r]!);
    sign.push(ncat > 0 ? ">=" : "< ");
    value.push(tree.splitIndex[r]!);
    improve.push(tree.splitImprove[r]!);
  }
  return Object.freeze({ criterion, sign, value, improve });
}

/**
 * Competing split criteria at a node, ranked by improvement (R `competes`).
 *
 * Pass a {@link CoaDtree} for all group roots, or a node id plus the dtree.
 */
export function competes(
  nodeId: number | CoaDtree,
  dtree?: CoaDtree,
): CoaCompetes | Record<string, CoaCompetes> {
  if (typeof nodeId !== "number") {
    return Object.fromEntries(
      Object.keys(nodeId.deviantGroups).map((g) => [g, competesAt(nodeId.groupRoots[g]!, nodeId)]),
    );
  }
  if (dtree === undefined) throw new Error("dtree is required when nodeId is an integer");
  return competesAt(nodeId, dtree);
}

/**
 * Mean construct scores per deviant group (R `plot_group_scores` matrix).
 *
 * Returns a `constructs x groups` NamedMatrix whose columns (group label
 * order) are the column means of `constructScores` over each group's 1-based
 * case indices. Pass a {@link CoaAnalysis} (its model is used) or a
 * {@link CoaDtree} plus its `model`. This carries the plot's computation so
 * the plot layer only marshals.
 */
export function groupScoreMeans(
  coaOrDtree: CoaAnalysis | CoaDtree,
  model?: PlsModel,
): NamedMatrix {
  let dtree: CoaDtree;
  if (coaOrDtree.kind === "coa_analysis") {
    dtree = coaOrDtree.devianceTree;
    model = coaOrDtree.plsModel;
  } else {
    dtree = coaOrDtree;
    if (model === undefined) throw new Error("model is required when passing a CoaDtree.");
  }
  const scores = model.constructScores;
  const groups = Object.keys(dtree.deviantGroups);
  const out = namedMatrix(scores.cols, groups);
  groups.forEach((group, j) => {
    const cases = dtree.deviantGroups[group]!;
    scores.cols.forEach((_, c) => {
      let sum = 0;
      for (const obs of cases) sum += scores.values[obs - 1]![c]!;
      out.values[c]![j] = sum / cases.length;
    });
  });
  return out;
}

// -----------------------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------------------

/** Options of {@link assessCoa}. */
export interface CoaOptions extends PredictiveDevianceOptions {
  devianceBounds?: readonly number[];
  params?: string | readonly string[];
}

/** Named-args form of {@link assessCoa}. */
export interface AssessCoaArgs extends CoaOptions {
  model: unknown;
  focalConstruct: string;
}

/** Full Composite Overfit Analysis (R `assess_coa`). */
export function assessCoa(args: AssessCoaArgs): CoaAnalysis | null;
export function assessCoa(
  model: unknown,
  focalConstruct: string,
  options?: CoaOptions,
): CoaAnalysis | null;
export function assessCoa(
  modelOrArgs: unknown,
  positionalFocal?: string,
  positionalOptions: CoaOptions = {},
): CoaAnalysis | null {
  const named = isNamedArgs(modelOrArgs);
  const seminrModel = named ? (modelOrArgs as AssessCoaArgs).model : modelOrArgs;
  const focalConstruct = named
    ? (modelOrArgs as AssessCoaArgs).focalConstruct
    : positionalFocal!;
  const options = named ? (modelOrArgs as AssessCoaArgs) : positionalOptions;

  if (!validateForPrediction(seminrModel, "assessCoa")) return null;
  const model = seminrModel;

  const constructNames = model.constructScores.cols;
  if (!constructNames.includes(focalConstruct)) {
    throw new Error(
      `focal_construct '${focalConstruct}' not found in model constructs: ` +
        constructNames.join(", "),
    );
  }

  const bounds = (options.devianceBounds ?? [0.025, 0.975]).map(Number);
  if (
    bounds.length !== 2 ||
    bounds.some((v) => v < 0) ||
    bounds.some((v) => v > 1) ||
    bounds[0]! >= bounds[1]!
  ) {
    throw new Error(
      "deviance_bounds must be two values in [0, 1] with first < second, e.g. c(0.025, 0.975)",
    );
  }

  const paramsT = normalizeParams(options.params ?? "path_coef");

  const pd = predictiveDeviance(model, focalConstruct, options);
  if (pd === null) return null; // unreachable: the model already passed validation

  const dt = devianceTree(pd, bounds);
  const unstable = unstableParams(model, dt.deviantGroups, paramsT);
  const devianceBounds: readonly [number, number] = [bounds[0]!, bounds[1]!];
  const nObs = model.data.values.length;

  const g = (v: number) => String(Number(v.toPrecision(6)));
  const g4 = (v: number) => String(Number(v.toPrecision(4)));

  return Object.freeze({
    kind: "coa_analysis" as const,
    plsModel: model,
    focalConstruct,
    devianceBounds,
    predictiveDeviance: pd,
    devianceTree: dt,
    unstable,
    toString(): string {
      const lines = [
        "Composite Overfit Analysis (COA)",
        "================================",
        `Focal construct: ${focalConstruct}`,
        `Deviance bounds: ${g(devianceBounds[0])} - ${g(devianceBounds[1])}`,
        `Observations: ${nObs}`,
        "",
        "Prediction Metrics:",
        `  In-sample MSE:  ${g4(pd.isMse)}`,
        `  Out-of-sample MSE: ${g4(pd.oosMse)}`,
        `  Overfit ratio:     ${g4(pd.overfitRatio)}`,
        "",
        "Deviant Cases:",
      ];
      const groupEntries = Object.entries(dt.deviantGroups);
      let groupsLine = `  Groups: ${groupEntries.length}`;
      if (groupEntries.length > 0) {
        const sizes = groupEntries.map(([gl, c]) => `${gl} = ${c.length}`).join(", ");
        groupsLine += ` ( ${sizes} )`;
      }
      lines.push(groupsLine);
      lines.push(`  Unique deviants: ${dt.uniqueDeviants.length}`);
      return lines.join("\n");
    },
    summarize(): string {
      const g6 = (v: number) => String(Number(v.toPrecision(6)));
      const lines = [
        "Composite Overfit Analysis (COA) Summary",
        "=========================================",
        `Focal construct: ${focalConstruct}`,
        `Observations: ${nObs}`,
        `Deviance bounds: ${g(devianceBounds[0])} - ${g(devianceBounds[1])}`,
        "",
        "Prediction Metrics:",
        `  IS_MSE        ${g6(pd.isMse)}`,
        `  OOS_MSE       ${g6(pd.oosMse)}`,
        `  Overfit_Ratio ${g6(pd.overfitRatio)}`,
        "",
      ];
      const groupEntries = Object.entries(dt.deviantGroups);
      if (groupEntries.length > 0) {
        lines.push("Deviant Groups:", "  Group  Size  Root_Node");
        for (const [gl, cases] of groupEntries) {
          lines.push(`  ${gl.padEnd(5)}  ${String(cases.length).padEnd(4)}  ${dt.groupRoots[gl]}`);
        }
        lines.push("");
      }
      lines.push(`Unique deviants: ${dt.uniqueDeviants.length}`);
      const unstableEntries = Object.entries(unstable.groups);
      if (unstableEntries.length > 0) {
        lines.push("", "Parameter Instability (max |path_coef diff|):", "  Group  Max_Path_Diff");
        for (const [gl, grp] of unstableEntries) {
          const diff = grp.paramDiffs["path_coef"];
          const maxDiff =
            diff !== undefined
              ? Math.max(...diff.values.flat().filter((v) => !Number.isNaN(v)).map(Math.abs))
              : NaN;
          lines.push(`  ${gl.padEnd(5)}  ${g6(maxDiff)}`);
        }
      }
      return lines.join("\n");
    },
  });
}
