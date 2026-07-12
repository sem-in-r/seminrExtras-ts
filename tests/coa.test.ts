/**
 * COA (Slice 8): assessCoa pipeline against R goldens + behavioral spec.
 *
 * Parity fixture tests/fixtures/coa/coa.json carries, per case, the single R
 * fold permutation predictPls consumed (1-based, decremented here and injected
 * via `ordering`), the resulting PD vector and MSE metrics, R's pd_data
 * verbatim, and per deviance-bounds block the full deviance-tree outputs.
 * Observation indices are 1-BASED throughout (R convention).
 *
 * Two parity layers (documented divergence, same family as the cart
 * c1_pd_real decoupling): the tree stage is chaotic under the ~1e-15
 * PD/construct-score noise left by the predict stage, so EXACT structural
 * tree goldens are asserted on R's pd_data inputs (rPds), while the composed
 * pipeline (pds) is held to tolerance-robust outputs: PD/MSE at 1e-9 and the
 * deviant-group labels/sizes.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import { predictDA, predictPls } from "@seminr/core";
import {
  assessCoa,
  competes,
  devianceTree,
  groupRules,
  mainAncestors,
  pathTo,
  predictiveDeviance,
  unstableParams,
  type CoaCompetes,
  type CoaDeviance,
  type CoaRules,
} from "../src/featureCoa.ts";
import {
  FIXTURES_DIR,
  expectNamedClose,
  toMatrix,
  type FixtureMatrixNode,
} from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

const RTOL = 1e-9;
const ATOL = 1e-12;

interface TreeBlock {
  bounds: [number, number];
  sortedPd: number[];
  groups: { labels: string[]; roots: number[]; cases: number[][] };
  uniqueDeviants: number[];
  deviantNodes: number[];
  params?: string[];
  rules?: { group: string; construct: string[]; gte: (number | null)[]; lt: (number | null)[] }[];
  competes?: {
    group: string;
    criterion: string[];
    sign: string[];
    value: (number | null)[];
    improve: (number | null)[];
  }[];
  unstable?: { group: string; cases: number[]; paramDiffs: Record<string, FixtureMatrixNode> }[];
}

interface CoaCase {
  name: string;
  focal: string;
  noFolds: number;
  ordering: number[];
  pd: number[];
  isMse: number;
  oosMse: number;
  overfitRatio: number;
  pdData: FixtureMatrixNode;
  trees: TreeBlock[];
}

const fx = JSON.parse(readFileSync(join(FIXTURES_DIR, "coa", "coa.json"), "utf8")) as {
  cases: CoaCase[];
};

const CASE_NAMES = ["c1_cusl_123", "c1_cusl_42", "m3_loyalty_123", "m16_loyalty_123"];
const MODEL_ID: Record<string, string> = {
  c1_cusl_123: "C1",
  c1_cusl_42: "C1",
  m3_loyalty_123: "M3",
  m16_loyalty_123: "M16",
};

function getCase(name: string): CoaCase {
  return fx.cases.find((c) => c.name === name)!;
}

const models: Record<string, ReturnType<typeof estimateRegistryModel>> = {};
for (const [caseName, modelId] of Object.entries(MODEL_ID)) {
  models[modelId] ??= estimateRegistryModel(modelId);
  models[caseName] = models[modelId]!;
}

function ordering(c: CoaCase): number[] {
  return c.ordering.map((i) => i - 1);
}

/** Composed-pipeline PD records (predict stage live). */
const pds: Record<string, CoaDeviance> = {};
for (const name of CASE_NAMES) {
  const c = getCase(name);
  pds[name] = predictiveDeviance(models[name]!, c.focal, {
    technique: predictDA,
    noFolds: c.noFolds,
    ordering: ordering(c),
  })!;
}

/** CoaDeviance records rebuilt from R's exact pd_data (tree-stage inputs). */
const rPds: Record<string, CoaDeviance> = {};
for (const name of CASE_NAMES) {
  const c = getCase(name);
  const pdData = toMatrix(c.pdData);
  const pdIdx = pdData.cols.indexOf("PD");
  const pd = pdData.values.map((row) => row[pdIdx]!);
  rPds[name] = {
    kind: "coa_deviance",
    pd,
    pdData,
    isMse: c.isMse,
    oosMse: c.oosMse,
    overfitRatio: c.overfitRatio,
    fittedScore: pd,
    predictedScore: pd.map(() => 0),
  };
}

function isClose(a: number, b: number, rtol = RTOL, atol = ATOL): boolean {
  return Math.abs(a - b) <= Math.max(rtol * Math.max(Math.abs(a), Math.abs(b)), atol);
}

function assertVecClose(
  got: readonly number[],
  want: readonly (number | null)[],
  label: string,
  rtol = RTOL,
  atol = ATOL,
): void {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < got.length; i++) {
    const w = want[i] === null ? NaN : want[i]!;
    if (Number.isNaN(w)) {
      if (!Number.isNaN(got[i]!)) throw new Error(`${label}[${i}]: got ${got[i]}, expected NaN`);
    } else if (!isClose(got[i]!, w, rtol, atol)) {
      throw new Error(`${label}[${i}]: got ${got[i]}, expected ${w}`);
    }
  }
}

// --- predictive deviance parity -------------------------------------------------

describe("predictive deviance parity", () => {
  for (const name of CASE_NAMES) {
    test(`PD and metrics match R: ${name}`, () => {
      const c = getCase(name);
      const pd = pds[name]!;
      assertVecClose(pd.pd, c.pd, `${name}.pd`);
      expect(isClose(pd.isMse, c.isMse)).toBe(true);
      expect(isClose(pd.oosMse, c.oosMse)).toBe(true);
      expect(isClose(pd.overfitRatio, c.overfitRatio, 1e-6)).toBe(true);
    });
  }

  test("pd_data structure", () => {
    for (const name of ["c1_cusl_123", "m3_loyalty_123"]) {
      const pd = pds[name]!;
      const model = models[name]!;
      expect(pd.kind).toBe("coa_deviance");
      expect([...pd.pdData.cols]).toEqual([...model.constructScores.cols, "PD"]);
      expect(pd.pd.length).toBe(model.constructScores.values.length);
      pd.pd.forEach((v, i) => {
        expect(isClose(v, pd.fittedScore[i]! - pd.predictedScore[i]!, 1e-12, 0)).toBe(true);
      });
      expect(pd.isMse).toBeGreaterThanOrEqual(0);
      expect(pd.oosMse).toBeGreaterThanOrEqual(0);
      const expectedRatio = (pd.oosMse - pd.isMse) / pd.isMse;
      expect(isClose(pd.overfitRatio, expectedRatio, 1e-12, 0)).toBe(true);
    }
  });

  test("accepts a precomputed prediction", () => {
    const c = getCase("c1_cusl_123");
    const model = models.c1_cusl_123!;
    const pred = predictPls(model, {
      technique: predictDA,
      noFolds: c.noFolds,
      ordering: ordering(c),
    });
    const pd = predictiveDeviance(model, "CUSL", { predictModel: pred });
    expect(pd).not.toBeNull();
    expect([...pd!.pd]).toEqual([...pds.c1_cusl_123!.pd]);
    expect(pd!.isMse).toBe(pds.c1_cusl_123!.isMse);
  });
});

// --- deviance tree parity (on R pd_data inputs) ---------------------------------

describe("deviance tree parity", () => {
  for (const name of CASE_NAMES) {
    test(`tree outputs match R: ${name}`, () => {
      const c = getCase(name);
      for (const blk of c.trees) {
        const dt = devianceTree(rPds[name]!, blk.bounds);
        const label = `${name} bounds=${blk.bounds}`;
        assertVecClose(dt.sortedPd, blk.sortedPd, `${label}.sortedPd`);
        expect(Object.keys(dt.deviantGroups)).toEqual(blk.groups.labels);
        expect(Object.keys(dt.deviantGroups).map((g) => dt.groupRoots[g]!)).toEqual(
          blk.groups.roots,
        );
        blk.groups.labels.forEach((gl, i) => {
          expect([...dt.deviantGroups[gl]!]).toEqual(blk.groups.cases[i]!);
        });
        expect([...dt.uniqueDeviants]).toEqual(blk.uniqueDeviants);
        expect([...dt.deviantNodes]).toEqual(blk.deviantNodes);
      }
    });
  }

  test("tree structural invariants", () => {
    const dt = devianceTree(rPds.c1_cusl_123!);
    expect(dt.kind).toBe("coa_dtree");
    expect([...dt.sortedPd]).toEqual([...dt.sortedPd].sort((a, b) => b - a));
    const nLeaves = dt.tree.var.filter((v) => v === "<leaf>").length;
    expect(dt.sortedPd.length).toBe(nLeaves);
    const labels = Object.keys(dt.deviantGroups);
    expect(labels).toEqual(labels.map((_, i) => String.fromCharCode(65 + i)));
    expect(Object.keys(dt.groupRoots)).toEqual(labels);
    const grouped = new Set(Object.values(dt.deviantGroups).flat());
    for (const u of dt.uniqueDeviants) expect(grouped.has(u)).toBe(false);
  });
});

// --- rules and competing splits parity -------------------------------------------

function fullBlocks(): [string, TreeBlock][] {
  const out: [string, TreeBlock][] = [];
  for (const name of CASE_NAMES) {
    for (const blk of getCase(name).trees) {
      if (blk.rules !== undefined) out.push([name, blk]);
    }
  }
  return out;
}

describe("rules and competes parity", () => {
  test("group rules match R", () => {
    for (const [name, blk] of fullBlocks()) {
      const dt = devianceTree(rPds[name]!, blk.bounds);
      const rules = groupRules(dt) as Record<string, CoaRules>;
      expect(Object.keys(rules)).toEqual(blk.rules!.map((r) => r.group));
      for (const ref of blk.rules!) {
        const got = rules[ref.group]!;
        const label = `${name} group ${ref.group}`;
        expect([...got.construct]).toEqual(ref.construct);
        assertVecClose(got.gte, ref.gte, `${label}.gte`, RTOL, 0);
        assertVecClose(got.lt, ref.lt, `${label}.lt`, RTOL, 0);
      }
    }
  });

  test("competes match R", () => {
    for (const [name, blk] of fullBlocks()) {
      const dt = devianceTree(rPds[name]!, blk.bounds);
      const comp = competes(dt) as Record<string, CoaCompetes>;
      expect(Object.keys(comp)).toEqual(blk.competes!.map((c) => c.group));
      for (const ref of blk.competes!) {
        const got = comp[ref.group]!;
        const label = `${name} group ${ref.group}`;
        expect([...got.criterion]).toEqual(ref.criterion);
        expect([...got.sign]).toEqual(ref.sign);
        assertVecClose(got.value, ref.value, `${label}.value`, RTOL, 0);
        assertVecClose(got.improve, ref.improve, `${label}.improve`, RTOL, 0);
      }
    }
  });

  test("competes single-node form", () => {
    const c = getCase("c1_cusl_42");
    const blk = c.trees[0]!;
    const dt = devianceTree(rPds.c1_cusl_42!, blk.bounds);
    const ref = blk.competes![0]!;
    const got = competes(dt.groupRoots[ref.group]!, dt) as CoaCompetes;
    expect([...got.criterion]).toEqual(ref.criterion);
  });

  test("competes errors on root", () => {
    const dt = devianceTree(rPds.c1_cusl_123!);
    expect(() => competes(1, dt)).toThrow(/No splits before root/);
  });
});

// --- unstable parameters parity ----------------------------------------------------

describe("unstable parameters", () => {
  test("re-estimation diffs match R", () => {
    for (const [name, blk] of fullBlocks()) {
      const dt = devianceTree(rPds[name]!, blk.bounds);
      const params = blk.params!;
      const un = unstableParams(models[name]!, dt, params);
      expect(Object.keys(un.groups)).toEqual(blk.unstable!.map((u) => u.group));
      for (const ref of blk.unstable!) {
        const got = un.groups[ref.group]!;
        const label = `${name} group ${ref.group}`;
        expect([...got.cases]).toEqual(ref.cases);
        expect(Object.keys(got.paramDiffs)).toEqual(params);
        for (const param of params) {
          expectNamedClose(
            got.paramDiffs[param]!,
            toMatrix(ref.paramDiffs[param]!),
            1e-9,
            `${label} ${param}`,
          );
        }
      }
    }
  });

  test("empty groups yield an empty record", () => {
    const un = unstableParams(models.c1_cusl_123!, {});
    expect(un.kind).toBe("coa_unstable");
    expect(Object.keys(un.groups)).toEqual([]);
  });

  test("accepts a plain mapping like a dtree", () => {
    const dt = devianceTree(rPds.c1_cusl_123!);
    const viaDtree = unstableParams(models.c1_cusl_123!, dt);
    const viaMapping = unstableParams(models.c1_cusl_123!, { ...dt.deviantGroups });
    expect(Object.keys(viaDtree.groups)).toEqual(Object.keys(viaMapping.groups));
    const first = Object.keys(viaDtree.groups)[0]!;
    expectNamedClose(
      viaDtree.groups[first]!.paramDiffs["path_coef"]!,
      viaMapping.groups[first]!.paramDiffs["path_coef"]!,
      0,
      "mapping-equivalence",
    );
  });
});

// --- assessCoa integration -----------------------------------------------------------

describe("assessCoa integration", () => {
  test("composes the pipeline (tolerance-robust assertions)", () => {
    const c = getCase("c1_cusl_123");
    const blk = c.trees[0]!;
    const result = assessCoa(models.c1_cusl_123!, "CUSL", {
      params: blk.params!,
      noFolds: c.noFolds,
      ordering: ordering(c),
    });
    expect(result).not.toBeNull();
    expect(result!.kind).toBe("coa_analysis");
    expect(result!.focalConstruct).toBe("CUSL");
    expect(result!.devianceBounds).toEqual([0.025, 0.975]);
    expect([...result!.predictiveDeviance.pd]).toEqual([...pds.c1_cusl_123!.pd]);
    const gotGroups = result!.devianceTree.deviantGroups;
    expect(Object.keys(gotGroups)).toEqual(blk.groups.labels);
    expect(Object.values(gotGroups).map((cs) => cs.length)).toEqual(
      blk.groups.cases.map((cs) => cs.length),
    );
    expect(Object.keys(result!.unstable.groups)).toEqual(blk.groups.labels);
    expect(result!.plsModel).toBe(models.c1_cusl_123!);
  });

  test("accepts a precomputed prediction", () => {
    const c = getCase("c1_cusl_123");
    const model = models.c1_cusl_123!;
    const pred = predictPls(model, {
      technique: predictDA,
      noFolds: c.noFolds,
      ordering: ordering(c),
    });
    const result = assessCoa(model, "CUSL", { predictModel: pred });
    expect(result).not.toBeNull();
    assertVecClose(result!.predictiveDeviance.pd, c.pd, "precomputed.pd", RTOL, 0);
  });

  test("groupRules works from the analysis record", () => {
    const c = getCase("c1_cusl_123");
    const blk = c.trees[0]!;
    const model = models.c1_cusl_123!;
    const pred = predictPls(model, {
      technique: predictDA,
      noFolds: c.noFolds,
      ordering: ordering(c),
    });
    const result = assessCoa(model, "CUSL", { predictModel: pred })!;
    const allRules = groupRules(result) as Record<string, CoaRules>;
    expect(Object.keys(allRules)).toEqual(blk.groups.labels);
    const label = blk.rules![0]!.group;
    const single = groupRules(label, result) as CoaRules;
    expect(single).toEqual(allRules[label]!);
  });
});

// --- validation ------------------------------------------------------------------------

describe("validation", () => {
  test("rejects a non-seminr model with a warning", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessCoa({ not: "a_model" }, "X")).toBeNull();
      expect(predictiveDeviance({ not: "a_model" }, "X")).toBeNull();
      expect(
        warn.mock.calls.filter((cl) => String(cl[0]).includes("only works with SEMinR")).length,
      ).toBe(2);
    } finally {
      warn.mockRestore();
    }
  });

  test("rejects a HOC model", () => {
    const hocModel = estimateRegistryModel("C3");
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessCoa(hocModel, "CUSL")).toBeNull();
      expect(warn.mock.calls.some((cl) => String(cl[0]).includes("higher-order"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("errors on an invalid focal construct", () => {
    expect(() => assessCoa(models.c1_cusl_123!, "NONEXISTENT")).toThrow(
      /focal_construct 'NONEXISTENT' not found/,
    );
  });

  test("errors on invalid deviance bounds", () => {
    for (const bounds of [[0.975, 0.025], [-0.1, 1.2], [0.5]]) {
      expect(() =>
        assessCoa(models.c1_cusl_123!, "CUSL", { devianceBounds: bounds }),
      ).toThrow(/deviance_bounds/);
    }
  });

  test("errors on invalid params", () => {
    expect(() =>
      assessCoa(models.c1_cusl_123!, "CUSL", { params: "nonexistent_param" }),
    ).toThrow(/Invalid params: nonexistent_param/);
  });
});

// --- tree traversal helpers (R unit oracles) ---------------------------------------------

describe("tree traversal helpers", () => {
  test("pathTo", () => {
    expect(pathTo(468)).toEqual([1, 3, 7, 14, 29, 58, 117, 234, 468]);
    expect(pathTo(469)).toEqual([1, 3, 7, 14, 29, 58, 117, 234, 469]);
    expect(pathTo(1)).toEqual([1]);
  });

  test("mainAncestors removes descendants", () => {
    expect(mainAncestors(["4", "40", "81", "12", "24", "119", "239", "31"])).toEqual([
      "4",
      "40",
      "12",
      "119",
      "31",
    ]);
    expect(
      mainAncestors(["2", "12", "24", "204", "26", "29", "117", "469", "15", "31", "62", "124", "63"]),
    ).toEqual(["2", "12", "26", "29", "15"]);
    expect(mainAncestors(["5"])).toEqual(["5"]);
  });
});

// --- rendering ------------------------------------------------------------------------------

describe("rendering", () => {
  test("toString and summarize carry the R print surfaces", () => {
    const c = getCase("c1_cusl_123");
    const model = models.c1_cusl_123!;
    const pred = predictPls(model, {
      technique: predictDA,
      noFolds: c.noFolds,
      ordering: ordering(c),
    });
    const result = assessCoa(model, "CUSL", { predictModel: pred })!;
    const text = String(result);
    expect(text).toContain("Composite Overfit Analysis (COA)");
    expect(text).toContain("Focal construct: CUSL");
    expect(text).toContain("Overfit ratio");
    const summary = result.summarize();
    expect(summary).toContain("Composite Overfit Analysis (COA) Summary");
    expect(summary.includes("Max_Path_Diff") || summary.includes("Parameter Instability")).toBe(
      true,
    );
  });
});
