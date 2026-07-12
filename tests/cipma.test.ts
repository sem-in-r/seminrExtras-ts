/**
 * IPMA / cIPMA (Slice 6): assessIpma / assessCipma against R goldens.
 *
 * Parity fixtures (tests/fixtures/cipma/cipma.json) carry the R importance
 * (unstandardized/standardized total effects), 0-100 rescaled performance, the
 * four-way construct classification, and the embedded NCA effect sizes /
 * necessary predictors for eight registry-model cases (M1-M4). cIPMA is
 * deterministic here (ncaTestRep === 0), so parity holds at rel 1e-9 /
 * abs 1e-12 on the numeric vectors and exact on the rounded classification
 * table. Behavioral cases mirror the R suite (test-cipma-comprehensive.R).
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import type { NamedMatrix } from "@seminr/core";
import {
  assessCipma,
  assessIpma,
  checkPositiveWeights,
  computeTotalEffects,
  isInteractionConstruct,
  type CipmaAnalysis,
} from "../src/featureCipma.ts";
import { FIXTURES_DIR, toMatrix, type FixtureMatrixNode } from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

interface VecNode {
  names: string[];
  values: number[];
}

interface CipmaCase {
  name: string;
  modelId: string;
  entry: string;
  target: string;
  scaleMin: number;
  scaleMax: number;
  seed: number;
  constructs: string[];
  excludedInteractions: string[];
  negativeWeightConstructs: string[];
  importanceUnstd: VecNode;
  importanceStd: VecNode;
  performance: VecNode;
  classification: {
    construct: string[];
    importance: number[];
    performance: number[];
    highImportance: boolean[];
    necessary: boolean[];
    priority: string[];
  };
  ncaEffectSizes: FixtureMatrixNode | null;
  ncaNecessaryPredictors: string[] | null;
}

const fx = JSON.parse(
  readFileSync(join(FIXTURES_DIR, "cipma", "cipma.json"), "utf8"),
) as { cases: CipmaCase[] };

function runCase(c: CipmaCase): CipmaAnalysis {
  const model = estimateRegistryModel(c.modelId);
  const warn = spyOn(console, "warn").mockImplementation(() => {});
  try {
    const result =
      c.entry === "ipma"
        ? assessIpma(model, {
            target: c.target,
            scaleMin: c.scaleMin,
            scaleMax: c.scaleMax,
            seed: c.seed,
          })
        : assessCipma(model, {
            target: c.target,
            scaleMin: c.scaleMin,
            scaleMax: c.scaleMax,
            nca: true,
            ncaTestRep: 0,
            seed: c.seed,
          });
    expect(result).not.toBeNull();
    return result as CipmaAnalysis;
  } finally {
    warn.mockRestore();
  }
}

function isClose(a: number, b: number, rtol = 1e-9, atol = 1e-12): boolean {
  return Math.abs(a - b) <= Math.max(rtol * Math.max(Math.abs(a), Math.abs(b)), atol);
}

function assertVec(got: Record<string, number>, node: VecNode): void {
  expect(Object.keys(got)).toEqual(node.names);
  node.names.forEach((nm, i) => {
    if (!isClose(got[nm]!, node.values[i]!)) {
      throw new Error(`${nm}: got ${got[nm]}, expected ${node.values[i]}`);
    }
  });
}

function assertNamedClose(got: NamedMatrix, node: FixtureMatrixNode): void {
  const exp = toMatrix(node);
  expect(new Set(got.cols)).toEqual(new Set(exp.cols));
  expect(new Set(got.rows)).toEqual(new Set(exp.rows));
  for (let i = 0; i < exp.rows.length; i++) {
    for (let j = 0; j < exp.cols.length; j++) {
      const a = got.values[got.rows.indexOf(exp.rows[i]!)]![got.cols.indexOf(exp.cols[j]!)]!;
      const e = exp.values[i]![j]!;
      if (Number.isNaN(e)) {
        expect(Number.isNaN(a)).toBe(true);
      } else if (!isClose(a, e)) {
        throw new Error(`[${exp.rows[i]},${exp.cols[j]}]: got ${a}, expected ${e}`);
      }
    }
  }
}

describe("cIPMA R parity", () => {
  for (const [idx, c] of fx.cases.entries()) {
    test(`vectors match R golden: ${c.name} (case ${idx})`, () => {
      const r = runCase(c);
      expect(r.kind).toBe("cipma_analysis");
      expect([...r.constructs]).toEqual(c.constructs);
      expect([...r.excludedInteractions]).toEqual(c.excludedInteractions);
      expect([...r.negativeWeightConstructs]).toEqual(c.negativeWeightConstructs);
      assertVec(r.importanceUnstd, c.importanceUnstd);
      assertVec(r.importanceStd, c.importanceStd);
      assertVec(r.performance, c.performance);
    });

    test(`classification matches R golden: ${c.name} (case ${idx})`, () => {
      const r = runCase(c);
      const cls = c.classification;
      expect(r.classification.map((row) => row.construct)).toEqual(cls.construct);
      r.classification.forEach((row, i) => {
        expect(row.importance).toBe(cls.importance[i]!);
        expect(row.performance).toBe(cls.performance[i]!);
        expect(row.highImportance).toBe(cls.highImportance[i]!);
        expect(row.necessary).toBe(cls.necessary[i]!);
        expect(row.priority).toBe(cls.priority[i]!);
      });
    });

    test(`NCA block matches R golden: ${c.name} (case ${idx})`, () => {
      const r = runCase(c);
      if (c.entry === "ipma") {
        expect(r.nca).toBeNull();
        expect(c.ncaEffectSizes).toBeNull();
        return;
      }
      expect(r.nca).not.toBeNull();
      assertNamedClose(r.nca!.effectSizes, c.ncaEffectSizes!);
      expect([...r.nca!.necessaryPredictors]).toEqual(c.ncaNecessaryPredictors!);
    });
  }
});

describe("independent oracles (test-cipma-comprehensive.R)", () => {
  test("Image performance == weighted rescaled indicator means (M1, scale 1-10)", () => {
    const model = estimateRegistryModel("M1");
    const ow = model.outerWeights;
    const items = model.mmMatrix.constructItems("Image");
    const idx = items.map((it) => model.data.columns.indexOf(it));
    const w = items.map((it) => ow.values[ow.rows.indexOf(it)]![ow.cols.indexOf("Image")]!);
    const n = model.data.values.length;
    const means = idx.map((j) => model.data.values.reduce((s, row) => s + row[j]!, 0) / n);
    const num = w.reduce((s, wi, k) => s + wi * (((means[k]! - 1) / 9) * 100), 0);
    const expected = num / w.reduce((s, wi) => s + wi, 0);

    const r = assessCipma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10, nca: false });
    expect(r).not.toBeNull();
    expect(isClose(r!.performance["Image"]!, expected, 1e-10, 0)).toBe(true);
  });

  test("standardized importance == computeTotalEffects(pathCoef)[c, target] (M1)", () => {
    const model = estimateRegistryModel("M1");
    const pc = model.pathCoef;
    const total = computeTotalEffects(pc.values);
    const ti = pc.cols.indexOf("Loyalty");
    const r = assessCipma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10, nca: false });
    expect(r).not.toBeNull();
    for (const [c, imp] of Object.entries(r!.importanceStd)) {
      expect(isClose(imp, total[pc.rows.indexOf(c)]![ti]!, 1e-10, 0)).toBe(true);
    }
  });

  test("HOC performance chains through LOCs (M4 Quality)", () => {
    const model = estimateRegistryModel("M4");
    const ow = model.outerWeights;
    const locPerf = (loc: string): number => {
      const items = model.mmMatrix.constructItems(loc);
      const idx = items.map((it) => model.data.columns.indexOf(it));
      const w = items.map((it) => ow.values[ow.rows.indexOf(it)]![ow.cols.indexOf(loc)]!);
      const n = model.data.values.length;
      const means = idx.map((j) => model.data.values.reduce((s, row) => s + row[j]!, 0) / n);
      const num = w.reduce((s, wi, k) => s + wi * (((means[k]! - 1) / 9) * 100), 0);
      return num / w.reduce((s, wi) => s + wi, 0);
    };
    const locs = model.mmMatrix.constructItems("Quality");
    const hocW = locs.map((loc) => ow.values[ow.rows.indexOf(loc)]![ow.cols.indexOf("Quality")]!);
    const lp = locs.map(locPerf);
    const expected =
      hocW.reduce((s, wi, k) => s + wi * lp[k]!, 0) / hocW.reduce((s, wi) => s + wi, 0);

    const r = assessCipma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10, nca: false });
    expect(r).not.toBeNull();
    expect(isClose(r!.performance["Quality"]!, expected, 1e-10, 0)).toBe(true);
  });

  test("highImportance == strictly above nan-median", () => {
    const model = estimateRegistryModel("M1");
    const r = assessCipma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10, nca: false });
    expect(r).not.toBeNull();
    const imps = Object.values(r!.importanceUnstd).filter((v) => !Number.isNaN(v)).sort((a, b) => a - b);
    const mid = imps.length % 2 === 1
      ? imps[(imps.length - 1) / 2]!
      : (imps[imps.length / 2 - 1]! + imps[imps.length / 2]!) / 2;
    for (const row of r!.classification) {
      expect(row.highImportance).toBe(r!.importanceUnstd[row.construct]! > mid);
    }
  });
});

describe("IPMA/cIPMA equivalence + determinism", () => {
  test("assessIpma equals assessCipma with nca=false", () => {
    const model = estimateRegistryModel("M1");
    const a = assessIpma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10, seed: 42 });
    const b = assessCipma(model, {
      target: "Loyalty",
      scaleMin: 1,
      scaleMax: 10,
      nca: false,
      seed: 42,
    });
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    expect(a!.nca).toBeNull();
    expect(a!.importanceUnstd).toEqual(b!.importanceUnstd);
    expect(a!.importanceStd).toEqual(b!.importanceStd);
    expect(a!.performance).toEqual(b!.performance);
    expect(a!.classification).toEqual(b!.classification);
    expect([...a!.constructs]).toEqual([...b!.constructs]);
    expect(a!.scaleRange).toEqual(b!.scaleRange);
  });

  test("same inputs are deterministic", () => {
    const model = estimateRegistryModel("M1");
    const r1 = assessCipma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10, ncaTestRep: 0 });
    const r2 = assessCipma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10, ncaTestRep: 0 });
    expect(r1).not.toBeNull();
    expect(r2).not.toBeNull();
    expect(r1!.importanceUnstd).toEqual(r2!.importanceUnstd);
    expect(r1!.performance).toEqual(r2!.performance);
    expect(r1!.classification).toEqual(r2!.classification);
    expect(r1!.nca).not.toBeNull();
    expect(r1!.nca!.effectSizes.values).toEqual(r2!.nca!.effectSizes.values);
  });
});

describe("interaction exclusion (M3)", () => {
  test("interaction construct excluded from constructs and NCA predictors", () => {
    const model = estimateRegistryModel("M3");
    const r = assessCipma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10, ncaTestRep: 0 });
    expect(r).not.toBeNull();
    expect([...r!.excludedInteractions]).toContain("Image*Value");
    expect([...r!.constructs]).not.toContain("Image*Value");
    expect(r!.nca).not.toBeNull();
    expect([...r!.nca!.predictors]).not.toContain("Image*Value");
  });
});

describe("validation", () => {
  test("non-model warns and returns null", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessCipma("not a model", { target: "Loyalty" })).toBeNull();
      expect(warn.mock.calls.some((c) => String(c[0]).includes("SEMinR models"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("bad target throws", () => {
    const model = estimateRegistryModel("M1");
    expect(() => assessCipma(model, { target: "NonExistent" })).toThrow(
      /not found in model constructs/,
    );
  });

  test("scaleMin not less than scaleMax throws", () => {
    const model = estimateRegistryModel("M1");
    for (const [smin, smax] of [
      [10, 1],
      [5, 5],
    ] as const) {
      expect(() =>
        assessCipma(model, { target: "Loyalty", scaleMin: smin, scaleMax: smax }),
      ).toThrow(/scale_min must be less than scale_max/);
    }
  });

  test("non-numeric scale throws", () => {
    const model = estimateRegistryModel("M1");
    expect(() =>
      assessCipma(model, {
        target: "Loyalty",
        scaleMin: "a" as unknown as number,
        scaleMax: 10,
      }),
    ).toThrow(/single numeric values/);
  });

  test("non-model short-circuits before scale check", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(
        assessCipma("not a model", { target: "Loyalty", scaleMin: 10, scaleMax: 1 }),
      ).toBeNull();
      expect(warn.mock.calls.some((c) => String(c[0]).includes("SEMinR models"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });
});

describe("checkPositiveWeights (duck-typed fakes)", () => {
  const fakeModel = (
    items: Record<string, string[]>,
    rows: string[],
    cols: string[],
    values: number[][],
  ) => ({
    mmMatrix: { constructItems: (c: string) => items[c]! },
    outerWeights: { rows, cols, values },
  });

  test("flags a construct with a negative indicator weight", () => {
    const model = fakeModel(
      { A: ["a1", "a2"], B: ["b1"] },
      ["a1", "a2", "b1"],
      ["A", "B"],
      [
        [0.5, 0],
        [-0.3, 0],
        [0, 0.7],
      ],
    );
    expect(checkPositiveWeights(model, ["A", "B"])).toEqual(["A"]);
    expect(checkPositiveWeights(model, ["B"])).toEqual([]);
  });

  test("flags a HOC whose LOC has a negative indicator weight", () => {
    const model = fakeModel(
      { H: ["L"], L: ["l1", "l2"] },
      ["L", "l1", "l2"],
      ["H", "L"],
      [
        [0.9, 0],
        [0, 0.6],
        [0, -0.2],
      ],
    );
    expect(checkPositiveWeights(model, ["H"])).toEqual(["H"]);
  });
});

describe("record surface / rendering", () => {
  test("cIPMA and IPMA render with correct headings", () => {
    const cipma = runCase(fx.cases[0]!);
    expect(cipma.kind).toBe("cipma_analysis");
    expect(cipma.scaleRange).toEqual([1, 10]);
    const text = String(cipma);
    expect(text).toContain("cIPMA");
    expect(text).toContain("Top priority");
    expect(cipma.summarize()).toContain("Bottleneck");

    const ipma = runCase(fx.cases[1]!);
    const itext = String(ipma);
    expect(itext).toContain("Importance-Performance Map Analysis (IPMA)");
    expect(itext).not.toContain("cIPMA");
  });

  test("isInteractionConstruct", () => {
    expect(isInteractionConstruct("Image*Value")).toBe(true);
    expect(isInteractionConstruct("Image")).toBe(false);
  });
});
