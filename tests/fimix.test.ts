/**
 * FIMIX-PLS (Slice 9): assessFimix / assessFimixCompare against R goldens.
 *
 * Parity fixture tests/fixtures/fimix/fimix.json carries, per case, the nstart
 * NORMALIZED initial posterior matrices R's init_random_posteriors() produced
 * under the seed (the feature's only RNG — run_fimix_em consumes none),
 * injected here via `inits`. EM from identical inits is deterministic, so all
 * outputs are pinned: proportions, hard sizes/assignments (exact ints),
 * posteriors, segment path/intercept/variance parameters, log-likelihood,
 * information criteria, convergence flags and iteration counts.
 *
 * Compare cases re-seed identically per K (R calls set.seed inside
 * assess_fimix), so `inits` is a per-K mapping.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import {
  assessFimix,
  assessFimixCompare,
  computeEntropy,
  computeFimixCriteria,
  countFimixParameters,
  extractStructuralEquations,
  logSumExp,
  type FimixAnalysis,
} from "../src/featureFimix.ts";
import {
  FIXTURES_DIR,
  expectNamedClose,
  toMatrix,
  type FixtureMatrixNode,
} from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

const RTOL = 1e-9;
const ATOL = 1e-12;

const CASE_NAMES = ["m1_k2_s123", "m1_k3_s42", "m3_k2_s123", "m4_k2_s123"];
const MODEL_ID: Record<string, string> = {
  m1_k2_s123: "M1",
  m1_k3_s42: "M1",
  m3_k2_s123: "M3",
  m4_k2_s123: "M4",
};

const IC_KEYS = ["lnL", "AIC", "AIC3", "AIC4", "BIC", "CAIC", "HQ", "MDL5", "EN"];

interface FimixCase {
  name: string;
  K: number;
  nstart: number;
  maxIter: number;
  seed: number;
  inits: FixtureMatrixNode[];
  nParameters: number;
  iterations: number;
  segmentAssignment: number[];
  segmentSizes: number[];
  logLikelihood: number;
  segmentProportions: Record<string, number>;
  infoCriteria: Record<string, number>;
  posterior: FixtureMatrixNode;
  segmentPaths: FixtureMatrixNode[];
  segmentIntercepts: Record<string, number>[];
  segmentVariances: FixtureMatrixNode;
}

const fx = JSON.parse(readFileSync(join(FIXTURES_DIR, "fimix", "fimix.json"), "utf8")) as {
  cases: FimixCase[];
  compareCases: {
    KRange: number[];
    nstart: number;
    maxIter: number;
    seed: number;
    inits: Record<string, FixtureMatrixNode[]>;
    fitTable: FixtureMatrixNode;
    solutions: Record<
      string,
      { iterations: number; segmentSizes: number[]; logLikelihood: number }
    >;
  }[];
  nonConvergence: { model: string; K: number; nstart: number; maxIter: number; seed: number };
};

function getCase(name: string): FimixCase {
  return fx.cases.find((c) => c.name === name)!;
}

const models: Record<string, ReturnType<typeof estimateRegistryModel>> = {};
for (const [caseName, modelId] of Object.entries(MODEL_ID)) {
  models[modelId] ??= estimateRegistryModel(modelId);
  models[caseName] = models[modelId]!;
}
models.M2 = estimateRegistryModel("M2");

function inits(raw: FixtureMatrixNode[]): number[][][] {
  return raw.map((node) => toMatrix(node).values);
}

const results: Record<string, FimixAnalysis> = {};
for (const name of CASE_NAMES) {
  const c = getCase(name);
  results[name] = assessFimix(models[name]!, {
    K: c.K,
    nstart: c.nstart,
    maxIter: c.maxIter,
    seed: c.seed,
    inits: inits(c.inits),
  })!;
}

function isClose(a: number, b: number, rtol = RTOL, atol = ATOL): boolean {
  return Math.abs(a - b) <= Math.max(rtol * Math.max(Math.abs(a), Math.abs(b)), atol);
}

describe("assessFimix R parity", () => {
  for (const name of CASE_NAMES) {
    test(`segment structure: ${name}`, () => {
      const c = getCase(name);
      const res = results[name]!;
      expect(res.kind).toBe("fimix_analysis");
      expect(res.k).toBe(c.K);
      expect(res.converged).toBe(true);
      expect(res.nStartsCompleted).toBe(c.nstart);
      expect(res.nParameters).toBe(c.nParameters);
      expect(res.iterations).toBe(c.iterations);
      expect([...res.segmentAssignment]).toEqual(c.segmentAssignment);
      expect(Object.values(res.segmentSizes)).toEqual(c.segmentSizes);
      expect(Object.keys(res.segmentSizes)).toEqual(
        Array.from({ length: c.K }, (_, k) => `Segment_${k + 1}`),
      );
    });

    test(`proportions, likelihood, criteria: ${name}`, () => {
      const c = getCase(name);
      const res = results[name]!;
      expect(isClose(res.logLikelihood, c.logLikelihood)).toBe(true);
      expect(Object.keys(res.segmentProportions)).toEqual(Object.keys(c.segmentProportions));
      for (const [seg, value] of Object.entries(c.segmentProportions)) {
        expect(isClose(res.segmentProportions[seg]!, value)).toBe(true);
      }
      expect(Object.keys(res.infoCriteria)).toEqual(IC_KEYS);
      for (const [key, value] of Object.entries(c.infoCriteria)) {
        if (!isClose(res.infoCriteria[key]!, value)) {
          throw new Error(`${name} IC ${key}: got ${res.infoCriteria[key]}, expected ${value}`);
        }
      }
    });

    test(`posterior matrix: ${name}`, () => {
      const c = getCase(name);
      const res = results[name]!;
      const expected = toMatrix(c.posterior);
      expect([...res.posterior.cols]).toEqual(
        Array.from({ length: c.K }, (_, k) => `Segment_${k + 1}`),
      );
      for (let i = 0; i < expected.values.length; i++) {
        for (let j = 0; j < c.K; j++) {
          if (!isClose(res.posterior.values[i]![j]!, expected.values[i]![j]!)) {
            throw new Error(`${name} posterior[${i},${j}]`);
          }
        }
      }
    });

    test(`segment parameters: ${name}`, () => {
      const c = getCase(name);
      const res = results[name]!;
      expect(res.segmentPaths.length).toBe(c.K);
      for (let k = 0; k < c.K; k++) {
        expectNamedClose(
          res.segmentPaths[k]!,
          toMatrix(c.segmentPaths[k]!),
          1e-9,
          `${name} segmentPaths[${k}]`,
        );
        const expectedInts = c.segmentIntercepts[k]!;
        expect(Object.keys(res.segmentIntercepts[k]!)).toEqual(Object.keys(expectedInts));
        for (const [eq, value] of Object.entries(expectedInts)) {
          expect(isClose(res.segmentIntercepts[k]![eq]!, value)).toBe(true);
        }
      }
      expectNamedClose(
        res.segmentVariances,
        toMatrix(c.segmentVariances),
        1e-9,
        `${name} segmentVariances`,
      );
    });
  }
});

describe("assessFimixCompare R parity", () => {
  test("fit table and per-K solutions", () => {
    const c = fx.compareCases[0]!;
    const comp = assessFimixCompare(models.M2!, {
      KRange: c.KRange,
      nstart: c.nstart,
      maxIter: c.maxIter,
      seed: c.seed,
      inits: Object.fromEntries(c.KRange.map((k) => [k, inits(c.inits[`K${k}`]!)])),
    });
    expect(comp).not.toBeNull();
    expect(comp!.kind).toBe("fimix_comparison");
    expect([...comp!.kRange]).toEqual(c.KRange);
    expect(Object.keys(comp!.solutions)).toEqual(c.KRange.map((k) => `K${k}`));

    const expectedFit = toMatrix(c.fitTable);
    expect([...comp!.fitTable.cols]).toEqual([...expectedFit.cols]);
    for (let i = 0; i < expectedFit.values.length; i++) {
      for (let j = 0; j < expectedFit.cols.length; j++) {
        const e = expectedFit.values[i]![j]!;
        const a = comp!.fitTable.values[i]![j]!;
        if (Number.isNaN(e)) expect(Number.isNaN(a)).toBe(true);
        else if (!isClose(a, e)) throw new Error(`fitTable[${i},${j}]: got ${a}, expected ${e}`);
      }
    }

    for (const k of c.KRange) {
      const sol = comp!.solutions[`K${k}`]!;
      const expectedSol = c.solutions[`K${k}`]!;
      expect(sol).not.toBeNull();
      expect(sol.iterations).toBe(expectedSol.iterations);
      expect(Object.values(sol.segmentSizes)).toEqual(expectedSol.segmentSizes);
      expect(isClose(sol.logLikelihood, expectedSol.logLikelihood)).toBe(true);
    }
  });
});

describe("structural invariants", () => {
  for (const name of CASE_NAMES) {
    test(`probability invariants: ${name}`, () => {
      const res = results[name]!;
      const propSum = Object.values(res.segmentProportions).reduce((a, b) => a + b, 0);
      expect(Math.abs(propSum - 1)).toBeLessThan(1e-10);
      expect(Object.values(res.segmentProportions).every((p) => p > 0)).toBe(true);
      for (const row of res.posterior.values) {
        expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-10);
        for (const v of row) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(1);
        }
      }
      expect(Object.values(res.segmentSizes).reduce((a, b) => a + b, 0)).toBe(res.nObs);
      const argmax = res.posterior.values.map((row) => {
        let arg = 0;
        for (let s = 1; s < row.length; s++) if (row[s]! > row[arg]!) arg = s;
        return arg + 1;
      });
      expect([...res.segmentAssignment]).toEqual(argmax);
      for (const row of res.segmentVariances.values) for (const v of row) expect(v).toBeGreaterThan(0);
    });

    test(`segment paths share the pooled zero structure: ${name}`, () => {
      const modelPaths = models[name]!.pathCoef;
      for (const segPaths of results[name]!.segmentPaths) {
        expect([...segPaths.rows]).toEqual([...modelPaths.rows]);
        expect([...segPaths.cols]).toEqual([...modelPaths.cols]);
        for (let i = 0; i < modelPaths.rows.length; i++) {
          for (let j = 0; j < modelPaths.cols.length; j++) {
            expect(segPaths.values[i]![j] !== 0).toBe(modelPaths.values[i]![j] !== 0);
          }
        }
      }
    });
  }
});

describe("behavioral spec", () => {
  test("non-model warns and returns null", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessFimix("not a model")).toBeNull();
      expect(assessFimixCompare([1, 2, 3])).toBeNull();
      expect(
        warn.mock.calls.filter((c) => String(c[0]).includes("only works with SEMinR")).length,
      ).toBe(2);
    } finally {
      warn.mockRestore();
    }
  });

  test("invalid K throws", () => {
    for (const badK of [1, 0, -2, 2.5]) {
      expect(() => assessFimix(models.M2!, { K: badK })).toThrow(/K must be an integer >= 2\./);
    }
  });

  test("invalid nstart throws", () => {
    expect(() => assessFimix(models.M2!, { K: 2, nstart: 0 })).toThrow(
      /nstart must be an integer >= 1\./,
    );
  });

  test("invalid K_range throws", () => {
    for (const badRange of [[1, 2], [0], [-3, 2], []]) {
      expect(() => assessFimixCompare(models.M2!, { KRange: badRange })).toThrow(
        /K_range must be a vector of integers >= 2\./,
      );
    }
  });

  test("non-convergence warns and returns null", () => {
    const nc = fx.nonConvergence;
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      const res = assessFimix(models[nc.model]!, {
        K: nc.K,
        nstart: nc.nstart,
        maxIter: nc.maxIter,
        seed: nc.seed,
      });
      expect(res).toBeNull();
      expect(warn.mock.calls.some((c) => String(c[0]).includes("did not converge for K = 2"))).toBe(
        true,
      );
    } finally {
      warn.mockRestore();
    }
  });

  test("small sample warns (K = 26)", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      const res = assessFimix(models.M1!, { K: 26, nstart: 1, maxIter: 1, seed: 123 });
      expect(res).toBeNull();
      const messages = warn.mock.calls.map((c) => String(c[0]));
      expect(messages.some((m) => m.includes("may be too small for K = 26"))).toBe(true);
      expect(messages.some((m) => m.includes("did not converge"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("default path is seed-reproducible", () => {
    const a = assessFimix(models.M2!, { K: 2, nstart: 2, maxIter: 200, seed: 7 });
    const b = assessFimix(models.M2!, { K: 2, nstart: 2, maxIter: 200, seed: 7 });
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    expect(a!.logLikelihood).toBe(b!.logLikelihood);
    expect([...a!.segmentAssignment]).toEqual([...b!.segmentAssignment]);
  });
});

describe("internals", () => {
  test("logSumExp", () => {
    const x = [1, 2, 3];
    const direct = Math.log(x.reduce((s, v) => s + Math.exp(v), 0));
    expect(isClose(logSumExp(x), direct, 1e-12, 0)).toBe(true);
    expect(isClose(logSumExp([1000, 1000]), 1000 + Math.log(2), 1e-12, 0)).toBe(true);
    expect(logSumExp([-Infinity, -Infinity])).toBe(-Infinity);
  });

  test("extractStructuralEquations order (M1)", () => {
    const eqs = extractStructuralEquations(models.M1!);
    expect([...eqs.keys()]).toEqual(["Expectation", "Satisfaction", "Loyalty", "Value"]);
    const sat = eqs.get("Satisfaction")!;
    expect([...sat.predictors]).toEqual(["Image", "Expectation", "Value"]);
    const n = models.M1!.constructScores.values.length;
    expect(sat.y.length).toBe(n);
    expect(sat.x.length).toBe(n);
    expect(sat.x[0]!.length).toBe(4); // intercept + 3 predictors
    for (const row of sat.x) expect(row[0]).toBe(1);
  });

  test("countFimixParameters (M1)", () => {
    const eqs = extractStructuralEquations(models.M1!);
    expect(countFimixParameters(eqs, 2)).toBe(31);
    expect(countFimixParameters(eqs, 3)).toBe(47);
  });

  test("computeEntropy bounds", () => {
    const hard = [
      [1, 0],
      [0, 1],
      [1, 0],
    ];
    expect(Math.abs(computeEntropy(hard, 2) - 1)).toBeLessThan(1e-12);
    const uniform = Array.from({ length: 4 }, () => [0.5, 0.5]);
    expect(Math.abs(computeEntropy(uniform, 2))).toBeLessThan(1e-12);
    expect(Number.isNaN(computeEntropy(hard.map((r) => [r[0]!]), 1))).toBe(true);
  });

  test("computeFimixCriteria formulas", () => {
    const post = Array.from({ length: 10 }, () => [0.5, 0.5]);
    const ic = computeFimixCriteria(-100, 5, 10, 2, post);
    expect(Object.keys(ic)).toEqual(IC_KEYS);
    expect(ic["AIC"]).toBeCloseTo(210, 10);
    expect(ic["AIC3"]).toBeCloseTo(215, 10);
    expect(ic["AIC4"]).toBeCloseTo(220, 10);
    expect(ic["BIC"]).toBeCloseTo(200 + 5 * Math.log(10), 10);
    expect(ic["CAIC"]).toBeCloseTo(200 + 5 * (Math.log(10) + 1), 10);
    expect(ic["HQ"]).toBeCloseTo(200 + 10 * Math.log(Math.log(10)), 10);
    expect(ic["MDL5"]).toBeCloseTo(200 + 2.5 * Math.log(10), 10);
    expect(Math.abs(ic["EN"]!)).toBeLessThan(1e-12);
  });
});

describe("rendering", () => {
  test("toString and summarize smoke", () => {
    const res = results.m1_k2_s123!;
    const text = String(res);
    expect(text).toContain("FIMIX-PLS Analysis");
    expect(text).toContain("Segments: 2");
    expect(text).toContain("Segment Proportions");
    const summary = res.summarize();
    expect(summary).toContain("FIMIX-PLS Analysis Summary");
    expect(summary).toContain("Free parameters: 31");

    const c = fx.compareCases[0]!;
    const comp = assessFimixCompare(models.M2!, {
      KRange: c.KRange,
      nstart: c.nstart,
      maxIter: c.maxIter,
      seed: c.seed,
      inits: Object.fromEntries(c.KRange.map((k) => [k, inits(c.inits[`K${k}`]!)])),
    })!;
    const ctext = String(comp);
    expect(ctext).toContain("FIMIX-PLS Segment Selection");
    expect(ctext).toContain("Best K by criterion");
    expect(comp.summarize()).toContain("FIMIX-PLS Comparison Summary");
  });
});
