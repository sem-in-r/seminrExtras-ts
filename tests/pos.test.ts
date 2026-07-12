/**
 * PLS-POS (Slice 10): assessPos / assessPosCompare against R goldens.
 *
 * Parity fixture tests/fixtures/pos/pos.json carries, per case, the nstart
 * 1-based initial partitions R's random_partition() drew (the feature's only
 * RNG), injected here via `partitions`. Given identical initial partitions
 * every downstream stage (rerun estimation, distances, hill climb) is
 * deterministic, so assignments, sizes, objectives, R² and paths are pinned;
 * `start1` additionally pins the first start's distance matrix and
 * post-distance reassignment stage.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import {
  assessPos,
  assessPosCompare,
  buildCandidateList,
  computePosDistances,
  computePosObjective,
  computeStructuralResiduals,
  estimateSegmentModels,
  posEndogenous,
  posSegments,
  type PosAnalysis,
  type PosComparison,
} from "../src/featurePos.ts";
import {
  FIXTURES_DIR,
  expectNamedClose,
  toMatrix,
  type FixtureMatrixNode,
} from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

const RTOL = 1e-9;
const ATOL = 1e-12;

const CASE_NAMES = ["m1_k2_s123", "m2_k2_s42", "m3_k2_s42", "m4_k2_s42"];
const MODEL_ID: Record<string, string> = {
  m1_k2_s123: "M1",
  m2_k2_s42: "M2",
  m3_k2_s42: "M3",
  m4_k2_s42: "M4",
};

interface PosCase {
  name: string;
  K: number;
  nstart: number;
  maxIter: number;
  seed: number;
  minSegmentSize: number;
  partitions: number[][];
  start1: { distances: FixtureMatrixNode; postDistanceAssignment: number[] };
  segmentAssignment: number[];
  segmentSizes: number[];
  segmentRsquared: FixtureMatrixNode;
  segmentPaths: FixtureMatrixNode[];
  objective: number;
  allObjectives: (number | string | null)[];
  converged: boolean;
  iterations: number;
  endogenous: string[];
  nObs: number;
}

const fx = JSON.parse(readFileSync(join(FIXTURES_DIR, "pos", "pos.json"), "utf8")) as {
  cases: PosCase[];
  compareCases: {
    name: string;
    KRange: number[];
    nstart: number;
    maxIter: number;
    seed: number;
    partitions: Record<string, number[][]>;
    fitTable: {
      K: number;
      sumR2: number | null;
      avgR2PerSegment: number | null;
      converged: boolean | null;
      iterations: number | null;
    }[];
    solutions: Record<
      string,
      { segmentAssignment: number[]; objective: number; iterations: number }
    >;
  }[];
};

function getCase(name: string): PosCase {
  return fx.cases.find((c) => c.name === name)!;
}

const models: Record<string, ReturnType<typeof estimateRegistryModel>> = {};
for (const [caseName, modelId] of Object.entries(MODEL_ID)) {
  models[modelId] ??= estimateRegistryModel(modelId);
  models[caseName] = models[modelId]!;
}

const results: Record<string, PosAnalysis> = {};
for (const name of CASE_NAMES) {
  const c = getCase(name);
  results[name] = assessPos(models[name]!, {
    K: c.K,
    nstart: c.nstart,
    maxIter: c.maxIter,
    seed: c.seed,
    partitions: c.partitions,
  })!;
}

function isClose(a: number, b: number, rtol = RTOL, atol = ATOL): boolean {
  return Math.abs(a - b) <= Math.max(rtol * Math.max(Math.abs(a), Math.abs(b)), atol);
}

describe("assessPos R parity", () => {
  for (const name of CASE_NAMES) {
    test(`structure: ${name}`, () => {
      const c = getCase(name);
      const res = results[name]!;
      expect(res.kind).toBe("pos_analysis");
      expect(res.k).toBe(c.K);
      expect(res.nstart).toBe(c.nstart);
      expect(res.nObs).toBe(c.nObs);
      expect(res.converged).toBe(c.converged);
      expect(res.iterations).toBe(c.iterations);
      expect([...res.endogenous]).toEqual(c.endogenous);
    });

    test(`assignment and sizes: ${name}`, () => {
      const c = getCase(name);
      const res = results[name]!;
      expect([...res.segmentAssignment]).toEqual(c.segmentAssignment);
      expect(Object.values(res.segmentSizes)).toEqual(c.segmentSizes);
      expect(Object.keys(res.segmentSizes)).toEqual(
        Array.from({ length: c.K }, (_, k) => `Segment ${k + 1}`),
      );
    });

    test(`objective: ${name}`, () => {
      const c = getCase(name);
      const res = results[name]!;
      expect(isClose(res.objective, c.objective)).toBe(true);
      expect(res.allObjectives.length).toBe(c.nstart);
      res.allObjectives.forEach((got, i) => {
        const ref = c.allObjectives[i]!;
        if (ref === null || ref === "-Inf") expect(got).toBe(-Infinity);
        else expect(isClose(got, ref as number)).toBe(true);
      });
    });

    test(`segment R-squared: ${name}`, () => {
      const c = getCase(name);
      expectNamedClose(
        results[name]!.segmentRsquared,
        toMatrix(c.segmentRsquared),
        1e-9,
        `${name} segmentRsquared`,
      );
    });

    test(`segment paths: ${name}`, () => {
      const c = getCase(name);
      const res = results[name]!;
      expect(res.segmentPaths.length).toBe(c.K);
      c.segmentPaths.forEach((node, seg) => {
        expectNamedClose(res.segmentPaths[seg]!, toMatrix(node), 1e-9, `${name} paths[${seg}]`);
      });
    });

    test(`start-1 stage parity: ${name}`, () => {
      const c = getCase(name);
      const model = models[name]!;
      const endo = posEndogenous(model);
      expect(endo).toEqual(c.endogenous);
      const segModels = estimateSegmentModels(model, c.partitions[0]!, c.K);
      expect(segModels).not.toBeNull();
      const sqResid = computeStructuralResiduals(model, segModels!, endo);
      const distances = computePosDistances(sqResid);
      const expected = toMatrix(c.start1.distances);
      for (let i = 0; i < expected.values.length; i++) {
        for (let j = 0; j < c.K; j++) {
          if (!isClose(distances[i]![j]!, expected.values[i]![j]!)) {
            throw new Error(`${name} distances[${i},${j}]`);
          }
        }
      }
      const post = distances.map((row) => {
        let arg = 0;
        for (let s = 1; s < c.K; s++) if (row[s]! < row[arg]!) arg = s;
        return arg + 1;
      });
      expect(post).toEqual(c.start1.postDistanceAssignment);
    });
  }
});

describe("assessPosCompare R parity", () => {
  const cc = fx.compareCases[0]!;
  const compareResult = assessPosCompare(models.M1!, {
    KRange: cc.KRange,
    nstart: cc.nstart,
    maxIter: cc.maxIter,
    seed: cc.seed,
    partitions: Object.fromEntries(
      Object.entries(cc.partitions).map(([label, parts]) => [Number(label.slice(1)), parts]),
    ),
  }) as PosComparison;

  test("structure", () => {
    expect(compareResult.kind).toBe("pos_comparison");
    expect([...compareResult.kRange]).toEqual(cc.KRange);
    expect(Object.keys(compareResult.solutions)).toEqual(Object.keys(cc.solutions));
  });

  test("fit table", () => {
    expect(compareResult.fitTable.length).toBe(cc.fitTable.length);
    compareResult.fitTable.forEach((row, i) => {
      const ref = cc.fitTable[i]!;
      expect(row.k).toBe(ref.K);
      if (ref.sumR2 === null) expect(Number.isNaN(row.sumR2)).toBe(true);
      else expect(isClose(row.sumR2, ref.sumR2)).toBe(true);
      if (ref.avgR2PerSegment === null) expect(Number.isNaN(row.avgR2PerSegment)).toBe(true);
      else expect(isClose(row.avgR2PerSegment, ref.avgR2PerSegment)).toBe(true);
      expect(row.converged).toBe(ref.converged);
      expect(row.iterations).toBe(ref.iterations);
    });
  });

  test("per-K solutions", () => {
    for (const [label, ref] of Object.entries(cc.solutions)) {
      const sol = compareResult.solutions[label]!;
      expect(sol).not.toBeNull();
      expect([...sol.segmentAssignment]).toEqual(ref.segmentAssignment);
      expect(isClose(sol.objective, ref.objective)).toBe(true);
      expect(sol.iterations).toBe(ref.iterations);
    }
  });
});

describe("invariants", () => {
  test("sizes match assignment", () => {
    const res = results.m1_k2_s123!;
    const counts = new Array(res.k).fill(0);
    for (const a of res.segmentAssignment) counts[a - 1] += 1;
    expect(Object.values(res.segmentSizes)).toEqual(counts);
    expect(Object.values(res.segmentSizes).reduce((a, b) => a + b, 0)).toBe(res.nObs);
    expect(Object.values(res.segmentSizes).every((s) => s > 0)).toBe(true);
    expect(res.segmentAssignment.every((a) => a >= 1 && a <= res.k)).toBe(true);
  });

  test("objective equals R-squared sum", () => {
    const res = results.m1_k2_s123!;
    const total = res.segmentRsquared.values
      .flat()
      .filter((v) => !Number.isNaN(v))
      .reduce((a, b) => a + b, 0);
    expect(isClose(res.objective, total, 1e-9, 0)).toBe(true);
  });

  test("best objective is the max finite start objective", () => {
    const res = results.m1_k2_s123!;
    const finite = res.allObjectives.filter(Number.isFinite);
    expect(res.objective).toBe(Math.max(...finite));
  });

  test("segment paths share the global nonzero structure", () => {
    const res = results.m1_k2_s123!;
    const globalPaths = models.M1!.pathCoef;
    for (const segPaths of res.segmentPaths) {
      expect([...segPaths.rows]).toEqual([...globalPaths.rows]);
      expect([...segPaths.cols]).toEqual([...globalPaths.cols]);
      for (let i = 0; i < globalPaths.rows.length; i++) {
        for (let j = 0; j < globalPaths.cols.length; j++) {
          if (globalPaths.values[i]![j] !== 0) expect(segPaths.values[i]![j]).not.toBe(0);
        }
      }
    }
  });

  test("R-squared values are within [0, 1]", () => {
    for (const row of results.m1_k2_s123!.segmentRsquared.values) {
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
    }
  });

  test("moderation endogenous excludes the interaction; HOC stays a predictor", () => {
    const m3 = results.m3_k2_s42!;
    expect(m3.endogenous.every((nm) => !nm.includes("*"))).toBe(true);
    expect(m3.endogenous).toContain("Satisfaction");
    expect(m3.endogenous).toContain("Loyalty");
    expect(m3.segmentPaths[0]!.rows).toContain("Image*Value");

    const m4 = results.m4_k2_s42!;
    expect(m4.segmentPaths[0]!.rows).toContain("Quality");
    expect(m4.endogenous).toContain("Satisfaction");
    expect(m4.endogenous).toContain("Loyalty");
  });

  test("segment models are estimated PLS models", () => {
    const res = results.m1_k2_s123!;
    expect(res.segmentModels.length).toBe(res.k);
    for (const m of res.segmentModels) {
      expect(m.pathCoef).toBeDefined();
      expect(m.rSquared).toBeDefined();
    }
  });
});

describe("posSegments", () => {
  test("returns the segment models", () => {
    const segs = posSegments(results.m1_k2_s123!);
    expect(segs).toBe(results.m1_k2_s123!.segmentModels);
    expect(segs.length).toBe(2);
  });

  test("rejects a non-pos_analysis input", () => {
    expect(() => posSegments("not_pos" as unknown as PosAnalysis)).toThrow(
      /pos_analysis object/,
    );
  });
});

describe("validation and failure paths", () => {
  test("non-model warns and returns null", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessPos("not a model", { K: 2 })).toBeNull();
      expect(assessPosCompare("not a model")).toBeNull();
      expect(
        warn.mock.calls.filter((c) => String(c[0]).includes("only works with SEMinR")).length,
      ).toBe(2);
    } finally {
      warn.mockRestore();
    }
  });

  test("invalid K throws", () => {
    for (const badK of [1, 0, -2, 2.5]) {
      expect(() => assessPos(models.M1!, { K: badK })).toThrow(/K must be an integer >= 2\./);
    }
  });

  test("sample too small throws", () => {
    expect(() => assessPos(models.M1!, { K: 50, minSegmentSize: 20 })).toThrow(/too small/);
  });

  test("partitions length mismatch throws", () => {
    const n = models.M1!.constructScores.values.length;
    const onePartition = [Array.from({ length: n }, (_, i) => (i % 2) + 1)];
    expect(() => assessPos(models.M1!, { K: 2, nstart: 3, partitions: onePartition })).toThrow(
      /partitions must supply exactly nstart/,
    );
  });

  test("all starts failing throws", () => {
    const model = models.M1!;
    const n = model.constructScores.values.length;
    const balanced = Array.from({ length: n }, (_, i) => (i % 2) + 1);
    expect(() =>
      assessPos(model, {
        K: 2,
        nstart: 1,
        minSegmentSize: Math.floor(n / 2),
        partitions: [balanced],
      }),
    ).toThrow(/failed to find a valid segmentation/);
  });

  test("default path is seed-reproducible", () => {
    const opts = { K: 2, nstart: 1, maxIter: 1, searchDepth: 1, seed: 7 };
    const r1 = assessPos(models.M2!, opts)!;
    const r2 = assessPos(models.M2!, opts)!;
    expect(r1.objective).toBe(r2.objective);
    expect([...r1.segmentAssignment]).toEqual([...r2.segmentAssignment]);
    expect(Object.values(r1.segmentSizes)).toEqual(Object.values(r2.segmentSizes));
  });
});

describe("internals", () => {
  test("computePosObjective handles null models", () => {
    const res = results.m1_k2_s123!;
    expect(computePosObjective([null, null], res.endogenous)).toBe(-Infinity);
    const obj = computePosObjective([...res.segmentModels], res.endogenous);
    expect(Number.isFinite(obj)).toBe(true);
    expect(obj).toBeGreaterThan(0);
  });

  test("computePosDistances shape and non-negativity", () => {
    const res = results.m1_k2_s123!;
    const sq = computeStructuralResiduals(models.M1!, [...res.segmentModels], res.endogenous);
    const dists = computePosDistances(sq);
    const n = models.M1!.constructScores.values.length;
    expect(dists.length).toBe(n);
    expect(dists[0]!.length).toBe(2);
    for (const row of dists) for (const v of row) expect(v).toBeGreaterThanOrEqual(0);
  });

  test("buildCandidateList orders by improvement, converges to null", () => {
    const distances = [
      [0.5, 0.1], // improvement 0.4 -> first
      [0.2, 0.4], // current is best -> excluded
      [0.3, 0.2], // improvement 0.1
      [0.6, 0.3], // improvement 0.3
    ];
    const candidates = buildCandidateList(distances, [1, 1, 1, 1], 2)!;
    expect(candidates.map((c) => c.obs)).toEqual([0, 3, 2]);
    expect(candidates.every((c) => c.toK === 2 && c.fromK === 1)).toBe(true);
    expect(buildCandidateList([[0.1, 0.9]], [1], 2)).toBeNull();
  });

  test("default random partition is valid", () => {
    const res = assessPos(models.M2!, { K: 3, nstart: 1, maxIter: 1, searchDepth: 1, seed: 11 })!;
    expect(Object.values(res.segmentSizes).reduce((a, b) => a + b, 0)).toBe(res.nObs);
  });
});

describe("rendering", () => {
  test("toString contains key sections", () => {
    const text = String(results.m1_k2_s123!);
    expect(text).toContain("PLS-POS Analysis");
    expect(text).toContain("Segments: 2");
    expect(text).toContain("Objective");
    expect(text).toContain("Segment Sizes");
    expect(text).toContain("Path Coefficients");
  });

  test("summarize contains details", () => {
    const text = results.m1_k2_s123!.summarize();
    expect(text).toContain("Detailed Summary");
    expect(text).toContain("Global");
    expect(text).toContain("Path Coefficients per Segment");
    expect(text).toContain("Start objectives:");
  });
});
