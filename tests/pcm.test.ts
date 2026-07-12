/**
 * PCM (Slice 2): assessPcm against R goldens + behavioral spec (test-pcm.R).
 *
 * Parity fixtures (tests/fixtures/pcm/pcm.json) carry the R cross-validation
 * fold orderings per predict call (path1-DA, path1-EA, path2-DA, ...), injected
 * via `orderings` (1-based R permutations, decremented here). `reps` is a
 * numeric no-op in prediction (plan F10) — validated and stored only. RMSE/MAE
 * match R's summary(predict_pls) metrics to ~1e-15, so results compare at
 * rel 1e-6. Behavioral cases mirror ../seminrExtras/tests/testthat/test-pcm.R
 * via the py port's test_pcm.py.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import type { NamedMatrix } from "@seminr/core";
import {
  assessPcm,
  buildIsolatedSubModel,
  classifyPcm,
  detectFinalEndogenous,
  findMediationPaths,
  type PcmAnalysis,
  type PcmPath,
} from "../src/featurePcm.ts";
import { FIXTURES_DIR } from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

const RESULT_COLS = ["RMSE_DA", "RMSE_EA", "PCM_RMSE", "MAE_DA", "MAE_EA", "PCM_MAE"];

interface TableRef {
  rows: string[];
  cols: string[];
  values: number[][];
}

interface PcmCase {
  name: string;
  target: string;
  noFolds: number;
  reps: number;
  seed: number;
  orderings: number[][];
  paths: { antecedent: string; mediator: string; target: string }[];
  results: TableRef[];
}

const fx = JSON.parse(
  readFileSync(join(FIXTURES_DIR, "pcm", "pcm.json"), "utf8"),
) as { cases: PcmCase[] };

function findCase(name: string): PcmCase {
  const c = fx.cases.find((x) => x.name === name);
  if (!c) throw new Error(`missing fixture case: ${name}`);
  return c;
}

/** R permutations are 1-based; predictPls wants 0-based row indices. */
function orderings0(c: PcmCase): number[][] {
  return c.orderings.map((o) => o.map((i) => i - 1));
}

/** Python math.isclose semantics. */
function isClose(a: number, b: number, relTol: number, absTol: number): boolean {
  return Math.abs(a - b) <= Math.max(relTol * Math.max(Math.abs(a), Math.abs(b)), absTol);
}

function expectResults(got: NamedMatrix, ref: TableRef, label: string): void {
  expect(got.rows).toEqual(ref.rows);
  expect(got.cols).toEqual(ref.cols);
  for (let i = 0; i < ref.rows.length; i++) {
    for (let j = 0; j < ref.cols.length; j++) {
      const g = got.values[i]![j]!;
      const r = ref.values[i]![j]!;
      if (!isClose(g, r, 1e-6, 1e-9)) {
        throw new Error(`${label}[${ref.rows[i]}, ${ref.cols[j]}]: got ${g}, expected ${r}`);
      }
    }
  }
}

// --- Numeric parity against R goldens -------------------------------------------

const PARITY_CASES: [string, string, string | undefined][] = [
  ["m7_auto_target", "M7", undefined],
  ["m1_loyalty", "M1", "Loyalty"],
  ["m8_moderation", "M8", "Loyalty"],
];

describe("assessPcm R parity", () => {
  for (const [caseName, modelId, target] of PARITY_CASES) {
    test(`matches R golden: ${caseName}`, () => {
      const c = findCase(caseName);
      const model = estimateRegistryModel(modelId);
      const result = assessPcm(model, {
        target,
        noFolds: c.noFolds,
        reps: c.reps,
        orderings: orderings0(c),
      });
      expect(result).not.toBeNull();
      const r = result as PcmAnalysis;
      expect(r.target).toBe(c.target);

      const gotPaths = r.mediationPaths.map((p) => [p.antecedent, p.mediator, p.target]);
      const refPaths = c.paths.map((p) => [p.antecedent, p.mediator, p.target]);
      expect(gotPaths).toEqual(refPaths);

      expect(r.pcmResults.length).toBe(c.results.length);
      for (let i = 0; i < c.results.length; i++) {
        expectResults(r.pcmResults[i]!.results, c.results[i]!, `${caseName}.path${i}`);
      }
    });
  }
});

// --- Path discovery / helpers (spec: test-pcm.R) --------------------------------

describe("path discovery and classification", () => {
  test("detectFinalEndogenous finds the single terminal outcome", () => {
    expect(detectFinalEndogenous(estimateRegistryModel("M7"))).toBe("Loyalty");
    expect(detectFinalEndogenous(estimateRegistryModel("M1"))).toBe("Loyalty");
  });

  test("findMediationPaths on a simple mediation triangle", () => {
    const paths = findMediationPaths(estimateRegistryModel("M7"), "Loyalty");
    expect(paths.length).toBe(1);
    expect([paths[0]!.antecedent, paths[0]!.mediator, paths[0]!.target]).toEqual([
      "Image",
      "Satisfaction",
      "Loyalty",
    ]);
  });

  test("findMediationPaths with multiple antecedents", () => {
    const paths = findMediationPaths(estimateRegistryModel("M1"), "Loyalty");
    expect(paths.length).toBe(3);
    expect(paths.every((p) => p.mediator === "Satisfaction")).toBe(true);
    expect(new Set(paths.map((p) => p.antecedent))).toEqual(
      new Set(["Image", "Expectation", "Value"]),
    );
  });

  test("findMediationPaths empty for an unmediated target", () => {
    // Satisfaction's only predictor (Image) has no antecedent of its own.
    expect(findMediationPaths(estimateRegistryModel("M7"), "Satisfaction")).toEqual([]);
  });

  test("classifyPcm boundaries (Danks 2021)", () => {
    expect(classifyPcm(NaN)).toBe("NA");
    expect(classifyPcm(-0.01)).toBe("Negative");
    expect(classifyPcm(0.0)).toBe("Weak");
    expect(classifyPcm(0.049)).toBe("Weak");
    expect(classifyPcm(0.05)).toBe("Moderate");
    expect(classifyPcm(0.099)).toBe("Moderate");
    expect(classifyPcm(0.1)).toBe("Strong");
    expect(classifyPcm(0.2)).toBe("Strong");
  });
});

// --- Sub-model construction -----------------------------------------------------

describe("isolated sub-model construction", () => {
  test("rebuilds the partial-mediation triangle", () => {
    const model = estimateRegistryModel("M7");
    const sub = buildIsolatedSubModel(model, { antecedent: "Image", mediator: "Satisfaction", target: "Loyalty" });
    expect([...sub.constructScores.cols].sort()).toEqual(["Image", "Loyalty", "Satisfaction"]);
    // X->M, {X,M}->Y  (three structural paths total)
    expect(sub.smMatrix.constructAntecedents("Satisfaction")).toEqual(["Image"]);
    expect(sub.smMatrix.constructAntecedents("Loyalty")).toEqual(["Image", "Satisfaction"]);
  });

  test("preserves mode_B measurement", () => {
    const model = estimateRegistryModel("M7b");
    const sub = buildIsolatedSubModel(model, { antecedent: "Image", mediator: "Satisfaction", target: "Loyalty" });
    expect(sub.mmMatrix.isModeB("Image")).toBe(true);
  });
});

// --- assessPcm output structure -------------------------------------------------

describe("assessPcm output structure", () => {
  test("auto-detects target and shapes the result record", () => {
    const c = findCase("m7_auto_target");
    const result = assessPcm(estimateRegistryModel("M7"), {
      target: undefined,
      noFolds: c.noFolds,
      reps: c.reps,
      orderings: orderings0(c),
    });
    expect(result).not.toBeNull();
    const r = result as PcmAnalysis;
    expect(r.kind).toBe("pcm_analysis");
    expect(r.target).toBe("Loyalty");
    expect(r.noFolds).toBe(c.noFolds);
    expect(r.reps).toBe(c.reps);
    expect(r.mediationPaths.length).toBe(1);
    expect(r.pcmResults.length).toBe(1);

    const res = r.pcmResults[0]!;
    expect([res.antecedent, res.mediator, res.target]).toEqual(["Image", "Satisfaction", "Loyalty"]);
    expect(res.results.rows).toEqual(["CUSL1", "CUSL2", "CUSL3"]);
    expect(res.results.cols).toEqual(RESULT_COLS);

    // PCM = (EA - DA) / EA, and all DA/EA metrics strictly positive.
    for (const row of res.results.values) {
      expect(row[2]!).toBeCloseTo((row[1]! - row[0]!) / row[1]!, 12);
      expect(row[5]!).toBeCloseTo((row[4]! - row[3]!) / row[4]!, 12);
      for (const k of [0, 1, 3, 4]) expect(row[k]!).toBeGreaterThan(0);
    }
    expect(res.pcmRmse.every((v) => Math.abs(v) < 1)).toBe(true);
    expect(res.pcmMae.every((v) => Math.abs(v) < 1)).toBe(true);
  });

  test("auto-detect equals explicit target", () => {
    const c = findCase("m7_auto_target");
    const model = estimateRegistryModel("M7");
    const auto = assessPcm(model, { noFolds: c.noFolds, reps: 1, orderings: orderings0(c) });
    const explicit = assessPcm(model, {
      target: "Loyalty",
      noFolds: c.noFolds,
      reps: 1,
      orderings: orderings0(c),
    });
    expect(auto).not.toBeNull();
    expect(explicit).not.toBeNull();
    expect((auto as PcmAnalysis).pcmResults[0]!.results.values).toEqual(
      (explicit as PcmAnalysis).pcmResults[0]!.results.values,
    );
  });
});

// --- Interaction / HOC handling -------------------------------------------------

describe("interaction and HOC handling", () => {
  test("moderation model excludes interaction constructs from paths", () => {
    const paths = findMediationPaths(estimateRegistryModel("M8"), "Loyalty");
    expect(paths.length).toBeGreaterThanOrEqual(1);
    expect(paths.some((p) => p.mediator.includes("*") || p.antecedent.includes("*"))).toBe(false);
  });

  test("HOC models are rejected by validation (assessPcm returns null)", () => {
    for (const modelId of ["M4", "M9"]) {
      const warn = spyOn(console, "warn").mockImplementation(() => {});
      try {
        expect(assessPcm(estimateRegistryModel(modelId))).toBeNull();
        expect(warn.mock.calls.some((call) => String(call[0]).includes("higher-order"))).toBe(true);
      } finally {
        warn.mockRestore();
      }
    }
  });

  test("findMediationPaths skips HOC-containing triples with a warning", () => {
    // HOC-in-triple guard is only reachable via the internal helper (assessPcm
    // blocks HOC models earlier), mirroring test-pcm.R's direct call on M9.
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      const paths = findMediationPaths(estimateRegistryModel("M9"), "Loyalty");
      expect(paths).toEqual([]);
      expect(warn.mock.calls.some((call) => String(call[0]).includes("higher-order"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });
});

// --- Validation failures --------------------------------------------------------

describe("validation failures", () => {
  test("rejects a non-seminr model with a warning", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessPcm({ not: "a model" })).toBeNull();
      expect(warn.mock.calls.some((c) => String(c[0]).includes("only works with SEMinR"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("rejects an invalid target", () => {
    expect(() => assessPcm(estimateRegistryModel("M7"), { target: "Nonexistent" })).toThrow(
      /not found in model/,
    );
  });

  test("rejects a target without mediation", () => {
    expect(() => assessPcm(estimateRegistryModel("M7"), { target: "Satisfaction" })).toThrow(
      /No mediation paths/,
    );
  });

  test("rejects multiple final endogenous constructs", () => {
    // M13: Image -> {Expectation, Satisfaction}; both are final endogenous.
    expect(() => assessPcm(estimateRegistryModel("M13"))).toThrow(/Multiple final endogenous/);
  });

  test("rejects an invalid noFolds", () => {
    const model = estimateRegistryModel("M7");
    expect(() => assessPcm(model, { noFolds: 1 })).toThrow(/noFolds must be/);
    // @ts-expect-error deliberately invalid type
    expect(() => assessPcm(model, { noFolds: "a" })).toThrow(/noFolds must be/);
  });

  test("rejects an invalid reps", () => {
    expect(() => assessPcm(estimateRegistryModel("M7"), { reps: 0 })).toThrow(/reps must be/);
  });
});

// --- Record surface -------------------------------------------------------------

describe("record surface", () => {
  test("renders and self-describes", () => {
    const c = findCase("m7_auto_target");
    const result = assessPcm(estimateRegistryModel("M7"), {
      noFolds: c.noFolds,
      reps: c.reps,
      orderings: orderings0(c),
    });
    expect(result).not.toBeNull();
    const r = result as PcmAnalysis;

    const text = String(r);
    expect(text).toContain("Predictive Contribution");
    expect(text).toContain("Loyalty");
    expect(text).toContain("Image -> Satisfaction -> Loyalty");
    expect(text).toContain("PCM");

    const summary = r.summarize();
    expect(summary).toContain("PCM thresholds");
    expect(summary).toContain("RMSE_DA");
    expect(summary).toContain("RMSE_EA");
    expect(summary).toContain("PCM_RMSE");
    expect(summary).toContain("CUSL");
    expect(summary).toContain("Danks");
  });
});

// --- Default-rng reproducibility ------------------------------------------------

describe("default-rng path", () => {
  test("same seed reproduces", () => {
    const model = estimateRegistryModel("M7");
    const a = assessPcm(model, { seed: 7 }) as PcmAnalysis;
    const b = assessPcm(model, { seed: 7 }) as PcmAnalysis;
    expect(a.pcmResults[0]!.results.values).toEqual(b.pcmResults[0]!.results.values);
  });
});

// Reference the imported type so tsc treats it as used in named-arg helpers.
export type { PcmPath };
