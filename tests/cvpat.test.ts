/**
 * CVPAT (Slice 1): assessCvpat / assessCvpatCompare against R goldens.
 *
 * Parity fixtures (tests/fixtures/cvpat/cvpat.json) were generated with
 * noFolds=NULL (LOOCV — fold-order independent) and carry the R bootstrap index
 * streams (1-based; converted here), injected via `draws`. Tolerances: R's
 * assess_cvpat loses precision in its LOSS columns (7 significant digits, an
 * `as.matrix`-on-data.frame formatting artifact) so those compare at rel 1e-6;
 * everything else at rel 1e-9. Structure/error behavior follows the R test
 * suite (test-cvpat*.R), via the py port's test_cvpat.py.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import { predictDA, predictEA, type NamedMatrix } from "@seminr/core";
import {
  assessCvpat,
  assessCvpatCompare,
  type CvpatAssessment,
  type CvpatComparison,
} from "../src/featureCvpat.ts";
import type { CvpatDraws } from "../src/helpers.ts";
import { FIXTURES_DIR } from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

const LOSS_COLS = new Set([
  "PLS Loss",
  "LM Loss",
  "IA Loss",
  "Base Model Loss",
  "Alt Model Loss",
  "Diff",
]);

interface TableRef {
  rows: string[];
  cols: string[];
  values: number[][];
}

interface CvpatCase {
  name: string;
  technique: string;
  testtype: string;
  nboot: number;
  seed: number;
  pairIndices: number[][];
  dnullIndices: number[][];
  lm?: TableRef;
  ia?: TableRef;
  table?: TableRef;
}

const fx = JSON.parse(
  readFileSync(join(FIXTURES_DIR, "cvpat", "cvpat.json"), "utf8"),
) as { nObs: number; assess: CvpatCase[]; compare: CvpatCase[] };

const modelOne = estimateRegistryModel("C1");
const modelTwo = estimateRegistryModel("C2");

function caseDraws(c: CvpatCase): CvpatDraws {
  return {
    pairIndices: c.pairIndices.map((row) => row.map((i) => i - 1)),
    dnullIndices: c.dnullIndices.map((row) => row.map((i) => i - 1)),
  };
}

function technique(name: string) {
  return name === "EA" ? predictEA : predictDA;
}

/** Python math.isclose semantics. */
function isClose(a: number, b: number, relTol: number, absTol: number): boolean {
  return Math.abs(a - b) <= Math.max(relTol * Math.max(Math.abs(a), Math.abs(b)), absTol);
}

function expectTable(got: NamedMatrix, ref: TableRef, label: string, lossRel: number): void {
  expect(got.rows).toEqual(ref.rows);
  expect(got.cols).toEqual(ref.cols);
  for (let i = 0; i < ref.rows.length; i++) {
    for (let j = 0; j < ref.cols.length; j++) {
      const rel = LOSS_COLS.has(ref.cols[j]!) ? lossRel : 1e-9;
      const g = got.values[i]![j]!;
      const r = ref.values[i]![j]!;
      if (!isClose(g, r, rel, 1e-9)) {
        throw new Error(`${label}[${ref.rows[i]}, ${ref.cols[j]}]: got ${g}, expected ${r}`);
      }
    }
  }
}

describe("assessCvpat R parity", () => {
  for (const [idx, c] of fx.assess.entries()) {
    test(`matches R golden: ${c.name} (case ${idx})`, () => {
      const result = assessCvpat(modelOne, {
        testtype: c.testtype,
        nboot: c.nboot,
        technique: technique(c.technique),
        draws: caseDraws(c),
      });
      expect(result).not.toBeNull();
      const r = result as CvpatAssessment;
      // R truncates assess loss columns to 7 significant digits (as.matrix on
      // a mixed data.frame); TS keeps full precision -> rel 1e-6 on those.
      expectTable(r.cvpatCompareLm, c.lm!, `${c.name}.lm`, 1e-6);
      expectTable(r.cvpatCompareIa, c.ia!, `${c.name}.ia`, 1e-6);
    });
  }
});

describe("assessCvpatCompare R parity", () => {
  for (const [idx, c] of fx.compare.entries()) {
    test(`matches R golden: ${c.name} (case ${idx})`, () => {
      const result = assessCvpatCompare(modelOne, modelTwo, {
        testtype: c.testtype,
        nboot: c.nboot,
        technique: technique(c.technique),
        draws: caseDraws(c),
      });
      expect(result).not.toBeNull();
      // The compare table keeps full precision in R (unlist path, no as.matrix).
      expectTable((result as CvpatComparison).table, c.table!, c.name, 1e-9);
    });
  }
});

describe("validation / failure modes", () => {
  test("assessCvpat rejects a non-model with a warning", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessCvpat({ not: "a model" }, { nboot: 5 })).toBeNull();
      expect(warn.mock.calls.some((c) => String(c[0]).includes("only works with SEMinR models"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("assessCvpatCompare rejects a non-model with a warning", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessCvpatCompare(modelOne, [1, 2, 3], { nboot: 5 })).toBeNull();
      expect(warn.mock.calls.some((c) => String(c[0]).includes("only works with SEMinR models"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("HOC models return null with a warning", () => {
    const hocModel = estimateRegistryModel("C3");
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessCvpat(hocModel, { nboot: 5 })).toBeNull();
      expect(assessCvpatCompare(hocModel, hocModel, { nboot: 5 })).toBeNull();
      expect(warn.mock.calls.filter((c) => String(c[0]).includes("higher-order")).length).toBe(2);
    } finally {
      warn.mockRestore();
    }
  });

  test("compare with mismatched endogenous constructs throws", () => {
    const modelDiff = estimateRegistryModel("C4");
    expect(() => assessCvpatCompare(modelOne, modelDiff, { nboot: 5 })).toThrow(
      /identical endogenous/,
    );
  });

  test("reps is an accepted no-op; invalid reps throws", () => {
    // R's reps re-runs identical folds and averages identical matrices (a
    // numeric no-op, verified in R); accepted and ignored here.
    const a = assessCvpat(modelOne, { nboot: 10, seed: 5 });
    const b = assessCvpat(modelOne, { nboot: 10, seed: 5, reps: 2 });
    expect(a).not.toBeNull();
    expect(b).not.toBeNull();
    expect((a as CvpatAssessment).cvpatCompareLm.values).toEqual(
      (b as CvpatAssessment).cvpatCompareLm.values,
    );
    expect(() => assessCvpat(modelOne, { nboot: 5, reps: 0 })).toThrow(/reps/);
  });
});

describe("default-rng path", () => {
  test("same seed reproduces; different seed changes bootstrap only", () => {
    const a = assessCvpat(modelOne, { nboot: 20, seed: 123 }) as CvpatAssessment;
    const b = assessCvpat(modelOne, { nboot: 20, seed: 123 }) as CvpatAssessment;
    const c = assessCvpat(modelOne, { nboot: 20, seed: 99 }) as CvpatAssessment;
    expect(a.cvpatCompareLm.values).toEqual(b.cvpatCompareLm.values);
    const bootT = a.cvpatCompareLm.cols.indexOf("Boot T value");
    const colOf = (m: NamedMatrix, j: number) => m.values.map((row) => row[j]!);
    expect(colOf(a.cvpatCompareLm, bootT)).not.toEqual(colOf(c.cvpatCompareLm, bootT));
    // Loss columns do not depend on the bootstrap seed (LOOCV is deterministic).
    const lossA = colOf(a.cvpatCompareLm, 0);
    const lossC = colOf(c.cvpatCompareLm, 0);
    for (let i = 0; i < lossA.length; i++) expect(lossA[i]!).toBeCloseTo(lossC[i]!, 12);
  });
});

describe("record surface", () => {
  test("assessment and comparison records render and self-describe", () => {
    const c0 = fx.assess[0]!;
    const result = assessCvpat(modelOne, { nboot: c0.nboot, draws: caseDraws(c0) });
    expect(result).not.toBeNull();
    const r = result as CvpatAssessment;
    expect(r.kind).toBe("cvpat_assessment");
    expect(r.description).toContain("CVPAT");
    const text = String(r);
    expect(text).toContain("CVPAT");
    expect(text).toContain("PLS Loss");
    expect(text).toContain("Overall");
    expect(r.summarize()).toContain("Boot P Value");

    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      const cc = fx.compare[0]!;
      const comp = assessCvpatCompare(modelOne, modelTwo, {
        nboot: cc.nboot,
        draws: caseDraws(cc),
      });
      expect(comp).not.toBeNull();
      expect((comp as CvpatComparison).kind).toBe("cvpat_comparison");
      expect(String(comp)).toContain("Base Model Loss");
      // A clean compare run emits no stray warnings.
      expect(warn.mock.calls.length).toBe(0);
    } finally {
      warn.mockRestore();
    }
  });
});
