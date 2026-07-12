/**
 * Helpers kernel (Slice 0.5b): validation, endogenous extraction, losses, CVPAT
 * bootstrap. Ported from `../seminrExtras-py/tests/test_helpers.py`.
 *
 * Deterministic helpers are pinned by hand-computed expectations; the stochastic
 * kernel (`bootstrapCvpat`/`cvpatPerConstruct`) and `confInt` are pinned by R
 * goldens in `tests/fixtures/helpers/helpers.json` with the R index streams
 * injected via `draws` (plan F4/F8). Fixture indices are 1-based; tests convert.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import type { Dataset } from "@seminr/core";
import { FIXTURES_DIR } from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";
import {
  type CvpatBoot,
  type CvpatDraws,
  bootstrapCvpat,
  calculateLvLosses,
  confInt,
  cvpatPerConstruct,
  getEndogenousConstructs,
  getEndogenousItems,
  hasHigherOrder,
  itemsOfConstruct,
  lvLoss,
  overallLoss,
  rMean,
  rSd,
  seqSum,
  validateForPrediction,
  validateSeminrModel,
} from "../src/helpers.ts";

// --- fixtures ----------------------------------------------------------------

interface HelpersFixture {
  n: number;
  loss1: number[];
  loss2: number[];
  loss3: number[];
  bootstrapCvpat: Record<
    string,
    {
      testtype: string;
      nboot: number;
      pairIndices: number[][];
      dnullIndices: number[][];
      result: Record<string, number>;
    }
  >;
  cvpatPerConstruct: {
    nboot: number;
    pairIndices: number[][];
    dnullIndices: number[][];
    lossOne: { cols: string[]; values: number[][] };
    lossTwo: { cols: string[]; values: number[][] };
    results: { construct: string; result: Record<string, number> }[];
  };
  confInt: {
    names: string[];
    values: number[][][]; // (B, rows, cols)
    cases: {
      from: string;
      to: string;
      through: string | null;
      alpha: number;
      lower: number;
      upper: number;
    }[];
  };
}

const helpersFx: HelpersFixture = JSON.parse(
  readFileSync(join(FIXTURES_DIR, "helpers", "helpers.json"), "utf8"),
) as HelpersFixture;

const c1Model = estimateRegistryModel("C1");

/** Convert a fixture's 1-based index streams to the 0-based `CvpatDraws`. */
function drawsFrom(node: { pairIndices: number[][]; dnullIndices: number[][] }): CvpatDraws {
  return {
    pairIndices: node.pairIndices.map((row) => row.map((v) => v - 1)),
    dnullIndices: node.dnullIndices.map((row) => row.map((v) => v - 1)),
  };
}

/** `math.isclose(rel_tol=1e-9, abs_tol=1e-12)`. */
function isClose(a: number, b: number, relTol = 1e-9, absTol = 1e-12): boolean {
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  return Math.abs(a - b) <= Math.max(relTol * Math.max(Math.abs(a), Math.abs(b)), absTol);
}

function assertCvpatMatches(result: CvpatBoot, expected: Record<string, number>): void {
  const got: Record<string, number> = {
    "Std. T value": result.stdTValue,
    "Std. P value": result.stdPValue,
    "Boot T value": result.bootTValue,
    "Boot P Value": result.bootPValue,
    "Perc. P Value": result.percPValue,
  };
  for (const [key, ref] of Object.entries(expected)) {
    expect(isClose(got[key]!, ref), `${key}: got ${got[key]}, expected ${ref}`).toBe(true);
  }
}

// --- validation --------------------------------------------------------------

describe("validation", () => {
  test("validateSeminrModel accepts a model, warns + rejects a non-model", () => {
    expect(validateSeminrModel(c1Model, "assess_foo")).toBe(true);
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(validateSeminrModel({}, "assess_foo")).toBe(false);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(warn.mock.calls[0]![0]).toMatch(/assess_foo only works with SEMinR models/);
    } finally {
      warn.mockRestore();
    }
  });

  test("validateForPrediction rejects HOC models", () => {
    const hocModel = estimateRegistryModel("M4");
    expect(hasHigherOrder(hocModel)).toBe(true);
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(validateForPrediction(hocModel, "assess_cvpat")).toBe(false);
      expect(warn.mock.calls.at(-1)![0]).toMatch(/no published solution .* higher-order/);
    } finally {
      warn.mockRestore();
    }
  });

  test("validateForPrediction accepts a plain model", () => {
    expect(hasHigherOrder(c1Model)).toBe(false);
    expect(validateForPrediction(c1Model, "assess_cvpat")).toBe(true);
  });
});

// --- endogenous extraction ---------------------------------------------------

describe("endogenous extraction", () => {
  test("endogenous constructs and items (C1)", () => {
    expect(getEndogenousConstructs(c1Model)).toEqual(["CUSA", "CUSL"]);
    expect(getEndogenousItems(c1Model)).toEqual(["cusa", "cusl_1", "cusl_2", "cusl_3"]);
    expect(getEndogenousItems(c1Model, ["CUSL"])).toEqual(["cusl_1", "cusl_2", "cusl_3"]);
    expect(itemsOfConstruct("COMP", c1Model)).toEqual(["comp_1", "comp_2", "comp_3"]);
  });

  test("endogenous items exclude interaction constructs (M3)", () => {
    const m3 = estimateRegistryModel("M3");
    const items = getEndogenousItems(m3, ["Satisfaction", "Image*Value"]);
    expect(items).toEqual(["CUSA1", "CUSA2", "CUSA3"]);
  });
});

// --- losses ------------------------------------------------------------------

function errorDataset(): Dataset {
  return {
    columns: ["cusa", "cusl_1", "cusl_2", "cusl_3"],
    values: [
      [1.0, 2.0, -1.0, 0.0],
      [-2.0, 1.0, 3.0, 1.0],
      [0.5, 0.0, -0.5, 2.0],
    ],
  };
}

describe("losses", () => {
  test("lvLoss multi- and single-item", () => {
    const error = errorDataset();
    const cusl = lvLoss("CUSL", c1Model, error);
    const eCusl = [(4 + 1 + 0) / 3, (1 + 9 + 1) / 3, (0 + 0.25 + 4) / 3];
    cusl.forEach((v, i) => expect(isClose(v, eCusl[i]!)).toBe(true));
    const cusa = lvLoss("CUSA", c1Model, error);
    [1.0, 4.0, 0.25].forEach((ref, i) => expect(isClose(cusa[i]!, ref)).toBe(true));
  });

  test("calculateLvLosses and overallLoss", () => {
    const error = errorDataset();
    const losses = calculateLvLosses(["CUSA", "CUSL"], c1Model, error);
    expect(losses.columns).toEqual(["CUSA", "CUSL"]);
    losses.values.forEach((row, i) => expect(isClose(row[0]!, [1.0, 4.0, 0.25][i]!)).toBe(true));
    const overall = overallLoss(losses);
    losses.values.forEach((row, i) => expect(isClose(overall[i]!, (row[0]! + row[1]!) / 2)).toBe(true));
    const vec = [1.0, 2.0, 3.0];
    expect(overallLoss(vec)).toEqual(vec);
  });
});

// --- bootstrapCvpat R parity -------------------------------------------------

describe("bootstrapCvpat R parity", () => {
  for (const caseName of ["twoSided", "greater", "twoSidedWeak", "greaterWeak"]) {
    test(`matches R golden: ${caseName}`, () => {
      const c = helpersFx.bootstrapCvpat[caseName]!;
      const loss2 = caseName.endsWith("Weak") ? helpersFx.loss3 : helpersFx.loss2;
      const result = bootstrapCvpat(helpersFx.loss1, loss2, c.testtype, c.nboot, {
        draws: drawsFrom(c),
      });
      assertCvpatMatches(result, c.result);
    });
  }

  test("rejects unsupported testtype", () => {
    const loss = Array.from({ length: 10 }, (_, i) => i);
    expect(() =>
      bootstrapCvpat(
        loss,
        loss.map((v) => v + 0.1),
        "less",
        5,
        { rng: 1 },
      ),
    ).toThrow(/testtype/);
  });

  test("degenerate variance warns and yields NaN", () => {
    // loss values are arbitrary; a constant d_null resample forces zero variance
    const loss1 = Array.from({ length: 20 }, (_, i) => 0.3 + 0.01 * i);
    const loss2 = loss1.map((v, i) => v + 0.05 + 0.001 * i);
    const nboot = 10;
    const identity = Array.from({ length: 20 }, (_, i) => i);
    const zeros = Array.from({ length: 20 }, () => 0);
    const draws: CvpatDraws = {
      pairIndices: Array.from({ length: nboot }, () => [...identity]),
      dnullIndices: Array.from({ length: nboot }, () => [...zeros]),
    };
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      const result = bootstrapCvpat(loss1, loss2, "two.sided", nboot, { draws });
      expect(warn.mock.calls.at(-1)![0]).toMatch(/Bootstrap variance near zero/);
      expect(Number.isNaN(result.bootTValue)).toBe(true);
      expect(Number.isNaN(result.bootPValue)).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("default rng is reproducible for a numeric seed", () => {
    const gen = (() => {
      // small deterministic loss vectors; only reproducibility is under test
      let s = 7;
      const r = () => {
        s = (s * 1103515245 + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
      };
      const loss1 = Array.from({ length: 30 }, () => r());
      const loss2 = loss1.map((v) => v + r() * 0.05);
      return { loss1, loss2 };
    })();
    const a = bootstrapCvpat(gen.loss1, gen.loss2, "two.sided", 25, { rng: 42 });
    const b = bootstrapCvpat(gen.loss1, gen.loss2, "two.sided", 25, { rng: 42 });
    expect(a.stdTValue).toBe(b.stdTValue);
    expect(a.stdPValue).toBe(b.stdPValue);
    expect(a.bootTValue).toBe(b.bootTValue);
    expect(a.bootPValue).toBe(b.bootPValue);
    expect(a.percPValue).toBe(b.percPValue);
  });
});

// --- cvpatPerConstruct R parity ----------------------------------------------

describe("cvpatPerConstruct R parity", () => {
  test("matches R golden (CUSA, CUSL)", () => {
    const pc = helpersFx.cvpatPerConstruct;
    const lossOne: Dataset = { columns: [...pc.lossOne.cols], values: pc.lossOne.values };
    const lossTwo: Dataset = { columns: [...pc.lossTwo.cols], values: pc.lossTwo.values };
    const results = cvpatPerConstruct(lossOne, lossTwo, "two.sided", pc.nboot, {
      draws: drawsFrom(pc),
    });
    expect(Object.keys(results)).toEqual(["CUSA", "CUSL"]);
    for (const entry of pc.results) {
      assertCvpatMatches(results[entry.construct]!, entry.result);
    }
  });
});

// --- confInt R parity --------------------------------------------------------

describe("confInt R parity", () => {
  test("matches R goldens", () => {
    const ci = helpersFx.confInt;
    const names = ci.names;
    const nRows = names.length;
    const nCols = names.length;
    const nB = ci.values.length;
    // (B, rows, cols) -> [rowIdx][colIdx][b]
    const boot: number[][][] = Array.from({ length: nRows }, (_, r) =>
      Array.from({ length: nCols }, (_, cc) =>
        Array.from({ length: nB }, (_, b) => ci.values[b]![r]![cc]!),
      ),
    );
    for (const cse of ci.cases) {
      const [lower, upper] = confInt(boot, names, names, cse.from, cse.to, cse.through, cse.alpha);
      expect(isClose(lower, cse.lower)).toBe(true);
      expect(isClose(upper, cse.upper)).toBe(true);
    }
  });
});

// --- numeric core exports ----------------------------------------------------

describe("numeric core", () => {
  test("seqSum, rMean, rSd basic behaviour", () => {
    expect(seqSum([])).toBe(0);
    expect(seqSum([1, 2, 3, 4])).toBe(10);
    expect(isClose(rMean([1, 2, 3, 4]), 2.5)).toBe(true);
    expect(isClose(rSd([2, 4, 4, 4, 5, 5, 7, 9]), 2.138089935299395)).toBe(true);
  });
});
