/**
 * CTA-PLS (Slice 4): assessCta against R goldens + behavioral spec (test-cta.R).
 *
 * Parity fixtures (tests/fixtures/cta/cta.json) carry the R bootstrap row-index
 * streams (1-based, one vector per iteration — assessCta's only RNG), injected
 * via `draws` (converted to 0-based here). Numerics compare at rel/abs 1e-9. The
 * fixture also pins R `p.adjust` semantics (BH/bonferroni, incl. the lazy-`n` NA
 * behavior). Behavioral cases mirror ../seminrExtras/tests/testthat/test-cta.R
 * via the py port's test_cta.py, including its deterministic tau_1342 oracle and
 * the tetrad sanity checks on simulated common-factor vs composite data.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import { mulberry32 } from "@seminr/core";
import { cov as csCov, fromRows, toRows } from "@compstats/core/linalg";
import {
  assessCta,
  computeTetrads,
  enumerateBorrowedTetrads,
  enumerateTetrads,
  findDonor,
  getStructurallyConnected,
  pAdjust,
  resolveIndicators,
  type CtaAnalysis,
  type TetradSpec,
} from "../src/featureCta.ts";
import { FIXTURES_DIR } from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

interface DetailRef {
  rows: string | string[];
  cols: string[];
  values: number[][] | number[];
  significant: boolean | boolean[];
}

interface CtaCase {
  name: string;
  modelId: string;
  constructs: string[] | string | null;
  nboot: number;
  seed: number;
  alpha: number;
  correction: string;
  borrow: boolean;
  nObs: number;
  bootIndices: number[][];
  constructResults: {
    construct: string[];
    mode: string[];
    indicators: number[];
    tetrads: number[];
    significant: number[];
    verdict: string[];
  };
  tetradDetails: Record<string, DetailRef>;
  detailCols: string[];
  skipped: string[];
  borrowing: Record<string, unknown> | unknown[];
}

interface PAdjustCase {
  name: string;
  method: string;
  p: (number | null)[];
  expected: number[];
}

const fx = JSON.parse(
  readFileSync(join(FIXTURES_DIR, "cta", "cta.json"), "utf8"),
) as { cases: CtaCase[]; padjust: PAdjustCase[] };

function caseByName(name: string): CtaCase {
  const c = fx.cases.find((x) => x.name === name);
  if (!c) throw new Error(`no fixture case ${name}`);
  return c;
}

function draws(c: CtaCase): number[][] {
  return c.bootIndices.map((row) => row.map((i) => i - 1));
}

/** Python math.isclose with NaN/None handling from the py test. */
function nanClose(got: number, want: number | null, rel = 1e-9, abs = 1e-9): boolean {
  if (want === null || (typeof want === "number" && Number.isNaN(want))) {
    return Number.isNaN(got);
  }
  return Math.abs(got - want) <= Math.max(rel * Math.max(Math.abs(got), Math.abs(want)), abs);
}

function as2d(v: number[][] | number[]): number[][] {
  return Array.isArray(v[0]) ? (v as number[][]) : [v as number[]];
}

function asRows(v: string | string[]): string[] {
  return typeof v === "string" ? [v] : v;
}

function asSig(v: boolean | boolean[]): boolean[] {
  return typeof v === "boolean" ? [v] : v;
}

// --- Numeric parity against R goldens ---------------------------------------------

const PARITY_CASES: readonly string[] = [
  "m1_borrow_default",
  "m1_noborrow",
  "m1_image_bonferroni_alpha10",
  "m3b_moderation",
  "m11_hoc_super",
];

describe("assessCta R parity", () => {
  for (const name of PARITY_CASES) {
    test(`matches R golden: ${name}`, () => {
      const c = caseByName(name);
      const model = estimateRegistryModel(c.modelId);
      const result = assessCta(model, {
        constructs: c.constructs === null ? undefined : c.constructs,
        nboot: c.nboot,
        seed: c.seed,
        alpha: c.alpha,
        correction: c.correction,
        borrow: c.borrow,
        draws: draws(c),
      });
      expect(result).not.toBeNull();
      const res = result as CtaAnalysis;

      const cr = c.constructResults;
      const rows = res.constructResults;
      expect(rows.map((r) => r.construct)).toEqual(cr.construct);
      expect(rows.map((r) => r.mode)).toEqual(cr.mode);
      expect(rows.map((r) => r.indicators)).toEqual(cr.indicators);
      expect(rows.map((r) => r.tetrads)).toEqual(cr.tetrads);
      expect(rows.map((r) => r.significant)).toEqual(cr.significant);
      expect(rows.map((r) => r.verdict)).toEqual(cr.verdict);

      expect([...res.tetradDetails.keys()]).toEqual(cr.construct);
      const detailCols = c.detailCols;
      for (const [construct, ref] of Object.entries(c.tetradDetails)) {
        const got = res.tetradDetails.get(construct)!;
        const refRows = asRows(ref.rows);
        const refValues = as2d(ref.values);
        expect(got.table.rows).toEqual(refRows);
        expect(got.table.cols).toEqual(detailCols);
        for (let i = 0; i < refRows.length; i++) {
          for (let j = 0; j < detailCols.length; j++) {
            const g = got.table.values[i]![j]!;
            const w = refValues[i]![j]!;
            if (!nanClose(g, w)) {
              throw new Error(
                `${name}.${construct}[${refRows[i]}, ${detailCols[j]}]: got ${g}, expected ${w}`,
              );
            }
          }
        }
        expect(got.significant).toEqual(asSig(ref.significant));
      }

      expect(res.skipped).toEqual(c.skipped);
      const gotBorrowing: Record<string, unknown> = {};
      for (const [nm, b] of res.borrowing) {
        gotBorrowing[nm] = {
          donor: b.donor,
          donor_mode: b.donorMode,
          vanishing_pattern: b.vanishingPattern,
          n_vanishing: b.nVanishing,
        };
      }
      const refBorrowing =
        c.borrowing && !Array.isArray(c.borrowing) ? c.borrowing : {};
      expect(gotBorrowing).toEqual(refBorrowing as Record<string, unknown>);

      expect(res.nboot).toBe(c.nboot);
      expect(res.alpha).toBe(c.alpha);
      expect(res.correction).toBe(c.correction);
    });
  }
});

describe("pAdjust matches R", () => {
  for (const pcase of fx.padjust) {
    test(pcase.name, () => {
      const p = pcase.p.map((v) => (v === null ? NaN : v));
      const got = pAdjust(p, pcase.method);
      for (let k = 0; k < pcase.expected.length; k++) {
        if (!nanClose(got[k]!, pcase.expected[k]!, 1e-12, 1e-15)) {
          throw new Error(`${pcase.name}[${k}]: got ${got[k]}, expected ${pcase.expected[k]}`);
        }
      }
    });
  }
});

// --- Validation / failure modes (spec: test-cta.R 286-308) ------------------------

describe("validation / failure modes", () => {
  test("rejects a non-model with a warning", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessCta({ not: "a model" }, { nboot: 10 })).toBeNull();
      expect(
        warn.mock.calls.some((call) => String(call[0]).includes("only works with SEMinR models")),
      ).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("partial invalid constructs warns and tests the valid ones", () => {
    const model = estimateRegistryModel("M1");
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      const result = assessCta(model, { constructs: ["Image", "Nonexistent"], nboot: 10, seed: 123 });
      expect(result).not.toBeNull();
      expect((result as CtaAnalysis).constructResults.map((r) => r.construct)).toEqual(["Image"]);
      expect(warn.mock.calls.some((call) => String(call[0]).includes("Constructs not found"))).toBe(
        true,
      );
    } finally {
      warn.mockRestore();
    }
  });

  test("all invalid constructs returns null with both warnings", () => {
    const model = estimateRegistryModel("M1");
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(assessCta(model, { constructs: ["Nonexistent"], nboot: 10 })).toBeNull();
      const messages = warn.mock.calls.map((call) => String(call[0]));
      expect(messages.some((m) => m.includes("Constructs not found"))).toBe(true);
      expect(messages.some((m) => m.includes("No valid constructs"))).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("invalid correction throws", () => {
    const model = estimateRegistryModel("M1");
    expect(() => assessCta(model, { nboot: 10, correction: "holm" })).toThrow(/correction/);
  });
});

// --- Reproducibility (default RNG path; spec: 215-233) ----------------------------

describe("default-rng path", () => {
  test("same seed reproduces; different seed changes SD but not the estimate", () => {
    const model = estimateRegistryModel("M1");
    const a = assessCta(model, { constructs: "Image", nboot: 10, seed: 42 }) as CtaAnalysis;
    const b = assessCta(model, { constructs: "Image", nboot: 10, seed: 42 }) as CtaAnalysis;
    const cc = assessCta(model, { constructs: "Image", nboot: 10, seed: 99 }) as CtaAnalysis;
    const tableA = a.tetradDetails.get("Image")!.table;
    const sdCol = tableA.cols.indexOf("Boot_SD");
    const estCol = tableA.cols.indexOf("Estimate");
    const av = tableA.values;
    const bv = b.tetradDetails.get("Image")!.table.values;
    const cv = cc.tetradDetails.get("Image")!.table.values;
    expect(av).toEqual(bv);
    // Estimates are deterministic (original covariance); Boot_SD depends on seed.
    expect(av.map((r) => r[estCol]!)).toEqual(cv.map((r) => r[estCol]!));
    expect(av.map((r) => r[sdCol]!)).not.toEqual(cv.map((r) => r[sdCol]!));
  });
});

// --- Correction behavior (spec: 239-259) -------------------------------------------

describe("correction behavior", () => {
  test("none passes raw P through; BH/bonferroni share the raw P column", () => {
    const model = estimateRegistryModel("M1");
    const tables: Record<string, ReturnType<() => CtaAnalysis>> = {};
    for (const method of ["BH", "bonferroni", "none"]) {
      tables[method] = assessCta(model, {
        constructs: "Image",
        nboot: 10,
        seed: 123,
        correction: method,
      }) as CtaAnalysis;
    }
    const t = (m: string) => (tables[m] as CtaAnalysis).tetradDetails.get("Image")!.table;
    const pCol = t("BH").cols.indexOf("P_Value");
    const adjCol = t("BH").cols.indexOf("Adj_P");
    const pOf = (m: string) => t(m).values.map((r) => r[pCol]!);
    expect(pOf("BH")).toEqual(pOf("none"));
    expect(pOf("bonferroni")).toEqual(pOf("none"));
    expect(t("none").values.map((r) => r[adjCol]!)).toEqual(pOf("none"));
  });

  test("alpha changes the CI column names", () => {
    const model = estimateRegistryModel("M1");
    const r05 = assessCta(model, { constructs: "Image", nboot: 10, seed: 123, alpha: 0.05 }) as CtaAnalysis;
    const r10 = assessCta(model, { constructs: "Image", nboot: 10, seed: 123, alpha: 0.1 }) as CtaAnalysis;
    const cols05 = new Set(r05.tetradDetails.get("Image")!.table.cols);
    const cols10 = new Set(r10.tetradDetails.get("Image")!.table.cols);
    expect(cols05.has("2.5% CI") && cols05.has("97.5% CI")).toBe(true);
    expect(cols10.has("5% CI") && cols10.has("95% CI")).toBe(true);
  });
});

// --- Borrowing / skipping across model types (spec: 362-522) -----------------------

describe("borrowing / skipping", () => {
  test("mediation model borrows for 2-3 indicator constructs; borrow=false skips them", () => {
    const model = estimateRegistryModel("M10");
    const r = assessCta(model, { nboot: 10, seed: 123 }) as CtaAnalysis;
    const tested = r.constructResults.map((row) => row.construct);
    expect(tested).toContain("Image");
    expect(tested).toContain("Satisfaction");
    expect(tested).toContain("Loyalty");
    expect(r.borrowing.has("Satisfaction")).toBe(true);
    expect(r.borrowing.has("Loyalty")).toBe(true);
    expect(r.borrowing.has("Image")).toBe(false);

    const rNb = assessCta(model, { nboot: 10, seed: 123, borrow: false }) as CtaAnalysis;
    expect(rNb.skipped).toContain("Satisfaction");
    expect(rNb.skipped).toContain("Loyalty");
    expect(rNb.constructResults.map((row) => row.construct)).toEqual(["Image"]);
  });

  test("formative focal cannot borrow", () => {
    const model = estimateRegistryModel("M12");
    const r = assessCta(model, { constructs: "Value", nboot: 10, seed: 123 }) as CtaAnalysis;
    expect(r.skipped).toContain("Value");
    expect(r.constructResults.length).toBe(0);
  });

  test("no reflective donor for a 3-indicator focal is skipped", () => {
    const model = estimateRegistryModel("M13");
    const r = assessCta(model, { constructs: "Expectation", nboot: 10, seed: 123 }) as CtaAnalysis;
    expect(r.skipped).toContain("Expectation");
    expect(r.constructResults.length).toBe(0);
  });

  test("HOC with too few LOCs is skipped in an empty record (not null)", () => {
    const model = estimateRegistryModel("M4");
    const r = assessCta(model, { constructs: "Quality", nboot: 10, seed: 123 });
    expect(r).not.toBeNull();
    const res = r as CtaAnalysis;
    expect(res.kind).toBe("cta_analysis");
    expect(res.skipped).toContain("Quality");
    expect(res.constructResults.length).toBe(0);
    expect(res.tetradDetails.size).toBe(0);
  });

  test("a LOC under a HOC is tested independently over its raw indicators", () => {
    const model = estimateRegistryModel("M4");
    const r = assessCta(model, { constructs: "Image", nboot: 10, seed: 123 }) as CtaAnalysis;
    expect(r.constructResults.map((row) => row.construct)).toEqual(["Image"]);
    expect(r.constructResults[0]!.indicators).toBe(5);
  });
});

// --- Internal tetrad machinery (spec: 559-665) --------------------------------------

describe("tetrad machinery", () => {
  test("enumerateTetrads counts", () => {
    const four = enumerateTetrads(["a", "b", "c", "d"]);
    expect(four.length).toBe(2);
    expect(four.map((t) => t.tetradId)).toEqual([1, 2]);
    const five = enumerateTetrads(["a", "b", "c", "d", "e"]);
    expect(five.length).toBe(10);
  });

  test("tau_1342 exact oracle", () => {
    const cov = [
      [1.0, 0.5, 0.3, 0.2],
      [0.5, 1.0, 0.4, 0.6],
      [0.3, 0.4, 1.0, 0.7],
      [0.2, 0.6, 0.7, 1.0],
    ];
    const names = ["a", "b", "c", "d"];
    const spec: TetradSpec = { i: "a", j: "b", k: "c", l: "d", tetradId: 3 };
    const vals = computeTetrads(cov, names, [spec]);
    expect(Math.abs(vals[0]! - 0.1)).toBeLessThan(1e-15);
  });

  test("enumerateBorrowedTetrads", () => {
    const allPattern = enumerateBorrowedTetrads(["x1", "x2", "x3"], ["x4"], "all");
    expect(allPattern.length).toBe(2);
    const tau = enumerateBorrowedTetrads(["x1", "x2"], ["x3", "x4"], "tau_1342");
    expect(tau.length).toBe(1);
    const t = tau[0]!;
    expect([t.i, t.j, t.k, t.l, t.tetradId]).toEqual(["x1", "x2", "x3", "x4", 3]);
  });

  test("structurally connected order (targets then sources, unique)", () => {
    const model = estimateRegistryModel("M1");
    expect(getStructurallyConnected("Image", model)).toEqual([
      "Expectation",
      "Satisfaction",
      "Loyalty",
    ]);
  });

  test("findDonor returns null for a formative focal", () => {
    const model = estimateRegistryModel("M12");
    const info = resolveIndicators("Value", model);
    expect(findDonor("Value", info, model)).toBeNull();
  });

  test("common-factor tetrads are near zero", () => {
    const gen = mulberry32(42);
    const normal = () => {
      // Box-Muller off mulberry32 for a deterministic standard normal.
      const u1 = Math.max(gen(), Number.MIN_VALUE);
      const u2 = gen();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };
    const n = 500;
    const lambdas = [0.8, 0.7, 0.9, 0.85];
    const x: number[][] = [];
    for (let r = 0; r < n; r++) {
      const factor = normal();
      x.push(lambdas.map((lam) => lam * factor + Math.sqrt(1 - lam * lam) * normal()));
    }
    const covMat = toRows(csCov(fromRows(x)));
    const names = ["x1", "x2", "x3", "x4"];
    const vals = computeTetrads(covMat, names, enumerateTetrads(names));
    expect(vals.every((v) => Math.abs(v) < 0.05)).toBe(true);
  });
});

// --- Record surface (spec: 528-553) -------------------------------------------------

describe("record surface", () => {
  test("record renders and self-describes; clean run emits no warnings", () => {
    const c = caseByName("m1_borrow_default");
    const model = estimateRegistryModel("M1");
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    let result: CtaAnalysis | null;
    try {
      result = assessCta(model, { nboot: c.nboot, seed: c.seed, draws: draws(c) });
      expect(warn.mock.calls.length).toBe(0);
    } finally {
      warn.mockRestore();
    }
    expect(result).not.toBeNull();
    const res = result as CtaAnalysis;
    expect(res.kind).toBe("cta_analysis");
    const text = String(res);
    expect(text).toContain("Confirmatory Tetrad Analysis");
    expect(text).toContain("Borrowing");
    expect(text).toContain("Reflective");
    const summary = res.summarize();
    expect(summary).toContain("Detailed Results");
    expect(summary).toContain("Borrowed");
    expect(summary).toContain("Tetrad");
  });
});
