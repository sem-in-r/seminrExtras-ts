/**
 * Congruence testing (Slice 3): congruenceTest against R goldens.
 *
 * Parity fixtures (tests/fixtures/congruence/congruence.json) carry the R
 * bootstrap row-index streams (1-based; converted here) injected via `draws`.
 * The bootstrap re-estimation matches R's `rerun` to machine precision, so
 * parity holds at rel 1e-9 / abs 1e-9. Behavioral cases follow the R suite
 * (test-congruence.R) via the py port's test_congruence.py, including its
 * independent `referenceRc` oracle that guards estimate-to-pair labeling.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, spyOn, test } from "bun:test";
import { rhoCAve, type NamedMatrix, type PlsModel } from "@seminr/core";
import { colCor } from "@seminr/core/math";
import { congruenceTest, type CongruenceTest } from "../src/featureCongruence.ts";
import { FIXTURES_DIR } from "./helpers/fixtures.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

const HEADING = "Congruence coefficient test (Franke, Sarstedt & Danks, 2021)";

interface TableRef {
  rows: string[];
  cols: string[];
  values: number[][];
}

interface CongruenceCase {
  name: string;
  nboot: number;
  seed: number;
  alpha: number;
  threshold: number;
  bootIndices: number[][];
  results: TableRef;
}

const fx = JSON.parse(
  readFileSync(join(FIXTURES_DIR, "congruence", "congruence.json"), "utf8"),
) as { nObs: number; cases: CongruenceCase[] };

const c1Model = estimateRegistryModel("C1");

function caseDraws(c: CongruenceCase): number[][] {
  return c.bootIndices.map((row) => row.map((i) => i - 1));
}

/** Python math.isclose semantics. */
function isClose(a: number, b: number, relTol: number, absTol: number): boolean {
  return Math.abs(a - b) <= Math.max(relTol * Math.max(Math.abs(a), Math.abs(b)), absTol);
}

// --- name-based oracle (ports reference_rc from test-congruence.R:195-199) -------

/** cor matrix of construct scores with rhoC on the diagonal, then Franke Eq. 2. */
function referenceRc(model: PlsModel, x: string, y: string): number {
  const scores = model.constructScores;
  const names = [...scores.cols];
  const mat = colCor(scores.values, scores.values).map((row) => [...row]);
  const rca = rhoCAve(model.outerLoadings, names);
  for (let i = 0; i < names.length; i++) mat[i]![i] = rca.values[i]![0]!;
  const xi = names.indexOf(x);
  const yi = names.indexOf(y);
  const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i]!, 0);
  const cx = mat.map((row) => row[xi]!);
  const cy = mat.map((row) => row[yi]!);
  return dot(cx, cy) / Math.sqrt(dot(cx, cx) * dot(cy, cy));
}

// --- parity ---------------------------------------------------------------------

describe("congruenceTest R parity", () => {
  for (const [idx, c] of fx.cases.entries()) {
    test(`matches R golden: ${c.name} (case ${idx})`, () => {
      const result = congruenceTest(c1Model, {
        nboot: c.nboot,
        alpha: c.alpha,
        threshold: c.threshold,
        draws: caseDraws(c),
      });
      expect(result).not.toBeNull();
      const got = (result as CongruenceTest).results;
      const ref = c.results;
      // Labels must match exactly: double-space pair labels and alpha-dependent
      // CI column names ("2.5% CI"/"97.5% CI" vs "5% CI"/"95% CI").
      expect(got.rows).toEqual(ref.rows);
      expect(got.cols).toEqual(ref.cols);
      for (let i = 0; i < ref.rows.length; i++) {
        for (let j = 0; j < ref.cols.length; j++) {
          const g = got.values[i]![j]!;
          const r = ref.values[i]![j]!;
          if (!isClose(g, r, 1e-9, 1e-9)) {
            throw new Error(`${c.name}[${ref.rows[i]}, ${ref.cols[j]}]: got ${g}, expected ${r}`);
          }
        }
      }
    });
  }
});

// --- validation / failure modes (spec: test-congruence.R) -----------------------

describe("validation / failure modes", () => {
  test("rejects a non-model with a warning", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(congruenceTest({ not: "a_model" }, { nboot: 5 })).toBeNull();
      expect(
        warn.mock.calls.some((c) => String(c[0]).includes("only works with SEMinR models")),
      ).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });

  test("rejects null with a warning", () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(congruenceTest(null, { nboot: 5 })).toBeNull();
      expect(
        warn.mock.calls.some((c) => String(c[0]).includes("only works with SEMinR models")),
      ).toBe(true);
    } finally {
      warn.mockRestore();
    }
  });
});

// --- higher-order models are accepted -------------------------------------------

describe("higher-order models", () => {
  test("HOC model is accepted (no warning) and yields all pairs", () => {
    const hocModel = estimateRegistryModel("M4");
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      const result = congruenceTest(hocModel, { nboot: 5, seed: 123 });
      expect(result).not.toBeNull();
      // 3 construct scores (Quality/Satisfaction/Loyalty) -> choose(3, 2) = 3 pairs.
      const results = (result as CongruenceTest).results;
      expect(results.values.length).toBe(3);
      expect(results.cols.length).toBe(6);
      expect(warn.mock.calls.length).toBe(0);
    } finally {
      warn.mockRestore();
    }
  });
});

// --- reproducibility (default RNG path) -----------------------------------------

describe("default-rng path", () => {
  test("same seed reproduces; different seed changes SD but not the estimate", () => {
    const a = congruenceTest(c1Model, { nboot: 20, seed: 42 }) as CongruenceTest;
    const b = congruenceTest(c1Model, { nboot: 20, seed: 42 }) as CongruenceTest;
    const c = congruenceTest(c1Model, { nboot: 20, seed: 99 }) as CongruenceTest;
    expect(a.results.values).toEqual(b.results.values);
    const sd = a.results.cols.indexOf("Bootstrap SD");
    const colOf = (m: NamedMatrix, j: number) => m.values.map((row) => row[j]!);
    expect(colOf(a.results, sd)).not.toEqual(colOf(c.results, sd));
    // Original estimate is deterministic (independent of the bootstrap seed).
    const estA = colOf(a.results, 0);
    const estC = colOf(c.results, 0);
    for (let i = 0; i < estA.length; i++) expect(estA[i]!).toBeCloseTo(estC[i]!, 12);
  });
});

// --- original estimates land on the correct construct pair ----------------------

describe("estimate labeling", () => {
  test("original estimates match the reference oracle", () => {
    const result = congruenceTest(c1Model, { nboot: 5, seed: 123 }) as CongruenceTest;
    const est = result.results.cols.indexOf("Original Est.");
    for (let i = 0; i < result.results.rows.length; i++) {
      const label = result.results.rows[i]!;
      const [x, y] = label.split("->").map((p) => p.trim());
      const ref = referenceRc(c1Model, x!, y!);
      expect(isClose(result.results.values[i]![est]!, ref, 1e-8, 1e-10)).toBe(true);
    }
  });
});

// --- Diff / T Stat internal consistency -----------------------------------------

describe("Diff / T Stat consistency", () => {
  test("Diff = threshold - |est|; T Stat = Diff / SD", () => {
    const threshold = 0.95;
    const result = congruenceTest(c1Model, { nboot: 20, seed: 7, threshold }) as CongruenceTest;
    const cols = result.results.cols;
    const vals = result.results.values;
    const est = cols.indexOf("Original Est.");
    const diff = cols.indexOf("Diff");
    const sd = cols.indexOf("Bootstrap SD");
    const t = cols.indexOf("T Stat.");
    for (const row of vals) {
      expect(row[diff]!).toBeCloseTo(threshold - Math.abs(row[est]!), 12);
      if (row[sd]! > 0) expect(row[t]!).toBeCloseTo(row[diff]! / row[sd]!, 10);
    }
  });
});

// --- record surface -------------------------------------------------------------

describe("record surface", () => {
  test("record renders and self-describes", () => {
    const result = congruenceTest(c1Model, { nboot: 5, seed: 123 }) as CongruenceTest;
    expect(result.kind).toBe("congruence_test");
    const text = String(result);
    expect(text).toContain(HEADING);
    expect(text).toContain("Original Est.");
    expect(result.summarize()).toBe(text);
  });
});
