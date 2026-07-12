/**
 * CART engine parity tests against rpart 4.1.24 goldens (py plan F15).
 *
 * Fixture tests/fixtures/cart/cart.json carries, per case, the input data
 * verbatim plus rpart's frame / splits / where grown with COA's control
 * (minsplit = 2, cp = 0). Structure (node ids, variable choices,
 * competitor/surrogate counts, leaf assignment) must match EXACTLY; floating
 * columns match at rel 1e-9 / abs 1e-12. The real case (c1_pd_real) is the C1
 * predictive-deviance data — 613 frame nodes, 1203 split rows.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import { rpartAnova, type CartTree } from "../src/cart.ts";
import { FIXTURES_DIR } from "./helpers/fixtures.ts";

interface CartCase {
  name: string;
  response: string;
  data: { cols: string[]; values: number[][] };
  frame: {
    nodeIds: number[];
    var: string[];
    n: number[];
    wt: number[];
    dev: number[];
    yval: number[];
    complexity: (number | null)[];
    ncompete: number[];
    nsurrogate: number[];
  };
  splits: {
    var: string[];
    count: number[];
    ncat: number[];
    improve: number[];
    index: number[];
    adj: number[];
  } | null;
  where: number[];
}

const fx = JSON.parse(readFileSync(join(FIXTURES_DIR, "cart", "cart.json"), "utf8")) as {
  cases: CartCase[];
};

const CASE_NAMES = [
  "midpoint",
  "tie_duplicate_predictor",
  "constant_y",
  "correlated_surrogates",
  "deep_five_predictors",
  "c1_pd_real",
];

function getCase(name: string): CartCase {
  return fx.cases.find((c) => c.name === name)!;
}

function inputs(c: CartCase): { x: number[][]; y: number[]; names: string[] } {
  const predNames = c.data.cols.filter((col) => col !== c.response);
  const predIdx = predNames.map((col) => c.data.cols.indexOf(col));
  const yIdx = c.data.cols.indexOf(c.response);
  return {
    x: c.data.values.map((row) => predIdx.map((j) => row[j]!)),
    y: c.data.values.map((row) => row[yIdx]!),
    names: predNames,
  };
}

function grow(c: CartCase): CartTree {
  const { x, y, names } = inputs(c);
  return rpartAnova(x, y, names, { minsplit: 2, cp: 0 });
}

/** NaN-aware allclose at rel 1e-9 / abs 1e-12 (R NA arrives as JSON null). */
function assertAllClose(got: readonly number[], want: readonly (number | null)[], label: string) {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < got.length; i++) {
    const w = want[i] === null ? NaN : want[i]!;
    const a = got[i]!;
    if (Number.isNaN(w)) {
      if (!Number.isNaN(a)) throw new Error(`${label}[${i}]: got ${a}, expected NaN`);
    } else if (!(Math.abs(a - w) <= Math.max(1e-9 * Math.max(Math.abs(a), Math.abs(w)), 1e-12))) {
      throw new Error(`${label}[${i}]: got ${a}, expected ${w}`);
    }
  }
}

describe("rpart golden parity", () => {
  for (const name of CASE_NAMES) {
    test(`frame parity: ${name}`, () => {
      const c = getCase(name);
      const tree = grow(c);
      expect([...tree.nodeIds]).toEqual(c.frame.nodeIds);
      expect([...tree.var]).toEqual(c.frame.var);
      expect([...tree.n]).toEqual(c.frame.n);
      assertAllClose(tree.wt, c.frame.wt, "wt");
      assertAllClose(tree.dev, c.frame.dev, "dev");
      assertAllClose(tree.yval, c.frame.yval, "yval");
      assertAllClose(tree.complexity, c.frame.complexity, "complexity");
      expect([...tree.ncompete]).toEqual(c.frame.ncompete);
      expect([...tree.nsurrogate]).toEqual(c.frame.nsurrogate);
    });

    test(`splits parity: ${name}`, () => {
      const c = getCase(name);
      const tree = grow(c);
      if (c.splits === null) {
        expect(tree.splitVar.length).toBe(0);
        return;
      }
      expect([...tree.splitVar]).toEqual(c.splits.var);
      assertAllClose(tree.splitCount, c.splits.count, "count");
      expect(tree.splitNcat.map((v) => v)).toEqual(c.splits.ncat);
      assertAllClose(tree.splitImprove, c.splits.improve, "improve");
      assertAllClose(tree.splitIndex, c.splits.index, "index");
      assertAllClose(tree.splitAdj, c.splits.adj, "adj");
    });

    test(`where parity: ${name}`, () => {
      const c = getCase(name);
      const tree = grow(c);
      expect([...tree.where]).toEqual(c.where);
    });
  }
});

describe("structural invariants and unit behavior", () => {
  test("split index lands on the midpoint", () => {
    const tree = rpartAnova([[1], [2], [4], [8]], [1, 2, 3, 10], ["x"], { minsplit: 2, cp: 0 });
    expect(tree.splitIndex[0]).toBe(6);
  });

  test("constant y yields a single leaf", () => {
    const x = [1, 2, 3, 4, 5, 6].map((v) => [v]);
    const tree = rpartAnova(x, [5, 5, 5, 5, 5, 5], ["x"], { minsplit: 2, cp: 0 });
    expect([...tree.nodeIds]).toEqual([1]);
    expect([...tree.var]).toEqual(["<leaf>"]);
    expect(tree.splitVar.length).toBe(0);
    expect([...tree.where]).toEqual([1, 1, 1, 1, 1, 1]);
  });

  test("minsplit is respected", () => {
    const x = [1, 2, 3, 4, 5, 6].map((v) => [v]);
    const tree = rpartAnova(x, [1, 1.2, 0.8, 9, 9.2, 8.8], ["x"], { minsplit: 6, cp: 0 });
    expect([...tree.nodeIds]).toEqual([1, 2, 3]);
    expect([...tree.var]).toEqual(["x", "<leaf>", "<leaf>"]);
  });

  test("where indexes leaf frame rows", () => {
    const tree = rpartAnova([[1], [2], [4], [8]], [1, 2, 3, 10], ["x"], { minsplit: 2, cp: 0 });
    for (const w of tree.where) expect(tree.var[w - 1]).toBe("<leaf>");
    const leafRows = [...new Set(tree.where)];
    expect(leafRows.reduce((s, r) => s + tree.n[r - 1]!, 0)).toBe(4);
  });

  test("children follow 2k/2k+1 numbering", () => {
    const tree = rpartAnova([[1], [2], [4], [8]], [1, 2, 3, 10], ["x"], { minsplit: 2, cp: 0 });
    const ids = new Set(tree.nodeIds);
    tree.nodeIds.forEach((nodeId, i) => {
      if (tree.var[i] !== "<leaf>") {
        expect(ids.has(2 * nodeId)).toBe(true);
        expect(ids.has(2 * nodeId + 1)).toBe(true);
      }
    });
  });

  test("split rows form one block per internal frame row", () => {
    const tree = grow(getCase("correlated_surrogates"));
    let expectedRows = 0;
    for (let i = 0; i < tree.nodeIds.length; i++) {
      if (tree.var[i] !== "<leaf>") expectedRows += 1 + tree.ncompete[i]! + tree.nsurrogate[i]!;
    }
    expect(tree.splitVar.length).toBe(expectedRows);
    let offset = 0;
    for (let i = 0; i < tree.nodeIds.length; i++) {
      if (tree.var[i] === "<leaf>") continue;
      expect(tree.splitVar[offset]).toBe(tree.var[i]!);
      offset += 1 + tree.ncompete[i]! + tree.nsurrogate[i]!;
    }
  });

  test("dev is the node SS and yval the node mean", () => {
    const c = getCase("midpoint");
    const tree = grow(c);
    const { y } = inputs(c);
    const mean = y.reduce((s, v) => s + v, 0) / y.length;
    const rootSs = y.reduce((s, v) => s + (v - mean) * (v - mean), 0);
    expect(Math.abs(tree.dev[0]! - rootSs) / rootSs).toBeLessThan(1e-12);
    expect(Math.abs(tree.yval[0]! - mean) / Math.abs(mean)).toBeLessThan(1e-12);
  });

  test("splitRowOffsets matches the cumulative block layout", () => {
    const tree = grow(getCase("correlated_surrogates"));
    const offsets = tree.splitRowOffsets();
    let pos = 1;
    for (let i = 0; i < tree.nodeIds.length; i++) {
      expect(offsets[i]).toBe(pos);
      if (tree.var[i] !== "<leaf>") pos += 1 + tree.ncompete[i]! + tree.nsurrogate[i]!;
    }
  });

  test("rejects NaN inputs and shape mismatches", () => {
    expect(() => rpartAnova([[NaN]], [1], ["x"], {})).toThrow(/complete data/);
    expect(() => rpartAnova([[1], [2]], [1], ["x"], {})).toThrow(/aligned/);
    expect(() => rpartAnova([[1, 2]], [1], ["x"], {})).toThrow(/var_names|varNames/);
  });
});
