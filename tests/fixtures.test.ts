/**
 * Unit tests for the parity-test harness (tests/helpers/fixtures.ts): fixture
 * decoding of jsonlite {rows, cols, values} nodes and label-aligned matrix
 * comparison. Ported from the py port's tests of tests/helpers/fixtures.py.
 */

import { describe, expect, test } from "bun:test";
import { namedMatrix } from "@seminr/core";
import {
  PLS_TOL,
  expectNamedClose,
  loadModelFixture,
  toMatrix,
} from "./helpers/fixtures.ts";

describe("toMatrix", () => {
  test("decodes a full matrix node", () => {
    const m = toMatrix({
      rows: ["a", "b"],
      cols: ["x", "y", "z"],
      values: [
        [1, 2, 3],
        [4, 5, 6],
      ],
    });
    expect(m.rows).toEqual(["a", "b"]);
    expect(m.cols).toEqual(["x", "y", "z"]);
    expect(m.values).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  test("re-inflates a fully collapsed k x 1 matrix (flat values, scalar col)", () => {
    // jsonlite auto_unbox collapses a single-column matrix to a flat vector
    // and its lone column name to a bare string.
    const m = toMatrix({ rows: ["a", "b", "c"], cols: "w", values: [1, 2, 3] });
    expect(m.rows).toEqual(["a", "b", "c"]);
    expect(m.cols).toEqual(["w"]);
    expect(m.values).toEqual([[1], [2], [3]]);
  });

  test("re-inflates a collapsed single-row matrix (scalar row name)", () => {
    const m = toMatrix({ rows: "r", cols: ["x", "y"], values: [[1, 2]] });
    expect(m.rows).toEqual(["r"]);
    expect(m.cols).toEqual(["x", "y"]);
    expect(m.values).toEqual([[1, 2]]);
  });

  test("re-inflates scalar row entries (k x 1 with nested scalars)", () => {
    const m = toMatrix({ rows: ["a", "b"], cols: ["w"], values: [1, 2] });
    expect(m.values).toEqual([[1], [2]]);
  });

  test("maps R NA (JSON null or the string 'NA') to NaN", () => {
    const m = toMatrix({
      rows: ["a"],
      cols: ["x", "y", "z"],
      values: [[null, "NA", 1.5]],
    });
    expect(Number.isNaN(m.values[0]![0]!)).toBe(true);
    expect(Number.isNaN(m.values[0]![1]!)).toBe(true);
    expect(m.values[0]![2]).toBe(1.5);
  });

  test("defaults missing row/col names to index strings", () => {
    const m = toMatrix({ rows: null, cols: null, values: [[1, 2]] });
    expect(m.rows).toEqual(["0"]);
    expect(m.cols).toEqual(["0", "1"]);
  });

  test("rejects a shape/name mismatch", () => {
    expect(() =>
      toMatrix({ rows: ["a", "b"], cols: ["x"], values: [[1]] }),
    ).toThrow(/shape/);
  });
});

describe("expectNamedClose", () => {
  const expected = namedMatrix(["a", "b"], ["x", "y"], [
    [1, 2],
    [3, 4],
  ]);

  test("passes on an exact match", () => {
    expectNamedClose(expected, expected, PLS_TOL, "self");
  });

  test("aligns by label under benign row/col reordering", () => {
    const actual = namedMatrix(["b", "a"], ["y", "x"], [
      [4, 3],
      [2, 1],
    ]);
    expectNamedClose(actual, expected, PLS_TOL, "reordered");
  });

  test("fails when a cell differs beyond tolerance", () => {
    const actual = namedMatrix(["a", "b"], ["x", "y"], [
      [1, 2],
      [3, 4.001],
    ]);
    expect(() => expectNamedClose(actual, expected, 1e-5, "off")).toThrow(
      /off\[b, y\]/,
    );
  });

  test("falls back to positional rows when row labels differ (Rsq case)", () => {
    // seminr-ts names rSquared rows Rsq/AdjRsq where R uses R^2/AdjR^2.
    const r = namedMatrix(["R^2", "AdjR^2"], ["x"], [[0.5], [0.4]]);
    const ts = namedMatrix(["Rsq", "AdjRsq"], ["x"], [[0.5], [0.4]]);
    expectNamedClose(ts, r, PLS_TOL, "rsq");
  });

  test("requires NaN to match NaN on both sides", () => {
    const withNan = namedMatrix(["a", "b"], ["x", "y"], [
      [1, 2],
      [3, NaN],
    ]);
    expectNamedClose(withNan, withNan, PLS_TOL, "nan-nan");
    expect(() => expectNamedClose(expected, withNan, PLS_TOL, "num-vs-na")).toThrow(
      /expected NA/,
    );
    expect(() => expectNamedClose(withNan, expected, PLS_TOL, "na-vs-num")).toThrow(
      /num-vs-na|na-vs-num/,
    );
  });

  test("fails on column-label set mismatch", () => {
    const other = namedMatrix(["a", "b"], ["x", "q"], [
      [1, 2],
      [3, 4],
    ]);
    expect(() => expectNamedClose(other, expected, PLS_TOL, "cols")).toThrow(
      /column labels differ/,
    );
  });

  test("fails on shape mismatch", () => {
    const other = namedMatrix(["a"], ["x", "y"], [[1, 2]]);
    expect(() => expectNamedClose(other, expected, PLS_TOL, "shape")).toThrow(
      /shape/,
    );
  });
});

describe("loadModelFixture", () => {
  test("parses a committed model golden", () => {
    const f = loadModelFixture("M1");
    expect(f.id).toBe("M1");
    expect(f.dataset).toBe("mobi");
    const pc = toMatrix(f.pathCoef);
    expect(pc.rows.length).toBeGreaterThan(0);
    expect(pc.rows).toEqual(pc.cols);
  });

  test("throws for a missing golden", () => {
    expect(() => loadModelFixture("NOPE")).toThrow();
  });
});
