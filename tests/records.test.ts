/**
 * Unit tests for the shared record-rendering helpers (src/records.ts), ported
 * from the py port's records.py: aligned rounded table rendering, R-style CI
 * column labels, and column-major non-zero path lines.
 */

import { describe, expect, test } from "bun:test";
import { namedMatrix } from "@seminr/core";
import { ciColumnLabels, formatTable, nonzeroPathLines } from "../src/records.ts";

describe("formatTable", () => {
  test("renders an aligned table with rounded cells", () => {
    const table = namedMatrix(["Image", "Sat"], ["beta", "p"], [
      [0.12345, 0.0499],
      [-1.5, 0.5],
    ]);
    expect(formatTable(table)).toBe(
      [
        "        beta     p",
        "Image  0.123 0.050",
        "Sat   -1.500 0.500",
      ].join("\n"),
    );
  });

  test("honors the digits argument", () => {
    const table = namedMatrix(["r"], ["c"], [[0.98765]]);
    expect(formatTable(table, 1)).toBe(["    c", "r 1.0"].join("\n"));
  });

  test("widens columns to fit long headers and long row names", () => {
    const table = namedMatrix(["a", "longname"], ["wide-header"], [[1], [2]]);
    const lines = formatTable(table).split("\n");
    expect(lines[0]).toBe("         wide-header");
    expect(lines[1]).toBe("a              1.000");
    expect(lines[2]).toBe("longname       2.000");
  });
});

describe("ciColumnLabels", () => {
  test("formats R-style percentile labels", () => {
    expect(ciColumnLabels(0.05)).toEqual(["2.5% CI", "97.5% CI"]);
    expect(ciColumnLabels(0.1)).toEqual(["5% CI", "95% CI"]);
    expect(ciColumnLabels(0.01)).toEqual(["0.5% CI", "99.5% CI"]);
  });
});

describe("nonzeroPathLines", () => {
  test("lists non-zero cells in R column-major order", () => {
    const paths = namedMatrix(["A", "B"], ["X", "Y"], [
      [0.5, 0],
      [0, -0.25],
    ]);
    expect(nonzeroPathLines(paths, "  ")).toEqual([
      "  A -> X: 0.5000",
      "  B -> Y: -0.2500",
    ]);
  });

  test("returns no lines for an all-zero matrix", () => {
    const paths = namedMatrix(["A"], ["X"], [[0]]);
    expect(nonzeroPathLines(paths, "")).toEqual([]);
  });
});
