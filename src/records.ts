/**
 * Shared rendering helpers for result records.
 *
 * R prints result matrices through seminr's `print.table_output` (rounded,
 * name-aligned). Every seminrExtras-ts record renders its tables through
 * `formatTable` so `toString()`/`summarize()` stay consistent across features
 * (ported from the py port's records.py, which mirrors the same formatter).
 */

import type { NamedMatrix } from "@seminr/core";

/** Render a NamedMatrix as an aligned text table with rounded cells. */
export function formatTable(table: NamedMatrix, digits: number = 3): string {
  const cells = table.values.map((row) => row.map((value) => value.toFixed(digits)));
  const colWidths = table.cols.map((col, j) =>
    Math.max(col.length, ...cells.map((row) => row[j]!.length)),
  );
  const rowWidth = Math.max(0, ...table.rows.map((r) => r.length));
  const header = [
    " ".repeat(rowWidth),
    ...table.cols.map((col, j) => col.padStart(colWidths[j]!)),
  ].join(" ");
  const lines = [header];
  for (let i = 0; i < table.rows.length; i++) {
    const body = table.cols.map((_, j) => cells[i]![j]!.padStart(colWidths[j]!));
    lines.push([table.rows[i]!.padEnd(rowWidth), ...body].join(" "));
  }
  return lines.join("\n");
}

/** R-`%g`-style number rendering (6 significant digits, trailing zeros stripped). */
export function gFormat(v: number): string {
  return String(Number(v.toPrecision(6)));
}

/** R-style percentile-CI column labels, e.g. `alpha=0.05` -> `["2.5% CI", "97.5% CI"]`. */
export function ciColumnLabels(alpha: number): [string, string] {
  return [`${gFormat((alpha / 2) * 100)}% CI`, `${gFormat((1 - alpha / 2) * 100)}% CI`];
}

/** Non-zero cells in R's `which(arr.ind = TRUE)` column-major order. */
export function nonzeroPathLines(pathMat: NamedMatrix, indent: string): string[] {
  const lines: string[] = [];
  for (let j = 0; j < pathMat.cols.length; j++) {
    for (let i = 0; i < pathMat.rows.length; i++) {
      const coef = pathMat.values[i]![j]!;
      if (coef !== 0) {
        lines.push(`${indent}${pathMat.rows[i]} -> ${pathMat.cols[j]}: ${coef.toFixed(4)}`);
      }
    }
  }
  return lines;
}
