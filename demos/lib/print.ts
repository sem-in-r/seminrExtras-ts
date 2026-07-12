/**
 * Plain-text rendering helpers for the seminrExtras demos.
 *
 * A blank-line-prefixed section header, a pure aligned-table renderer for
 * `NamedMatrix` values (NaN cells printed as "." — R's na.print default), and
 * a one-line confirmation for rendered SVG plots. Ported from the py demo lib;
 * kept in the demos, never in `src` — the package returns data objects and the
 * demos render them.
 */

import type { NamedMatrix, SvgPlot } from "@seminr/core";

/** A blank-line-prefixed `== title ==` section header. */
export function heading(title: string): string {
  return `\n== ${title} ==`;
}

/** Render a NamedMatrix as an aligned text table; NaN cells as `naPrint`. */
export function formatMatrix(m: NamedMatrix, digits = 3, naPrint = "."): string {
  const header = ["", ...m.cols.map(String)];
  const body = m.rows.map((row, i) => [
    String(row),
    ...m.values[i]!.map((v) => (Number.isNaN(v) ? naPrint : v.toFixed(digits))),
  ]);
  const table = [header, ...body];
  const widths = header.map((_, j) => Math.max(...table.map((line) => line[j]!.length)));
  return table
    .map((line) => line.map((cell, j) => cell.padStart(widths[j]!)).join("  "))
    .join("\n");
}

/** One-line confirmation for a rendered SVG plot (the plot is never shown). */
export function rendered(p: SvgPlot | null, what: string): string {
  if (p === null) return `Rendered ${what} (nothing to plot)`;
  const m = p.svg.match(/width="(\d+)" height="(\d+)"/);
  return `Rendered ${what} (SVG ${m?.[1]}x${m?.[2]})`;
}
