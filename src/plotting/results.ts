/**
 * SVG ports of seminrExtras' base-R statistical plots (R `plot.*` methods),
 * following the py port's matplotlib layer (plotting/results.py) shape for
 * shape: same panels, series, reference lines, titles and "nothing to plot"
 * null returns. Every function consumes only public fields of the feature
 * result records (plus the record-layer geometry helpers
 * `CartTree.plotLayout`/label methods, `groupScoreMeans` and `crFdhLine`).
 *
 * Each function returns a `@seminr/core` SvgPlot (`.svg` source + `.save()`),
 * or null on R's `message()` + no-op paths. `plot()` dispatches on
 * `record.kind`. Data-bearing shapes carry `class`/`data-value` attributes;
 * tests assert on those, never on pixel geometry.
 */

import type { NamedMatrix, SvgPlot } from "@seminr/core";
import { quantile } from "@compstats/core/stats";
import type { CipmaAnalysis } from "../featureCipma.ts";
import { groupScoreMeans, type CoaAnalysis } from "../featureCoa.ts";
import type { CtaAnalysis } from "../featureCta.ts";
import type { FimixAnalysis, FimixComparison } from "../featureFimix.ts";
import { computeCeFdh, crFdhLine, type NcaAnalysis, type NcaEsse } from "../featureNca.ts";
import type { PcmAnalysis } from "../featurePcm.ts";
import type { PosAnalysis, PosComparison } from "../featurePos.ts";
import { gray, paletteColors, posPalette } from "./rColors.ts";
import {
  axes,
  DEVICE_SIZE,
  extendRange,
  frame,
  panelGrid,
  plotRegion,
  starPoints,
  SvgFigure,
  textLegend,
  xPos,
  yPos,
  type Frame,
  type LegendCorner,
  type LegendEntry,
  type PanelRect,
} from "./svg.ts";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const DEVICE: PanelRect = { x0: 0, y0: 0, width: DEVICE_SIZE, height: DEVICE_SIZE };

function col(m: NamedMatrix, name: string): number[] {
  const j = m.cols.indexOf(name);
  if (j === -1) throw new Error(`Unknown column name: ${name}`);
  return m.values.map((row) => row[j]!);
}

function finiteOf(values: readonly number[]): number[] {
  return values.filter((v) => Number.isFinite(v));
}

/** Dashed horizontal reference line across the plot region. */
function hline(fig: SvgFigure, f: Frame, y: number, attrs: Record<string, string | number>): void {
  const [left, right] = plotRegion(f);
  fig.line(left, yPos(f, y), right, yPos(f, y), { class: "refline", ...attrs });
}

/** Vertical reference line across the plot region. */
function vline(fig: SvgFigure, f: Frame, x: number, attrs: Record<string, string | number>): void {
  const [, , top, bottom] = plotRegion(f);
  fig.line(xPos(f, x), top, xPos(f, x), bottom, { class: "refline", ...attrs });
}

/** A data-marked series polyline plus optional per-vertex markers. */
function seriesLine(
  fig: SvgFigure,
  f: Frame,
  xs: readonly number[],
  ys: readonly number[],
  attrs: Record<string, string | number>,
  marker?: "circle" | "triangle",
): void {
  const pts: [number, number][] = [];
  for (let i = 0; i < xs.length; i++) {
    if (Number.isNaN(ys[i]!)) continue;
    pts.push([xPos(f, xs[i]!), yPos(f, ys[i]!)]);
  }
  if (pts.length > 1) fig.polyline(pts, { class: "series", ...attrs });
  if (marker === undefined) return;
  const color = attrs["stroke"] ?? "black";
  for (const [cx, cy] of pts) {
    if (marker === "circle") {
      fig.circle(cx, cy, 3, { class: "marker", fill: color });
    } else {
      fig.polygon(
        [
          [cx, cy - 3.5],
          [cx - 3, cy + 2.5],
          [cx + 3, cy + 2.5],
        ],
        { class: "marker", fill: color },
      );
    }
  }
}

/**
 * Grouped (`beside = TRUE`) bar chart into `f`; `matrix` is groups x series
 * (row-major arrays). NaN heights draw as 0 (py `nan_to_num`). Bars carry
 * `class="bar"` and `data-value` (the drawn height).
 */
function groupedBars(
  fig: SvgFigure,
  f: Frame,
  nSeries: number,
  matrix: readonly (readonly number[])[],
  colors: readonly string[],
): void {
  const width = 0.8 / nSeries;
  const y0px = yPos(f, Math.max(f.ylim[0], 0));
  for (let s = 0; s < nSeries; s++) {
    const offset = (s - (nSeries - 1) / 2) * width;
    for (let g = 0; g < matrix.length; g++) {
      const raw = matrix[g]![s]!;
      const h = Number.isNaN(raw) ? 0 : raw;
      const xLeft = xPos(f, g + offset - width / 2);
      const xRight = xPos(f, g + offset + width / 2);
      const yTop = yPos(f, Math.max(h, 0));
      const yBottom = h < 0 ? yPos(f, h) : y0px;
      fig.rect(xLeft, yTop, xRight - xLeft, Math.abs(yBottom - yTop), {
        class: "bar",
        "data-value": h,
        fill: colors[s]!,
      });
    }
  }
}

/** Simple (one series) bar chart at integer positions 0..n-1. */
function simpleBars(
  fig: SvgFigure,
  f: Frame,
  heights: readonly number[],
  colors: readonly string[],
  opacity?: number,
): void {
  const y0px = yPos(f, Math.max(f.ylim[0], 0));
  for (let g = 0; g < heights.length; g++) {
    const raw = heights[g]!;
    const h = Number.isNaN(raw) ? 0 : raw;
    const xLeft = xPos(f, g - 0.4);
    const xRight = xPos(f, g + 0.4);
    const yTop = yPos(f, Math.max(h, 0));
    const yBottom = h < 0 ? yPos(f, h) : y0px;
    const attrs: Record<string, string | number> = {
      class: "bar",
      "data-value": h,
      fill: colors[g % colors.length]!,
    };
    if (opacity !== undefined) attrs["fill-opacity"] = opacity;
    fig.rect(xLeft, yTop, xRight - xLeft, Math.abs(yBottom - yTop), attrs);
  }
}

/** Panel grid geometry shared by the multi-panel plots (py `_grid` sizing). */
function panelLayout(n: number): { rects: PanelRect[]; fig: SvgFigure; ncols: number } {
  const ncols = n > 1 ? Math.min(n, 3) : 1;
  const { rects, width, height } = panelGrid(Math.max(n, 1), ncols);
  return { rects, fig: new SvgFigure(width, height), ncols };
}

// ---------------------------------------------------------------------------
// CTA-PLS
// ---------------------------------------------------------------------------

/**
 * Per-construct adjusted-p dot plots (R `plot.cta_analysis`).
 *
 * Returns null when no constructs were tested (R message + no-op).
 */
export function plotCta(record: CtaAnalysis): SvgPlot | null {
  if (record.constructResults.length === 0) return null;
  const constructs = [...record.tetradDetails.keys()];
  const n = constructs.length;
  const ncols = n <= 4 ? Math.min(n, 2) : 3;
  const { rects, width, height } = panelGrid(n, ncols, 384, 336);
  const fig = new SvgFigure(width, height);
  const verdicts = new Map(record.constructResults.map((r) => [r.construct, r.verdict]));

  constructs.forEach((construct, p) => {
    const detail = record.tetradDetails.get(construct)!;
    const adjP = col(detail.table, "Adj_P");
    const xs = adjP.map((_, i) => i + 1);
    const finite = finiteOf(adjP);
    const top = finite.length > 0 ? Math.max(1, ...finite) : 1;
    const f = frame(rects[p]!, extendRange([1, Math.max(xs.length, 1)]), [0, top]);
    axes(fig, f, {
      xTicks: xs,
      xLabel: "Tetrad index",
      yLabel: "Adjusted p-value",
      title: construct,
    });
    hline(fig, f, record.alpha, { stroke: gray(40), "stroke-dasharray": "4,3" });
    adjP.forEach((pVal, i) => {
      if (Number.isNaN(pVal)) return;
      fig.circle(xPos(f, xs[i]!), yPos(f, pVal), 4, {
        class: "pt",
        "data-value": pVal,
        fill: pVal <= record.alpha ? "firebrick" : "steelblue",
      });
    });
    const verdict = verdicts.get(construct)!;
    textLegend(
      fig,
      f,
      [{ label: verdict, color: verdict.includes("supported") ? "steelblue" : "firebrick" }],
      "topright",
    );
  });
  return fig.finish();
}

// ---------------------------------------------------------------------------
// IPMA / cIPMA
// ---------------------------------------------------------------------------

export interface PlotCipmaOptions {
  /** `"cipma"` (default; falls back to `"ipma"` without an NCA overlay) or `"ipma"`. */
  type?: string;
  /** `"unstandardized"` (default) or `"standardized"` total effects on x. */
  importanceMetric?: string;
}

/** Importance-Performance scatter (R `plot.cipma_analysis`). */
export function plotCipma(record: CipmaAnalysis, options: PlotCipmaOptions = {}): SvgPlot {
  const importanceMetric = options.importanceMetric ?? "unstandardized";
  let type = options.type ?? "cipma";
  if (type === "cipma" && record.nca === null) type = "ipma";
  const impMap =
    importanceMetric === "unstandardized" ? record.importanceUnstd : record.importanceStd;
  const constructs = [...record.constructs];
  const imp = constructs.map((c) => impMap[c]!);
  const perf = constructs.map((c) => record.performance[c]!);

  const impPad = imp.length > 0 ? Math.max(0.05, (Math.max(...imp) - Math.min(...imp)) * 0.15) : 0.05;
  const xlim: [number, number] = [Math.min(...imp) - impPad, Math.max(...imp) + impPad];
  const ylim: [number, number] = [
    Math.max(0, Math.min(...perf) - 5),
    Math.min(100, Math.max(...perf) + 5),
  ];

  const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
  const f = frame(DEVICE, xlim, ylim);
  const meanOf = (v: readonly number[]): number => {
    const fin = finiteOf(v);
    return fin.reduce((a, b) => a + b, 0) / fin.length;
  };
  axes(fig, f, {
    xLabel:
      importanceMetric === "unstandardized"
        ? "Importance (unstandardized total effect)"
        : "Importance (standardized total effect)",
    yLabel: "Performance (0-100)",
    title: `${type === "cipma" ? "cIPMA" : "IPMA"}: ${record.target}`,
  });
  vline(fig, f, meanOf(imp), { stroke: gray(70), "stroke-dasharray": "4,3" });
  hline(fig, f, meanOf(perf), { stroke: gray(70), "stroke-dasharray": "4,3" });

  const necessary = new Map(record.classification.map((row) => [row.construct, row.necessary]));
  constructs.forEach((c, i) => {
    const cx = xPos(f, imp[i]!);
    const cy = yPos(f, perf[i]!);
    if (type === "cipma") {
      if (necessary.get(c) === true) {
        fig.circle(cx, cy, 6, { class: "pt", "data-x": imp[i]!, "data-y": perf[i]!, fill: "red", stroke: "red" });
      } else {
        fig.circle(cx, cy, 6, { class: "pt", "data-x": imp[i]!, "data-y": perf[i]!, fill: "none", stroke: "steelblue" });
      }
    } else {
      fig.circle(cx, cy, 6, { class: "pt", "data-x": imp[i]!, "data-y": perf[i]!, fill: "steelblue" });
    }
    fig.text(cx, cy - 9, c, { class: "ann", "text-anchor": "middle", "font-size": 10 });
  });

  if (type === "cipma") {
    textLegend(
      fig,
      f,
      [
        { label: "Necessary + sufficient", color: "red" },
        { label: "Sufficient only", color: "steelblue" },
      ],
      "bottomleft",
    );
  }
  return fig.finish();
}

// ---------------------------------------------------------------------------
// COA
// ---------------------------------------------------------------------------

export interface PlotCoaOptions {
  /** `"pd"` (default), `"groups"` or `"tree"`. */
  type?: string;
  /** Constructs dropped from the `"groups"` plot. */
  remove?: readonly string[];
}

/**
 * Composite Overfit Analysis plots (R `plot.coa_analysis`): `"pd"`
 * (predictive-deviance scatter), `"groups"` (group mean construct scores;
 * null with no deviant groups) or `"tree"` (deviance tree diagram).
 */
export function plotCoa(record: CoaAnalysis, options: PlotCoaOptions = {}): SvgPlot | null {
  const type = options.type ?? "pd";
  if (type === "pd") return plotCoaPd(record);
  if (type === "groups") return plotCoaGroups(record, options.remove);
  if (type === "tree") return plotCoaTree(record);
  throw new Error(`type must be 'pd', 'groups' or 'tree'; got '${type}'`);
}

function plotCoaPd(record: CoaAnalysis): SvgPlot {
  const pd = record.predictiveDeviance.pd;
  const dt = record.devianceTree;
  const n = pd.length;
  const groupId = new Array<number>(n).fill(0);
  Object.values(dt.deviantGroups).forEach((cases, i) => {
    for (const c of cases) groupId[c - 1] = i + 2;
  });
  for (const c of dt.uniqueDeviants) groupId[c - 1] = 1;

  const order = pd.map((_, i) => i).sort((a, b) => pd[a]! - pd[b]!);
  const pdSorted = order.map((i) => pd[i]!);
  const grpSorted = order.map((i) => groupId[i]!);

  const lo = quantile([...pd], record.devianceBounds[0]);
  const hi = quantile([...pd], record.devianceBounds[1]);

  const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
  const f = frame(DEVICE, [1, n], extendRange([Math.min(...pd), Math.max(...pd)]));
  axes(fig, f, {
    xLabel: "Cases",
    yLabel: "Predictive Deviance",
    title: `Predictive Deviance: ${record.focalConstruct}`,
  });
  hline(fig, f, lo, { stroke: "darkgray", "stroke-dasharray": "4,3" });
  hline(fig, f, hi, { stroke: "darkgray", "stroke-dasharray": "4,3" });

  const nGroups = Object.keys(dt.deviantGroups).length;
  const groupCols = nGroups > 0 ? ["black", ...paletteColors(nGroups, "Set1")] : ["black"];
  for (let i = 0; i < n; i++) {
    const cx = xPos(f, i + 1);
    const cy = yPos(f, pdSorted[i]!);
    const g = grpSorted[i]!;
    if (g === 0) {
      fig.circle(cx, cy, 1.8, { class: "pt", fill: "lightgray" });
    } else if (g === 1) {
      fig.polygon(starPoints(cx, cy, 4.5), { class: "pt-unique", fill: "black" });
    } else {
      const color = groupCols[g - 1]!;
      fig.circle(cx, cy, 2.8, { class: "pt-group", fill: color });
      fig.text(cx, cy - 5, String.fromCharCode(65 + g - 2), {
        class: "ann",
        fill: color,
        "font-size": 8,
        "text-anchor": "middle",
      });
    }
  }

  const labels = Object.keys(dt.deviantGroups);
  if (labels.length > 0) {
    const entries: LegendEntry[] = labels.map((label, j) => ({
      label,
      color: groupCols[j + 1]!,
      marker: "●",
    }));
    if (dt.uniqueDeviants.length > 0) entries.push({ label: "Unique", color: "black", marker: "*" });
    textLegend(fig, f, entries, "topleft");
  }
  return fig.finish();
}

function plotCoaGroups(record: CoaAnalysis, remove?: readonly string[]): SvgPlot | null {
  const dt = record.devianceTree;
  if (Object.keys(dt.deviantGroups).length === 0) return null;
  const scores = groupScoreMeans(record); // constructs x groups
  let constructs = [...scores.rows];
  const groups = [...scores.cols];
  let values = scores.values.map((row) => [...row]);

  if (remove !== undefined && remove.length > 0) {
    const keep = constructs.map((c, i) => (remove.includes(c) ? -1 : i)).filter((i) => i >= 0);
    constructs = keep.map((i) => constructs[i]!);
    values = keep.map((i) => values[i]!);
  }

  const focal = record.focalConstruct;
  const focalIdx = constructs.indexOf(focal);
  if (focalIdx !== -1 && focalIdx < constructs.length - 1) {
    const order = constructs.map((_, i) => i).filter((i) => i !== focalIdx);
    order.push(focalIdx);
    constructs = order.map((i) => constructs[i]!);
    values = order.map((i) => values[i]!);
  }

  const numConstructs = constructs.length;
  const maxAbs = Math.max(...values.flat().map((v) => (Number.isNaN(v) ? 0 : Math.abs(v))));
  const ylim = Math.max(3.5, maxAbs + 0.5);
  const cols = paletteColors(groups.length, "Set1");

  const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
  const f = frame(DEVICE, extendRange([1, numConstructs]), [-ylim, ylim]);
  const [left, right] = plotRegion(f);
  // 50% reference zones behind the series (R adjustcolor("gray", ...) bands)
  fig.rect(left, yPos(f, 1.15), right - left, yPos(f, -1.15) - yPos(f, 1.15), {
    fill: "gray",
    "fill-opacity": 0.1,
  });
  fig.rect(left, yPos(f, 0.67), right - left, yPos(f, -0.67) - yPos(f, 0.67), {
    fill: "gray",
    "fill-opacity": 0.15,
  });
  axes(fig, f, {
    xTicks: constructs.map((_, i) => i + 1),
    xTickLabels: constructs,
    yLabel: "Average Construct Score",
    title: "Deviant Group Construct Scores",
  });
  if (focalIdx !== -1) {
    vline(fig, f, numConstructs - 0.5, { stroke: gray(80), "stroke-width": 2 });
  }
  const xs = constructs.map((_, i) => i + 1);
  groups.forEach((group, j) => {
    const ys = values.map((row) => row[j]!);
    seriesLine(fig, f, xs, ys, { stroke: cols[j]!, "data-group": group });
    // py uses the group letter itself as the marker glyph
    xs.forEach((x, i) => {
      if (Number.isNaN(ys[i]!)) return;
      fig.text(xPos(f, x), yPos(f, ys[i]!) + 3, group, {
        class: "marker",
        fill: cols[j]!,
        "font-size": 9,
        "text-anchor": "middle",
      });
    });
  });
  textLegend(
    fig,
    f,
    groups.map((group, j) => ({ label: group, color: cols[j]! })),
    "topright",
  );
  return fig.finish();
}

function plotCoaTree(record: CoaAnalysis): SvgPlot {
  const tree = record.devianceTree.tree;
  const layout = tree.plotLayout();
  const splitLabels = tree.splitRuleLabels();
  const nodeLabels = tree.nodeLabels(true);

  const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
  const f = frame(
    DEVICE,
    extendRange([Math.min(...layout.x), Math.max(...layout.x)]),
    extendRange([Math.min(...layout.y), Math.max(...layout.y)]),
  );
  const [left, right, top] = plotRegion(f);
  fig.text((left + right) / 2, top - 16, `Deviance Tree: ${record.focalConstruct}`, {
    class: "title",
    "text-anchor": "middle",
    "font-size": 14,
    "font-weight": "bold",
  });
  for (const [parent, elbow, child] of layout.edges) {
    fig.polyline(
      [
        [xPos(f, parent[0]), yPos(f, parent[1])],
        [xPos(f, elbow[0]), yPos(f, elbow[1])],
        [xPos(f, child[0]), yPos(f, child[1])],
      ],
      { class: "edge", "stroke-width": 0.8 },
    );
  }
  layout.nodeIds.forEach((nid, i) => {
    const x = xPos(f, layout.x[i]!);
    const y = yPos(f, layout.y[i]!);
    if (layout.isLeaf[i]) {
      fig.text(x, y + 10, nodeLabels[nid]!, {
        class: "node-label",
        "font-size": 7,
        "text-anchor": "middle",
      });
    } else {
      fig.text(x, y - 4, splitLabels[nid]!, {
        class: "node-label",
        "font-size": 7,
        "text-anchor": "middle",
      });
    }
  });
  return fig.finish();
}

// ---------------------------------------------------------------------------
// NCA
// ---------------------------------------------------------------------------

export interface PlotNcaOptions {
  /** `"scatter"` (default) or `"effects"`. */
  type?: string;
}

/**
 * NCA plots (R `plot.nca_analysis`): `"scatter"` or `"effects"`.
 * `"effects"` returns null when there are no predictors.
 */
export function plotNca(record: NcaAnalysis, options: PlotNcaOptions = {}): SvgPlot | null {
  const type = options.type ?? "scatter";
  if (type === "scatter") return plotNcaScatter(record);
  if (type === "effects") return plotNcaEffects(record);
  throw new Error(`type must be 'scatter' or 'effects'; got '${type}'`);
}

function plotNcaScatter(record: NcaAnalysis): SvgPlot {
  const scores = record.plsModel.constructScores;
  const y = col(scores, record.target);
  const predictors = [...record.predictors];
  const { rects, fig } = panelLayout(predictors.length);

  predictors.forEach((pred, p) => {
    const x = col(scores, pred);
    const f = frame(
      rects[p]!,
      extendRange([Math.min(...x), Math.max(...x)]),
      extendRange([Math.min(...y), Math.max(...y)]),
    );
    axes(fig, f, { xLabel: pred, yLabel: record.target, title: `NCA: ${pred} -> ${record.target}` });
    for (let i = 0; i < x.length; i++) {
      fig.circle(xPos(f, x[i]!), yPos(f, y[i]!), 2.4, {
        class: "pt",
        fill: "black",
        "fill-opacity": 0.4,
      });
    }
    const entries: LegendEntry[] = [];
    if (record.ceilings.includes("ce_fdh")) {
      const { ux, cy } = computeCeFdh(x, y);
      const pts: [number, number][] = [];
      for (let i = 0; i < ux.length; i++) {
        pts.push([xPos(f, ux[i]!), yPos(f, cy[i]!)]);
        if (i + 1 < ux.length) pts.push([xPos(f, ux[i + 1]!), yPos(f, cy[i]!)]);
      }
      fig.polyline(pts, { class: "ceiling-ce", stroke: "red", "stroke-width": 2 });
      entries.push({ label: "ce_fdh", color: "red" });
    }
    if (record.ceilings.includes("cr_fdh")) {
      const line = crFdhLine(x, y);
      if (line !== null) {
        const [a, b] = line;
        const [xMin, xMax] = [Math.min(...x), Math.max(...x)];
        const [yMin, yMax] = [Math.min(...y), Math.max(...y)];
        const pts: [number, number][] = [];
        for (let i = 0; i < 100; i++) {
          const xi = xMin + ((xMax - xMin) * i) / 99;
          const yi = Math.min(Math.max(a + b * xi, yMin), yMax);
          pts.push([xPos(f, xi), yPos(f, yi)]);
        }
        fig.polyline(pts, {
          class: "ceiling-cr",
          stroke: "blue",
          "stroke-width": 2,
          "stroke-dasharray": "6,4",
        });
      }
      entries.push({ label: "cr_fdh", color: "blue" });
    }
    if (entries.length > 0) textLegend(fig, f, entries, "bottomright");
  });
  return fig.finish();
}

function plotNcaEffects(record: NcaAnalysis): SvgPlot | null {
  const es = record.effectSizes;
  if (es.rows.length === 0) return null;
  const nCeil = es.cols.length;
  const colors = nCeil <= 2 ? ["steelblue", "coral"].slice(0, nCeil) : paletteColors(nCeil, "Set1");

  const allVals = finiteOf(es.values.flat());
  const ylimMax = allVals.length > 0 ? Math.max(0.5, Math.max(...allVals) * 1.2) : 0.5;
  const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
  const f = frame(DEVICE, [-0.5, es.rows.length - 0.5], [0, ylimMax]);
  axes(fig, f, {
    xTicks: es.rows.map((_, i) => i),
    xTickLabels: es.rows,
    xTickAngle: 90,
    yLabel: "Effect Size (d)",
    title: `NCA Effect Sizes: ${record.target}`,
  });
  groupedBars(fig, f, nCeil, es.values, colors);
  hline(fig, f, 0.1, { stroke: "darkgray", "stroke-dasharray": "4,3" });
  fig.text(xPos(f, es.rows.length - 0.6), yPos(f, 0.1) - 3, "d = 0.1", {
    class: "ann",
    fill: "darkgray",
    "font-size": 7,
  });
  if (nCeil > 1) {
    textLegend(
      fig,
      f,
      es.cols.map((c, j) => ({ label: c, color: colors[j]! })),
      "topright",
    );
  }
  return fig.finish();
}

export interface PlotNcaEsseOptions {
  /** `"sensitivity"` (default) or `"difference"`. */
  type?: string;
}

/** NCA-ESSE plots (R `plot.nca_esse`): `"sensitivity"` or `"difference"`. */
export function plotNcaEsse(record: NcaEsse, options: PlotNcaEsseOptions = {}): SvgPlot {
  const type = options.type ?? "sensitivity";
  if (type === "sensitivity") return plotEsseSensitivity(record);
  if (type === "difference") return plotEsseDifference(record);
  throw new Error(`type must be 'sensitivity' or 'difference'; got '${type}'`);
}

function plotEsseSensitivity(record: NcaEsse): SvgPlot {
  const predictors = [...record.predictors];
  const thresholds = [...record.thresholds];
  const { rects, fig } = panelLayout(predictors.length);

  predictors.forEach((pred, p) => {
    const emp = col(record.effectSizes, pred);
    const bench = col(record.benchmark, pred);
    const finite = [...finiteOf(emp), ...finiteOf(bench), 0.1];
    const f = frame(
      rects[p]!,
      extendRange([Math.min(...thresholds), Math.max(...thresholds)]),
      [0, Math.max(...finite) * 1.2],
    );
    axes(fig, f, {
      xLabel: "ECDF threshold",
      yLabel: "NCA effect size (d)",
      title: `NCA-ESSE: ${pred} -> ${record.target}`,
    });
    seriesLine(fig, f, thresholds, emp, { stroke: "black", "data-series": "empirical" }, "circle");
    seriesLine(
      fig,
      f,
      thresholds,
      bench,
      { stroke: gray(50), "stroke-dasharray": "6,4", "data-series": "benchmark" },
      "triangle",
    );
    hline(fig, f, 0.1, { stroke: "darkgray", "stroke-dasharray": "1,3" });
    textLegend(
      fig,
      f,
      [
        { label: "Empirical", color: "black" },
        { label: "Benchmark (uniform)", color: gray(50) },
      ],
      "topleft",
    );
  });
  return fig.finish();
}

function plotEsseDifference(record: NcaEsse): SvgPlot {
  const predictors = [...record.predictors];
  const thresholds = record.thresholds.slice(1);
  const { rects, fig } = panelLayout(predictors.length);

  predictors.forEach((pred, p) => {
    const emp = col(record.effectSizes, pred);
    const bench = col(record.benchmark, pred);
    const deltaDiff = thresholds.map(
      (_, i) => emp[i + 1]! - emp[i]! - (bench[i + 1]! - bench[i]!),
    );
    const finite = [...finiteOf(deltaDiff), 0];
    const lo = Math.min(...finite);
    const hi = Math.max(...finite);
    const pad = Math.max(0.05, (hi - lo) * 0.1);
    const f = frame(
      rects[p]!,
      extendRange([Math.min(...thresholds), Math.max(...thresholds)]),
      [lo - pad, hi + pad],
    );
    axes(fig, f, {
      xLabel: "ECDF threshold",
      yLabel: "Δ Empirical - Δ Benchmark",
      title: `ESSE Difference: ${pred} -> ${record.target}`,
    });
    hline(fig, f, 0, { stroke: "darkgray", "stroke-dasharray": "4,3" });
    seriesLine(fig, f, thresholds, deltaDiff, { stroke: "black" }, "circle");
  });
  return fig.finish();
}

// ---------------------------------------------------------------------------
// PCM
// ---------------------------------------------------------------------------

export interface PlotPcmOptions {
  /** `"RMSE"` (default) or `"MAE"`. */
  metric?: string;
  /** Legend corner on the first panel; null suppresses the legend. */
  legendPos?: LegendCorner | null;
  barCol?: string;
  negCol?: string;
  /** Font-size multipliers (R cex; base sizes 8 and 10 px). */
  cexLabels?: number;
  cexLegend?: number;
}

/** Per-path PCM bar charts (R `plot.pcm_analysis`); `metric` RMSE or MAE. */
export function plotPcm(record: PcmAnalysis, options: PlotPcmOptions = {}): SvgPlot {
  const metric = options.metric ?? "RMSE";
  if (metric !== "RMSE" && metric !== "MAE") {
    throw new Error(`metric must be 'RMSE' or 'MAE'; got '${metric}'`);
  }
  const legendPos = options.legendPos === undefined ? "topright" : options.legendPos;
  const barCol = options.barCol ?? "steelblue";
  const negCol = options.negCol ?? "tomato";
  const cexLabels = options.cexLabels ?? 0.8;

  const nPaths = record.pcmResults.length;
  const { rects, width, height } = panelGrid(nPaths, nPaths);
  const fig = new SvgFigure(width, height);

  record.pcmResults.forEach((res, p) => {
    const vals = [...(metric === "RMSE" ? res.pcmRmse : res.pcmMae)];
    const yMin = Math.min(...vals, 0, 0.12);
    const yMax = Math.max(...vals, 0, 0.12);
    const f = frame(rects[p]!, [-0.5, vals.length - 0.5], [
      Math.min(yMin, -0.02),
      Math.max(yMax * 1.15, 0.12),
    ]);
    axes(fig, f, {
      xTicks: vals.map((_, i) => i),
      xTickLabels: res.results.rows,
      xTickAngle: 90,
      yLabel: `PCM (${metric})`,
      title: `${res.antecedent} -> ${res.mediator} -> ${res.target}`,
    });
    simpleBars(
      fig,
      f,
      vals,
      vals.map((v) => (v >= 0 ? barCol : negCol)),
      0.7,
    );
    hline(fig, f, 0, { stroke: gray(40) });
    hline(fig, f, 0.05, { stroke: "orange", "stroke-dasharray": "4,3" });
    hline(fig, f, 0.1, { stroke: "darkgreen", "stroke-dasharray": "4,3" });
    vals.forEach((v, i) => {
      fig.text(xPos(f, i), yPos(f, v) + (v >= 0 ? -3 : 10), v.toFixed(3), {
        class: "ann",
        "font-size": (8 * cexLabels) / 0.8,
        "text-anchor": "middle",
      });
    });
    if (legendPos !== null && p === 0) {
      textLegend(
        fig,
        f,
        [
          { label: "Weak (< 0.05)", color: "black" },
          { label: "Moderate (0.05-0.10)", color: "orange" },
          { label: "Strong (> 0.10)", color: "darkgreen" },
        ],
        legendPos,
      );
    }
  });
  return fig.finish();
}

// ---------------------------------------------------------------------------
// FIMIX-PLS / PLS-POS shared segment-paths panel
// ---------------------------------------------------------------------------

/** Non-zero cells of a path matrix in R `which(arr.ind = TRUE)` (column-major). */
function nonzeroPaths(pathMat: NamedMatrix): { labels: string[]; cells: [number, number][] } {
  const labels: string[] = [];
  const cells: [number, number][] = [];
  pathMat.cols.forEach((to, j) => {
    pathMat.rows.forEach((from, i) => {
      if (pathMat.values[i]![j]! !== 0) {
        labels.push(`${from} -> ${to}`);
        cells.push([i, j]);
      }
    });
  });
  return { labels, cells };
}

function plotSegmentPaths(
  segmentPaths: readonly NamedMatrix[],
  k: number,
  title: string,
): SvgPlot | null {
  const { labels, cells } = nonzeroPaths(segmentPaths[0]!);
  if (cells.length === 0) return null;
  const coef = cells.map(([i, j]) =>
    Array.from({ length: k }, (_, s) => segmentPaths[s]!.values[i]![j]!),
  );
  const flat = finiteOf(coef.flat());
  const lo = Math.min(0, ...flat);
  const hi = Math.max(0, ...flat);
  const pad = Math.max(0.05, (hi - lo) * 0.08);
  const width = Math.max(576, Math.round(115 * labels.length));
  const fig = new SvgFigure(width, 432);
  const f = frame({ x0: 0, y0: 0, width, height: 432 }, [-0.5, labels.length - 0.5], [
    lo - (lo < 0 ? pad : 0),
    hi + pad,
  ]);
  axes(fig, f, {
    xTicks: labels.map((_, i) => i),
    xTickLabels: labels,
    xTickAngle: 90,
    yLabel: "Path Coefficient",
    title,
  });
  groupedBars(fig, f, k, coef, posPalette(k));
  textLegend(
    fig,
    f,
    Array.from({ length: k }, (_, s) => ({ label: `Seg. ${s + 1}`, color: posPalette(k)[s]! })),
    "topright",
  );
  return fig.finish();
}

/** Single-panel percent bars (segment proportions). */
function proportionBars(heights: readonly number[], colors: readonly string[], title: string): SvgPlot {
  const k = heights.length;
  const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
  const f = frame(DEVICE, [-0.5, k - 0.5], [0, 100]);
  axes(fig, f, {
    xTicks: heights.map((_, s) => s),
    xTickLabels: heights.map((_, s) => `Seg. ${s + 1}`),
    yLabel: "Proportion (%)",
    title,
  });
  simpleBars(fig, f, heights, colors);
  return fig.finish();
}

// ---------------------------------------------------------------------------
// FIMIX-PLS
// ---------------------------------------------------------------------------

export interface PlotFimixOptions {
  /** `"segments"` (default) or `"paths"`. */
  type?: string;
}

/**
 * FIMIX-PLS plots (R `plot.fimix_analysis`): `"segments"` or `"paths"`.
 * `"paths"` returns null when the first segment has no non-zero paths.
 */
export function plotFimix(record: FimixAnalysis, options: PlotFimixOptions = {}): SvgPlot | null {
  const type = options.type ?? "segments";
  const k = record.k;
  if (type === "segments") {
    const heights = Object.values(record.segmentProportions).map((p) => p * 100);
    return proportionBars(heights, posPalette(k), `FIMIX-PLS: Segment Proportions (K = ${k} )`);
  }
  if (type === "paths") {
    return plotSegmentPaths(record.segmentPaths, k, `FIMIX-PLS: Path Coefficients (K = ${k} )`);
  }
  throw new Error(`type must be 'segments' or 'paths'; got '${type}'`);
}

export interface PlotFimixCompareOptions {
  /** `"criteria"` (default) or `"entropy"`. */
  type?: string;
  /** Information criteria overlaid by the `"criteria"` plot. */
  criteria?: readonly string[];
}

/**
 * FIMIX-PLS comparison plots (R `plot.fimix_comparison`): `"criteria"`
 * overlays information criteria vs K (throws when none of `criteria` are
 * present); `"entropy"` plots EN vs K.
 */
export function plotFimixCompare(
  record: FimixComparison,
  options: PlotFimixCompareOptions = {},
): SvgPlot {
  const type = options.type ?? "criteria";
  const criteria = options.criteria ?? ["AIC3", "AIC4", "BIC", "CAIC"];
  const ft = record.fitTable;
  const kVals = col(ft, "K");
  if (type === "criteria") {
    const valid = criteria.filter((c) => ft.cols.includes(c));
    if (valid.length === 0) throw new Error("No valid criteria found to plot.");
    const colors = paletteColors(valid.length, "Set1");
    const series = valid.map((c) => col(ft, c));
    const finite = finiteOf(series.flat());
    let ylim: [number, number] = [0, 1];
    if (finite.length > 0) {
      const lo = Math.min(...finite);
      const hi = Math.max(...finite);
      const pad = (hi - lo) * 0.05 || 0.5;
      ylim = [lo - pad, hi + pad];
    }
    const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
    const f = frame(DEVICE, extendRange([Math.min(...kVals), Math.max(...kVals)]), ylim);
    axes(fig, f, {
      xTicks: kVals,
      xLabel: "Number of Segments (K)",
      yLabel: "Information Criterion",
      title: "FIMIX-PLS: Model Selection Criteria",
    });
    valid.forEach((c, i) => {
      seriesLine(fig, f, kVals, series[i]!, { stroke: colors[i]!, "data-series": c }, "circle");
    });
    textLegend(
      fig,
      f,
      valid.map((c, i) => ({ label: c, color: colors[i]! })),
      "topright",
    );
    return fig.finish();
  }
  if (type === "entropy") {
    const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
    const f = frame(DEVICE, extendRange([Math.min(...kVals), Math.max(...kVals)]), [0, 1]);
    axes(fig, f, {
      xTicks: kVals,
      xLabel: "Number of Segments (K)",
      yLabel: "Normed Entropy (EN)",
      title: "FIMIX-PLS: Classification Quality",
    });
    hline(fig, f, 0.5, { stroke: gray(70), "stroke-dasharray": "4,3" });
    fig.text(xPos(f, Math.max(...kVals)), yPos(f, 0.5) + 10, "EN = 0.50", {
      class: "ann",
      fill: gray(50),
      "font-size": 8,
      "text-anchor": "end",
    });
    seriesLine(fig, f, kVals, col(ft, "EN"), { stroke: "steelblue" }, "circle");
    return fig.finish();
  }
  throw new Error(`type must be 'criteria' or 'entropy'; got '${type}'`);
}

// ---------------------------------------------------------------------------
// PLS-POS
// ---------------------------------------------------------------------------

export interface PlotPosOptions {
  /** `"segments"` (default), `"rsquared"` or `"paths"`. */
  type?: string;
}

/**
 * PLS-POS plots (R `plot.pos_analysis`): `"segments"`, `"rsquared"` or
 * `"paths"`. `"paths"` returns null when the first segment has no non-zero
 * paths.
 */
export function plotPos(record: PosAnalysis, options: PlotPosOptions = {}): SvgPlot | null {
  const type = options.type ?? "segments";
  const k = record.k;
  const cols = posPalette(k);
  if (type === "segments") {
    const heights = Object.values(record.segmentSizes).map((s) => (s / record.nObs) * 100);
    return proportionBars(heights, cols, `PLS-POS: Segment Proportions (K = ${k} )`);
  }
  if (type === "rsquared") {
    const rsq = record.segmentRsquared;
    const flat = finiteOf(rsq.values.flat());
    const hi = flat.length > 0 ? Math.max(...flat, 0) : 1;
    const width = Math.max(576, Math.round(115 * rsq.rows.length));
    const fig = new SvgFigure(width, 432);
    const f = frame({ x0: 0, y0: 0, width, height: 432 }, [-0.5, rsq.rows.length - 0.5], [
      0,
      hi * 1.15 || 1,
    ]);
    axes(fig, f, {
      xTicks: rsq.rows.map((_, i) => i),
      xTickLabels: rsq.rows,
      xTickAngle: 90,
      yLabel: "R²",
      title: `PLS-POS: R² per Construct (K = ${k} )`,
    });
    groupedBars(fig, f, k, rsq.values, cols);
    textLegend(
      fig,
      f,
      Array.from({ length: k }, (_, s) => ({ label: `Seg. ${s + 1}`, color: cols[s]! })),
      "topright",
    );
    return fig.finish();
  }
  if (type === "paths") {
    return plotSegmentPaths(record.segmentPaths, k, `PLS-POS: Path Coefficients (K = ${k} )`);
  }
  throw new Error(`type must be 'segments', 'rsquared' or 'paths'; got '${type}'`);
}

/**
 * PLS-POS objective-vs-K line plot (R `plot.pos_comparison`).
 * Returns null when fewer than two solutions have a valid objective.
 */
export function plotPosCompare(record: PosComparison): SvgPlot | null {
  const valid = record.fitTable.filter((row) => !Number.isNaN(row.sumR2));
  if (valid.length < 2) return null;
  const kVals = valid.map((row) => row.k);
  const sumR2 = valid.map((row) => row.sumR2);
  const fig = new SvgFigure(DEVICE_SIZE, DEVICE_SIZE);
  const f = frame(
    DEVICE,
    extendRange([Math.min(...kVals), Math.max(...kVals)]),
    extendRange([Math.min(...sumR2), Math.max(...sumR2)]),
  );
  axes(fig, f, {
    xTicks: kVals,
    xLabel: "Number of Segments (K)",
    yLabel: "Objective (Sum R²)",
    title: "PLS-POS: Objective Criterion vs K",
  });
  seriesLine(fig, f, kVals, sumR2, { stroke: "black", "stroke-width": 2 }, "circle");
  return fig.finish();
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

type PlotOptions = PlotCipmaOptions &
  PlotCoaOptions &
  PlotNcaOptions &
  PlotNcaEsseOptions &
  PlotPcmOptions &
  PlotFimixOptions &
  PlotFimixCompareOptions &
  PlotPosOptions;

/**
 * Dispatch to the right plot function by `record.kind`.
 *
 * Throws for objects without a known `kind` (py raises TypeError).
 */
export function plot(record: unknown, options: PlotOptions = {}): SvgPlot | null {
  const kind =
    typeof record === "object" && record !== null && "kind" in record
      ? (record as { kind: unknown }).kind
      : null;
  switch (kind) {
    case "cta_analysis":
      return plotCta(record as CtaAnalysis);
    case "cipma_analysis":
      return plotCipma(record as CipmaAnalysis, options);
    case "coa_analysis":
      return plotCoa(record as CoaAnalysis, options);
    case "nca_analysis":
      return plotNca(record as NcaAnalysis, options);
    case "nca_esse":
      return plotNcaEsse(record as NcaEsse, options);
    case "pcm_analysis":
      return plotPcm(record as PcmAnalysis, options);
    case "fimix_analysis":
      return plotFimix(record as FimixAnalysis, options);
    case "fimix_comparison":
      return plotFimixCompare(record as FimixComparison, options);
    case "pos_analysis":
      return plotPos(record as PosAnalysis, options);
    case "pos_comparison":
      return plotPosCompare(record as PosComparison);
    default:
      throw new Error(`plot() does not support object of kind ${JSON.stringify(kind)}`);
  }
}
