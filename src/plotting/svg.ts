/**
 * Internal SVG-string chart primitives for the seminrExtras result plots.
 *
 * A tiny dependency-free layer in the spirit of seminr-ts' internal
 * `plot/charts/svg.ts` (base-R look: white background, boxed plot region,
 * outward ticks, Helvetica text), extended with the multi-panel grid the
 * seminrExtras plots need (R `par(mfrow)` / py `Figure.subplots`). Not a
 * charting library — just what `results.ts` draws. Data-bearing shapes carry
 * `class`/`data-*` attributes so tests can assert structure on the SVG source.
 */

import { SvgPlot } from "@seminr/core";

export const FONT_FAMILY = "Helvetica, Arial, sans-serif";
export const FONT_SIZE = 12;

/** Single-panel device: R's default 7x7in at 96 dpi. */
export const DEVICE_SIZE = 672;
/** Grid panels: py's 4.5x4.0in figsize at 96 dpi. */
export const PANEL_WIDTH = 432;
export const PANEL_HEIGHT = 384;

export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export const MARGIN: Margin = { top: 48, right: 24, bottom: 72, left: 64 };

/** A panel's pixel rectangle within the device. */
export interface PanelRect {
  x0: number;
  y0: number;
  width: number;
  height: number;
}

/** A panel's pixel frame plus data limits; maps data coords to pixels. */
export interface Frame extends PanelRect {
  margin: Margin;
  xlim: readonly [number, number];
  ylim: readonly [number, number];
}

export function frame(
  rect: PanelRect,
  xlim: readonly [number, number],
  ylim: readonly [number, number],
  margin: Margin = MARGIN,
): Frame {
  return { ...rect, margin: { ...margin }, xlim, ylim };
}

export function xPos(f: Frame, x: number): number {
  const [x0, x1] = f.xlim;
  const plotWidth = f.width - f.margin.left - f.margin.right;
  return f.x0 + f.margin.left + ((x - x0) / (x1 - x0)) * plotWidth;
}

export function yPos(f: Frame, y: number): number {
  const [y0, y1] = f.ylim;
  const plotHeight = f.height - f.margin.top - f.margin.bottom;
  return f.y0 + f.height - f.margin.bottom - ((y - y0) / (y1 - y0)) * plotHeight;
}

/** Plot-region pixel bounds `[left, right, top, bottom]`. */
export function plotRegion(f: Frame): [number, number, number, number] {
  return [
    f.x0 + f.margin.left,
    f.x0 + f.width - f.margin.right,
    f.y0 + f.margin.top,
    f.y0 + f.height - f.margin.bottom,
  ];
}

/**
 * R's default `xaxs`/`yaxs = "r"`: extend data limits by 4% of the range each
 * side; a degenerate (zero-range) limit widens by ±0.5.
 */
export function extendRange(
  lim: readonly [number, number],
  fraction = 0.04,
): [number, number] {
  const range = lim[1] - lim[0];
  if (range === 0) return [lim[0] - 0.5, lim[1] + 0.5];
  const pad = range * fraction;
  return [lim[0] - pad, lim[1] + pad];
}

/**
 * Tick positions in the spirit of R's `pretty()`: ~n intervals on a
 * 1/2/5 x 10^k step, expanded to whole steps inside the limits.
 */
export function prettyTicks(min: number, max: number, n = 5): number[] {
  if (!(max > min)) return [min];
  const rawStep = (max - min) / n;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const stepUnit = normalized < 1.5 ? 1 : normalized < 3 ? 2 : normalized < 7 ? 5 : 10;
  const step = stepUnit * magnitude;
  const first = Math.ceil(min / step - 1e-9);
  const last = Math.floor(max / step + 1e-9);
  const ticks: number[] = [];
  for (let k = first; k <= last; k++) {
    // strip float noise (0.2 * 3 = 0.6000000000000001) and normalize -0
    ticks.push(Number((k * step).toPrecision(12)) + 0);
  }
  return ticks;
}

/** Compact tick-label formatting (strips float noise). */
export function tickLabel(value: number): string {
  return String(Math.round(value * 1e10) / 1e10);
}

/** Round pixel coordinates to keep the SVG source compact. */
export const px = (value: number): string => String(Math.round(value * 100) / 100);

export function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export type Attrs = Record<string, string | number>;

function attrString(attrs: Attrs): string {
  return Object.entries(attrs)
    .map(([key, value]) => ` ${key}="${value}"`)
    .join("");
}

/** Accumulates SVG source; `finish()` wraps it into a `@seminr/core` SvgPlot. */
export class SvgFigure {
  private parts: string[] = [];

  constructor(
    readonly width: number,
    readonly height: number,
  ) {
    this.parts.push(
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" ` +
        `viewBox="0 0 ${width} ${height}" font-family="${FONT_FAMILY}" font-size="${FONT_SIZE}">\n`,
      `<rect width="${width}" height="${height}" fill="white"/>\n`,
    );
  }

  raw(source: string): void {
    this.parts.push(source);
  }

  line(x1: number, y1: number, x2: number, y2: number, attrs: Attrs = {}): void {
    this.parts.push(
      `<line x1="${px(x1)}" y1="${px(y1)}" x2="${px(x2)}" y2="${px(y2)}"` +
        `${attrString({ stroke: "black", ...attrs })}/>\n`,
    );
  }

  polyline(points: readonly (readonly [number, number])[], attrs: Attrs = {}): void {
    const path = points.map(([x, y]) => `${px(x)},${px(y)}`).join(" ");
    this.parts.push(
      `<polyline points="${path}"${attrString({ fill: "none", stroke: "black", ...attrs })}/>\n`,
    );
  }

  circle(cx: number, cy: number, r: number, attrs: Attrs = {}): void {
    this.parts.push(`<circle cx="${px(cx)}" cy="${px(cy)}" r="${r}"${attrString(attrs)}/>\n`);
  }

  rect(x: number, y: number, width: number, height: number, attrs: Attrs = {}): void {
    this.parts.push(
      `<rect x="${px(x)}" y="${px(y)}" width="${px(width)}" height="${px(height)}"` +
        `${attrString(attrs)}/>\n`,
    );
  }

  polygon(points: readonly (readonly [number, number])[], attrs: Attrs = {}): void {
    const path = points.map(([x, y]) => `${px(x)},${px(y)}`).join(" ");
    this.parts.push(`<polygon points="${path}"${attrString(attrs)}/>\n`);
  }

  /** Text; embedded newlines become stacked `<tspan>` lines. */
  text(x: number, y: number, content: string, attrs: Attrs = {}): void {
    const lines = content.split("\n");
    if (lines.length === 1) {
      this.parts.push(`<text x="${px(x)}" y="${px(y)}"${attrString(attrs)}>${escapeXml(content)}</text>\n`);
      return;
    }
    const spans = lines
      .map(
        (line, i) =>
          `<tspan x="${px(x)}" dy="${i === 0 ? 0 : "1.1em"}">${escapeXml(line)}</tspan>`,
      )
      .join("");
    this.parts.push(`<text x="${px(x)}" y="${px(y)}"${attrString(attrs)}>${spans}</text>\n`);
  }

  finish(): SvgPlot {
    return new SvgPlot(this.parts.join("") + "</svg>\n");
  }
}

/**
 * Panel rectangles for an n-plot grid with `ncols` columns (row-major), plus
 * the device size that contains them. Mirrors py's `_grid` (trailing panels
 * are simply absent).
 */
export function panelGrid(
  n: number,
  ncols: number,
  panelWidth = PANEL_WIDTH,
  panelHeight = PANEL_HEIGHT,
): { rects: PanelRect[]; width: number; height: number } {
  const nrows = Math.ceil(n / ncols);
  const rects: PanelRect[] = [];
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / ncols);
    const c = i % ncols;
    rects.push({ x0: c * panelWidth, y0: r * panelHeight, width: panelWidth, height: panelHeight });
  }
  return { rects, width: ncols * panelWidth, height: nrows * panelHeight };
}

const TICK_LENGTH = 6;

export interface AxesOptions {
  xTicks?: readonly number[];
  yTicks?: readonly number[];
  /** Custom x tick labels (defaults to numeric labels). */
  xTickLabels?: readonly string[];
  /** Rotate x tick labels by this many degrees (right-anchored), as R's las/srt. */
  xTickAngle?: number;
  xLabel?: string;
  yLabel?: string;
  title?: string;
  /** Draw the full box (R bty = "o") instead of just the two axis lines. */
  box?: boolean;
}

/** Axis box, outward ticks, tick labels, axis labels, and panel title. */
export function axes(fig: SvgFigure, f: Frame, options: AxesOptions = {}): void {
  const {
    xTicks = prettyTicks(f.xlim[0], f.xlim[1]),
    yTicks = prettyTicks(f.ylim[0], f.ylim[1]),
    xTickLabels,
    xTickAngle = 0,
    xLabel,
    yLabel,
    title,
    box = true,
  } = options;
  const [left, right, top, bottom] = plotRegion(f);

  if (box) {
    fig.rect(left, top, right - left, bottom - top, { fill: "none", stroke: "black" });
  } else {
    fig.line(left, bottom, right, bottom);
    fig.line(left, top, left, bottom);
  }

  for (let i = 0; i < xTicks.length; i++) {
    const tick = xTicks[i]!;
    const x = xPos(f, tick);
    if (x < left - 1e-6 || x > right + 1e-6) continue;
    fig.line(x, bottom, x, bottom + TICK_LENGTH);
    const label = xTickLabels?.[i] ?? tickLabel(tick);
    if (xTickAngle !== 0) {
      fig.text(x, bottom + TICK_LENGTH + 12, label, {
        "text-anchor": "end",
        transform: `rotate(${-xTickAngle} ${px(x)} ${px(bottom + TICK_LENGTH + 12)})`,
      });
    } else {
      fig.text(x, bottom + TICK_LENGTH + 14, label, { "text-anchor": "middle" });
    }
  }
  for (const tick of yTicks) {
    const y = yPos(f, tick);
    if (y < top - 1e-6 || y > bottom + 1e-6) continue;
    fig.line(left - TICK_LENGTH, y, left, y);
    fig.text(left - TICK_LENGTH - 4, y + 4, tickLabel(tick), { "text-anchor": "end" });
  }

  if (xLabel !== undefined) {
    fig.text((left + right) / 2, bottom + 44, xLabel, { "text-anchor": "middle" });
  }
  if (yLabel !== undefined) {
    const y = (top + bottom) / 2;
    fig.text(left - 44, y, yLabel, {
      "text-anchor": "middle",
      transform: `rotate(-90 ${px(left - 44)} ${px(y)})`,
    });
  }
  if (title !== undefined) {
    fig.text((left + right) / 2, top - 16, title, {
      class: "title",
      "text-anchor": "middle",
      "font-size": 14,
      "font-weight": "bold",
    });
  }
}

/** R legend corner keywords used by the seminrExtras plots. */
export type LegendCorner =
  | "topleft"
  | "topright"
  | "bottomleft"
  | "bottomright";

export interface LegendEntry {
  label: string;
  color: string;
  /** Optional marker glyph prefixed to the label (e.g. "●", "*"). */
  marker?: string;
}

/** Text-only legend (R `legend(bty = "n")`): colored labels stacked at a corner. */
export function textLegend(
  fig: SvgFigure,
  f: Frame,
  entries: readonly LegendEntry[],
  corner: LegendCorner,
): void {
  const [left, right, top, bottom] = plotRegion(f);
  const pad = 8;
  const lineHeight = 14;
  const anchorRight = corner === "topright" || corner === "bottomright";
  const x = anchorRight ? right - pad : left + pad;
  const yStart =
    corner === "topleft" || corner === "topright"
      ? top + pad + 10
      : bottom - pad - (entries.length - 1) * lineHeight;
  entries.forEach((entry, i) => {
    const label = entry.marker !== undefined ? `${entry.marker} ${entry.label}` : entry.label;
    fig.text(x, yStart + i * lineHeight, label, {
      class: "legend",
      fill: entry.color,
      "font-size": 10,
      "text-anchor": anchorRight ? "end" : "start",
    });
  });
}

/** A 5-point star polygon's vertices (matplotlib `marker="*"` analog). */
export function starPoints(
  cx: number,
  cy: number,
  r: number,
): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < 10; i++) {
    const radius = i % 2 === 0 ? r : r * 0.4;
    const angle = -Math.PI / 2 + (i * Math.PI) / 5;
    pts.push([cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)]);
  }
  return pts;
}
