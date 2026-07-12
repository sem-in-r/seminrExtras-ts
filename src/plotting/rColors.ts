/**
 * R grDevices colour helpers, pinned for SVG parity (py plotting/_r_colors.py).
 *
 * Reproduces the exact `palette.colors()` hexes seminrExtras' base-R plots use
 * (`Set1`/`Set2`/`Okabe-Ito`), the shared `pos_palette` rule (Set2 up to K = 8,
 * else Okabe-Ito, error past 9) and R's `grayNN` shades. Named R colours
 * (firebrick, steelblue, ...) are valid SVG/CSS colour names already; R's
 * `adjustcolor` alpha blending is rendered via SVG opacity attributes at the
 * call sites instead of colour arithmetic.
 */

/** grDevices `palette.colors(palette = "Set1")` (max 9). */
export const SET1: readonly string[] = [
  "#E41A1C",
  "#377EB8",
  "#4DAF4A",
  "#984EA3",
  "#FF7F00",
  "#FFFF33",
  "#A65628",
  "#F781BF",
  "#999999",
];

/** grDevices `palette.colors(palette = "Set2")` (max 8). */
export const SET2: readonly string[] = [
  "#66C2A5",
  "#FC8D62",
  "#8DA0CB",
  "#E78AC3",
  "#A6D854",
  "#FFD92F",
  "#E5C494",
  "#B3B3B3",
];

/** grDevices default `palette.colors()` (Okabe-Ito, max 9). */
export const OKABE_ITO: readonly string[] = [
  "#000000",
  "#E69F00",
  "#56B4E9",
  "#009E73",
  "#F0E442",
  "#0072B2",
  "#D55E00",
  "#CC79A7",
  "#999999",
];

const PALETTES: Record<string, readonly string[]> = {
  Set1: SET1,
  Set2: SET2,
  "Okabe-Ito": OKABE_ITO,
};

/**
 * First `n` colours of an R palette (`palette.colors(n, palette=)`).
 *
 * Throws when `n` exceeds the palette length, matching R's error.
 */
export function paletteColors(n: number, palette = "Okabe-Ito"): string[] {
  const colors = PALETTES[palette];
  if (colors === undefined) {
    throw new Error(
      `Unknown palette '${palette}'. Use one of ${Object.keys(PALETTES).join(", ")}.`,
    );
  }
  if (n > colors.length) {
    throw new Error(`palette '${palette}' has only ${colors.length} colours; requested ${n}.`);
  }
  return colors.slice(0, n);
}

/** Segment palette (R `pos_palette`): Set2 for K <= 8, else Okabe-Ito. */
export function posPalette(k: number): string[] {
  return k <= 8 ? paletteColors(k, "Set2") : paletteColors(k, "Okabe-Ito");
}

/** R `grayNN` shade as a hex colour (e.g. `gray(40)` -> `"#666666"`). */
export function gray(pct: number): string {
  const v = Math.round((pct / 100) * 255);
  const hex = v.toString(16).padStart(2, "0");
  return `#${hex}${hex}${hex}`;
}
