/**
 * Primer Chapter 4: Evaluating reflective measurement models.
 *
 * Ports the R workbook chapter 4: unidimensionality via principal-components
 * eigenvalues and Horn's parallel analysis, indicator loadings and
 * reliability, HTMT discriminant validity, and a bootstrapped HTMT plus a
 * congruence test.
 *
 * The R chapter uses the psych and paran packages. Here PCA eigenvalues come
 * from `@seminr/core/math` (Jacobi eigendecomposition of the correlation
 * matrix), and Horn's parallel analysis is a small inline helper (no extra
 * dependency). Not ported: psych::iclust's Revelle beta (a clustering-based
 * unidimensionality index with no lightweight analogue).
 *
 * Accompanies: Partial Least Squares SEM Using R (2nd Ed., 2026), Hair et al.
 *
 * Run: bun run build && bun run demos/primer-chap4.ts
 */

import {
  bootstrapModel,
  composite,
  constructs,
  estimatePls,
  meanReplacement,
  mulberry32,
  multiItems,
  paths,
  relationships,
  singleItem,
  summarizePls,
  summarizePlsBoot,
} from "@seminr/core";
import { colCor, jacobiEigenSym, quantile } from "@seminr/core/math";
import { congruenceTest } from "@seminr/extras";
import { loadCorpRep } from "./lib/data.ts";
import { formatMatrix, heading } from "./lib/print.ts";

/** Descending eigenvalues of a data block's correlation matrix. */
function eigenvalues(block: number[][]): number[] {
  return [...jacobiEigenSym(colCor(block, block)).values].sort((a, b) => b - a);
}

/**
 * Horn's parallel analysis on a data block, mirroring `paran::paran` in
 * spirit: compare the observed correlation-matrix eigenvalues against the
 * given centile of eigenvalues from random standard-normal data of the same
 * shape. Returns observed, random-centile, and the retained mask.
 */
function hornParallelAnalysis(
  block: number[][],
  iterations = 200,
  centile = 0.95,
  seed = 123,
): { observed: number[]; random: number[]; retained: boolean[] } {
  const n = block.length;
  const p = block[0]!.length;
  const observed = eigenvalues(block);
  const rng = mulberry32(seed);
  // standard normals via Box-Muller on the seeded uniform stream
  const normal = (): number =>
    Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
  const randomEvs: number[][] = [];
  for (let it = 0; it < iterations; it++) {
    const r = Array.from({ length: n }, () => Array.from({ length: p }, normal));
    randomEvs.push(eigenvalues(r));
  }
  const random = Array.from({ length: p }, (_, j) =>
    quantile(randomEvs.map((row) => row[j]!), centile),
  );
  return { observed, random, retained: observed.map((v, j) => v > random[j]!) };
}

const corpRep = await loadCorpRep();

const mm = constructs(
  composite("COMP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
);
const sm = relationships(paths(["COMP", "LIKE"], ["CUSA", "CUSL"]), paths("CUSA", "CUSL"));
const model = estimatePls(corpRep, mm, sm, { missing: meanReplacement, missingValue: -99 });
const summary = summarizePls(model);
console.log(`Estimated the reflective model in ${model.iterations} iterations.`);

// Reflective construct item blocks (multi-item ones only), over the
// cleaned/imputed data the model actually used.
const conList: Record<string, string[]> = {
  COMP: multiItems("comp_", [1, 2, 3]),
  LIKE: multiItems("like_", [1, 2, 3]),
  CUSL: multiItems("cusl_", [1, 2, 3]),
};
const block = (items: string[]): number[][] => {
  const idx = items.map((item) => model.data.columns.indexOf(item));
  return model.data.values.map((row) => idx.map((j) => row[j]!));
};

console.log(heading("Principal components: eigenvalues per construct"));
for (const [name, items] of Object.entries(conList)) {
  const evs = eigenvalues(block(items));
  console.log(`  ${name}: ${evs.map((v, i) => `PC${i + 1} EV = ${v.toFixed(2)}`).join("  ")}`);
}

console.log(heading("Horn's parallel analysis (retain if observed > random 95th centile)"));
for (const [name, items] of Object.entries(conList)) {
  const { observed, random, retained } = hornParallelAnalysis(block(items));
  const fmt = (v: number[]): string => `[${v.map((x) => x.toFixed(2)).join(", ")}]`;
  const nRetain = retained.filter(Boolean).length;
  console.log(
    `  ${name}: observed EVs ${fmt(observed)} vs random ${fmt(random)} -> retain ${nRetain}`,
  );
}

console.log(heading("Outer loadings"));
console.log(formatMatrix(summary.loadings));

console.log(heading("Internal consistency and reliability"));
console.log(formatMatrix(summary.reliability));

console.log(heading("HTMT discriminant validity"));
console.log(formatMatrix(summary.validity.htmt));

console.log(heading("Bootstrapped HTMT"));
const boot = bootstrapModel({ model, nboot: 1000, seed: 123 });
console.log(formatMatrix(summarizePlsBoot(boot).bootstrappedHtmt));

console.log(heading("Congruence test (alpha = 0.10)"));
const cong = congruenceTest(model, { alpha: 0.1 })!;
console.log(formatMatrix(cong.results));
