/**
 * Primer Chapter 8: Mediation analysis.
 *
 * Ports the R workbook chapter 8: total indirect effects, specific indirect
 * effects (COMP/LIKE -> CUSA -> CUSL) with bootstrap significance, the direct
 * effects, sign and effect-size checks for each mediation, then a moderated
 * mediation with a CUSA x SC interaction and its index of moderated mediation
 * (point estimate, bootstrap CI, and p-value).
 *
 * Accompanies: Partial Least Squares SEM Using R (2nd Ed., 2026), Hair et al.
 *
 * Run: bun run build && bun run demos/primer-chap8.ts
 */

import {
  bootstrapModel,
  composite,
  constructs,
  estimatePls,
  interactionTerm,
  meanReplacement,
  modeB,
  multiItems,
  nmGet,
  paths,
  relationships,
  singleItem,
  specificEffectSignificance,
  summarizePls,
  twoStage,
  type BootModel,
} from "@seminr/core";
import { quantile } from "@compstats/core/stats";
import { loadCorpRep } from "./lib/data.ts";
import { formatMatrix, heading } from "./lib/print.ts";

function measurement() {
  return constructs(
    composite("QUAL", multiItems("qual_", [1, 2, 3, 4, 5, 6, 7, 8]), modeB),
    composite("PERF", multiItems("perf_", [1, 2, 3, 4, 5]), modeB),
    composite("CSOR", multiItems("csor_", [1, 2, 3, 4, 5]), modeB),
    composite("ATTR", multiItems("attr_", [1, 2, 3]), modeB),
    composite("COMP", multiItems("comp_", [1, 2, 3])),
    composite("LIKE", multiItems("like_", [1, 2, 3])),
    composite("CUSA", singleItem("cusa")),
    composite("CUSL", multiItems("cusl_", [1, 2, 3])),
  );
}

/** The per-replication distribution of a single structural path. */
function bootPathSeries(boot: BootModel, source: string, target: string): number[] {
  return boot.bootPaths.map((bp) => nmGet(bp, source, target));
}

/** Two-sided bootstrap p-value: 2 * min(P(x <= 0), P(x >= 0)). */
function pVal(x: readonly number[]): number {
  const le = x.filter((v) => v <= 0).length / x.length;
  const ge = x.filter((v) => v >= 0).length / x.length;
  return 2 * Math.min(le, ge);
}

const corpRep = await loadCorpRep();

const sm = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths("CUSA", "CUSL"),
);
const model = estimatePls(corpRep, measurement(), sm, {
  missing: meanReplacement,
  missingValue: -99,
});
const summary = summarizePls(model);
const boot = bootstrapModel({ model, nboot: 1000, seed: 123 });

console.log(heading("Total indirect effects"));
console.log(formatMatrix(summary.totalIndirectEffects));

console.log(heading("Specific indirect effects through CUSA"));
for (const [source, alpha] of [
  ["COMP", 0.1],
  ["LIKE", 0.05],
] as const) {
  const eff = specificEffectSignificance(boot, { from: source, through: ["CUSA"], to: "CUSL", alpha });
  console.log(
    `  ${source} -> CUSA -> CUSL (alpha=${alpha}): est=${eff.originalEst.toFixed(4)}` +
      `  CI=[${eff.ciLower.toFixed(4)}, ${eff.ciUpper.toFixed(4)}]  p=${eff.bootstrapP.toFixed(4)}`,
  );
}

console.log(heading("Direct effects (structural paths, R^2 rows on top)"));
console.log(formatMatrix(summary.paths));

console.log(heading("Sign and effect size v of each mediation"));
for (const source of ["LIKE", "COMP"]) {
  const pDirect = nmGet(summary.paths, source, "CUSL");
  const pA = nmGet(summary.paths, source, "CUSA");
  const pB = nmGet(summary.paths, "CUSA", "CUSL");
  const sign = pDirect * pA * pB;
  const v = pA ** 2 * pB ** 2;
  console.log(
    `  ${source}: sign(p1*p2*p3) = ${sign >= 0 ? "+" : ""}${sign.toFixed(5)}   ` +
      `effect size v = ${v.toFixed(5)}`,
  );
}

// --- Moderated mediation: add a CUSA x SC interaction ---
const mmMod = constructs(
  composite("QUAL", multiItems("qual_", [1, 2, 3, 4, 5, 6, 7, 8]), modeB),
  composite("PERF", multiItems("perf_", [1, 2, 3, 4, 5]), modeB),
  composite("CSOR", multiItems("csor_", [1, 2, 3, 4, 5]), modeB),
  composite("ATTR", multiItems("attr_", [1, 2, 3]), modeB),
  composite("COMP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("SC", multiItems("switch_", [1, 2, 3, 4])),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
  interactionTerm({ iv: "CUSA", moderator: "SC", method: twoStage }),
);
const smMod = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths(["CUSA", "SC", "CUSA*SC"], "CUSL"),
);
const modelMod = estimatePls(corpRep, mmMod, smMod, {
  missing: meanReplacement,
  missingValue: -99,
});
const sumMod = summarizePls(modelMod);
const bootMod = bootstrapModel({ model: modelMod, nboot: 1000, seed: 12345 });

console.log(heading("Index of moderated mediation (p1 * p5)"));
const p5 = bootPathSeries(bootMod, "CUSA*SC", "CUSL");
for (const source of ["COMP", "LIKE"]) {
  const point = nmGet(sumMod.paths, source, "CUSA") * nmGet(sumMod.paths, "CUSA*SC", "CUSL");
  const p1p5 = bootPathSeries(bootMod, source, "CUSA").map((v, i) => v * p5[i]!);
  const lo = quantile(p1p5, 0.025);
  const hi = quantile(p1p5, 0.975);
  console.log(
    `  ${source}: index = ${point >= 0 ? "+" : ""}${point.toFixed(5)}  ` +
      `95% CI=[${lo.toFixed(5)}, ${hi.toFixed(5)}]  p=${pVal(p1p5).toFixed(4)}`,
  );
}
