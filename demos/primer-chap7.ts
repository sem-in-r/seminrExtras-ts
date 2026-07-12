/**
 * Primer Chapter 7: Moderation analysis.
 *
 * Ports the R workbook chapter 7: add a switching-cost (SC) moderator and a
 * two-stage CUSA x SC interaction to the corporate reputation model, bootstrap
 * the structural paths, inspect the interaction's effect size, and render a
 * simple slope-analysis plot of the moderation.
 *
 * Accompanies: Partial Least Squares SEM Using R (2nd Ed., 2026), Hair et al.
 *
 * Run: bun run build && bun run demos/primer-chap7.ts
 */

import {
  bootstrapModel,
  composite,
  constructs,
  estimatePls,
  interactionTerm,
  meanReplacement,
  modeA,
  modeB,
  multiItems,
  paths,
  relationships,
  singleItem,
  slopeAnalysis,
  summarizePls,
  summarizePlsBoot,
  twoStage,
} from "@seminr/core";
import { formatMatrix, heading, rendered } from "./lib/print.ts";
import { loadCorpRep } from "./lib/data.ts";

const corpRep = await loadCorpRep();

const mm = constructs(
  composite("QUAL", multiItems("qual_", [1, 2, 3, 4, 5, 6, 7, 8]), modeB),
  composite("PERF", multiItems("perf_", [1, 2, 3, 4, 5]), modeB),
  composite("CSOR", multiItems("csor_", [1, 2, 3, 4, 5]), modeB),
  composite("ATTR", multiItems("attr_", [1, 2, 3]), modeB),
  composite("COMP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("SC", multiItems("switch_", [1, 2, 3, 4])),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
  interactionTerm({ iv: "CUSA", moderator: "SC", method: twoStage, weights: modeA }),
);
const sm = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths(["CUSA", "SC", "CUSA*SC"], "CUSL"),
);
const model = estimatePls(corpRep, mm, sm, { missing: meanReplacement, missingValue: -99 });
const summary = summarizePls(model);
console.log(`Estimated the moderated model in ${model.iterations} iterations.`);

console.log(heading("Bootstrapped structural paths"));
const boot = bootstrapModel({ model, nboot: 1000, seed: 123 });
console.log(formatMatrix(summarizePlsBoot(boot).bootstrappedPaths));

console.log(heading("Effect sizes (f-squared)"));
console.log(formatMatrix(summary.fSquare));

console.log(heading("Simple slope analysis"));
// Render the two-way interaction (CUSA x SC) on CUSL; the plot is never shown.
const fig = slopeAnalysis(model, "CUSL", "SC", "CUSA", "bottomright");
console.log(rendered(fig, "slope analysis"));
