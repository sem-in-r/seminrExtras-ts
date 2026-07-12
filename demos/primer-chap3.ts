/**
 * Primer Chapter 3: Introduction to seminr.
 *
 * Ports the R workbook chapter 3: load the corporate reputation data, specify
 * a simple measurement and structural model, estimate it, see how a
 * mis-specified model is caught, then summarize and bootstrap the model.
 *
 * Accompanies: Partial Least Squares SEM Using R (2nd Ed., 2026), Hair et al.
 *
 * Run: bun run build && bun run demos/primer-chap3.ts
 */

import {
  bootstrapModel,
  composite,
  constructs,
  estimatePls,
  meanReplacement,
  multiItems,
  paths,
  relationships,
  singleItem,
  summarizePls,
  summarizePlsBoot,
} from "@seminr/core";
import { loadCorpRep } from "./lib/data.ts";
import { formatMatrix, heading } from "./lib/print.ts";

const corpRep = await loadCorpRep();

console.log(heading("First rows of the corporate reputation data"));
// R: head(corp_rep_data). The Dataset exposes its value matrix and columns.
console.log(`  ${corpRep.values.length} observations x ${corpRep.columns.length} columns`);
console.log(`  first columns: ${corpRep.columns.slice(0, 8).join(", ")} ...`);

// Simple 4-construct model.
const simpleMm = constructs(
  composite("COMP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
);
const simpleSm = relationships(paths(["COMP", "LIKE"], ["CUSA", "CUSL"]), paths("CUSA", "CUSL"));

console.log(heading("Estimating the simple model"));
const model = estimatePls(corpRep, simpleMm, simpleSm, {
  missing: meanReplacement,
  missingValue: -99,
});
console.log(`  Converged in ${model.iterations} iterations.`);

// How a mis-specified model is caught: the MM names the construct "COP" but
// the structural model references "COMP"; estimatePls throws immediately.
console.log(heading("Catching a mis-specified model"));
const errorMm = constructs(
  composite("COP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
);
try {
  estimatePls(corpRep, errorMm, simpleSm, { missing: meanReplacement, missingValue: -99 });
} catch (error) {
  console.log(`  ${(error as Error).message}`);
}

// Summaries.
const summary = summarizePls(model);
console.log(heading("Structural paths (R^2 rows on top)"));
console.log(formatMatrix(summary.paths));
console.log(heading("Construct reliability"));
console.log(formatMatrix(summary.reliability));

// Bootstrap the model.
console.log(heading("Bootstrapping the model (1000 replications)"));
const boot = bootstrapModel({ model, nboot: 1000, seed: 123 });
const bootSummary = summarizePlsBoot(boot);
console.log("Bootstrapped structural paths:");
console.log(formatMatrix(bootSummary.bootstrappedPaths));
console.log("\nBootstrapped indicator loadings:");
console.log(formatMatrix(bootSummary.bootstrappedLoadings));
