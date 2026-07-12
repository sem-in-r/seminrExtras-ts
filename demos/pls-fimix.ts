/**
 * FIMIX-PLS (Finite Mixture PLS) on the corporate reputation model, mirroring
 * seminrExtras' `seminr-pls-fimix` demo. FIMIX-PLS uncovers unobserved
 * heterogeneity with an EM algorithm: observations are probabilistically
 * assigned to K latent segments, each with its own structural path
 * coefficients. Comparing information criteria across K helps choose the
 * number of segments.
 *
 * The R demo uses nstart = 10 and K_range = 2:4; here nstart = 5 keeps the run
 * fast while still exercising multiple random starts.
 *
 * Run: bun run build && bun run demos/pls-fimix.ts
 */

import {
  composite,
  constructs,
  estimatePls,
  meanReplacement,
  modeB,
  multiItems,
  paths,
  relationships,
  singleItem,
} from "@seminr/core";
import { assessFimix, assessFimixCompare, plot } from "@seminr/extras";
import { loadCorpRep } from "./lib/data.ts";
import { heading, rendered } from "./lib/print.ts";

const NSTART = 5; // R demo uses 10

const corpRep = await loadCorpRep();

const mm = constructs(
  composite("QUAL", multiItems("qual_", [1, 2, 3, 4, 5, 6, 7, 8]), modeB),
  composite("PERF", multiItems("perf_", [1, 2, 3, 4, 5]), modeB),
  composite("CSOR", multiItems("csor_", [1, 2, 3, 4, 5]), modeB),
  composite("ATTR", multiItems("attr_", [1, 2, 3]), modeB),
  composite("COMP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
);
const sm = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths("CUSA", "CUSL"),
);
const model = estimatePls(corpRep, mm, sm, { missing: meanReplacement, missingValue: -99 });

// --- FIMIX for a single K ---
console.log(heading("FIMIX-PLS for K = 2"));
const fimix = assessFimix(model, { K: 2, nstart: NSTART, seed: 123 })!;
console.log(String(fimix));
console.log(heading("FIMIX-PLS (K = 2) summary"));
console.log(fimix.summarize());
console.log(rendered(plot(fimix), "FIMIX segment proportions / paths"));

// --- Compare fit criteria across K ---
console.log(heading("FIMIX-PLS comparison across K = 2, 3, 4"));
const compare = assessFimixCompare(model, { KRange: [2, 3, 4], nstart: NSTART, seed: 123 })!;
console.log(String(compare));
console.log(rendered(plot(compare), "FIMIX information-criteria comparison"));
