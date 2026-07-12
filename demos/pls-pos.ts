/**
 * PLS-POS (Prediction-Oriented Segmentation) on the corporate reputation
 * model, mirroring seminrExtras' `seminr-pls-pos` demo. PLS-POS is a
 * deterministic hill-climbing segmentation that maximizes the summed R-squared
 * of the endogenous constructs across K segments. Unlike FIMIX-PLS it makes no
 * distributional assumptions and can capture heterogeneity in formative
 * measurement models too.
 *
 * PLS-POS re-estimates sub-models on every candidate move, so it is the
 * slowest demo. The R demo uses nstart = 10 and max_iter = 100; here
 * nstart = 2 and a small maxIter keep the run to a few seconds.
 *
 * Run: bun run build && bun run demos/pls-pos.ts
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
  summarizePls,
} from "@seminr/core";
import { assessPos, assessPosCompare, plot, posSegments } from "@seminr/extras";
import { loadCorpRep } from "./lib/data.ts";
import { formatMatrix, heading, rendered } from "./lib/print.ts";

const NSTART = 2; // R demo uses 10
const MAX_ITER = 8; // R demo uses 100

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

// --- PLS-POS for a single K ---
console.log(heading("PLS-POS for K = 2"));
const pos = assessPos(model, { K: 2, nstart: NSTART, maxIter: MAX_ITER, seed: 123 })!;
console.log(String(pos));
console.log(heading("PLS-POS (K = 2) summary"));
console.log(pos.summarize());

console.log(rendered(plot(pos, { type: "segments" }), "POS segment proportions"));
console.log(rendered(plot(pos, { type: "rsquared" }), "POS R-squared per segment"));
console.log(rendered(plot(pos, { type: "paths" }), "POS path coefficients per segment"));

// --- Extract the re-estimated per-segment PLS models ---
console.log(heading("Segment-specific models (pos_segments)"));
const segModels = posSegments(pos);
console.log(`Recovered ${segModels.length} segment models.`);
console.log("Segment 1 structural paths (R^2 rows on top):");
console.log(formatMatrix(summarizePls(segModels[0]!).paths));

// --- Compare the objective across K ---
console.log(heading("PLS-POS comparison across K = 2, 3"));
const compare = assessPosCompare(model, {
  KRange: [2, 3],
  nstart: NSTART,
  maxIter: 5,
  seed: 123,
})!;
console.log(String(compare));
console.log(rendered(plot(compare), "POS objective vs K"));
