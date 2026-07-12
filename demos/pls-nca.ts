/**
 * NCA (Necessary Condition Analysis) on a MOBI PLS-SEM model, mirroring
 * seminrExtras' `seminr-pls-nca` demo. NCA asks whether a predictor is a
 * *necessary* (not merely sufficient) condition for an outcome by fitting a
 * ceiling line over the XY scatter and measuring the empty upper-left "ceiling
 * zone". The demo also runs NCA-ESSE, which stress-tests the effect size
 * against removal of extreme upper-left observations (Becker et al., 2026).
 *
 * The R demo uses test.rep = 1000; here testRep = 100 keeps the run fast.
 *
 * Run: bun run build && bun run demos/pls-nca.ts
 */

import { composite, constructs, estimatePls, multiItems, paths, relationships } from "@seminr/core";
import { assessNca, assessNcaEsse, plot } from "@seminr/extras";
import { loadMobi } from "./lib/data.ts";
import { heading, rendered } from "./lib/print.ts";

const TEST_REP = 100; // R demo uses 1000

const mobi = await loadMobi();

const mm = constructs(
  composite("Image", multiItems("IMAG", [1, 2, 3, 4, 5])),
  composite("Value", multiItems("PERV", [1, 2])),
  composite("Satisfaction", multiItems("CUSA", [1, 2, 3])),
  composite("Loyalty", multiItems("CUSL", [1, 2, 3])),
);
const sm = relationships(
  paths(["Image", "Value"], "Satisfaction"),
  paths("Satisfaction", "Loyalty"),
);
const model = estimatePls(mobi, mm, sm);

// --- NCA on Satisfaction (Image and Value are the predictors) ---
console.log(heading("NCA for target = Satisfaction"));
const ncaSat = assessNca(model, { target: "Satisfaction", testRep: TEST_REP, seed: 123 })!;
console.log(String(ncaSat));
console.log(heading("NCA (Satisfaction) summary with bottleneck tables"));
console.log(ncaSat.summarize());
console.log(rendered(plot(ncaSat, { type: "effects" }), "NCA effect-size bar plot"));
console.log(rendered(plot(ncaSat, { type: "scatter" }), "NCA ceiling scatter plots"));

// --- NCA on Loyalty (single predictor, Satisfaction) ---
console.log(heading("NCA for target = Loyalty"));
const ncaLoy = assessNca(model, { target: "Loyalty", testRep: TEST_REP, seed: 123 })!;
console.log(String(ncaLoy));

// --- NCA-ESSE sensitivity analysis on Satisfaction ---
console.log(heading("NCA-ESSE for target = Satisfaction"));
const thresholds = Array.from({ length: 11 }, (_, i) => Math.round(i * 5) / 1000); // seq(0, 0.05, by = 0.005)
const esse = assessNcaEsse(model, { target: "Satisfaction", thresholds, seed: 123 })!;
console.log(String(esse));
console.log(heading("NCA-ESSE summary"));
console.log(esse.summarize());
console.log(rendered(plot(esse, { type: "sensitivity" }), "ESSE sensitivity plot"));
console.log(rendered(plot(esse, { type: "difference" }), "ESSE difference plot"));
