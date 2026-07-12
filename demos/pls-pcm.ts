/**
 * PCM (Predictive Contribution of the Mediator) on MOBI mediation models,
 * mirroring seminrExtras' `seminr-pls-pcm` demo. PCM evaluates the predictive
 * contribution of a mediating construct by comparing the Direct Antecedents
 * (DA) and Earliest Antecedents (EA) prediction approaches on isolated
 * mediation sub-models (Danks, 2021).
 *
 * Run: bun run build && bun run demos/pls-pcm.ts
 */

import { composite, constructs, estimatePls, multiItems, paths, relationships } from "@seminr/core";
import { assessPcm, plot } from "@seminr/extras";
import { loadMobi } from "./lib/data.ts";
import { heading, rendered } from "./lib/print.ts";

const mobi = await loadMobi();

// --- Step 1: a single-mediator model Image -> Satisfaction -> Loyalty ---
const mm = constructs(
  composite("Image", multiItems("IMAG", [1, 2, 3, 4, 5])),
  composite("Satisfaction", multiItems("CUSA", [1, 2, 3])),
  composite("Loyalty", multiItems("CUSL", [1, 2, 3])),
);
const sm = relationships(
  paths("Image", "Satisfaction"),
  paths("Satisfaction", "Loyalty"),
  paths("Image", "Loyalty"),
);
const model = estimatePls(mobi, mm, sm);
console.log(`Estimated the single-mediator model in ${model.iterations} iterations.`);

// PCM auto-detects the mediation path Image -> Satisfaction -> Loyalty.
console.log(heading("PCM: single mediation path (target = Loyalty)"));
const pcm = assessPcm(model, { target: "Loyalty", noFolds: 10, reps: 10 })!;
console.log(String(pcm));
console.log(heading("PCM detailed per-indicator summary"));
console.log(pcm.summarize());
console.log(rendered(plot(pcm), "PCM barplot"));

// --- Step 2: a multi-mediator model with three paths into Loyalty ---
const mmFull = constructs(
  composite("Image", multiItems("IMAG", [1, 2, 3, 4, 5])),
  composite("Expectation", multiItems("CUEX", [1, 2, 3])),
  composite("Value", multiItems("PERV", [1, 2])),
  composite("Satisfaction", multiItems("CUSA", [1, 2, 3])),
  composite("Loyalty", multiItems("CUSL", [1, 2, 3])),
);
const smFull = relationships(
  paths("Image", ["Expectation", "Satisfaction", "Loyalty"]),
  paths("Expectation", ["Value", "Satisfaction"]),
  paths("Value", "Satisfaction"),
  paths("Satisfaction", "Loyalty"),
);
const modelFull = estimatePls(mobi, mmFull, smFull);

// PCM finds all three mediation paths through Satisfaction into Loyalty.
console.log(heading("PCM: multiple mediation paths (target = Loyalty)"));
const pcmFull = assessPcm(modelFull, { target: "Loyalty", noFolds: 10, reps: 10 })!;
console.log(String(pcmFull));
console.log(rendered(plot(pcmFull), "PCM comparison barplot"));
