/**
 * IPMA / cIPMA (Combined Importance-Performance Map Analysis) on the MOBI
 * model, mirroring seminrExtras' `seminr-pls-cipma` demo. IPMA plots each
 * predictor's importance (unstandardized total effect on the target) against
 * its performance (construct scores rescaled to 0-100). cIPMA overlays an NCA
 * to flag constructs that are *necessary* conditions for the outcome.
 *
 * The R demo uses nca_test.rep = 1000; here ncaTestRep = 100 keeps the run
 * fast (this is the only stochastic part).
 *
 * Run: bun run build && bun run demos/pls-cipma.ts
 */

import { composite, constructs, estimatePls, multiItems, paths, relationships } from "@seminr/core";
import { assessCipma, assessIpma, plot } from "@seminr/extras";
import { loadMobi } from "./lib/data.ts";
import { heading, rendered } from "./lib/print.ts";

const NCA_TEST_REP = 100; // R demo uses 1000

const mobi = await loadMobi();

const mm = constructs(
  composite("Image", multiItems("IMAG", [1, 2, 3, 4, 5])),
  composite("Expectation", multiItems("CUEX", [1, 2, 3])),
  composite("Value", multiItems("PERV", [1, 2])),
  composite("Satisfaction", multiItems("CUSA", [1, 2, 3])),
  composite("Loyalty", multiItems("CUSL", [1, 2, 3])),
);
const sm = relationships(
  paths("Image", ["Expectation", "Satisfaction", "Loyalty"]),
  paths("Expectation", ["Value", "Satisfaction"]),
  paths("Value", "Satisfaction"),
  paths("Satisfaction", "Loyalty"),
);
const model = estimatePls(mobi, mm, sm);

// --- IPMA only (importance + performance, no necessity) ---
console.log(heading("IPMA for target = Loyalty (1-10 scale)"));
const ipma = assessIpma(model, { target: "Loyalty", scaleMin: 1, scaleMax: 10 })!;
console.log(String(ipma));
console.log(heading("IPMA summary"));
console.log(ipma.summarize());
console.log(rendered(plot(ipma, { type: "ipma" }), "IPMA map"));

// --- cIPMA: IPMA + NCA necessity classification ---
console.log(heading("cIPMA for target = Loyalty (IPMA + NCA)"));
const cipma = assessCipma(model, {
  target: "Loyalty",
  scaleMin: 1,
  scaleMax: 10,
  ncaTestRep: NCA_TEST_REP,
  seed: 123,
})!;
console.log(String(cipma));
console.log(heading("cIPMA summary"));
console.log(cipma.summarize());

console.log(rendered(plot(cipma, { type: "cipma" }), "cIPMA map"));
console.log(rendered(plot(cipma, { type: "ipma" }), "IPMA map from cIPMA record"));
console.log(
  rendered(
    plot(cipma, { importanceMetric: "standardized" }),
    "cIPMA map (standardized importance)",
  ),
);
