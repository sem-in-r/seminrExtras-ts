/**
 * Congruence-coefficient testing on the MOBI model, mirroring seminrExtras'
 * `seminr-pls-congruence` demo. The congruence test asks, for each pair of
 * constructs, whether their correlation-pattern columns are congruent (cosine
 * similarity close to a threshold, 1 by default) — a discriminant-validity
 * style check bootstrapped for significance.
 *
 * The R demo uses nboot = 2000; here nboot = 200 keeps the run fast.
 *
 * Run: bun run build && bun run demos/pls-congruence.ts
 */

import { composite, constructs, estimatePls, multiItems, paths, relationships } from "@seminr/core";
import { congruenceTest } from "@seminr/extras";
import { loadMobi } from "./lib/data.ts";
import { heading } from "./lib/print.ts";

const NBOOT = 200; // R demo uses 2000

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

console.log(heading("Congruence test (threshold = 1, alpha = 0.05)"));
const result = congruenceTest(model, { nboot: NBOOT, seed: 123, alpha: 0.05, threshold: 1 })!;
console.log(String(result));

console.log(heading("Congruence test summary (bootstrap detail)"));
console.log(result.summarize());
