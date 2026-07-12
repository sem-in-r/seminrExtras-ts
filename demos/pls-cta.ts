/**
 * CTA-PLS (Confirmatory Tetrad Analysis) on the MOBI model, mirroring
 * seminrExtras' `seminr-pls-cta` demo. CTA-PLS tests whether each construct's
 * indicators are consistent with a reflective (common-factor) specification:
 * under that null every model-implied vanishing tetrad is zero. With
 * `borrow: true` (default), constructs with only 2-3 indicators borrow
 * indicators from structurally connected constructs to form testable 4-tuples.
 *
 * The R demo uses nboot = 5000; here nboot = 200 keeps the run fast.
 *
 * Run: bun run build && bun run demos/pls-cta.ts
 */

import { composite, constructs, estimatePls, multiItems, paths, relationships } from "@seminr/core";
import { assessCta, plot } from "@seminr/extras";
import { loadMobi } from "./lib/data.ts";
import { heading, rendered } from "./lib/print.ts";

const NBOOT = 200; // R demo uses 5000

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

// --- CTA-PLS with borrowing (default) ---
console.log(heading("CTA-PLS with borrowing (default)"));
const cta = assessCta(model, { nboot: NBOOT, seed: 123 })!;
console.log(String(cta));
console.log(heading("CTA-PLS detailed tetrad results"));
console.log(cta.summarize());
console.log(rendered(plot(cta), "CTA adjusted-p-value plot"));

// --- CTA-PLS without borrowing: only constructs with >= 4 indicators tested ---
console.log(heading("CTA-PLS without borrowing"));
const ctaNoBorrow = assessCta(model, { nboot: NBOOT, borrow: false, seed: 123 })!;
console.log(String(ctaNoBorrow));
