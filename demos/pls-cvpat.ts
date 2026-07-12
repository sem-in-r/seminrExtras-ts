/**
 * CVPAT (Cross-Validated Predictive Ability Test) on the corporate reputation
 * model, mirroring seminrExtras' `seminr-pls-cvpat` demo: build an established
 * model and a competing model, compare their predictive loss with
 * `assessCvpatCompare`, then benchmark the established model against LM/IA
 * baselines with `assessCvpat`.
 *
 * The R demo uses nboot = 2000; here nboot = 200 keeps the run fast. R also
 * passes reps = 10 to predict_pls (a numeric no-op re-running identical folds);
 * assessCvpat accepts reps for API parity but it changes nothing.
 *
 * Run: bun run build && bun run demos/pls-cvpat.ts
 */

import {
  composite,
  constructs,
  estimatePls,
  meanReplacement,
  modeB,
  multiItems,
  paths,
  predictDA,
  relationships,
  singleItem,
  type MeasurementModel,
} from "@seminr/core";
import { assessCvpat, assessCvpatCompare } from "@seminr/extras";
import { loadCorpRep } from "./lib/data.ts";
import { heading } from "./lib/print.ts";

const NBOOT = 200; // R demo uses 2000

// Formative (modeB) driver constructs feeding two reflective outcomes.
function corpRepMeasurement(): MeasurementModel {
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

const corpRep = await loadCorpRep();

// Established structural model: COMP/LIKE drive both CUSA and CUSL.
const establishedSm = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths("CUSA", "CUSL"),
);
// Competing model: COMP/LIKE drive only CUSA (a fully mediated alternative).
const altSm = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], "CUSA"),
  paths("CUSA", "CUSL"),
);

const establishedModel = estimatePls(corpRep, corpRepMeasurement(), establishedSm, {
  missing: meanReplacement,
  missingValue: -99,
});
const competingModel = estimatePls(corpRep, corpRepMeasurement(), altSm, {
  missing: meanReplacement,
  missingValue: -99,
});

// Compare the two models' out-of-sample loss (base vs. alternative).
console.log(heading("CVPAT model comparison (established vs. competing)"));
const compare = assessCvpatCompare(establishedModel, competingModel, {
  testtype: "two.sided",
  nboot: NBOOT,
  technique: predictDA,
  seed: 123,
  noFolds: 10,
});
console.log(String(compare));

// Benchmark the established model against the linear-model (LM) and
// indicator-average (IA) baselines.
console.log(heading("CVPAT assessment of the established model"));
const assessment = assessCvpat(establishedModel, {
  testtype: "two.sided",
  nboot: NBOOT,
  seed: 123,
  technique: predictDA,
  noFolds: 10,
});
console.log(String(assessment));
