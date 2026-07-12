/**
 * Primer Chapter 6: Evaluating the structural model.
 *
 * Ports the R workbook chapter 6: structural collinearity (VIF), bootstrapped
 * paths and total effects, R-squared and f-squared, a CVPAT assessment, and a
 * predictive model comparison across three competing structural models using
 * the BIC information criterion, Akaike weights, CVPAT, and predictive
 * overfit ratios.
 *
 * The CVPAT nboot is reduced from 2000 to 200 for speed; R's reps argument is
 * a numeric no-op and omitted.
 *
 * Accompanies: Partial Least Squares SEM Using R (2nd Ed., 2026), Hair et al.
 *
 * Run: bun run build && bun run demos/primer-chap6.ts
 */

import {
  bootstrapModel,
  composite,
  computeItCriteriaWeights,
  constructs,
  estimatePls,
  meanReplacement,
  modeB,
  multiItems,
  nmGet,
  paths,
  predictDA,
  predictPls,
  relationships,
  singleItem,
  summarizePls,
  summarizePlsBoot,
  summarizePlsPredict,
  type MeasurementModel,
} from "@seminr/core";
import { assessCvpat, assessCvpatCompare } from "@seminr/extras";
import { loadCorpRep } from "./lib/data.ts";
import { formatMatrix, heading } from "./lib/print.ts";

const NBOOT = 200; // R demo uses 2000

function measurement(): MeasurementModel {
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

const sm = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths("CUSA", "CUSL"),
);
const model = estimatePls(corpRep, measurement(), sm, {
  missing: meanReplacement,
  missingValue: -99,
});
const summary = summarizePls(model);
const boot = bootstrapModel({ model, nboot: NBOOT, seed: 123 });
const bootSummary = summarizePlsBoot(boot);

console.log(heading("Structural collinearity (VIF antecedents)"));
for (const [target, vifs] of Object.entries(summary.vifAntecedents)) {
  const line = Object.entries(vifs)
    .map(([antecedent, v]) => `${antecedent}=${v.toFixed(2)}`)
    .join(", ");
  console.log(`  ${target}: ${line}`);
}

console.log(heading("Bootstrapped structural paths"));
console.log(formatMatrix(bootSummary.bootstrappedPaths));

console.log(heading("Bootstrapped total effects"));
console.log(formatMatrix(bootSummary.bootstrappedTotalPaths));

console.log(heading("R-squared and effect sizes (f-squared)"));
console.log(formatMatrix(summary.paths));
console.log("");
console.log(formatMatrix(summary.fSquare));

console.log(heading("CVPAT assessment (testtype = greater)"));
const assessment = assessCvpat(model, {
  testtype: "greater",
  nboot: NBOOT,
  seed: 123,
  technique: predictDA,
  noFolds: 10,
});
console.log(String(assessment));

// --- Predictive comparison of three competing structural models ---
const sm1 = sm;
const sm2 = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE", "CUSA"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths("CUSA", "CUSL"),
);
const sm3 = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE", "CUSA", "CUSL"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths("CUSA", "CUSL"),
);
const models = {
  Model1: estimatePls(corpRep, measurement(), sm1, { missingValue: -99 }),
  Model2: estimatePls(corpRep, measurement(), sm2, { missingValue: -99 }),
  Model3: estimatePls(corpRep, measurement(), sm3, { missingValue: -99 }),
};

console.log(heading("BIC for CUSA across competing models"));
const bic: Record<string, number> = {};
for (const [name, m] of Object.entries(models)) {
  bic[name] = nmGet(summarizePls(m).itCriteria, "BIC", "CUSA");
  console.log(`  ${name}: BIC(CUSA) = ${bic[name]!.toFixed(3)}`);
}

console.log(heading("BIC-based Akaike weights"));
const weights = computeItCriteriaWeights(Object.values(bic));
Object.keys(bic).forEach((name, i) => {
  console.log(`  ${name}: weight = ${weights[i]!.toFixed(4)}`);
});

console.log(heading("CVPAT model comparison (Model1 vs Model3, testtype = greater)"));
const compare = assessCvpatCompare(models.Model1, models.Model3, {
  testtype: "greater",
  nboot: NBOOT,
  technique: predictDA,
  seed: 123,
  noFolds: 10,
});
console.log(String(compare));

console.log(heading("Predictive overfit ratios (construct-level metrics)"));
for (const name of ["Model1", "Model3"] as const) {
  console.log(`${name}:`);
  console.log(formatMatrix(summarizePlsPredict(predictPls(models[name])).constructError));
}
