/**
 * Primer Chapter 5: Evaluating formative measurement models.
 *
 * Ports the R workbook chapter 5 on the extended corporate reputation model,
 * whose QUAL/PERF/CSOR/ATTR drivers are formative (modeB): reliability, HTMT,
 * redundancy analyses for convergent validity, indicator collinearity (VIF),
 * and bootstrapped weights and loadings.
 *
 * Accompanies: Partial Least Squares SEM Using R (2nd Ed., 2026), Hair et al.
 *
 * Run: bun run build && bun run demos/primer-chap5.ts
 */

import {
  bootstrapModel,
  composite,
  constructs,
  estimatePls,
  meanReplacement,
  modeB,
  multiItems,
  nmGet,
  paths,
  relationships,
  singleItem,
  summarizePls,
  summarizePlsBoot,
  type Dataset,
} from "@seminr/core";
import { loadCorpRep } from "./lib/data.ts";
import { formatMatrix, heading } from "./lib/print.ts";

const corpRep = await loadCorpRep();

/** Formative-vs-global redundancy model; print its path (convergent validity). */
function redundancy(data: Dataset, prefix: string, count: number, globalItem: string): void {
  const items = Array.from({ length: count }, (_, i) => i + 1);
  const mm = constructs(
    composite(`${prefix}_F`, multiItems(`${prefix.toLowerCase()}_`, items), modeB),
    composite(`${prefix}_G`, singleItem(globalItem)),
  );
  const sm = relationships(paths(`${prefix}_F`, `${prefix}_G`));
  const model = estimatePls(data, mm, sm, { missing: meanReplacement, missingValue: -99 });
  const coef = nmGet(summarizePls(model).paths, `${prefix}_F`, `${prefix}_G`);
  console.log(
    `  ${prefix}: ${prefix}_F -> ${prefix}_G path = ${coef.toFixed(3)}  ` +
      "(>= 0.70 supports convergent validity)",
  );
}

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
const summary = summarizePls(model);
console.log(`Estimated the extended (formative) model in ${model.iterations} iterations.`);

console.log(heading("Internal consistency and reliability"));
console.log(formatMatrix(summary.reliability));

console.log(heading("HTMT discriminant validity"));
console.log(formatMatrix(summary.validity.htmt));

console.log(heading("Redundancy analysis (convergent validity of formative constructs)"));
redundancy(corpRep, "ATTR", 3, "attr_global");
redundancy(corpRep, "CSOR", 5, "csor_global");
redundancy(corpRep, "PERF", 5, "perf_global");
redundancy(corpRep, "QUAL", 8, "qual_global");

console.log(heading("Indicator collinearity (VIF)"));
for (const [construct, vifs] of Object.entries(summary.validity.vifItems)) {
  const line = Object.entries(vifs)
    .map(([item, v]) => `${item}=${v.toFixed(2)}`)
    .join(", ");
  console.log(`  ${construct}: ${line}`);
}

const boot = bootstrapModel({ model, nboot: 1000, seed: 123 });
const bootSummary = summarizePlsBoot(boot);

console.log(heading("Bootstrapped outer weights"));
console.log(formatMatrix(bootSummary.bootstrappedWeights));

console.log(heading("Bootstrapped outer loadings"));
console.log(formatMatrix(bootSummary.bootstrappedLoadings));
