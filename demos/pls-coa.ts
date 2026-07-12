/**
 * COA (Composite Overfit Analysis) on the corporate reputation model,
 * mirroring seminrExtras' `seminr-pls-coa` demo. COA detects observation-level
 * overfitting: it computes each observation's predictive deviance (in-sample
 * minus out-of-sample fit), grows a regression tree over the construct scores
 * to isolate deviant subgroups, and re-estimates the model without each group
 * to expose parameter instability.
 *
 * Run: bun run build && bun run demos/pls-coa.ts
 */

import {
  composite,
  constructs,
  estimatePls,
  meanReplacement,
  multiItems,
  paths,
  relationships,
  singleItem,
} from "@seminr/core";
import {
  assessCoa,
  competes,
  devianceTree,
  groupRules,
  plot,
  predictiveDeviance,
  unstableParams,
  type CoaCompetes,
  type CoaRules,
} from "@seminr/extras";
import { loadCorpRep } from "./lib/data.ts";
import { formatMatrix, heading, rendered } from "./lib/print.ts";

const corpRep = await loadCorpRep();

const mm = constructs(
  composite("COMP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
);
const sm = relationships(paths(["COMP", "LIKE"], "CUSA"), paths("CUSA", "CUSL"));
const model = estimatePls(corpRep, mm, sm, { missing: meanReplacement, missingValue: -99 });

// --- Full COA pipeline ---
// seed 1: with this fold RNG the run yields several deviant groups (the R/py
// demos' seed 123 happens to yield none here, which would make a dull demo).
console.log(heading("COA for focal construct = CUSL"));
const coa = assessCoa(model, "CUSL", { noFolds: 10, reps: 1, cores: 1, seed: 1 })!;
console.log(String(coa));
console.log(heading("COA summary"));
console.log(coa.summarize());

console.log(rendered(plot(coa, { type: "pd" }), "predictive-deviance distribution"));
console.log(rendered(plot(coa, { type: "tree" }), "deviance decision tree"));
console.log(rendered(plot(coa, { type: "groups" }), "deviant-group highlights"));

// --- Step-by-step: the same pipeline via individual functions ---
console.log(heading("Step 1: predictive deviance"));
const pd = predictiveDeviance(model, "CUSL", { noFolds: 10, reps: 1, cores: 1, seed: 1 })!;
console.log(
  `IS MSE: ${pd.isMse.toFixed(4)}  OOS MSE: ${pd.oosMse.toFixed(4)}  ` +
    `Overfit ratio: ${pd.overfitRatio.toFixed(4)}`,
);

console.log(heading("Step 2: deviance tree groups"));
const tree = devianceTree(pd, [0.025, 0.975]);
console.log(`Deviant groups: ${Object.keys(tree.deviantGroups).sort().join(", ")}`);

console.log(heading("Step 3: parameter instability (path_coef)"));
const instab = unstableParams(model, tree, "path_coef");
for (const [label, group] of Object.entries(instab.groups)) {
  console.log(`Group ${label} (cases ${group.cases.join(", ")}) path_coef diffs:`);
  console.log(formatMatrix(group.paramDiffs["path_coef"]!));
}

console.log(heading("Decision rules for all deviant groups"));
for (const [label, rules] of Object.entries(groupRules(tree) as Record<string, CoaRules>)) {
  console.log(`Group ${label}:`);
  console.log(
    rules.construct
      .map((c, i) => `  ${rules.gte[i]!.toFixed(3)} <= ${c} < ${rules.lt[i]!.toFixed(3)}`)
      .join("\n"),
  );
}

console.log(heading("Competing splits at all group root nodes"));
for (const [label, comp] of Object.entries(competes(tree) as Record<string, CoaCompetes>)) {
  console.log(`Group ${label}:`);
  console.log(
    comp.criterion
      .map(
        (c, i) =>
          `  ${c} ${comp.sign[i]} ${comp.value[i]!.toFixed(3)}  (improve ${comp.improve[i]!.toFixed(3)})`,
      )
      .join("\n"),
  );
}
