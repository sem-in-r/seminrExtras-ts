/**
 * Debugging seminr models: reading the errors a mis-specified model produces.
 *
 * Mirrors seminrExtras' `seminr-help-debugging` demo. The R demo relies on
 * `estimate_pls(..., assess_syntax = TRUE)` to pre-screen for common mistakes;
 * seminr-ts ports that screen as `assessModelSpecification(mm, sm, columns)`,
 * so this demo runs each mis-specification through it, catches the error, and
 * shows the exact message you would see (estimatePls throws equivalent errors
 * later if the screen is skipped). Finally it inspects a correctly specified
 * model before estimating it (the analogue of R's "plot the conceptual model
 * to eyeball it").
 *
 * Run: bun run build && bun run demos/help-debugging.ts
 */

import {
  assessModelSpecification,
  composite,
  constructs,
  estimatePls,
  interactionTerm,
  meanReplacement,
  multiItems,
  paths,
  relationships,
  singleItem,
  SmMatrix,
  twoStage,
  type MeasurementModel,
  type SMMatrix,
} from "@seminr/core";
import { loadCorpRep } from "./lib/data.ts";
import { heading } from "./lib/print.ts";

const corpRep = await loadCorpRep();

/** Run a specification screen expected to fail; print the caught error. */
function showError(label: string, mm: MeasurementModel, sm: SMMatrix): void {
  try {
    assessModelSpecification(mm, SmMatrix.from(sm), corpRep.columns);
    console.log(`${label}: unexpectedly succeeded (no error raised)`);
  } catch (error) {
    console.log(`${label}: ${(error as Error).message}`);
  }
}

// Problem 1: construct misspelled in the measurement model.
// MM defines "COP" but the structural model references "COMP".
console.log(heading("Problem 1: construct misspelled in the measurement model"));
const errorMm = constructs(
  composite("COP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
);
const simpleSm = relationships(paths(["COMP", "LIKE"], ["CUSA", "CUSL"]), paths("CUSA", "CUSL"));
showError("  assessModelSpecification", errorMm, simpleSm);

// Problem 2: construct misspelled in the structural model.
console.log(heading("Problem 2: construct misspelled in the structural model"));
const simpleMm = constructs(
  composite("COMP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
);
const errorSm = relationships(paths(["COP", "LIKE"], ["CUSA", "CUSL"]), paths("CUSA", "CUSL"));
showError("  assessModelSpecification", simpleMm, errorSm);

// Problem 3: interaction whose moderator is missing from the structural model.
console.log(heading("Problem 3: moderator omitted from an interaction"));
const interactionMm = constructs(
  composite("COMP", multiItems("comp_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  interactionTerm({ iv: "COMP", moderator: "LIKE", method: twoStage }),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
);
// The moderator LIKE never appears as a source in the structural model.
const interactionSm = relationships(paths(["COMP", "COMP*LIKE"], "CUSL"));
showError("  assessModelSpecification", interactionMm, interactionSm);

// Problem 4: indicator names that do not match the data columns.
// "co_1".."co_3" do not exist; the columns are "comp_1".."comp_3".
console.log(heading("Problem 4: indicator names that do not match the data"));
const badIndicatorMm = constructs(
  composite("COMP", multiItems("co_", [1, 2, 3])),
  composite("LIKE", multiItems("like_", [1, 2, 3])),
  composite("CUSA", singleItem("cusa")),
  composite("CUSL", multiItems("cusl_", [1, 2, 3])),
);
showError("  assessModelSpecification", badIndicatorMm, simpleSm);

// Inspecting the specification before estimating: the analogue of R's "plot
// the conceptual model". The measurement model is a list of construct specs
// and the structural model a list of source -> target rows, so we can eyeball
// names and paths. (seminr-ts's DOT plotting targets estimated models.)
console.log(heading("Inspecting a correctly specified model before estimating"));
console.log("Measurement model constructs and indicators:");
for (const spec of simpleMm) {
  if (spec.kind === "construct") {
    console.log(`  ${spec.name} (${spec.type}): ${spec.items.join(", ")}`);
  }
}
console.log("Structural paths:");
for (const row of simpleSm) {
  console.log(`  ${row.source} -> ${row.target}`);
}

const model = estimatePls(corpRep, simpleMm, simpleSm, {
  missing: meanReplacement,
  missingValue: -99,
});
console.log(`\nCorrected model estimated cleanly in ${model.iterations} iterations.`);
