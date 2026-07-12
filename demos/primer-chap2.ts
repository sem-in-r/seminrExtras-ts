/**
 * Primer Chapter 2: Getting started with seminr in TypeScript.
 *
 * The R workbook chapter 2 introduces R and RStudio: install.packages(),
 * library(), the ? help operator, vignette(), demo(), and the learnr
 * tutorials. This is the TypeScript analogue — how to install the packages,
 * import them, get help, and find the demos — kept honest to what the
 * TypeScript ecosystem provides.
 *
 * Accompanies: Partial Least Squares SEM Using R (2nd Ed., 2026), Hair et al.
 *
 * Run: bun run build && bun run demos/primer-chap2.ts
 */

import { version as coreVersion } from "@seminr/core";
import { version as extrasVersion } from "@seminr/extras";
import { heading } from "./lib/print.ts";

console.log(heading("Installing the packages"));
// In R you would run install.packages("seminr"); the TypeScript equivalents
// are npm packages, installed with bun / npm:
console.log("  bun add @seminr/core @seminr/extras       # or: npm install ...");

console.log(heading("Importing the packages"));
// library(seminr) / library(seminrExtras) becomes an ES import.
console.log(`  import { estimatePls } from "@seminr/core"     -> version ${coreVersion}`);
console.log(`  import { assessCvpat } from "@seminr/extras"   -> version ${extrasVersion}`);

console.log(heading("A first vector"));
// R: vector <- c(1, 2, 3, 4, 5). The TypeScript analogue is an array.
const vector = [1, 2, 3, 4, 5];
console.log(`  const vector = [${vector.join(", ")}]`);

console.log(heading("Getting help"));
// R uses ?estimate_pls and vignette(); here every export carries JSDoc that
// editors surface on hover, and the .d.ts types document the full signatures.
console.log("  Hover any import in your editor for its JSDoc (the ? analogue),");
console.log("  or read the declaration files under dist/.");

console.log(heading("Finding the demos"));
// R: demo() lists demos; demo("seminr-pls-ecsi") runs one. Here the demos are
// runnable scripts under demos/.
console.log("  The demos live in this demos/ directory. Run any of them, e.g.:");
console.log("    bun run demos/pls-cvpat.ts");
console.log("    bun run demos/primer-chap3.ts");

// Not ported: learnr interactive tutorials (run_tutorial()) and R vignettes —
// these are R/RStudio-specific and have no seminr-ts analogue.
