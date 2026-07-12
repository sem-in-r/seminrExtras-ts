# @seminr/extras

Supplementary evaluation and validation methods for **PLS-SEM** models,
compatible with [SEMinR](https://github.com/sem-in-r/seminr-ts). This is a
from-scratch TypeScript port of the R package
[`seminrExtras`](https://github.com/sem-in-r/seminrExtras); the R package is the
spec, and every entry point is pinned to R-generated golden fixtures for numeric
parity.

Like its R counterpart, `@seminr/extras` is **not a standalone package**. It does
not estimate models — it consumes an *already-estimated* SEMinR model and runs
further analysis on it. Estimation is provided by
[`@seminr/core`](https://www.npmjs.com/package/@seminr/core) (the TypeScript port
of the `seminr` estimation engine), which is a required dependency.

SEMinR (Ray, Danks, & Calero Valdez, 2026) is a domain-specific language for
building, estimating, and evaluating structural equation models.

## Methods

- **Cross-Validated Predictive Ability Test (CVPAT)** — Compare model predictive
  performance against benchmarks or alternative models (Liengaard et al., 2021;
  Sharma et al., 2022).
- **Predictive Contribution of the Mediator (PCM)** — Evaluate whether a
  mediating construct improves out-of-sample predictive accuracy by comparing
  Direct Antecedent (DA) and Earliest Antecedent (EA) prediction approaches
  (Danks, 2021).
- **Combined Importance-Performance Map Analysis (cIPMA)** — IPMA with NCA
  integration to identify constructs that are both important and necessary
  (Ringle & Sarstedt, 2016; Sarstedt et al., 2024; Hauff et al., 2024).
- **Composite Overfit Analysis (COA)** — Detect observation-level overfitting in
  PLS composite models via predictive deviance trees and parameter instability
  analysis.
- **Necessary Condition Analysis (NCA)** — Test whether predictors are necessary
  conditions for an outcome, complementing PLS-SEM's sufficiency logic (Dul,
  2016; Richter et al., 2020).
- **NCA-ESSE** — Effect Size Sensitivity Extension that assesses how robust NCA
  results are to extreme response patterns (Becker et al., 2026).
- **Confirmatory Tetrad Analysis (CTA-PLS)** — Empirically test whether a
  construct's measurement model is reflective or formative, with indicator
  borrowing for constructs with fewer than 4 indicators (Gudergan et al., 2008).
- **Unobserved Heterogeneity** — Two complementary segmentation approaches for
  detecting latent classes in PLS-SEM:
  - **FIMIX-PLS** — EM-based probabilistic segmentation assuming normally
    distributed residuals (Hahn et al., 2002; Sarstedt et al., 2011).
  - **PLS-POS** — Deterministic hill-climbing segmentation that maximizes the sum
    of R-squared across segments, making no distributional assumptions (Becker
    et al., 2013).
- **Congruence Testing** — Bootstrapped congruence coefficient testing for
  construct validity (Franke, Sarstedt, & Danks, 2021).

## Installation

```sh
npm install @seminr/core @seminr/extras
# or: bun add @seminr/core @seminr/extras
```

Pure ESM, targeting Node 18+ / Bun. `@seminr/core` is a required runtime
dependency (it provides model estimation).

## Entry points

| Function | Description |
|---|---|
| `assessCvpat()` | CVPAT against LM and IA benchmarks |
| `assessCvpatCompare()` | Compare predictive loss of two PLS models |
| `assessPcm()` | Predictive Contribution of the Mediator (Danks, 2021) |
| `assessIpma()` | Importance-Performance Map Analysis (IPMA) |
| `assessCipma()` | Combined IPMA with Necessary Condition Analysis (cIPMA) |
| `assessCoa()` | Composite Overfit Analysis (full pipeline) |
| `predictiveDeviance()` | Compute predictive deviance scores |
| `devianceTree()` | Identify deviant case groups via decision tree |
| `unstableParams()` | Parameter instability analysis |
| `groupRules()` | Extract decision rules for deviant groups |
| `competes()` | Show competing splits at tree nodes |
| `assessNca()` | Necessary Condition Analysis for PLS-SEM |
| `assessNcaEsse()` | NCA with Effect Size Sensitivity Extension |
| `assessCta()` | Confirmatory Tetrad Analysis (CTA-PLS) with indicator borrowing (Gudergan et al., 2008) |
| `assessFimix()` | FIMIX-PLS latent class segmentation |
| `assessFimixCompare()` | Compare FIMIX solutions across K values |
| `assessPos()` | PLS-POS prediction-oriented segmentation (Becker et al., 2013) |
| `assessPosCompare()` | Compare PLS-POS solutions across K values |
| `posSegments()` | Extract segment-specific re-estimated PLS models |
| `congruenceTest()` | Bootstrapped congruence coefficient testing |
| `plot()` | Render any result to an SVG string (dispatches by result type) |

## Conventions

A few patterns hold across every entry point; they mirror the R package's S3
`print()`/`summary()` design:

- **Result objects** are frozen records with a `kind` discriminant. Call
  `String(result)` (or `result.toString()`) for the R `print()` equivalent and
  `result.summarize()` for the `summary()` equivalent. `plot(result)` returns an
  SVG string (the R `plot.*` methods).
- **Call styles** — every entry point accepts both a trailing named-options
  object (shown below) and R-style positional arguments.
- **Validation** — on invalid input an entry point emits a `console.warn` and
  returns `null` rather than throwing (the R warning-and-`NULL` convention), so
  examples below use `!` to assert a non-null result.
- **Randomness** — by default the resampling paths seed a `mulberry32` PRNG, so
  results are reproducible run-to-run but **not bit-identical to R**. For exact R
  parity you can inject the RNG streams (`draws`, `ordering(s)`, `perms`,
  `inits`, `partitions`) the R code used.

## Example: CVPAT

Estimate the model with `@seminr/core`, then assess it. `parseCsv` and the model
DSL (`constructs`, `composite`, `multiItems`, `paths`, `relationships`,
`estimatePls`) all come from `@seminr/core` — see its README for the estimation
API; `@seminr/extras` only consumes the estimated model.

```ts
import {
  composite, constructs, estimatePls, meanReplacement, modeB, multiItems,
  paths, predictDA, relationships, singleItem, parseCsv,
} from "@seminr/core";
import { assessCvpat, assessCvpatCompare } from "@seminr/extras";

const data = parseCsv(/* your CSV text */);

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

const establishedSm = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], ["CUSA", "CUSL"]),
  paths("CUSA", "CUSL"),
);
const competingSm = relationships(
  paths(["QUAL", "PERF", "CSOR", "ATTR"], ["COMP", "LIKE"]),
  paths(["COMP", "LIKE"], "CUSA"),
  paths("CUSA", "CUSL"),
);

const opts = { missing: meanReplacement, missingValue: -99 };
const established = estimatePls(data, mm, establishedSm, opts);
const competing = estimatePls(data, mm, competingSm, opts);

// CVPAT of the established model against the LM and IA benchmarks
const assess = assessCvpat(established, {
  testtype: "two.sided", nboot: 2000, technique: predictDA, seed: 123, noFolds: 10,
})!;
console.log(String(assess));

// Compare the predictive loss of two models
const compare = assessCvpatCompare(established, competing, {
  testtype: "two.sided", nboot: 2000, technique: predictDA, seed: 123, noFolds: 10,
})!;
console.log(String(compare));
```

## Example: CTA-PLS

CTA-PLS (Gudergan et al., 2008) tests whether each construct's indicators are
consistent with a reflective specification: under that null every model-implied
vanishing tetrad is zero. With `borrow: true` (the default), constructs with
only 2–3 indicators borrow indicators from structurally connected constructs to
form testable 4-tuples.

```ts
import { assessCta, plot } from "@seminr/extras";

// mobiModel: an estimatePls(...) result from @seminr/core
const cta = assessCta(mobiModel, { nboot: 5000, seed: 123 })!;
console.log(String(cta));      // print
console.log(cta.summarize());  // per-tetrad detail
const svg = plot(cta);         // adjusted-p-value plot as an SVG string

// Without borrowing, only constructs with >= 4 indicators are tested
const ctaNoBorrow = assessCta(mobiModel, { nboot: 5000, borrow: false })!;
```

## Example: Unobserved heterogeneity (FIMIX-PLS and PLS-POS)

FIMIX-PLS and PLS-POS are complementary approaches for detecting unobserved
heterogeneity. FIMIX-PLS uses probabilistic (EM-based) assignment and assumes
normally distributed residuals; PLS-POS uses deterministic (hill-climbing)
assignment with no distributional assumptions. Both partition the sample into K
segments with segment-specific path coefficients.

```ts
import {
  assessFimix, assessFimixCompare, assessPos, assessPosCompare, posSegments,
} from "@seminr/extras";

// FIMIX with K = 2, then compare across K = 2..4 via information criteria
const fimix = assessFimix(model, { K: 2, nstart: 10, seed: 123 })!;
console.log(String(fimix));
const fimixCompare = assessFimixCompare(model, { KRange: [2, 3, 4], nstart: 10, seed: 123 })!;

// PLS-POS with K = 2, then compare across K, then re-estimate per segment
const pos = assessPos(model, { K: 2, nstart: 10, seed: 123 })!;
const posCompare = assessPosCompare(model, { KRange: [2, 3, 4], nstart: 10, seed: 123 })!;
const segments = posSegments(pos);
```

## Example: Congruence testing

```ts
import { congruenceTest } from "@seminr/extras";

const cong = congruenceTest(model, { nboot: 2000, seed: 123 })!;
console.log(String(cong));
console.log(cong.summarize());
```

## Demos

The `demos/` directory contains runnable TypeScript equivalents of the R
package's demos, including the chapter examples from the *PLS-SEM Using R*
workbook (Hair et al., 2026):

```
pls-cvpat  pls-pcm  pls-cipma  pls-coa  pls-nca  pls-cta  pls-fimix  pls-pos
pls-congruence  help-debugging  primer-chap2 … primer-chap8
```

Run one against the built package, e.g.:

```sh
bun run build && bun run demos/pls-cta.ts
```

## Development

```sh
bun install
bun run build          # strict tsc -> dist/
bun run typecheck
bun run typecheck:demos
bun test               # parity vs R golden fixtures
```

## References

- Becker, J.-M., Rai, A., Ringle, C. M., & Voelckner, F. (2013). Discovering
  Unobserved Heterogeneity in Structural Equation Models to Avert Validity
  Threats. *MIS Quarterly*, 37(3), 665-694.
- Becker, J.-M., Richter, N. F., Ringle, C. M., & Sarstedt, M. (2026). Must-have,
  or maybe not? A sensitivity-based extension to necessary condition analysis.
  *Journal of Business Research*, 206, 115920.
- Danks, N. P. (2021). The piggy in the middle: The role of mediators in
  PLS-SEM-based prediction. *ACM SIGMIS Database*, 52(SI), 24-42.
- Dul, J. (2016). Necessary Condition Analysis (NCA): Logic and methodology of
  "necessary but not sufficient" causality. *Organizational Research Methods*,
  19(1), 10-52.
- Franke, G. R., Sarstedt, M., & Danks, N. P. (2021). An empirical comparison of
  factor score estimation methods. *Journal of Business Research*, 130, 318-334.
- Hahn, C., Johnson, M. D., Herrmann, A., & Huber, F. (2002). Capturing Customer
  Heterogeneity using a Finite Mixture PLS Approach. *Schmalenbach Business
  Review*, 54, 243-269.
- Gudergan, S. P., Ringle, C. M., Wende, S. & Will, A. (2008). Confirmatory
  Tetrad Analysis in PLS Path Modeling. *Journal of Business Research*, 61(12),
  1238-1249.
- Hair, J.F. (Jr), Hult, T.M., Ringle, C.M., Sarstedt, M., Danks, N.P., and
  Adler, S. (2026). *Partial Least Squares Structural Equation Modeling (PLS-SEM)
  Using R (Second Edition) - A Workbook*. Springer.
- Hauff, S., Richter, N. F., Sarstedt, M., & Ringle, C. M. (2024). Importance and
  Performance in PLS-SEM and NCA: Introducing the Combined Importance-Performance
  Map Analysis (cIPMA). *Journal of Retailing and Consumer Services*, 78, 103723.
- Liengaard, B. D., Sharma, P. N., Hult, G. T. M., Jensen, M. B., Sarstedt, M.,
  Hair, J. F., & Ringle, C. M. (2021). Prediction: coveted, yet forsaken?
  Introducing a cross-validated predictive ability test in partial least squares
  path modeling. *Decision Sciences*, 52(2), 362-392.
- Ray, S., Danks, N.P., Calero Valdez, A. (2026). SEMinR: Domain-specific
  language for building, estimating, and visualizing structural equation models
  in R.
- Richter, N. F., Schubring, S., Hauff, S., Ringle, C. M., & Sarstedt, M. (2020).
  When predictors of outcomes are necessary: guidelines for the combined use of
  PLS-SEM and NCA. *Industrial Management & Data Systems*, 120(12), 2243-2267.
- Ringle, C. M. & Sarstedt, M. (2016). Gain More Insight from Your PLS-SEM
  Results: The Importance-Performance Map Analysis. *Industrial Management & Data
  Systems*, 119(9), 1865-1886.
- Sarstedt, M., Becker, J.-M., Ringle, C. M. & Schwaiger, M. (2011). Uncovering
  and Treating Unobserved Heterogeneity with FIMIX-PLS. *Schmalenbach Business
  Review*, 63(1), 34-62.
- Sarstedt, M., Richter, N. F., Hauff, S. & Ringle, C. M. (2024). Combined
  Importance-Performance Map Analysis (cIPMA): A SmartPLS 4 Tutorial. *Journal of
  Marketing Analytics*, 12, 746-760.
- Sharma, P. N., Liengaard, B. D., Hair, J. F., Sarstedt, M., & Ringle, C. M.
  (2022). Predictive model assessment and selection in composite-based modeling
  using PLS-SEM: extensions and guidelines for using CVPAT. *European Journal of
  Marketing*, 57(6), 1662-1677.

## License

[GPL-3.0-only](./LICENSE)
