# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

seminrExtras-ts (`@seminr/extras`) is a **from-scratch TypeScript port of the R package `seminrExtras`**,
located as a sibling repo at `../seminrExtras/`. Treat the R package as the spec: its `R/`, `tests/`,
`demo/`, and `README.md` are the source of truth for behavior to replicate, not a legacy codebase to
modify. The completed Python port (`../seminrExtras-py`) is the working reference where the hard porting
decisions were already made; the estimation engine is `../seminr-ts` (`@seminr/core`, a `file:` dependency).

`seminrExtras` is a supplementary package for **SEMinR**, a DSL for specifying and estimating PLS-SEM
(partial least squares structural equation models). seminrExtras does not estimate models itself — it
consumes an *already-estimated* SEMinR model object and runs further analysis/validation on it. Any
TypeScript port needs an equivalent "estimated model" data shape to operate on (i.e. a TS/JS SEMinR
equivalent, or a compatible interchange format), since none of these features make sense standalone.

### Features to port (see `../seminrExtras/README.md` for full descriptions and citations)

| Area | R entry points |
| --- | --- |
| CVPAT (Cross-Validated Predictive Ability Test) | `assess_cvpat()`, `assess_cvpat_compare()` |
| PCM (Predictive Contribution of the Mediator) | `assess_pcm()` |
| cIPMA / IPMA (Importance-Performance Map + NCA) | `assess_cipma()`, `assess_ipma()` |
| COA (Composite Overfit Analysis) | `assess_coa()`, `predictive_deviance()`, `deviance_tree()`, `unstable_params()`, `group_rules()`, `competes()` |
| NCA (Necessary Condition Analysis) | `assess_nca()`, `assess_nca_esse()` |
| CTA-PLS (Confirmatory Tetrad Analysis) | `assess_cta()` |
| FIMIX-PLS (latent class segmentation) | `assess_fimix()`, `assess_fimix_compare()` |
| PLS-POS (prediction-oriented segmentation) | `assess_pos()`, `assess_pos_compare()`, `pos_segments()` |
| Congruence testing | `congruence_test()` |

Source file sizes in `../seminrExtras/R/` (rough complexity signal): `feature_nca.R` (~37K), `feature_coa.R`
(~32K), `feature_cta.R` (~33K), `feature_fimix.R` (~31K), `feature_cipma.R` (~30K), `feature_pos.R` (~29K),
`feature_cvpat.R` (~22K), `feature_pcm.R` (~19K), `feature_congruence.R` (~11K), `helpers.R` (~8K).

## Design Philosophy to Carry Over from SEMinR/seminrExtras

### Three-stage pipeline: Specify → Estimate → Evaluate

Port targets are all **evaluation-stage** functions: they take an estimated model object and return an
analysis result. Do not fold model estimation into these functions.

### Result objects need a print/summary story

In R, results are S3-classed objects with `print()`/`summary()` methods (custom `"table_output"` class via
`comment()` for metadata). Design an equivalent TS convention early (e.g. a tagged result type with a
`.format()`/`.summary()` method or a dedicated formatter module) rather than ad hoc console logging per
feature — this pattern repeats across every `assess_*` function.

### Model type compatibility is a hard requirement, not an edge case

Every feature must be designed against all of these model shapes from day one — the R implementation treats
this as core, not optional:

- **Simple path models**
- **Mediation models** — indirect effects via `(I - B)^{-1} - I` on the path coefficient matrix
- **Moderation models (interaction terms)** — interaction construct names contain `*` (e.g.
  `"Image*Value"`); two-stage approach uses a single artificial indicator (`"Image*Value_intxn"`, weight
  1.0), product-indicator approach uses names like `"IMAG1*PERV1"`. Interaction constructs **are** present
  in construct scores / path coefficients. For IPMA/cIPMA, exclude interaction constructs (performance
  isn't meaningful on a 0–100 scale); for NCA/FIMIX/POS they're valid and should be included.
- **Higher-order constructs (HOC)** — when HOC is present, `items_of_construct("HOC", model)` returns LOC
  (lower-order construct) names, not raw indicators, for both two-stage and repeated-indicators approaches.
  LOC construct scores exist in the model data but are **not** on the original measurement scale — never
  treat them as raw indicators. LOC constructs typically do not appear in construct scores / path
  coefficients, only the HOC does. Indicator-level operations (e.g. IPMA performance) must recurse:
  HOC → LOC → actual indicators.

### Avoid reimplementing what the model estimator already provides

The R version deliberately avoids reaching into `seminr:::` internals — everything routes through
`seminr`'s exported API (e.g. `predict_pls()`), with any missing helper copied locally into `helpers.R`
rather than depending on unexported internals. If/when this port depends on a TS SEMinR equivalent, hold to
the same boundary: consume its public API only.

### Prefer implementing core algorithms internally over adding runtime dependencies

The R package implements CE-FDH/CR-FDH (NCA), EM-based segmentation (FIMIX), and decision trees (COA
deviance tree) itself rather than pulling in heavy dependencies, when the algorithm is well-defined enough
to control precisely. Apply the same bias in TypeScript: reach for a package only when an algorithm is
genuinely complex or out of scope, and prefer it as an optional/peer dependency over a hard one.

## Current State

**The port is complete.** All 20 R entry points are implemented and pinned to golden fixtures that are
**R's own output** — generated by `generate_fixtures.R`, which `devtools::load_all()`s the canonical
`../seminr` and `../seminrExtras`, runs them on `mobi`/`corp_rep_data`, and serialises at full double
precision. That script lives in the py port's repo and its outputs were copied into
`tests/fixtures/` from there, so the py port is the generator's *location* and never the parity
target: no fixture value in this package was produced by Python. One generator, one R session, one
set of goldens shared by both ports — which is why they are never regenerated here, plus an SVG plotting layer for
the 10 R `plot.*` methods and all 17 demos.

Two fixtures under `tests/fixtures/helpers/` are the exception to "never regenerated here", and are a
different kind of artefact: `arith.R` and `matstats.R` are generated in this repo by `Rscript` and pin
R's **arithmetic** — which of R's three means a routine follows, and the one-matrix `cor`/`cov` call
form — rather than seminrExtras' output. They assert exact doubles rather than a tolerance, which is
sound only because `capabilities("long.double")` is FALSE on arm64 macOS; each fixture records that
flag and its test asserts it first.

- **Toolchain** (mirrors seminr-ts): Bun test runner, strict `tsc` build to `dist/`, pure ESM,
  ES2022/NodeNext, explicit `.ts` import extensions, **no linter**. `bun test` / `bun run build` /
  `bun run typecheck` / `bun run typecheck:demos`.
- **Layout**: file-per-feature mirroring R (`src/featureCvpat.ts`, `src/featureCoa.ts`, ...);
  `src/cart.ts` (internal rpart-parity CART engine), `src/helpers.ts`, `src/records.ts` (all three
  deliberately NOT in the `src/index.ts` barrel — the R `:::` analog); `src/plotting/` (SVG emitters,
  exported via the barrel); `tests/helpers/{fixtures,models}.ts` (parity harness + 24-model registry);
  `demos/` (17 runnable scripts importing the built package).
- **Dependencies**: `@seminr/core` (the estimator, root barrel only) and `@compstats/core` (the
  low-level numerics, R-pinned). Scalars come from `@compstats/core/stats`, matrix work from
  `@compstats/core/linalg`; **never the root `@compstats/core` entry**, which reaches its canvas and
  interactive layers. `@seminr/core/math` is no longer imported anywhere — that facade existed for
  this package and is free for seminr-ts to retire. Cross the row-major/column-major boundary with
  `helpers.ts`'s `asMatrix`, never `fromRows`: on a bootstrap-loop shape `fromRows` costs three times
  the computation it feeds.
- **R has FOUR means, and which one a site needs is a parity decision.** Read the R line *and* the
  storage mode of its argument before porting a `mean()`:
  - `mean.default` on a **double** vector (`summary.c` `do_mean`, REALSXP branch) makes a correcting
    second pass → `@compstats/core`'s `mean`.
  - `mean.default` on an **integer or logical** vector takes the INTSXP branch — **one uncorrected
    pass**, i.e. exactly `sum/n`. Both bundled datasets are integer throughout
    (`sapply(mobi, storage.mode)`), and `mean(x >= 0)` over a logical is this branch too. On `mobi`
    the two branches differ on 8 of 24 columns.
  - `var`/`sd` (`cov.c`'s `MEAN` macro) is the same corrected body, and R coerces first, so integer
    and double agree → `@compstats/core`'s `sd`. **`sd` never needs the storage-mode question.**
  - `colMeans`/`rowMeans` (`array.c` `do_colsum`) is one uncorrected pass → the private helper in
    `featureCvpat.ts` and the plain `colSum / n` in `featureCipma.ts`; `helpers.ts`'s `seqSum` is
    what they and `cart.ts` are built on.

  Every site names the R function it follows in a comment. The tell is always the same: the branches
  differ by ulps, so no feature-level assertion can see it, and the delegation looks right because
  the *name* matches. Four instances found across three packages so far. Every call site names
  the R function it follows, and `tests/fixtures/helpers/arith.R` pins all three as exact doubles.
  Likewise `cor(x)`/`cov(x)`: R's one-matrix form is not its two-argument form, and this package
  needs the one-matrix one (`matstats.R`).
- **Conventions**: frozen result records with a `kind` discriminant + `toString()`/`summarize()`;
  dual positional/named call styles on every entry point; validation via `console.warn` + null return;
  exact R parity only via injected RNG streams (`draws`/`ordering(s)`/`perms`/`inits`/`partitions`) —
  default paths seed `mulberry32` and are documented as non-bit-identical to R.
- **Known limitation**: repeated-indicators HOC models are unsupported by `@seminr/core`
  (`higherComposite` is two-stage only) — the registry marks M4r unsupported and one test is skipped.

npm publication is user-owned and has NOT happened.

## Plans

Branch plans and their working docs live in `.claude/plans/` (gitignored, synced across machines by
Sideways). `CLAUDE.local.md` `@`-includes the active one.

**One folder per work stream**, named `NNN-PURPOSE-slug`:

- `NNN` — zero-padded sequence starting at `001`. It strictly increments and is **never reused**;
  list the directory and take the next unused number.
- `PURPOSE` — uppercase tag for the kind of document that started the stream (`PLAN`, `BUGFIX`,
  `REFACTOR`, `HOTFIX`). It does not change when a second kind of document joins the folder.
- `slug` — short kebab-case name, normally the branch name with `/` replaced by `-`.

**Inside the folder**: the main document takes the name of its kind (`PLAN.md`, `BUGFIX.md`). When a
folder holds two or more, prefix each with a letter giving the reading order (`a-PLAN.md`,
`b-BUGFIX.md`) — a folder gains letters the moment a second document arrives. Supporting files keep a
kind tag and take no letter (`SKETCHES.html`, `ASSET-og-card.html`). A reference within a folder uses
the bare filename; a reference across folders uses the full path from the repository root.

When a branch merges, mark its plan closed rather than deleting it: a
`> **CLOSED** (date): merged to <branch> as <sha>` note under the title, plus a final Current State
entry. Closed plans keep their number.
