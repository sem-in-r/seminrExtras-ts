/**
 * Canonical model registry for seminrExtras parity fixtures.
 *
 * Ported faithfully from `../seminrExtras-py/tests/helpers/models.py`. Every
 * distinct `(dataset, measurement model, structural model)` triple the R
 * `seminrExtras` test suite exercises, deduplicated into a single registry
 * keyed by short IDs (`M1`..`M17`, `C1`..`C4`). The same IDs are defined R-side
 * in `../seminrExtras-py/scripts/generate_fixtures.R`; the registries must stay
 * in lock-step so a golden JSON matches its model here.
 *
 * Each builder returns `{ measurementModel, structuralModel }` built with the
 * `@seminr/core` DSL. {@link estimateRegistryModel} loads the right dataset and
 * estimates with the correct missing-data handling (`corp_rep_data` is always
 * mean-replaced with a `-99` missing marker, mirroring the R tests).
 *
 * Datasets are loaded from committed CSVs in this repo (`@seminr/core` bundles
 * none): `tests/fixtures/data/mobi.csv` and `tests/fixtures/data/corp_rep_data.csv`.
 *
 * Known engine gap (decision D7): `higherComposite` in `@seminr/core` only
 * supports `method: "two_stage"`; R's `repeated_indicators` HOC method has no
 * port yet, so `M4r` is registered but flagged `unsupported`.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  composite,
  constructs,
  estimatePls,
  higherComposite,
  interactionTerm,
  meanReplacement,
  modeB,
  multiItems,
  parseCsv,
  paths,
  productIndicator,
  reflective,
  relationships,
  singleItem,
  twoStage,
  type Dataset,
  type MeasurementModel,
  type PlsModel,
  type SMMatrix,
} from "@seminr/core";
import { FIXTURES_DIR } from "./fixtures.ts";

export interface Model {
  measurementModel: MeasurementModel;
  structuralModel: SMMatrix;
}

export type ModelBuilder = () => Model;

/** The R missing marker used for `corp_rep_data` throughout the R tests. */
export const CORP_REP_MISSING_VALUE = -99;

// ---------------------------------------------------------------------------
// mobi measurement-model fragments (shared item blocks)
// ---------------------------------------------------------------------------

const image = () => composite("Image", multiItems("IMAG", [1, 2, 3, 4, 5]));
const expectation = () => composite("Expectation", multiItems("CUEX", [1, 2, 3]));
const value = () => composite("Value", multiItems("PERV", [1, 2]));
const satisfaction = () => composite("Satisfaction", multiItems("CUSA", [1, 2, 3]));
const loyalty = () => composite("Loyalty", multiItems("CUSL", [1, 2, 3]));

// ---------------------------------------------------------------------------
// Core mobi models
// ---------------------------------------------------------------------------

/** ECSI full mobi model (Image/Expectation/Value/Satisfaction/Loyalty). */
const m1: ModelBuilder = () => ({
  measurementModel: constructs(image(), expectation(), value(), satisfaction(), loyalty()),
  structuralModel: relationships(
    paths("Image", ["Expectation", "Satisfaction", "Loyalty"]),
    paths("Expectation", ["Value", "Satisfaction"]),
    paths("Value", "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
  ),
});

/** Simple mediation chain Image -> Expectation -> Satisfaction -> Loyalty. */
const m2: ModelBuilder = () => ({
  measurementModel: constructs(image(), expectation(), satisfaction(), loyalty()),
  structuralModel: relationships(
    paths("Image", ["Expectation", "Loyalty"]),
    paths("Expectation", "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
  ),
});

/** Two-stage moderation (Image x Value) on Satisfaction, plus Loyalty. */
const m3: ModelBuilder = () => ({
  measurementModel: constructs(
    image(),
    value(),
    satisfaction(),
    loyalty(),
    interactionTerm({ iv: "Image", moderator: "Value", method: twoStage }),
  ),
  structuralModel: relationships(
    paths(["Image", "Value", "Image*Value"], "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
  ),
});

/** Two-stage moderation, 3-construct (no Loyalty). */
const m3b: ModelBuilder = () => ({
  measurementModel: constructs(
    image(),
    value(),
    satisfaction(),
    interactionTerm({ iv: "Image", moderator: "Value", method: twoStage }),
  ),
  structuralModel: relationships(paths(["Image", "Value", "Image*Value"], "Satisfaction")),
});

/** Higher-order composite Quality = {Image, Expectation}, two-stage. */
const m4: ModelBuilder = () => ({
  measurementModel: constructs(
    image(),
    expectation(),
    higherComposite("Quality", ["Image", "Expectation"], "two_stage"),
    satisfaction(),
    loyalty(),
  ),
  structuralModel: relationships(
    paths("Quality", ["Satisfaction", "Loyalty"]),
    paths("Satisfaction", "Loyalty"),
  ),
});

/** M4 with repeated_indicators HOC method -- unsupported by @seminr/core. */
const m4r: ModelBuilder = () => {
  throw new Error(
    "higherComposite({ method: 'repeated_indicators' }) is not implemented in " +
      "@seminr/core (only method: 'two_stage'); M4r has no TS model yet.",
  );
};

/** NCA 4-construct: {Image, Value} -> Satisfaction -> Loyalty. */
const m5: ModelBuilder = () => ({
  measurementModel: constructs(image(), value(), satisfaction(), loyalty()),
  structuralModel: relationships(
    paths(["Image", "Value"], "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
  ),
});

/** NCA 3-construct: {Image, Value} -> Satisfaction. */
const m5b: ModelBuilder = () => ({
  measurementModel: constructs(image(), value(), satisfaction()),
  structuralModel: relationships(paths(["Image", "Value"], "Satisfaction")),
});

/** Higher-order composite Rep = {Image, Value}, two-stage, + Loyalty. */
const m6: ModelBuilder = () => ({
  measurementModel: constructs(
    image(),
    value(),
    higherComposite("Rep", ["Image", "Value"], "two_stage"),
    satisfaction(),
    loyalty(),
  ),
  structuralModel: relationships(paths("Rep", "Satisfaction"), paths("Satisfaction", "Loyalty")),
});

/** PCM mediation triangle: Image -> Satisfaction -> Loyalty, Image -> Loyalty. */
const m7: ModelBuilder = () => ({
  measurementModel: constructs(image(), satisfaction(), loyalty()),
  structuralModel: relationships(
    paths("Image", "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
    paths("Image", "Loyalty"),
  ),
});

/** M7 with Image measured mode_B. */
const m7b: ModelBuilder = () => ({
  measurementModel: constructs(
    composite("Image", multiItems("IMAG", [1, 2, 3, 4, 5]), modeB),
    satisfaction(),
    loyalty(),
  ),
  structuralModel: relationships(
    paths("Image", "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
    paths("Image", "Loyalty"),
  ),
});

/** PCM moderation with the DEFAULT interaction method (product_indicator). */
const m8: ModelBuilder = () => ({
  measurementModel: constructs(
    image(),
    value(),
    satisfaction(),
    loyalty(),
    interactionTerm({ iv: "Image", moderator: "Value", method: productIndicator }),
  ),
  structuralModel: relationships(
    paths("Image", "Satisfaction"),
    paths("Value", "Satisfaction"),
    paths("Image*Value", "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
    paths("Image", "Loyalty"),
  ),
});

/** PCM HOC with the default higher_composite method (two_stage). */
const m9: ModelBuilder = () => ({
  measurementModel: constructs(
    composite("Tangibles", multiItems("IMAG", [1, 2, 3])),
    composite("Intangibles", multiItems("IMAG", [4, 5])),
    higherComposite("Image", ["Tangibles", "Intangibles"]),
    satisfaction(),
    loyalty(),
  ),
  structuralModel: relationships(
    paths("Image", "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
    paths("Image", "Loyalty"),
  ),
});

/** CTA 3-construct mediation: Image -> {Satisfaction, Loyalty}; Satisfaction -> Loyalty. */
const m10: ModelBuilder = () => ({
  measurementModel: constructs(image(), satisfaction(), loyalty()),
  structuralModel: relationships(
    paths("Image", ["Satisfaction", "Loyalty"]),
    paths("Satisfaction", "Loyalty"),
  ),
});

/** CTA HOC over four LOCs (Super), two-stage; Super -> Loyalty. */
const m11: ModelBuilder = () => ({
  measurementModel: constructs(
    composite("IMAG1_2", multiItems("IMAG", [1, 2])),
    composite("IMAG3_4", multiItems("IMAG", [3, 4])),
    composite("CUEX_all", multiItems("CUEX", [1, 2, 3])),
    composite("CUSA_all", multiItems("CUSA", [1, 2, 3])),
    higherComposite("Super", ["IMAG1_2", "IMAG3_4", "CUEX_all", "CUSA_all"], "two_stage"),
    loyalty(),
  ),
  structuralModel: relationships(paths("Super", "Loyalty")),
});

/** CTA formative focal: Value is mode_B; {Image, Value} -> Satisfaction. */
const m12: ModelBuilder = () => ({
  measurementModel: constructs(
    image(),
    composite("Value", multiItems("PERV", [1, 2]), modeB),
    satisfaction(),
  ),
  structuralModel: relationships(paths(["Image", "Value"], "Satisfaction")),
});

/** CTA mixed formative: Image is mode_B; Image -> {Expectation, Satisfaction}. */
const m13: ModelBuilder = () => ({
  measurementModel: constructs(
    composite("Image", multiItems("IMAG", [1, 2, 3, 4, 5]), modeB),
    expectation(),
    satisfaction(),
  ),
  structuralModel: relationships(paths("Image", ["Expectation", "Satisfaction"])),
});

/** Serial mediation: Image -> Quality; {Image, Quality} -> Satisfaction -> Loyalty. */
const m14: ModelBuilder = () => ({
  measurementModel: constructs(
    image(),
    composite("Quality", multiItems("PERQ", [1, 2, 3, 4, 5, 6, 7])),
    satisfaction(),
    loyalty(),
  ),
  structuralModel: relationships(
    paths("Image", "Quality"),
    paths(["Image", "Quality"], "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
  ),
});

/** Both exogenous composites mode_B: {Image, Value} -> Satisfaction -> Loyalty. */
const m16: ModelBuilder = () => ({
  measurementModel: constructs(
    composite("Image", multiItems("IMAG", [1, 2, 3, 4, 5]), modeB),
    composite("Value", multiItems("PERV", [1, 2]), modeB),
    satisfaction(),
    loyalty(),
  ),
  structuralModel: relationships(
    paths(["Image", "Value"], "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
  ),
});

/** Reflective (PLSc) constructs: {Image, Value} -> Satisfaction -> Loyalty. */
const m17: ModelBuilder = () => ({
  measurementModel: constructs(
    reflective("Image", multiItems("IMAG", [1, 2, 3, 4, 5])),
    reflective("Value", multiItems("PERV", [1, 2])),
    reflective("Satisfaction", multiItems("CUSA", [1, 2, 3])),
    reflective("Loyalty", multiItems("CUSL", [1, 2, 3])),
  ),
  structuralModel: relationships(
    paths(["Image", "Value"], "Satisfaction"),
    paths("Satisfaction", "Loyalty"),
  ),
});

// ---------------------------------------------------------------------------
// corp_rep measurement-model fragments
// ---------------------------------------------------------------------------

const comp = () => composite("COMP", multiItems("comp_", [1, 2, 3]));
const like = () => composite("LIKE", multiItems("like_", [1, 2, 3]));
const cusaSingle = () => composite("CUSA", singleItem("cusa"));
const cusl = () => composite("CUSL", multiItems("cusl_", [1, 2, 3]));

/** corp_rep mediation: {COMP, LIKE} -> CUSA -> CUSL. */
const c1: ModelBuilder = () => ({
  measurementModel: constructs(comp(), like(), cusaSingle(), cusl()),
  structuralModel: relationships(paths(["COMP", "LIKE"], "CUSA"), paths("CUSA", "CUSL")),
});

/** corp_rep alternative: {COMP, LIKE} -> {CUSA, CUSL}; CUSA -> CUSL. */
const c2: ModelBuilder = () => ({
  measurementModel: constructs(comp(), like(), cusaSingle(), cusl()),
  structuralModel: relationships(paths(["COMP", "LIKE"], ["CUSA", "CUSL"]), paths("CUSA", "CUSL")),
});

/** corp_rep HOC: COKE = {COMP, LIKE} two-stage; COKE -> CUSA -> CUSL. */
const c3: ModelBuilder = () => ({
  measurementModel: constructs(
    comp(),
    like(),
    higherComposite("COKE", ["COMP", "LIKE"], "two_stage"),
    cusaSingle(),
    cusl(),
  ),
  structuralModel: relationships(paths("COKE", "CUSA"), paths("CUSA", "CUSL")),
});

/** corp_rep 3-construct: {COMP, LIKE} -> CUSA. */
const c4: ModelBuilder = () => ({
  measurementModel: constructs(comp(), like(), cusaSingle()),
  structuralModel: relationships(paths(["COMP", "LIKE"], "CUSA")),
});

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export type DatasetName = "mobi" | "corp_rep";

/** Metadata + builder for one registry model. */
export interface ModelMeta {
  readonly id: string;
  readonly dataset: DatasetName;
  readonly description: string;
  readonly build: ModelBuilder;
  readonly rTests: readonly string[];
  /** Non-null => engine gap; test is skipped / estimate throws with this reason. */
  readonly unsupported: string | null;
}

function meta(
  id: string,
  dataset: DatasetName,
  description: string,
  build: ModelBuilder,
  rTests: readonly string[],
  unsupported: string | null = null,
): ModelMeta {
  return Object.freeze({ id, dataset, description, build, rTests, unsupported });
}

export const REGISTRY: Readonly<Record<string, ModelMeta>> = Object.freeze({
  M1: meta("M1", "mobi", "ECSI full: Image/Expectation/Value/Satisfaction/Loyalty", m1, [
    "test-cvpat.R",
    "test-coa.R",
    "test-fimix.R",
    "test-pos.R",
  ]),
  M2: meta("M2", "mobi", "Simple mediation Image->Expectation->Satisfaction->Loyalty", m2, [
    "test-cvpat.R",
    "test-pcm.R",
  ]),
  M3: meta(
    "M3",
    "mobi",
    "Two-stage moderation (Image x Value) on Satisfaction + Loyalty",
    m3,
    ["test-cipma-comprehensive.R", "test-nca.R", "test-fimix.R"],
  ),
  M3b: meta("M3b", "mobi", "Two-stage moderation, 3-construct (no Loyalty)", m3b, [
    "test-cipma-comprehensive.R",
  ]),
  M4: meta("M4", "mobi", "HOC Quality={Image,Expectation}, two-stage", m4, [
    "test-cipma-comprehensive.R",
    "test-pcm.R",
  ]),
  M4r: meta(
    "M4r",
    "mobi",
    "HOC Quality={Image,Expectation}, repeated_indicators",
    m4r,
    ["test-cipma-comprehensive.R"],
    "repeated_indicators HOC method not implemented in @seminr/core",
  ),
  M5: meta("M5", "mobi", "NCA 4-construct: {Image,Value}->Satisfaction->Loyalty", m5, [
    "test-nca.R",
  ]),
  M5b: meta("M5b", "mobi", "NCA 3-construct: {Image,Value}->Satisfaction", m5b, ["test-nca.R"]),
  M6: meta("M6", "mobi", "HOC Rep={Image,Value}, two-stage + Loyalty", m6, [
    "test-cipma-comprehensive.R",
    "test-nca.R",
  ]),
  M7: meta("M7", "mobi", "PCM mediation triangle Image/Satisfaction/Loyalty", m7, ["test-pcm.R"]),
  M7b: meta("M7b", "mobi", "PCM mediation triangle, Image mode_B", m7b, ["test-pcm.R"]),
  M8: meta("M8", "mobi", "PCM moderation, default interaction method (product_indicator)", m8, [
    "test-pcm.R",
  ]),
  M9: meta("M9", "mobi", "PCM HOC Image={Tangibles,Intangibles}, default (two_stage)", m9, [
    "test-pcm.R",
  ]),
  M10: meta("M10", "mobi", "CTA 3-construct mediation Image/Satisfaction/Loyalty", m10, [
    "test-cta.R",
  ]),
  M11: meta("M11", "mobi", "CTA HOC Super over four LOCs, two-stage", m11, ["test-cta.R"]),
  M12: meta(
    "M12",
    "mobi",
    "CTA formative focal: Value mode_B; {Image,Value}->Satisfaction",
    m12,
    ["test-cta.R"],
  ),
  M13: meta(
    "M13",
    "mobi",
    "CTA mixed formative: Image mode_B; Image->{Expectation,Satisfaction}",
    m13,
    ["test-cta.R"],
  ),
  M14: meta("M14", "mobi", "Serial mediation Image->Quality->Satisfaction->Loyalty", m14, [
    "test-pcm.R",
    "test-coa.R",
  ]),
  // M15 is identical to M3 (mediated moderation); use M3, do not duplicate.
  M16: meta("M16", "mobi", "Both exogenous composites mode_B", m16, [
    "test-cipma-comprehensive.R",
  ]),
  M17: meta("M17", "mobi", "Reflective PLSc constructs", m17, ["test-cta.R", "test-nca.R"]),
  C1: meta("C1", "corp_rep", "corp_rep mediation {COMP,LIKE}->CUSA->CUSL", c1, [
    "test-cta.R",
    "test-cipma-comprehensive.R",
    "test-pcm.R",
  ]),
  C2: meta("C2", "corp_rep", "corp_rep alternative {COMP,LIKE}->{CUSA,CUSL}; CUSA->CUSL", c2, [
    "test-cta.R",
  ]),
  C3: meta("C3", "corp_rep", "corp_rep HOC COKE={COMP,LIKE} two-stage", c3, [
    "test-cipma-comprehensive.R",
  ]),
  C4: meta("C4", "corp_rep", "corp_rep 3-construct {COMP,LIKE}->CUSA", c4, ["test-cta.R"]),
});

/** All registry IDs in definition order. */
export const REGISTRY_IDS: readonly string[] = Object.freeze(Object.keys(REGISTRY));

/** IDs that estimate a mobi model. */
export const MOBI_IDS: readonly string[] = Object.freeze(
  REGISTRY_IDS.filter((id) => REGISTRY[id]!.dataset === "mobi"),
);

/** IDs that estimate a corp_rep model. */
export const CORP_REP_IDS: readonly string[] = Object.freeze(
  REGISTRY_IDS.filter((id) => REGISTRY[id]!.dataset === "corp_rep"),
);

// ---------------------------------------------------------------------------
// Dataset loading + estimation
// ---------------------------------------------------------------------------

const DATA_DIR = join(FIXTURES_DIR, "data");
const DATASET_FILES: Record<DatasetName, string> = {
  mobi: "mobi.csv",
  corp_rep: "corp_rep_data.csv",
};

const datasetCache = new Map<DatasetName, Dataset>();

function loadDataset(name: DatasetName): Dataset {
  const cached = datasetCache.get(name);
  if (cached) return cached;
  const text = readFileSync(join(DATA_DIR, DATASET_FILES[name]), "utf8");
  const data = parseCsv(text);
  datasetCache.set(name, data);
  return data;
}

/**
 * Estimate the registry model `modelId` with the correct dataset and
 * missing-data handling.
 *
 * `corp_rep` models use mean replacement with a `-99` missing marker; mobi
 * models estimate with defaults. Throws for `unsupported` models (engine gaps).
 */
export function estimateRegistryModel(modelId: string): PlsModel {
  const m = REGISTRY[modelId];
  if (!m) throw new Error(`unknown model id ${modelId}`);
  if (m.unsupported !== null) throw new Error(`${modelId}: ${m.unsupported}`);
  const { measurementModel, structuralModel } = m.build();
  const data = loadDataset(m.dataset);
  if (m.dataset === "corp_rep") {
    return estimatePls(data, measurementModel, structuralModel, {
      missing: meanReplacement,
      missingValue: CORP_REP_MISSING_VALUE,
    });
  }
  return estimatePls(data, measurementModel, structuralModel);
}
