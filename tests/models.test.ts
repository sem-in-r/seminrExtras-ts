/**
 * Baseline model-registry parity test (plan 0.3b/0.4).
 *
 * For every model in the canonical registry, estimate it with `@seminr/core`
 * and compare `pathCoef`, `outerWeights`, and `rSquared` against the committed
 * R-golden `tests/fixtures/models/<ID>.json` at the plan's 1e-5 tolerance.
 *
 * The registry is kept in lock-step with the R fixture generator
 * (`../seminrExtras-py/scripts/generate_fixtures.R`) and the Python port
 * (`../seminrExtras-py/tests/helpers/models.py`); the same IDs (`M1`..`M17`,
 * `C1`..`C4`) name the same `(dataset, measurement model, structural model)`
 * triples on both sides.
 *
 * `M4r` (repeated-indicators HOC) is skip-marked: `@seminr/core`'s
 * `higherComposite` only supports `method: "two_stage"`, the same engine gap
 * the Python port recorded (decision D7).
 */

import { describe, expect, test } from "bun:test";
import { expectNamedClose, loadModelFixture, PLS_TOL, toMatrix } from "./helpers/fixtures.ts";
import { estimateRegistryModel, REGISTRY, REGISTRY_IDS } from "./helpers/models.ts";

describe("model registry parity vs R goldens", () => {
  for (const id of REGISTRY_IDS) {
    const meta = REGISTRY[id]!;
    const runner = meta.unsupported ? test.skip : test;

    runner(`${id}: ${meta.description}`, () => {
      const model = estimateRegistryModel(id);
      const fixture = loadModelFixture(id);

      expect(fixture.dataset).toBe(meta.dataset);

      expectNamedClose(model.pathCoef, toMatrix(fixture.pathCoef), PLS_TOL, `${id}.pathCoef`);
      expectNamedClose(
        model.outerWeights,
        toMatrix(fixture.outerWeights),
        PLS_TOL,
        `${id}.outerWeights`,
      );
      expectNamedClose(model.rSquared, toMatrix(fixture.rSquared), PLS_TOL, `${id}.rSquared`);
    });
  }
});
