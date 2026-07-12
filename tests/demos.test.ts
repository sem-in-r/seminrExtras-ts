/**
 * Run every demo script end-to-end and assert it produces the expected output
 * (py port's test_demos.py). Each demo under `demos/` runs as a subprocess
 * (`bun demos/<name>.ts`) so the test exercises exactly what a user running
 * the demo would see: a clean exit, no error on stderr, and the marker strings
 * each demo is meant to print. Resampling is kept small in the demos so this
 * stays reasonably fast (the 1000-boot primer chapters dominate).
 *
 * The demos import the built package ("@seminr/extras" resolves to dist/, the
 * same self-reference convention as seminr-ts's demos), so a stale or missing
 * dist/ is rebuilt here first.
 */

import { beforeAll, describe, expect, test } from "bun:test";
import { readdirSync, statSync } from "node:fs";

const REPO_ROOT = new URL("..", import.meta.url).pathname;

/** (script, marker substrings that must appear in stdout) */
const DEMOS: [string, string[]][] = [
  [
    "pls-cvpat.ts",
    [
      "CVPAT model comparison (established vs. competing)",
      "CVPAT assessment of the established model",
      "Base Model Loss",
      "PLS Loss",
    ],
  ],
  [
    "pls-pcm.ts",
    [
      "PCM: single mediation path",
      "PCM: multiple mediation paths",
      "Image -> Satisfaction -> Loyalty",
      "Rendered PCM barplot",
    ],
  ],
  [
    "pls-cipma.ts",
    [
      "IPMA for target = Loyalty",
      "cIPMA for target = Loyalty",
      "Importance-Performance Map Analysis",
      "Rendered cIPMA map",
    ],
  ],
  [
    "pls-coa.ts",
    [
      "COA for focal construct = CUSL",
      "Composite Overfit Analysis (COA)",
      "Step 1: predictive deviance",
      "Competing splits at all group root nodes",
    ],
  ],
  [
    "pls-nca.ts",
    [
      "NCA for target = Satisfaction",
      "NCA-ESSE for target = Satisfaction",
      "Effect Sizes (d)",
      "Rendered NCA ceiling scatter plots",
    ],
  ],
  [
    "pls-cta.ts",
    [
      "CTA-PLS with borrowing (default)",
      "CTA-PLS without borrowing",
      "Borrowing:",
      "Rendered CTA adjusted-p-value plot",
    ],
  ],
  [
    "pls-fimix.ts",
    [
      "FIMIX-PLS for K = 2",
      "FIMIX-PLS comparison across K",
      "Segment Proportions",
      "Rendered FIMIX information-criteria comparison",
    ],
  ],
  [
    "pls-pos.ts",
    [
      "PLS-POS for K = 2",
      "Segment-specific models (pos_segments)",
      "PLS-POS comparison across K",
      "Rendered POS objective vs K",
    ],
  ],
  [
    "pls-congruence.ts",
    [
      "Congruence test (threshold = 1, alpha = 0.05)",
      "Congruence test summary",
    ],
  ],
  [
    "help-debugging.ts",
    [
      "Problem 1: construct misspelled in the measurement model",
      "Problem 3: moderator omitted from an interaction",
      "Corrected model estimated cleanly",
    ],
  ],
  [
    "primer-chap2.ts",
    ["Installing the packages", "Importing the packages", "Finding the demos"],
  ],
  [
    "primer-chap3.ts",
    [
      "First rows of the corporate reputation data",
      "Catching a mis-specified model",
      "Bootstrapping the model",
    ],
  ],
  [
    "primer-chap4.ts",
    [
      "Horn's parallel analysis",
      "Outer loadings",
      "HTMT discriminant validity",
      "Congruence test (alpha = 0.10)",
    ],
  ],
  [
    "primer-chap5.ts",
    [
      "Redundancy analysis (convergent validity of formative constructs)",
      "Indicator collinearity (VIF)",
      "Bootstrapped outer weights",
    ],
  ],
  [
    "primer-chap6.ts",
    [
      "Structural collinearity (VIF antecedents)",
      "BIC for CUSA across competing models",
      "BIC-based Akaike weights",
      "CVPAT model comparison (Model1 vs Model3, testtype = greater)",
    ],
  ],
  [
    "primer-chap7.ts",
    ["Bootstrapped structural paths", "Effect sizes (f-squared)", "Rendered slope analysis"],
  ],
  [
    "primer-chap8.ts",
    [
      "Total indirect effects",
      "Specific indirect effects through CUSA",
      "Index of moderated mediation (p1 * p5)",
    ],
  ],
];

function newestMtime(dir: string): number {
  let newest = 0;
  for (const entry of readdirSync(dir, { recursive: true }) as string[]) {
    const path = `${dir}/${entry}`;
    const st = statSync(path, { throwIfNoEntry: false });
    if (st?.isFile()) newest = Math.max(newest, st.mtimeMs);
  }
  return newest;
}

beforeAll(() => {
  // The demos run against dist/ — rebuild when it is missing or older than src.
  const distIndex = statSync(`${REPO_ROOT}dist/index.js`, { throwIfNoEntry: false });
  if (distIndex === undefined || distIndex.mtimeMs < newestMtime(`${REPO_ROOT}src`)) {
    const build = Bun.spawnSync(["bun", "run", "build"], { cwd: REPO_ROOT });
    if (build.exitCode !== 0) {
      throw new Error(`bun run build failed:\n${build.stderr.toString()}`);
    }
  }
});

describe("demos", () => {
  for (const [script, markers] of DEMOS) {
    test(
      script,
      () => {
        const result = Bun.spawnSync(["bun", `demos/${script}`], { cwd: REPO_ROOT });
        const stdout = result.stdout.toString();
        const stderr = result.stderr.toString();
        expect(
          result.exitCode,
          `${script} exited ${result.exitCode}\nstdout:\n${stdout}\nstderr:\n${stderr}`,
        ).toBe(0);
        // A demo must never surface an error; warnings are tolerated.
        expect(stderr, `${script} raised:\n${stderr}`).not.toContain("error:");
        for (const marker of markers) {
          expect(stdout, `${script} missing marker ${JSON.stringify(marker)}`).toContain(marker);
        }
      },
      { timeout: 240_000 },
    );
  }
});
