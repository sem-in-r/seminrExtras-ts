/**
 * Structural tests for the SVG plotting layer (Slice 12), mirroring the py
 * port's test_plotting.py. Assertions are SVG-structure level (element
 * classes/counts, `data-value` payloads, titles/labels, null returns on empty
 * paths), never pixel parity. Feature records come from cheap default-path
 * runs (no parity fixtures needed).
 */

import { describe, expect, test } from "bun:test";
import { SvgPlot } from "@seminr/core";
import { rpartAnova } from "../src/cart.ts";
import { assessCipma, assessIpma, type CipmaAnalysis } from "../src/featureCipma.ts";
import { assessCoa, groupScoreMeans, type CoaAnalysis } from "../src/featureCoa.ts";
import type { CtaAnalysis } from "../src/featureCta.ts";
import { assessCta } from "../src/featureCta.ts";
import {
  assessFimix,
  assessFimixCompare,
  type FimixAnalysis,
  type FimixComparison,
} from "../src/featureFimix.ts";
import { assessNca, assessNcaEsse, crFdhLine, type NcaAnalysis, type NcaEsse } from "../src/featureNca.ts";
import { assessPcm, type PcmAnalysis } from "../src/featurePcm.ts";
import {
  assessPos,
  assessPosCompare,
  type PosAnalysis,
  type PosComparison,
} from "../src/featurePos.ts";
import {
  plot,
  plotCipma,
  plotCoa,
  plotCta,
  plotFimix,
  plotFimixCompare,
  plotNca,
  plotNcaEsse,
  plotPcm,
  plotPos,
  plotPosCompare,
} from "../src/plotting/results.ts";
import { estimateRegistryModel } from "./helpers/models.ts";

// ---------------------------------------------------------------------------
// Module-scoped feature records (cheap default-path runs)
// ---------------------------------------------------------------------------

const ctaRecord = assessCta(estimateRegistryModel("M1"), { nboot: 20, seed: 123 }) as CtaAnalysis;
const cipmaRecord = assessCipma(estimateRegistryModel("M1"), {
  target: "Loyalty",
  scaleMin: 1,
  scaleMax: 10,
  ncaTestRep: 0,
}) as CipmaAnalysis;
const ipmaRecord = assessIpma(estimateRegistryModel("M1"), {
  target: "Loyalty",
  scaleMin: 1,
  scaleMax: 10,
}) as CipmaAnalysis;
// seed 1 (not the parity default 123): this default-path run must actually
// produce deviant groups for the groups/means tests to bite
const coaRecord = assessCoa(estimateRegistryModel("C1"), "CUSL", { seed: 1 }) as CoaAnalysis;
const ncaRecord = assessNca(estimateRegistryModel("M5"), {
  target: "Satisfaction",
  testRep: 0,
}) as NcaAnalysis;
const esseRecord = assessNcaEsse(estimateRegistryModel("M5"), {
  target: "Satisfaction",
  testRep: 0,
}) as NcaEsse;
const pcmRecord = assessPcm(estimateRegistryModel("M7"), { seed: 7 }) as PcmAnalysis;
const fimixRecord = assessFimix(estimateRegistryModel("M1"), {
  K: 2,
  nstart: 3,
  maxIter: 2000,
  seed: 123,
}) as FimixAnalysis;
const fimixCompareRecord = assessFimixCompare(estimateRegistryModel("M2"), {
  KRange: [2, 3],
  nstart: 3,
  maxIter: 2000,
  seed: 123,
}) as FimixComparison;
const posRecord = assessPos(estimateRegistryModel("M1"), {
  K: 2,
  nstart: 2,
  maxIter: 10,
  seed: 123,
}) as PosAnalysis;
const posCompareRecord = assessPosCompare(estimateRegistryModel("M1"), {
  KRange: [2, 3],
  nstart: 2,
  maxIter: 10,
  seed: 123,
}) as PosComparison;

// ---------------------------------------------------------------------------
// SVG-source probes
// ---------------------------------------------------------------------------

function unescapeXml(text: string): string {
  return text
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&");
}

function svgOf(p: SvgPlot | null): string {
  expect(p).not.toBeNull();
  return (p as SvgPlot).svg;
}

/** All panel titles in the SVG (unescaped). */
function titles(svg: string): string[] {
  return [...svg.matchAll(/<text [^>]*class="title"[^>]*>(.*?)<\/text>/g)].map((m) =>
    unescapeXml(m[1]!),
  );
}

/** All legend labels in the SVG (unescaped). */
function legendLabels(svg: string): string[] {
  return [...svg.matchAll(/<text [^>]*class="legend"[^>]*>(.*?)<\/text>/g)].map((m) =>
    unescapeXml(m[1]!),
  );
}

/** `data-value` payloads of all bars, in emission order. */
function barValues(svg: string): number[] {
  return [...svg.matchAll(/class="bar" data-value="([^"]+)"/g)].map((m) => Number(m[1]!));
}

function countClass(svg: string, cls: string): number {
  return (svg.match(new RegExp(`class="${cls}"`, "g")) ?? []).length;
}

// ---------------------------------------------------------------------------
// CTA
// ---------------------------------------------------------------------------

describe("plotCta", () => {
  const svg = svgOf(plotCta(ctaRecord));

  test("one titled panel per tested construct", () => {
    const constructs = [...ctaRecord.tetradDetails.keys()];
    const got = titles(svg);
    for (const construct of constructs) expect(got).toContain(construct);
  });

  test("scatter points match non-NaN tetrad count and carry adjusted p", () => {
    let expected = 0;
    const expectedVals: number[] = [];
    for (const detail of ctaRecord.tetradDetails.values()) {
      const j = detail.table.cols.indexOf("Adj_P");
      for (const row of detail.table.values) {
        if (!Number.isNaN(row[j]!)) {
          expected += 1;
          expectedVals.push(row[j]!);
        }
      }
    }
    const got = [...svg.matchAll(/class="pt" data-value="([^"]+)"/g)].map((m) => Number(m[1]!));
    expect(got.length).toBe(expected);
    expect([...got].sort()).toEqual([...expectedVals].sort());
  });

  test("draws the alpha threshold line in every panel", () => {
    expect(countClass(svg, "refline")).toBe(ctaRecord.tetradDetails.size);
  });

  test("returns null when no constructs were tested", () => {
    const empty = {
      kind: "cta_analysis",
      constructResults: [],
      tetradDetails: new Map(),
      nboot: 0,
      alpha: 0.05,
      correction: "BH",
      skipped: [],
      borrowing: new Map(),
    } as unknown as CtaAnalysis;
    expect(plotCta(empty)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// cIPMA / IPMA
// ---------------------------------------------------------------------------

describe("plotCipma", () => {
  test("points carry each construct's importance/performance", () => {
    const svg = svgOf(plotCipma(cipmaRecord));
    const pts = [...svg.matchAll(/class="pt" data-x="([^"]+)" data-y="([^"]+)"/g)].map((m) => [
      Number(m[1]!),
      Number(m[2]!),
    ]);
    for (const c of cipmaRecord.constructs) {
      const x = cipmaRecord.importanceUnstd[c]!;
      const y = cipmaRecord.performance[c]!;
      expect(
        pts.some(([px, py]) => Math.abs(px! - x) < 1e-12 && Math.abs(py! - y) < 1e-12),
      ).toBeTrue();
    }
    expect(titles(svg)[0]).toContain(cipmaRecord.target);
  });

  test("falls back to IPMA without an NCA overlay (no legend)", () => {
    const svg = svgOf(plotCipma(ipmaRecord));
    expect(titles(svg)[0]!.startsWith("IPMA:")).toBeTrue();
    expect(legendLabels(svg)).toEqual([]);
  });

  test("cIPMA has the necessity legend", () => {
    const svg = svgOf(plotCipma(cipmaRecord));
    expect(titles(svg)[0]!.startsWith("cIPMA:")).toBeTrue();
    expect(legendLabels(svg)).toContain("Necessary + sufficient");
  });
});

// ---------------------------------------------------------------------------
// COA
// ---------------------------------------------------------------------------

describe("plotCoa", () => {
  test("pd scatter has the focal title and both quantile lines", () => {
    const svg = svgOf(plotCoa(coaRecord, { type: "pd" }));
    expect(titles(svg)[0]).toContain(coaRecord.focalConstruct);
    expect(countClass(svg, "refline")).toBeGreaterThanOrEqual(2);
    const n = coaRecord.predictiveDeviance.pd.length;
    const drawn =
      countClass(svg, "pt") + countClass(svg, "pt-unique") + countClass(svg, "pt-group");
    expect(drawn).toBe(n);
  });

  test("groups plot draws one letter-marker series per group", () => {
    const groups = Object.keys(coaRecord.devianceTree.deviantGroups);
    expect(groups.length).toBeGreaterThan(0); // this run has deviant groups
    const svg = svgOf(plotCoa(coaRecord, { type: "groups" }));
    const means = groupScoreMeans(coaRecord);
    for (const group of groups) {
      expect(svg).toContain(`data-group="${group}"`);
      const letters = [
        ...svg.matchAll(new RegExp(`class="marker"[^>]*>${group}</text>`, "g")),
      ];
      expect(letters.length).toBe(means.rows.length);
    }
    expect(legendLabels(svg)).toEqual(groups);
  });

  test("groups plot honors remove", () => {
    const means = groupScoreMeans(coaRecord);
    const dropped = means.rows.find((r) => r !== coaRecord.focalConstruct)!;
    const svg = svgOf(plotCoa(coaRecord, { type: "groups", remove: [dropped] }));
    const group = Object.keys(coaRecord.devianceTree.deviantGroups)[0]!;
    const letters = [...svg.matchAll(new RegExp(`class="marker"[^>]*>${group}</text>`, "g"))];
    expect(letters.length).toBe(means.rows.length - 1);
  });

  test("groups plot returns null without deviant groups", () => {
    const dt = { ...coaRecord.devianceTree, deviantGroups: {}, groupRoots: {} };
    const coa = { ...coaRecord, devianceTree: dt } as CoaAnalysis;
    expect(plotCoa(coa, { type: "groups" })).toBeNull();
  });

  test("tree plot labels every node", () => {
    const svg = svgOf(plotCoa(coaRecord, { type: "tree" }));
    expect(titles(svg)[0]).toContain("Deviance Tree");
    expect(countClass(svg, "node-label")).toBe(coaRecord.devianceTree.tree.nodeIds.length);
  });

  test("bad type throws", () => {
    expect(() => plotCoa(coaRecord, { type: "BOGUS" })).toThrow(/pd/);
  });
});

// ---------------------------------------------------------------------------
// NCA + ESSE
// ---------------------------------------------------------------------------

describe("plotNca", () => {
  test("scatter has one panel per predictor with ceiling lines", () => {
    const svg = svgOf(plotNca(ncaRecord, { type: "scatter" }));
    const got = titles(svg);
    for (const pred of ncaRecord.predictors) {
      expect(got.some((t) => t.includes(`NCA: ${pred} ->`))).toBeTrue();
    }
    expect(countClass(svg, "ceiling-ce")).toBe(ncaRecord.predictors.length);
  });

  test("effects bar values equal the effect-size table", () => {
    const svg = svgOf(plotNca(ncaRecord, { type: "effects" }));
    const got = barValues(svg)
      .map((v) => Math.round(v * 1e6) / 1e6)
      .sort((a, b) => a - b);
    const expected = ncaRecord.effectSizes.values
      .flat()
      .map((v) => Math.round((Number.isNaN(v) ? 0 : v) * 1e6) / 1e6)
      .sort((a, b) => a - b);
    expect(got).toEqual(expected);
  });

  test("effects returns null without predictors", () => {
    const empty = {
      ...ncaRecord,
      predictors: [],
      effectSizes: { rows: [], cols: [...ncaRecord.effectSizes.cols], values: [] },
    } as unknown as NcaAnalysis;
    expect(plotNca(empty, { type: "effects" })).toBeNull();
  });

  test("esse sensitivity draws empirical + benchmark series per predictor", () => {
    const svg = svgOf(plotNcaEsse(esseRecord, { type: "sensitivity" }));
    const empirical = (svg.match(/data-series="empirical"/g) ?? []).length;
    const benchmark = (svg.match(/data-series="benchmark"/g) ?? []).length;
    expect(empirical).toBe(esseRecord.predictors.length);
    expect(benchmark).toBe(esseRecord.predictors.length);
  });

  test("esse difference runs", () => {
    const svg = svgOf(plotNcaEsse(esseRecord, { type: "difference" }));
    expect(titles(svg).some((t) => t.includes("ESSE Difference"))).toBeTrue();
  });
});

// ---------------------------------------------------------------------------
// PCM
// ---------------------------------------------------------------------------

describe("plotPcm", () => {
  test("bar values match pcmRmse per path panel", () => {
    const svg = svgOf(plotPcm(pcmRecord, { metric: "RMSE" }));
    expect(titles(svg).length).toBe(pcmRecord.pcmResults.length);
    const got = barValues(svg).map((v) => Math.round(v * 1e8) / 1e8);
    const expected = pcmRecord.pcmResults
      .flatMap((res) => [...res.pcmRmse])
      .map((v) => Math.round(v * 1e8) / 1e8);
    expect(got).toEqual(expected);
  });

  test("MAE metric switches the y label", () => {
    const svg = svgOf(plotPcm(pcmRecord, { metric: "MAE" }));
    expect(svg).toContain("PCM (MAE)");
  });

  test("bad metric throws", () => {
    expect(() => plotPcm(pcmRecord, { metric: "BOGUS" })).toThrow(/RMSE/);
  });
});

// ---------------------------------------------------------------------------
// FIMIX
// ---------------------------------------------------------------------------

describe("plotFimix", () => {
  test("segments bar values equal proportions x 100", () => {
    const svg = svgOf(plotFimix(fimixRecord, { type: "segments" }));
    const got = barValues(svg).map((v) => Math.round(v * 1e6) / 1e6);
    const expected = Object.values(fimixRecord.segmentProportions).map(
      (p) => Math.round(p * 100 * 1e6) / 1e6,
    );
    expect(got).toEqual(expected);
  });

  test("paths panel has one legend entry per segment", () => {
    const svg = svgOf(plotFimix(fimixRecord, { type: "paths" }));
    expect(legendLabels(svg)).toEqual(
      Array.from({ length: fimixRecord.k }, (_, s) => `Seg. ${s + 1}`),
    );
  });

  test("paths returns null when all segment paths are zero", () => {
    const zeroed = fimixRecord.segmentPaths.map((pm) => ({
      rows: [...pm.rows],
      cols: [...pm.cols],
      values: pm.values.map((row) => row.map(() => 0)),
    }));
    const rec = { ...fimixRecord, segmentPaths: zeroed } as FimixAnalysis;
    expect(plotFimix(rec, { type: "paths" })).toBeNull();
  });

  test("compare criteria legend lists the requested criteria", () => {
    const svg = svgOf(
      plotFimixCompare(fimixCompareRecord, { type: "criteria", criteria: ["AIC", "BIC"] }),
    );
    expect(legendLabels(svg)).toEqual(["AIC", "BIC"]);
    expect((svg.match(/data-series="/g) ?? []).length).toBe(2);
  });

  test("compare with no valid criteria throws", () => {
    expect(() =>
      plotFimixCompare(fimixCompareRecord, { type: "criteria", criteria: ["NOPE"] }),
    ).toThrow(/No valid criteria/);
  });

  test("compare entropy renders the EN panel", () => {
    const svg = svgOf(plotFimixCompare(fimixCompareRecord, { type: "entropy" }));
    expect(titles(svg)[0]).toBe("FIMIX-PLS: Classification Quality");
    expect(countClass(svg, "refline")).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// POS
// ---------------------------------------------------------------------------

describe("plotPos", () => {
  test("segments bar values equal size shares x 100", () => {
    const svg = svgOf(plotPos(posRecord, { type: "segments" }));
    const got = barValues(svg).map((v) => Math.round(v * 1e6) / 1e6);
    const expected = Object.values(posRecord.segmentSizes).map(
      (s) => Math.round((s / posRecord.nObs) * 100 * 1e6) / 1e6,
    );
    expect(got).toEqual(expected);
  });

  test("rsquared draws one bar per construct x segment", () => {
    const svg = svgOf(plotPos(posRecord, { type: "rsquared" }));
    expect(barValues(svg).length).toBe(
      posRecord.segmentRsquared.rows.length * posRecord.segmentRsquared.cols.length,
    );
  });

  test("paths returns null when all segment paths are zero", () => {
    const zeroed = posRecord.segmentPaths.map((pm) => ({
      rows: [...pm.rows],
      cols: [...pm.cols],
      values: pm.values.map((row) => row.map(() => 0)),
    }));
    const rec = { ...posRecord, segmentPaths: zeroed } as PosAnalysis;
    expect(plotPos(rec, { type: "paths" })).toBeNull();
  });

  test("compare draws the objective line", () => {
    const svg = svgOf(plotPosCompare(posCompareRecord));
    expect(titles(svg)[0]).toContain("Objective Criterion vs K");
  });

  test("compare returns null with fewer than two valid rows", () => {
    const rec = { ...posCompareRecord, fitTable: posCompareRecord.fitTable.slice(0, 1) };
    expect(plotPosCompare(rec as PosComparison)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Dispatch + CartTree layout + record-layer plot helpers
// ---------------------------------------------------------------------------

describe("plot dispatch", () => {
  test("routes by record kind", () => {
    expect(plot(fimixRecord, { type: "segments" })).not.toBeNull();
  });

  test("throws for unknown objects", () => {
    expect(() => plot({})).toThrow(/does not support/);
  });
});

describe("CartTree plot accessors", () => {
  const tree = rpartAnova([[1], [2], [3], [4]], [1, 2, 5, 6], ["a"], { minsplit: 2, cp: 0 });

  test("leaves are spread left-to-right starting at 1", () => {
    const layout = tree.plotLayout();
    const leafX = layout.x.filter((_, i) => layout.isLeaf[i]);
    expect(leafX[0]).toBe(1);
    expect([...leafX]).toEqual([...leafX].sort((a, b) => a - b));
  });

  test("internal x is the mean of the children", () => {
    const layout = tree.plotLayout();
    const pos = new Map(layout.nodeIds.map((nid, i) => [nid, layout.x[i]!]));
    layout.nodeIds.forEach((nid, i) => {
      if (!layout.isLeaf[i]) {
        expect(pos.get(nid)!).toBeCloseTo((pos.get(2 * nid)! + pos.get(2 * nid + 1)!) / 2, 12);
      }
    });
  });

  test("root sits at the top", () => {
    const layout = tree.plotLayout();
    const rootY = layout.y[layout.nodeIds.indexOf(1)]!;
    expect(rootY).toBe(Math.max(...layout.y));
  });

  test("split rule labels follow text.rpart's left-branch format", () => {
    const labels = tree.splitRuleLabels();
    for (const label of Object.values(labels)) {
      expect(label).toMatch(/^a(< |>=)\d+(\.\d+)?$/);
    }
    expect(Object.keys(labels).length).toBe(tree.var.filter((v) => v !== "<leaf>").length);
  });

  test("node labels add n= on leaves only", () => {
    const labels = tree.nodeLabels(true);
    tree.nodeIds.forEach((nid, i) => {
      expect(labels[nid]!.includes("n=")).toBe(tree.var[i] === "<leaf>");
    });
  });

  test("crFdhLine is null with fewer than two peers", () => {
    expect(crFdhLine([1, 1], [2, 2])).toBeNull();
  });
});

describe("groupScoreMeans", () => {
  test("matches a hand-computed group mean", () => {
    const means = groupScoreMeans(coaRecord);
    const groups = Object.keys(coaRecord.devianceTree.deviantGroups);
    expect([...means.cols]).toEqual(groups);
    const scores = coaRecord.plsModel.constructScores;
    expect([...means.rows]).toEqual([...scores.cols]);
    const cases = coaRecord.devianceTree.deviantGroups[groups[0]!]!;
    let sum = 0;
    for (const c of cases) sum += scores.values[c - 1]![0]!;
    expect(means.values[0]![0]!).toBeCloseTo(sum / cases.length, 12);
  });

  test("requires a model when given a bare dtree", () => {
    expect(() => groupScoreMeans(coaRecord.devianceTree)).toThrow(/model is required/);
  });

  test("accepts a dtree plus its model", () => {
    const viaDtree = groupScoreMeans(coaRecord.devianceTree, coaRecord.plsModel);
    const viaCoa = groupScoreMeans(coaRecord);
    expect(viaDtree.values).toEqual(viaCoa.values);
  });
});
