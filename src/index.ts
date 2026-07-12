/**
 * seminrExtras-ts — supplementary evaluation methods for SEMinR PLS-SEM models,
 * ported from the seminrExtras R package.
 *
 * All functions consume an already-estimated model from `@seminr/core`
 * (specify → estimate → **evaluate**); nothing here estimates models itself.
 */

export { version } from "./version.ts";

// CVPAT — Cross-Validated Predictive Ability Test (Liengaard et al. 2021;
// Sharma et al. 2023)
export {
  assessCvpat,
  assessCvpatCompare,
  type AssessCvpatArgs,
  type AssessCvpatCompareArgs,
  type CvpatAssessment,
  type CvpatComparison,
  type CvpatFeatureOptions,
} from "./featureCvpat.ts";

// PCM — Predictive Contribution of the Mediator (Danks 2021)
export {
  assessPcm,
  buildIsolatedSubModel,
  classifyPcm,
  detectFinalEndogenous,
  findMediationPaths,
  type AssessPcmArgs,
  type PcmAnalysis,
  type PcmFeatureOptions,
  type PcmPath,
  type PcmPathResult,
} from "./featurePcm.ts";

// Congruence — bootstrap congruence-coefficient test (Franke, Sarstedt & Danks
// 2021)
export {
  congruenceTest,
  type CongruenceOptions,
  type CongruenceTest,
  type CongruenceTestArgs,
} from "./featureCongruence.ts";

// CTA-PLS — Confirmatory Tetrad Analysis (Gudergan et al. 2008; Cefis et al.
// 2025)
export {
  assessCta,
  computeTetrads,
  enumerateBorrowedTetrads,
  enumerateTetrads,
  findDonor,
  formatTetradLabel,
  getStructurallyConnected,
  pAdjust,
  resolveIndicators,
  type AssessCtaArgs,
  type CtaAnalysis,
  type CtaBorrowing,
  type CtaConstructResult,
  type CtaOptions,
  type CtaTetradDetails,
  type ResolvedIndicators,
  type TetradSpec,
} from "./featureCta.ts";

// NCA — Necessary Condition Analysis + ESSE benchmark (Dul 2016; Richter
// et al. 2020; Becker et al. 2026)
export {
  assessNca,
  assessNcaEsse,
  benchmarkEffectSize,
  ceFdhEffectSize,
  computeBottleneckColumn,
  computeCeFdh,
  computeEcdfNca,
  crFdhEffectSize,
  crFdhLine,
  getCeFdhPeers,
  lineCeilingZone,
  ncaEffectSize,
  ncaPermutationTest,
  INTERNAL_CEILINGS,
  type AssessNcaArgs,
  type AssessNcaEsseArgs,
  type NcaAnalysis,
  type NcaEsse,
  type NcaEsseOptions,
  type NcaOptions,
  type PermTestOptions,
} from "./featureNca.ts";

// IPMA / cIPMA — (combined) Importance-Performance Map Analysis (Ringle &
// Sarstedt 2016; Sarstedt et al. 2024)
export {
  assessCipma,
  assessIpma,
  checkPositiveWeights,
  classifyCipmaConstructs,
  computeIpmaPerformance,
  computeObservationPerformance,
  computeTotalEffects,
  computeUnstdTotalEffects,
  isInteractionConstruct,
  DEFAULT_NCA_CEILINGS,
  type AssessCipmaArgs,
  type AssessIpmaArgs,
  type CipmaAnalysis,
  type CipmaClassificationRow,
  type CipmaOptions,
  type IpmaOptions,
  type WeightsModelLike,
} from "./featureCipma.ts";

// COA — Composite Overfit Analysis (Ray, Danks & Valdez 2022)
export {
  assessCoa,
  competes,
  devianceTree,
  groupRules,
  groupScoreMeans,
  mainAncestors,
  pathTo,
  predictiveDeviance,
  unstableParams,
  type AssessCoaArgs,
  type CoaAnalysis,
  type CoaCompetes,
  type CoaDeviance,
  type CoaDtree,
  type CoaGroupInstability,
  type CoaOptions,
  type CoaRules,
  type CoaUnstable,
  type PredictiveDevianceOptions,
} from "./featureCoa.ts";

// FIMIX-PLS — finite mixture latent-class segmentation (Hahn et al. 2002;
// Sarstedt et al. 2011)
export {
  assessFimix,
  assessFimixCompare,
  type AssessFimixArgs,
  type AssessFimixCompareArgs,
  type FimixAnalysis,
  type FimixCompareOptions,
  type FimixComparison,
  type FimixOptions,
} from "./featureFimix.ts";

// PLS-POS — prediction-oriented segmentation (Becker et al. 2013)
export {
  assessPos,
  assessPosCompare,
  posSegments,
  type AssessPosArgs,
  type AssessPosCompareArgs,
  type PosAnalysis,
  type PosCompareOptions,
  type PosComparison,
  type PosFitRow,
  type PosOptions,
} from "./featurePos.ts";

// Plotting — SVG emitters for the R `plot.*` methods (each returns a
// `@seminr/core` SvgPlot, or null on R's "nothing to plot" paths)
export {
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
  type PlotCipmaOptions,
  type PlotCoaOptions,
  type PlotFimixCompareOptions,
  type PlotFimixOptions,
  type PlotNcaEsseOptions,
  type PlotNcaOptions,
  type PlotPcmOptions,
  type PlotPosOptions,
} from "./plotting/results.ts";
