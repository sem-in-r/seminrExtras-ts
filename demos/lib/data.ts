/**
 * Demo data loading (Bun). The mobi/ECSI and corporate reputation datasets
 * ship as test fixtures in this repo (copied from the R packages' data).
 */

import { parseCsv, type Dataset } from "@seminr/core";

const mobiCsvUrl = new URL("../../tests/fixtures/data/mobi.csv", import.meta.url);
const corpRepCsvUrl = new URL("../../tests/fixtures/data/corp_rep_data.csv", import.meta.url);

export async function loadMobi(): Promise<Dataset> {
  return parseCsv(await Bun.file(mobiCsvUrl).text());
}

export async function loadCorpRep(): Promise<Dataset> {
  return parseCsv(await Bun.file(corpRepCsvUrl).text());
}
