# R conformance fixture for the matrix statistics `src/` calls.
#
# Like `arith.R`, this pins R's *arithmetic* rather than seminrExtras' output,
# so it is generated here and does not touch the rule that the ported goldens
# under `tests/fixtures/` are never regenerated.
#
# What it pins is the call FORM. R's `cor(x)` and `cov(x)` on one matrix read
# every spread off a single accumulation; the two-argument `cor(x, y)` walks
# each column pair again and lands on different last bits. seminrExtras' R
# code calls the one-matrix form throughout --- `stats::cor(construct_scores)`
# at feature_congruence.R:128,151 and `stats::cov(info$data)` at
# feature_cta.R:552,591 --- and through v0.1.1 the port called the two-argument
# form against itself, which is a different function.
#
# Measured on the real matrices when the swap was made: against R,
# `colCor(x, x)` was exact on 13 of 16 cells of the C1 construct-score
# correlation and `colCov(x, x)` on 16 of 25 cells of the M1 Image covariance,
# while `@compstats/core`'s one-matrix forms were exact on every cell of both.
#
# On the two shapes below, the two-argument form was exact on 5 of 16 and 7 of
# 16 covariance cells and on 2 of 16 and 7 of 16 correlation cells, where the
# one-matrix form was exact on all 64. That spread is why these shapes are here
# rather than a plain rnorm block.
#
# Two shapes, because the failure modes differ: a wide-ish matrix with columns
# on very different scales, and a near-collinear pair where the correlation
# denominator does the damage.
#
# Regenerate with:  Rscript tests/fixtures/helpers/matstats.R

seed <- 91
set.seed(seed)

# Columns on deliberately different scales: cov's accumulation has to carry
# products spanning several orders of magnitude.
a <- cbind(
  rnorm(200, 0, 1),
  rnorm(200, 100, 25),
  rnorm(200, -3, 0.01),
  runif(200, 0, 1e4)
)

# A near-collinear pair, so cor's denominator is the delicate part.
base <- rnorm(150, 5, 2)
b <- cbind(base, base * 1.0001 + rnorm(150, 0, 1e-3), rnorm(150, 0, 1), base * -0.5 + rnorm(150, 0, 0.2))

g <- function(m) paste(apply(m, 1, function(r) paste0("[", paste(sprintf("%.17g", r), collapse = ", "), "]")), collapse = ", ")

out <- c(
  "{",
  '  "generator": "tests/fixtures/helpers/matstats.R",',
  sprintf('  "rVersion": "%s",', paste(R.version$major, R.version$minor, sep = ".")),
  sprintf('  "longDouble": %s,', tolower(as.character(capabilities("long.double")))),
  sprintf('  "seed": %d,', seed),
  sprintf('  "scaled": [%s],', g(a)),
  sprintf('  "scaledCov": [%s],', g(cov(a))),
  sprintf('  "scaledCor": [%s],', g(cor(a))),
  sprintf('  "collinear": [%s],', g(b)),
  sprintf('  "collinearCov": [%s],', g(cov(b))),
  sprintf('  "collinearCor": [%s]', g(cor(b))),
  "}"
)

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)][1]))
writeLines(out, file.path(here, "matstats.json"))
