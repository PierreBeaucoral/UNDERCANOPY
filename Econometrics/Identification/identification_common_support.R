# =============================================================================
# scripts/identification_common_support.R
# Task A — Common-support intersection at (i,j,t)
#
# Implements §3.5(B.1) of strategy_review_2026-05-03_identification.md:
# aggregate Han / Rio / BERT panels to dyad-year, intersect on (i,j,t), and
# re-estimate the dyad-year hurdle on the common-support cells.
#
# Inputs : Econometrics/Data/reg{1,2,3}.csv (adaptation), Econometrics/Data/reg{1,2,3}_mitigation.csv
# Outputs: regressions/common_support/coef_adapt.csv
#          regressions/common_support/coef_miti.csv
#          regressions/common_support/cells_C_adapt.rds
#          regressions/common_support/cells_C_miti.rds
#          regressions/common_support/summary.rds
# =============================================================================

suppressPackageStartupMessages({
  library(here); library(data.table); library(mhurdle); library(dplyr)
})

set.seed(20260503L)

ROOT <- here::here()
source(here::here("Econometrics", "Identification", "_helpers.R"))
out_dir <- here::here("Econometrics", "regressions", "common_support")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

t0 <- Sys.time()
message("[", format(Sys.time()), "] Task A: common-support hurdle on (i,j,t)")

# ── 1. Load and aggregate ────────────────────────────────────────────────────
read_panel <- function(path) fread(path, na.strings = c("", "NA"))

reg1 <- read_panel(here::here("Econometrics", "Data", "reg1.csv"))
reg2 <- read_panel(here::here("Econometrics", "Data", "reg2.csv"))
reg3 <- read_panel(here::here("Econometrics", "Data", "reg3.csv"))
reg1m <- read_panel(here::here("Econometrics", "Data", "reg1_mitigation.csv"))
reg2m <- read_panel(here::here("Econometrics", "Data", "reg2_mitigation.csv"))
reg3m <- read_panel(here::here("Econometrics", "Data", "reg3_mitigation.csv"))

agg_han_a  <- aggregate_dyad_year(reg1, "AdaptAmount", BASELINE_RECIPIENT_HAN)
agg_rio_a  <- aggregate_dyad_year(reg2, "AdaptAmount", BASELINE_RECIPIENT_HAN)
agg_bert_a <- aggregate_dyad_year(reg3, "AdaptAmount", BASELINE_RECIPIENT_BERT)
agg_han_m  <- aggregate_dyad_year(reg1m, "MitiAmount", BASELINE_RECIPIENT_HAN)
agg_rio_m  <- aggregate_dyad_year(reg2m, "MitiAmount", BASELINE_RECIPIENT_HAN)
agg_bert_m <- aggregate_dyad_year(reg3m, "MitiAmount", BASELINE_RECIPIENT_BERT)

# Standardise outcome column to `tot_adapt_amount` / `tot_miti_amount`
setnames(agg_han_a,  "tot_amount", "tot_adapt_amount")
setnames(agg_rio_a,  "tot_amount", "tot_adapt_amount")
setnames(agg_bert_a, "tot_amount", "tot_adapt_amount")
setnames(agg_han_m,  "tot_amount", "tot_miti_amount")
setnames(agg_rio_m,  "tot_amount", "tot_miti_amount")
setnames(agg_bert_m, "tot_amount", "tot_miti_amount")
setnames(agg_han_a,  "any_flow", "any_flow_adapt")
setnames(agg_rio_a,  "any_flow", "any_flow_adapt")
setnames(agg_bert_a, "any_flow", "any_flow_adapt")
setnames(agg_han_m,  "any_flow", "any_flow_miti")
setnames(agg_rio_m,  "any_flow", "any_flow_miti")
setnames(agg_bert_m, "any_flow", "any_flow_miti")

message(sprintf("  panel sizes (i,j,t): Han=%d  Rio=%d  BERT=%d (adapt)",
                nrow(agg_han_a), nrow(agg_rio_a), nrow(agg_bert_a)))
message(sprintf("                       Han=%d  Rio=%d  BERT=%d (miti)",
                nrow(agg_han_m), nrow(agg_rio_m), nrow(agg_bert_m)))

# ── 2. Intersection ──────────────────────────────────────────────────────────
key_cols <- c("provider", "recipient", "year")
keys_C_adapt <- Reduce(
  function(a, b) merge(a, b, by = key_cols),
  list(unique(agg_han_a[, ..key_cols]),
       unique(agg_rio_a[, ..key_cols]),
       unique(agg_bert_a[, ..key_cols]))
)
keys_C_miti <- Reduce(
  function(a, b) merge(a, b, by = key_cols),
  list(unique(agg_han_m[, ..key_cols]),
       unique(agg_rio_m[, ..key_cols]),
       unique(agg_bert_m[, ..key_cols]))
)
n_C_adapt <- nrow(keys_C_adapt)
n_C_miti  <- nrow(keys_C_miti)
message(sprintf("  |C_adapt| = %d   |C_miti| = %d", n_C_adapt, n_C_miti))

saveRDS(keys_C_adapt, file.path(out_dir, "cells_C_adapt.rds"))
saveRDS(keys_C_miti,  file.path(out_dir, "cells_C_miti.rds"))

# ── 3. Re-estimate hurdle on intersection ────────────────────────────────────
restrict_to_C <- function(d, keys, outcome_col) {
  d <- merge(d, keys, by = key_cols)
  setnames(d, outcome_col, "tot_amount", skip_absent = TRUE)
  if (outcome_col == "tot_adapt_amount") {
    setnames(d, "any_flow_adapt", "any_flow", skip_absent = TRUE)
  } else {
    setnames(d, "any_flow_miti", "any_flow", skip_absent = TRUE)
  }
  d
}

panels_a <- list(
  Han  = restrict_to_C(agg_han_a,  keys_C_adapt, "tot_adapt_amount"),
  Rio  = restrict_to_C(agg_rio_a,  keys_C_adapt, "tot_adapt_amount"),
  BERT = restrict_to_C(agg_bert_a, keys_C_adapt, "tot_adapt_amount")
)
panels_m <- list(
  Han  = restrict_to_C(agg_han_m,  keys_C_miti, "tot_miti_amount"),
  Rio  = restrict_to_C(agg_rio_m,  keys_C_miti, "tot_miti_amount"),
  BERT = restrict_to_C(agg_bert_m, keys_C_miti, "tot_miti_amount")
)

fit_one <- function(d, label) {
  message("    fitting ", label, "  (n=", nrow(d), ")")
  res <- fit_hurdle_dyad(d, verbose = TRUE)
  if (!isTRUE(res$ok)) {
    # Fallback: estimate selection (probit) and outcome (OLS) separately
    message("    mhurdle failed; falling back to two-step probit + OLS")
    rhs <- paste(intersect(DYAD_COVARS, names(d)), collapse = " + ")
    sel <- tryCatch(glm(as.formula(paste0("any_flow ~ ", rhs)),
                        family = binomial(link = "probit"),
                        data = d), error = function(e) NULL)
    pos <- d[any_flow == 1L & is.finite(tot_amount) & tot_amount > 0]
    out <- tryCatch(lm(as.formula(paste0("tot_amount ~ ", rhs)), data = pos),
                    error = function(e) NULL)
    coefs_sel <- if (!is.null(sel)) coef(sel) else rep(NA_real_, length(DYAD_COVARS) + 1L)
    coefs_out <- if (!is.null(out)) coef(out) else rep(NA_real_, length(DYAD_COVARS) + 1L)
    se_sel <- if (!is.null(sel)) sqrt(diag(vcov(sel))) else rep(NA_real_, length(coefs_sel))
    se_out <- if (!is.null(out)) sqrt(diag(vcov(out))) else rep(NA_real_, length(coefs_out))
    # Native p-values: probit GLM uses Wald-z (Pr(>|z|)); OLS lm uses
    # t with residual df (Pr(>|t|)). Both come from the model's own
    # summary table — last column.
    p_sel <- if (!is.null(sel)) summary(sel)$coefficients[, 4L]
             else rep(NA_real_, length(coefs_sel))
    p_out <- if (!is.null(out)) summary(out)$coefficients[, 4L]
             else rep(NA_real_, length(coefs_out))
    return(list(ok = TRUE, fallback = TRUE,
                coefs_selection = setNames(coefs_sel, paste0("h1.", names(coefs_sel))),
                coefs_outcome   = setNames(coefs_out, paste0("h2.", names(coefs_out))),
                se_selection    = setNames(se_sel, paste0("h1.", names(coefs_sel))),
                se_outcome      = setNames(se_out, paste0("h2.", names(coefs_out))),
                pval_selection  = setNames(p_sel,  paste0("h1.", names(coefs_sel))),
                pval_outcome    = setNames(p_out,  paste0("h2.", names(coefs_out))),
                n = nrow(d)))
  }
  res$fallback <- FALSE
  res
}

fits_a <- lapply(names(panels_a), function(k) fit_one(panels_a[[k]], paste("adapt", k)))
names(fits_a) <- names(panels_a)
fits_m <- lapply(names(panels_m), function(k) fit_one(panels_m[[k]], paste("miti",  k)))
names(fits_m) <- names(panels_m)

# ── 4. Tabulate coefficients ─────────────────────────────────────────────────
build_coef_table <- function(fits) {
  rows <- list()
  for (spec in names(fits)) {
    f <- fits[[spec]]
    cs <- f$coefs_selection; co <- f$coefs_outcome
    ss <- f$se_selection;    so <- f$se_outcome
    ps <- f$pval_selection;  po <- f$pval_outcome
    pull <- function(v, eq, name) {
      key <- paste0(eq, ".", name)
      val <- if (!is.null(v)) v[key] else NA_real_
      if (is.null(val) || length(val) == 0L || is.na(val)) NA_real_ else unname(val)
    }
    for (cv in EXPANDED_COVARS) {
      rows[[length(rows) + 1L]] <- data.table(
        spec = spec, covariate = cv,
        beta_selection = pull(cs, "h1", cv),
        beta_outcome   = pull(co, "h2", cv),
        se_selection   = pull(ss, "h1", cv),
        se_outcome     = pull(so, "h2", cv),
        pval_selection = pull(ps, "h1", cv),
        pval_outcome   = pull(po, "h2", cv),
        n_obs = f$n,
        fallback = isTRUE(f$fallback)
      )
    }
  }
  rbindlist(rows)
}
coef_adapt <- build_coef_table(fits_a)
coef_miti  <- build_coef_table(fits_m)
fwrite(coef_adapt, file.path(out_dir, "coef_adapt.csv"))
fwrite(coef_miti,  file.path(out_dir, "coef_miti.csv"))

# ── 5. Summary ───────────────────────────────────────────────────────────────
summary_obj <- list(
  n_C_adapt = n_C_adapt,
  n_C_miti  = n_C_miti,
  n_input = list(
    han_adapt = nrow(agg_han_a), rio_adapt = nrow(agg_rio_a), bert_adapt = nrow(agg_bert_a),
    han_miti  = nrow(agg_han_m), rio_miti  = nrow(agg_rio_m), bert_miti  = nrow(agg_bert_m)
  ),
  fallbacks = list(
    adapt = sapply(fits_a, function(f) isTRUE(f$fallback)),
    miti  = sapply(fits_m, function(f) isTRUE(f$fallback))
  ),
  runtime_sec = as.numeric(difftime(Sys.time(), t0, units = "secs"))
)
saveRDS(summary_obj, file.path(out_dir, "summary.rds"))
message(sprintf("[%s] Task A done in %.1f s", format(Sys.time()),
                summary_obj$runtime_sec))
