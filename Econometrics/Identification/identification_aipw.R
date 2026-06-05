# =============================================================================
# scripts/identification_aipw.R
# Task B — AIPW with trim-stability at (i,j,t)
#
# Implements §3.5(B.2) of strategy_review_2026-05-03_identification.md:
# pool Han ∪ BERT at the dyad-year level, fit a logistic propensity for
# IN_BERT, then estimate the headline coefficients via the Robins-Rotnitzky-
# Zhao influence function on positive-flow cells. Trim p-hat at three
# thresholds and report stability.
#
# Inputs : Econometrics/Data/reg{1,3}.csv, Econometrics/Data/reg{1,3}_mitigation.csv
# Outputs: regressions/aipw/coef_adapt.csv
#          regressions/aipw/coef_miti.csv
#          regressions/aipw/trim_stability.csv
#          regressions/aipw/diagnostics.rds
# =============================================================================

suppressPackageStartupMessages({
  library(here); library(data.table); library(boot); library(dplyr)
})
set.seed(20260503L)

ROOT <- here::here()
source(here::here("Econometrics", "Identification", "_helpers.R"))
out_dir <- here::here("Econometrics", "regressions", "aipw")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

t0 <- Sys.time()
N_BOOT <- 300L                        # production setting per task spec
TRIM_GRID <- list(
  c(0.01, 0.99),
  c(0.05, 0.95),
  c(0.10, 0.90)
)
# Wall-clock guard: if the first trim grid takes longer than this many seconds,
# downgrade N_BOOT to N_BOOT_FALLBACK and log a flag in diagnostics.
WALLCLOCK_BUDGET_SEC <- 2 * 3600
N_BOOT_FALLBACK <- 200L

# Use the expanded covariate list (all variables that change sign or
# significance in §4 of the EARE Round-1 revision). 26 covariates.
APE_VARS <- EXPANDED_COVARS

message("[", format(Sys.time()), "] Task B: AIPW with trim-stability ",
        "(expanded to ", length(APE_VARS), " covariates, N_BOOT=", N_BOOT, ")")

# ── 1. Build pooled Han ∪ BERT panels ────────────────────────────────────────
reg1  <- fread(here::here("Econometrics", "Data", "reg1.csv"),  na.strings = c("", "NA"))
reg3  <- fread(here::here("Econometrics", "Data", "reg3.csv"),  na.strings = c("", "NA"))
reg1m <- fread(here::here("Econometrics", "Data", "reg1_mitigation.csv"), na.strings = c("", "NA"))
reg3m <- fread(here::here("Econometrics", "Data", "reg3_mitigation.csv"), na.strings = c("", "NA"))

agg_han_a  <- aggregate_dyad_year(reg1,  "AdaptAmount", BASELINE_RECIPIENT_HAN)
agg_bert_a <- aggregate_dyad_year(reg3,  "AdaptAmount", BASELINE_RECIPIENT_BERT)
agg_han_m  <- aggregate_dyad_year(reg1m, "MitiAmount",  BASELINE_RECIPIENT_HAN)
agg_bert_m <- aggregate_dyad_year(reg3m, "MitiAmount",  BASELINE_RECIPIENT_BERT)

build_pool <- function(han, bert, outcome_in, outcome_out, any_in, any_out) {
  setnames(han,  outcome_in,  outcome_out, skip_absent = TRUE)
  setnames(bert, outcome_in,  outcome_out, skip_absent = TRUE)
  setnames(han,  any_in,      any_out,     skip_absent = TRUE)
  setnames(bert, any_in,      any_out,     skip_absent = TRUE)
  han[,  IN_BERT := 0L]
  bert[, IN_BERT := 1L]
  cols <- intersect(names(han), names(bert))
  pooled <- rbindlist(list(han[, ..cols], bert[, ..cols]), use.names = TRUE)
  pooled
}
pool_a <- build_pool(copy(agg_han_a),  copy(agg_bert_a),
                     "tot_amount", "tot_adapt_amount",
                     "any_flow",   "any_flow")
pool_m <- build_pool(copy(agg_han_m),  copy(agg_bert_m),
                     "tot_amount", "tot_miti_amount",
                     "any_flow",   "any_flow")
message(sprintf("  pooled adapt n = %d (BERT share = %.3f)",
                nrow(pool_a), mean(pool_a$IN_BERT)))
message(sprintf("  pooled miti  n = %d (BERT share = %.3f)",
                nrow(pool_m), mean(pool_m$IN_BERT)))

# ── 2. Diagnostic propensity model (full sample, single trim) ────────────────
diagnose_ps <- function(pool) {
  rhs <- paste(intersect(DYAD_COVARS, names(pool)), collapse = " + ")
  ps_fit <- suppressWarnings(glm(as.formula(paste0("IN_BERT ~ ", rhs)),
                                 family = binomial(link = "logit"),
                                 data = pool[complete.cases(pool[, c(DYAD_COVARS, "IN_BERT"),
                                                                 with = FALSE])]))
  phat <- predict(ps_fit, type = "response")
  list(quartiles = quantile(phat, c(0.01, 0.05, 0.5, 0.95, 0.99), na.rm = TRUE),
       fit = ps_fit)
}
diag_a <- diagnose_ps(pool_a)
diag_m <- diagnose_ps(pool_m)
message("  PS quantiles (adapt): ",
        paste(sprintf("%.3f", diag_a$quartiles), collapse = ", "))
message("  PS quantiles (miti) : ",
        paste(sprintf("%.3f", diag_m$quartiles), collapse = ", "))

# ── 3. AIPW point estimates by trim threshold + bootstrap SE ─────────────────
#'
#' Per-covariate convergence is tracked as the share of bootstrap reps for
#' which `aipw_one_coef` returned a finite estimate (no glm/lm error, no
#' all-NA propensity, etc.). Stored in the returned `convergence` vector.
boot_aipw <- function(pool, outcome, headline_coefs = APE_VARS,
                      trim_lo, trim_hi, n_boot = N_BOOT) {
  pool <- as.data.table(pool)
  est_pt <- numeric(length(headline_coefs)); names(est_pt) <- headline_coefs
  for (cv in headline_coefs) {
    others <- setdiff(DYAD_COVARS, cv)
    res <- tryCatch(aipw_one_coef(pool, coef_name = cv, outcome = outcome,
                                  covars = others,
                                  trim_lo = trim_lo, trim_hi = trim_hi),
                    error = function(e) list(estimate = NA_real_))
    est_pt[cv] <- res$estimate
  }
  # Cluster bootstrap by recipient (the dominant cluster in this panel)
  rec_levels <- unique(pool$recipient)
  boot_mat <- matrix(NA_real_, nrow = n_boot, ncol = length(headline_coefs),
                     dimnames = list(NULL, headline_coefs))
  for (b in seq_len(n_boot)) {
    set.seed(20260503L + b)                      # deterministic per-replicate seed
    samp <- sample(rec_levels, length(rec_levels), replace = TRUE)
    keys_b <- data.table(recipient = samp)
    p_b <- pool[keys_b, on = "recipient", nomatch = NULL, allow.cartesian = TRUE]
    for (cv in headline_coefs) {
      others <- setdiff(DYAD_COVARS, cv)
      res <- tryCatch(aipw_one_coef(p_b, coef_name = cv, outcome = outcome,
                                    covars = others,
                                    trim_lo = trim_lo, trim_hi = trim_hi),
                      error = function(e) list(estimate = NA_real_))
      boot_mat[b, cv] <- res$estimate
    }
  }
  conv <- apply(boot_mat, 2L,
                function(v) sum(is.finite(v)) / length(v))
  list(point = est_pt,
       se    = apply(boot_mat, 2L, sd, na.rm = TRUE),
       convergence = conv,
       n_boot_used = n_boot,
       n_trimmed = NA_integer_)   # see trim_stability below
}

# n_trimmed at each threshold (computed once on the full pooled panel)
trim_counts <- function(pool, trim_lo, trim_hi) {
  pool <- pool[complete.cases(pool[, c(DYAD_COVARS, "IN_BERT"), with = FALSE])]
  rhs <- paste(intersect(DYAD_COVARS, names(pool)), collapse = " + ")
  ps_fit <- suppressWarnings(glm(as.formula(paste0("IN_BERT ~ ", rhs)),
                                 family = binomial(link = "logit"),
                                 data = pool))
  phat <- predict(ps_fit, type = "response")
  sum(phat < trim_lo | phat > trim_hi)
}

run_trim_stability <- function(pool, outcome, label, n_boot_use) {
  message("  >>> ", label, ": running ", length(TRIM_GRID), " trim grids x ",
          length(APE_VARS), " coefs x ", n_boot_use, " bootstrap reps")
  rows <- list()
  conv_all <- list()
  for (tg in TRIM_GRID) {
    t_grid <- Sys.time()
    message("      trim = [", tg[1L], ", ", tg[2L], "]")
    n_trim <- trim_counts(pool, tg[1L], tg[2L])
    res <- boot_aipw(pool, outcome, APE_VARS, tg[1L], tg[2L], n_boot_use)
    rows[[length(rows) + 1L]] <- data.table(
      trim_lo = tg[1L], trim_hi = tg[2L],
      n_trimmed = n_trim,
      covariate = APE_VARS,
      estimate  = res$point,
      se        = res$se,
      convergence = unname(res$convergence[APE_VARS])
    )
    conv_all[[paste(label, tg[1L], tg[2L], sep = "_")]] <- res$convergence
    message(sprintf("      done in %.1f s",
                    as.numeric(difftime(Sys.time(), t_grid, units = "secs"))))
  }
  out <- rbindlist(rows)
  out[, label := label]
  attr(out, "convergence") <- conv_all
  out
}

# Run adaptation panel first; if first trim grid alone exceeds budget/3 we
# downgrade to N_BOOT_FALLBACK for the remaining work and tag the diagnostics.
n_boot_used_a <- N_BOOT
n_boot_used_m <- N_BOOT
fix_flag_n_boot <- FALSE
t_pilot <- Sys.time()
trim_a <- run_trim_stability(pool_a, "tot_adapt_amount", "adapt", n_boot_used_a)
elapsed_a <- as.numeric(difftime(Sys.time(), t_pilot, units = "secs"))
# Budget check: adapt panel runs first; if it consumed > half of the wall-clock
# budget already, drop mitigation to N_BOOT_FALLBACK to stay <= 2 hr total.
if (elapsed_a > WALLCLOCK_BUDGET_SEC / 2) {
  fix_flag_n_boot <- TRUE
  n_boot_used_m <- N_BOOT_FALLBACK
  message(sprintf("  >>> wall-clock guard: adaptation took %.0f s; ",
                  elapsed_a),
          "downgrading mitigation to N_BOOT=", N_BOOT_FALLBACK)
}
trim_m <- run_trim_stability(pool_m, "tot_miti_amount",  "miti", n_boot_used_m)
trim_all <- rbind(trim_a, trim_m)
fwrite(trim_all, file.path(out_dir, "trim_stability.csv"))

# Headline coefficient table at the canonical 1/99 trim
build_headline <- function(trim_dt, lab) {
  d <- trim_dt[label == lab & trim_lo == 0.01 & trim_hi == 0.99]
  d[, .(covariate, estimate, se,
        ci_low  = estimate - 1.96 * se,
        ci_high = estimate + 1.96 * se,
        n_trimmed = n_trimmed)]
}
fwrite(build_headline(trim_a, "adapt"), file.path(out_dir, "coef_adapt.csv"))
fwrite(build_headline(trim_m, "miti"),  file.path(out_dir, "coef_miti.csv"))

# Per-covariate convergence summary (averaged across trim grids per panel)
conv_a <- attr(trim_a, "convergence")
conv_m <- attr(trim_m, "convergence")
convergence_summary <- list(
  adapt = sapply(APE_VARS, function(cv) {
    mean(sapply(conv_a, function(v) v[cv]), na.rm = TRUE)
  }),
  miti  = sapply(APE_VARS, function(cv) {
    mean(sapply(conv_m, function(v) v[cv]), na.rm = TRUE)
  })
)

saveRDS(list(diag_a = diag_a, diag_m = diag_m,
             trim_grid = TRIM_GRID,
             n_boot_target = N_BOOT,
             n_boot_used = list(adapt = n_boot_used_a, miti = n_boot_used_m),
             fix_flag_n_boot_downgraded = fix_flag_n_boot,
             expanded_covariates = APE_VARS,
             convergence_per_covariate = convergence_summary,
             runtime_sec = as.numeric(difftime(Sys.time(), t0, units = "secs"))),
        file.path(out_dir, "diagnostics.rds"))
message(sprintf("[%s] Task B done in %.1f s", format(Sys.time()),
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
