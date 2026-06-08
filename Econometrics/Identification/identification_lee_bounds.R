# =============================================================================
# scripts/identification_lee_bounds.R
# Task C — Lee (2009) trimming bounds for three headline coefficients
#
# Implements §3.5(B.3) of strategy_review_2026-05-03_identification.md:
# treat BERT-vs-Han classification as the "treatment", colonial-tie /
# CPIAPublicAdm-above-median / CPIAbudget-above-median as the binary
# outcome partitions, and apply Lee's sharp bounds under the monotonic
# selection assumption. CPIA covariates are dichotomised at the median for
# the differential-retention computation.
#
# Inputs : Econometrics/Data/reg{1,3}.csv, Econometrics/Data/reg{1,3}_mitigation.csv
# Outputs: regressions/lee_bounds/bounds_adapt.csv
#          regressions/lee_bounds/bounds_miti.csv
#          regressions/lee_bounds/bounds_diagnostics.rds
# =============================================================================

suppressPackageStartupMessages({
  library(here); library(data.table); library(dplyr); library(boot)
})
set.seed(20260503L)

ROOT <- here::here()
source(here::here("Econometrics", "Identification", "_helpers.R"))
out_dir <- here::here("Econometrics", "regressions", "lee_bounds")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

t0 <- Sys.time()
N_BOOT <- as.integer(Sys.getenv("UC_N_BOOT", unset = "300"))
stopifnot(!is.na(N_BOOT), N_BOOT >= 1L)   # guard against a malformed UC_N_BOOT override
# Expanded coverage 2026-05-04: every variable that changes sign or
# significance between Han / Rio / BERT in §4 of the EARE Round-1 revision.
COEFS <- EXPANDED_COVARS
COEFS_BINARY     <- EXPANDED_COVARS_BINARY
COEFS_CONTINUOUS <- EXPANDED_COVARS_CONTINUOUS
message("[", format(Sys.time()), "] Task C: Lee (2009) bounds for ",
        length(COEFS), " covariates (",
        length(COEFS_BINARY), " binary + ",
        length(COEFS_CONTINUOUS), " continuous; median split)")

reg1  <- fread(here::here("Econometrics", "Data", "reg1.csv"),  na.strings = c("", "NA"))
reg3  <- fread(here::here("Econometrics", "Data", "reg3.csv"),  na.strings = c("", "NA"))
reg1m <- fread(here::here("Econometrics", "Data", "reg1_mitigation.csv"), na.strings = c("", "NA"))
reg3m <- fread(here::here("Econometrics", "Data", "reg3_mitigation.csv"), na.strings = c("", "NA"))

agg_han_a  <- aggregate_dyad_year(reg1,  "AdaptAmount", BASELINE_RECIPIENT_HAN)
agg_bert_a <- aggregate_dyad_year(reg3,  "AdaptAmount", BASELINE_RECIPIENT_BERT)
agg_han_m  <- aggregate_dyad_year(reg1m, "MitiAmount",  BASELINE_RECIPIENT_HAN)
agg_bert_m <- aggregate_dyad_year(reg3m, "MitiAmount",  BASELINE_RECIPIENT_BERT)

setnames(agg_han_a,  "tot_amount", "tot_adapt_amount"); setnames(agg_han_a,  "any_flow", "any_flow_adapt")
setnames(agg_bert_a, "tot_amount", "tot_adapt_amount"); setnames(agg_bert_a, "any_flow", "any_flow_adapt")
setnames(agg_han_m,  "tot_amount", "tot_miti_amount");  setnames(agg_han_m,  "any_flow", "any_flow_miti")
setnames(agg_bert_m, "tot_amount", "tot_miti_amount");  setnames(agg_bert_m, "any_flow", "any_flow_miti")

# ── Wrapper: compute Lee bound for each covariate (binary or continuous) ─────
#
# For binary covariates: lee_bound_binary() reports the OLS coefficient on the
# binary variable, with q computed directly from BERT-vs-Han retention by 0/1.
#
# For continuous covariates: lee_bound_one() median-splits on the pooled
# positive-flow values for the q computation, but reports the OLS coefficient
# on the *original* continuous regressor — directly comparable to the §4
# headline estimate.
run_lee_pack <- function(bert, han, outcome) {
  out <- list()
  flow_col <- if (outcome == "tot_adapt_amount") "any_flow_adapt" else "any_flow_miti"
  for (cv in COEFS) {
    is_bin <- cv %in% COEFS_BINARY
    covars_lm <- setdiff(DYAD_COVARS, cv)
    res <- tryCatch(
      lee_bound_one(bert, han,
                    var = cv, is_binary = is_bin,
                    outcome = outcome, covars = covars_lm,
                    flow_col = flow_col),
      error = function(e) {
        list(point = NA_real_, lower = NA_real_, upper = NA_real_,
             q_trim = NA_real_, n_trimmed = 0L, median_value = NA_real_,
             over_is_one = NA, r1 = NA_real_, r0 = NA_real_,
             msg = paste0("error: ", conditionMessage(e)))
      }
    )
    out[[cv]] <- res
  }
  out
}

# Bootstrap helper (cluster on recipient).
# Per-covariate convergence is tracked as the share of bootstrap reps for which
# the lower/upper bound returned a finite value.
boot_lee <- function(bert, han, outcome, n_boot = N_BOOT) {
  rec_levels <- unique(c(bert$recipient, han$recipient))
  res_pt <- run_lee_pack(bert, han, outcome)
  boot_lower <- matrix(NA_real_, nrow = n_boot, ncol = length(COEFS),
                       dimnames = list(NULL, COEFS))
  boot_upper <- boot_lower
  for (b in seq_len(n_boot)) {
    set.seed(20260503L + b)                             # deterministic per-replicate seed
    samp <- sample(rec_levels, length(rec_levels), replace = TRUE)
    keys_b <- data.table(recipient = samp)
    bt_b <- bert[keys_b, on = "recipient", nomatch = NULL, allow.cartesian = TRUE]
    ht_b <- han [keys_b, on = "recipient", nomatch = NULL, allow.cartesian = TRUE]
    res_b <- tryCatch(run_lee_pack(bt_b, ht_b, outcome), error = function(e) NULL)
    if (is.null(res_b)) next
    for (cv in COEFS) {
      boot_lower[b, cv] <- res_b[[cv]]$lower
      boot_upper[b, cv] <- res_b[[cv]]$upper
    }
  }
  ci_low_l  <- apply(boot_lower, 2L, quantile, probs = 0.025, na.rm = TRUE)
  ci_high_u <- apply(boot_upper, 2L, quantile, probs = 0.975, na.rm = TRUE)
  conv <- sapply(COEFS, function(cv) {
    sum(is.finite(boot_lower[, cv]) & is.finite(boot_upper[, cv])) / n_boot
  })
  list(point = res_pt,
       boot_lower = boot_lower, boot_upper = boot_upper,
       ci_low = ci_low_l, ci_high = ci_high_u,
       convergence = conv,
       n_boot_used = n_boot)
}

message("  >>> adaptation: ", N_BOOT, " bootstrap reps")
res_a <- boot_lee(agg_bert_a, agg_han_a, "tot_adapt_amount", N_BOOT)
message("  >>> mitigation: ", N_BOOT, " bootstrap reps")
res_m <- boot_lee(agg_bert_m, agg_han_m, "tot_miti_amount",  N_BOOT)

# ── Tabulate ─────────────────────────────────────────────────────────────────
# Defensive accessor: every row has the same column shape regardless of which
# early-return path the underlying lee_bound_one call took.
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a
to_table <- function(res, label) {
  rows <- lapply(COEFS, function(cv) {
    p <- res$point[[cv]]
    is_bin <- cv %in% COEFS_BINARY
    data.table(
      label = label,
      covariate = cv,
      is_binary      = as.integer(is_bin),
      point_estimate = p$point %||% NA_real_,
      lower_bound    = p$lower %||% NA_real_,
      upper_bound    = p$upper %||% NA_real_,
      q_trim         = p$q_trim %||% NA_real_,
      n_trimmed      = p$n_trimmed %||% 0L,
      median_value   = if (is_bin) NA_real_ else (p$median_value %||% NA_real_),
      ci_low_lower   = unname(res$ci_low[cv]),
      ci_high_upper  = unname(res$ci_high[cv]),
      r1 = p$r1 %||% NA_real_,
      r0 = p$r0 %||% NA_real_,
      convergence    = unname(res$convergence[cv]),
      msg = p$msg %||% NA_character_
    )
  })
  rbindlist(rows)
}
tab_a <- to_table(res_a, "adapt")
tab_m <- to_table(res_m, "miti")
fwrite(tab_a, file.path(out_dir, "bounds_adapt.csv"))
fwrite(tab_m, file.path(out_dir, "bounds_miti.csv"))

# Per-covariate diagnostic summary (q value, median split, convergence)
diag_summary <- list(
  adapt = data.table(
    covariate = COEFS,
    is_binary = as.integer(COEFS %in% COEFS_BINARY),
    q_trim    = sapply(COEFS, function(cv) res_a$point[[cv]]$q_trim),
    median_value = sapply(COEFS, function(cv) {
      if (cv %in% COEFS_BINARY) NA_real_ else res_a$point[[cv]]$median_value
    }),
    n_trimmed    = sapply(COEFS, function(cv) res_a$point[[cv]]$n_trimmed),
    convergence  = unname(res_a$convergence[COEFS])
  ),
  miti = data.table(
    covariate = COEFS,
    is_binary = as.integer(COEFS %in% COEFS_BINARY),
    q_trim    = sapply(COEFS, function(cv) res_m$point[[cv]]$q_trim),
    median_value = sapply(COEFS, function(cv) {
      if (cv %in% COEFS_BINARY) NA_real_ else res_m$point[[cv]]$median_value
    }),
    n_trimmed    = sapply(COEFS, function(cv) res_m$point[[cv]]$n_trimmed),
    convergence  = unname(res_m$convergence[COEFS])
  )
)

saveRDS(list(adapt = res_a, miti = res_m,
             n_boot_target = N_BOOT,
             expanded_covariates = COEFS,
             binary_covariates = COEFS_BINARY,
             continuous_covariates = COEFS_CONTINUOUS,
             diag_summary = diag_summary,
             runtime_sec = as.numeric(difftime(Sys.time(), t0, units = "secs"))),
        file.path(out_dir, "bounds_diagnostics.rds"))
message(sprintf("[%s] Task C done in %.1f s", format(Sys.time()),
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
