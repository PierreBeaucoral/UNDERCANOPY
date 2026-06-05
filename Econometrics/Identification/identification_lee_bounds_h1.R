# =============================================================================
# scripts/identification_lee_bounds_h1.R
# Path 2 — h1 (selection equation) Lee (2009) trimming bounds.
#
# Mirror of identification_lee_bounds.R but on the binary outcome `any_flow`
# (= cell has any positive climate finance) instead of log-positive-amount.
# Uses linear-probability OLS on the binary outcome — same sign and significance
# interpretation as a probit but tractable under the Lee trimming scheme.
#
# For each covariate (binary or median-dichotomised continuous):
#   1. Compute differential BERT retention rate q across the two groups.
#   2. Lower bound: trim the over-retained group's any_flow=1 cells by q.
#      Refit OLS on the trimmed sample.
#   3. Upper bound: trim the over-retained group's any_flow=0 cells by q.
#      Refit OLS on the trimmed sample.
#
# N_BOOT = 100 here (vs 300 in h2 version) for tractability.
#
# Outputs: regressions/lee_bounds_h1/bounds_adapt.csv
#          regressions/lee_bounds_h1/bounds_miti.csv
#          regressions/lee_bounds_h1/bounds_diagnostics.rds
# =============================================================================

suppressPackageStartupMessages({
  library(here); library(data.table); library(dplyr)
})
set.seed(20260503L)

ROOT <- here::here()
source(here::here("Econometrics", "Identification", "_helpers.R"))
out_dir <- here::here("Econometrics", "regressions", "lee_bounds_h1")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

t0 <- Sys.time()
N_BOOT <- 100L
COEFS <- EXPANDED_COVARS
COEFS_BINARY <- EXPANDED_COVARS_BINARY
COEFS_CONTINUOUS <- EXPANDED_COVARS_CONTINUOUS

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

# ── Lee h1 (binary outcome): retention asymmetry × OLS-on-binary trimming ────
lee_h1_one <- function(bert, han, var, is_binary, covars, flow_col,
                       median_value = NULL) {
  bert <- as.data.table(bert); han <- as.data.table(han)
  if (!var %in% names(bert) || !var %in% names(han)) {
    return(list(point = NA_real_, lower = NA_real_, upper = NA_real_,
                q_trim = NA_real_, n_trimmed = 0L,
                median_value = NA_real_, r1 = NA_real_, r0 = NA_real_,
                msg = "missing column"))
  }
  if (is_binary) {
    grp_b <- as.integer(bert[[var]])
    grp_h <- as.integer(han[[var]])
  } else {
    if (is.null(median_value)) {
      median_value <- median(c(bert[[var]], han[[var]]), na.rm = TRUE)
    }
    grp_b <- as.integer(bert[[var]] > median_value)
    grp_h <- as.integer(han[[var]] > median_value)
  }
  n_b1 <- sum(grp_b == 1L, na.rm = TRUE); n_h1 <- sum(grp_h == 1L, na.rm = TRUE)
  n_b0 <- sum(grp_b == 0L, na.rm = TRUE); n_h0 <- sum(grp_h == 0L, na.rm = TRUE)
  r1 <- if (n_h1 > 0L) n_b1 / n_h1 else NA_real_
  r0 <- if (n_h0 > 0L) n_b0 / n_h0 else NA_real_
  if (is.na(r1) || is.na(r0)) {
    return(list(point = NA_real_, lower = NA_real_, upper = NA_real_,
                q_trim = NA_real_, n_trimmed = 0L,
                median_value = if (is_binary) NA_real_ else median_value,
                r1 = r1, r0 = r0,
                msg = "tight: degenerate retention (one cell empty)"))
  }
  if (r1 >= r0) {
    over_is_one <- TRUE
    q <- (r1 - r0) / r1
  } else {
    over_is_one <- FALSE
    q <- (r0 - r1) / r0
  }
  # Point: untrimmed BERT OLS on binary outcome
  rhs <- paste(intersect(c(covars, var), names(bert)), collapse = " + ")
  bert_use <- bert[complete.cases(bert[, c(flow_col, covars, var), with = FALSE])]
  if (nrow(bert_use) < 100L) {
    return(list(point = NA_real_, lower = NA_real_, upper = NA_real_,
                q_trim = q, n_trimmed = 0L,
                median_value = if (is_binary) NA_real_ else median_value,
                r1 = r1, r0 = r0, msg = "too few obs after NA drop"))
  }
  pt_fit <- tryCatch(lm(as.formula(paste0(flow_col, " ~ ", rhs)), data = bert_use),
                     error = function(e) NULL)
  pt <- if (!is.null(pt_fit)) unname(coef(pt_fit)[var]) else NA_real_
  if (q < 1e-6) {
    return(list(point = pt, lower = pt, upper = pt,
                q_trim = q, n_trimmed = 0L,
                median_value = if (is_binary) NA_real_ else median_value,
                r1 = r1, r0 = r0, msg = "tight: q approx 0"))
  }
  # Trim the over-retained group's any_flow=1 (lower) or any_flow=0 (upper)
  if (is_binary) {
    over_grp <- as.integer(bert_use[[var]] == over_is_one)
  } else {
    over_grp <- as.integer((bert_use[[var]] > median_value) == over_is_one)
  }
  out_y <- bert_use[[flow_col]]
  trim_lower <- function() {
    idx_over_pos <- which(over_grp == 1L & out_y == 1L)
    if (!length(idx_over_pos)) return(NA_real_)
    n_drop <- min(length(idx_over_pos), max(1L, round(q * length(idx_over_pos))))
    drop_set <- sample(idx_over_pos, n_drop)   # random; same seed across reps
    keep <- setdiff(seq_len(nrow(bert_use)), drop_set)
    fit <- tryCatch(lm(as.formula(paste0(flow_col, " ~ ", rhs)),
                       data = bert_use[keep]), error = function(e) NULL)
    if (is.null(fit)) NA_real_ else unname(coef(fit)[var])
  }
  trim_upper <- function() {
    idx_over_neg <- which(over_grp == 1L & out_y == 0L)
    if (!length(idx_over_neg)) return(NA_real_)
    n_drop <- min(length(idx_over_neg), max(1L, round(q * length(idx_over_neg))))
    drop_set <- sample(idx_over_neg, n_drop)
    keep <- setdiff(seq_len(nrow(bert_use)), drop_set)
    fit <- tryCatch(lm(as.formula(paste0(flow_col, " ~ ", rhs)),
                       data = bert_use[keep]), error = function(e) NULL)
    if (is.null(fit)) NA_real_ else unname(coef(fit)[var])
  }
  lo <- trim_lower(); hi <- trim_upper()
  if (!is.na(lo) && !is.na(hi) && lo > hi) { tmp <- lo; lo <- hi; hi <- tmp }
  list(point = pt, lower = lo, upper = hi,
       q_trim = q, n_trimmed = round(q *
                                       sum(over_grp == 1L & out_y %in% c(0L,1L))),
       median_value = if (is_binary) NA_real_ else median_value,
       r1 = r1, r0 = r0, msg = "ok")
}

run_lee_h1_pack <- function(bert, han, flow_col) {
  out <- list()
  for (cv in COEFS) {
    is_bin <- cv %in% COEFS_BINARY
    covars_lm <- setdiff(DYAD_COVARS, cv)
    res <- tryCatch(
      lee_h1_one(bert, han, var = cv, is_binary = is_bin,
                 covars = covars_lm, flow_col = flow_col),
      error = function(e) list(point = NA_real_, lower = NA_real_,
                               upper = NA_real_, q_trim = NA_real_,
                               n_trimmed = 0L, median_value = NA_real_,
                               r1 = NA_real_, r0 = NA_real_,
                               msg = paste0("error: ", conditionMessage(e)))
    )
    out[[cv]] <- res
  }
  out
}

boot_lee_h1 <- function(bert, han, flow_col) {
  rec_levels <- unique(c(bert$recipient, han$recipient))
  res_pt <- run_lee_h1_pack(bert, han, flow_col)
  boot_lower <- matrix(NA_real_, nrow = N_BOOT, ncol = length(COEFS),
                       dimnames = list(NULL, COEFS))
  boot_upper <- boot_lower
  for (b in seq_len(N_BOOT)) {
    set.seed(20260503L + b)
    samp <- sample(rec_levels, length(rec_levels), replace = TRUE)
    keys_b <- data.table(recipient = samp)
    bt_b <- bert[keys_b, on = "recipient", nomatch = NULL, allow.cartesian = TRUE]
    ht_b <- han [keys_b, on = "recipient", nomatch = NULL, allow.cartesian = TRUE]
    res_b <- tryCatch(run_lee_h1_pack(bt_b, ht_b, flow_col), error = function(e) NULL)
    if (is.null(res_b)) next
    for (cv in COEFS) {
      boot_lower[b, cv] <- res_b[[cv]]$lower
      boot_upper[b, cv] <- res_b[[cv]]$upper
    }
    if (b %% 25L == 0L) message("    boot ", b, "/", N_BOOT)
  }
  ci_low_l  <- apply(boot_lower, 2L, quantile, probs = 0.025, na.rm = TRUE)
  ci_high_u <- apply(boot_upper, 2L, quantile, probs = 0.975, na.rm = TRUE)
  conv <- sapply(COEFS, function(cv) {
    sum(is.finite(boot_lower[, cv]) & is.finite(boot_upper[, cv])) / N_BOOT
  })
  list(point = res_pt, ci_low = ci_low_l, ci_high = ci_high_u,
       convergence = conv, n_boot_used = N_BOOT)
}

message("  >>> adaptation h1: ", N_BOOT, " bootstrap reps")
res_a <- boot_lee_h1(agg_bert_a, agg_han_a, "any_flow_adapt")
message("  >>> mitigation h1: ", N_BOOT, " bootstrap reps")
res_m <- boot_lee_h1(agg_bert_m, agg_han_m, "any_flow_miti")

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a
to_table <- function(res, label) {
  rbindlist(lapply(COEFS, function(cv) {
    p <- res$point[[cv]]
    is_bin <- cv %in% COEFS_BINARY
    data.table(
      label = label, covariate = cv, is_binary = as.integer(is_bin),
      point_estimate = p$point %||% NA_real_,
      lower_bound = p$lower %||% NA_real_,
      upper_bound = p$upper %||% NA_real_,
      q_trim = p$q_trim %||% NA_real_,
      n_trimmed = p$n_trimmed %||% 0L,
      median_value = if (is_bin) NA_real_ else (p$median_value %||% NA_real_),
      ci_low_lower = unname(res$ci_low[cv]),
      ci_high_upper = unname(res$ci_high[cv]),
      r1 = p$r1 %||% NA_real_, r0 = p$r0 %||% NA_real_,
      convergence = unname(res$convergence[cv]),
      msg = p$msg %||% NA_character_
    )
  }))
}
fwrite(to_table(res_a, "adapt"), file.path(out_dir, "bounds_adapt.csv"))
fwrite(to_table(res_m, "miti"),  file.path(out_dir, "bounds_miti.csv"))

saveRDS(list(
  n_boot_target = N_BOOT,
  expanded_covariates = COEFS,
  runtime_sec = as.numeric(difftime(Sys.time(), t0, units = "secs"))
), file.path(out_dir, "bounds_diagnostics.rds"))

message(sprintf("[%s] h1 Lee done in %.1f min", format(Sys.time()),
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))
