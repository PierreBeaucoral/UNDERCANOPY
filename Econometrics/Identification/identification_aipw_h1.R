# =============================================================================
# scripts/identification_aipw_h1.R
# Path 2 — h1 (selection equation) version of AIPW with trim-stability.
#
# Mirror of identification_aipw.R but with binary outcome (any_flow) instead of
# log-positive-flow. Outcome regression is OLS-on-binary (linear-probability),
# which gives a coefficient with the same sign and significance interpretation
# as a probit but is faster and avoids convergence pitfalls under bootstrapping.
#
# N_BOOT = 100 here for tractability (vs 300 in h2 version). The expanded
# coverage of 25 covariates × 2 panels × 3 trim grids × 100 reps takes ~25 min.
#
# Outputs: regressions/aipw_h1/coef_adapt.csv
#          regressions/aipw_h1/coef_miti.csv
#          regressions/aipw_h1/trim_stability.csv
#          regressions/aipw_h1/diagnostics.rds
# =============================================================================

suppressPackageStartupMessages({
  library(here); library(data.table); library(dplyr)
})
set.seed(20260503L)

ROOT <- here::here()
source(here::here("Econometrics", "Identification", "_helpers.R"))
out_dir <- here::here("Econometrics", "regressions", "aipw_h1")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

t0 <- Sys.time()
N_BOOT <- as.integer(Sys.getenv("UC_N_BOOT", unset = "100"))
stopifnot(!is.na(N_BOOT), N_BOOT >= 1L)   # guard against a malformed UC_N_BOOT override
TRIM_GRID <- list(c(0.01, 0.99), c(0.05, 0.95), c(0.10, 0.90))
APE_VARS <- EXPANDED_COVARS
EPS_PROB_LOCAL <- 1e-12

message("[", format(Sys.time()), "] Path 2 / h1 AIPW (",
        length(APE_VARS), " covariates, N_BOOT=", N_BOOT, ")")

# ── Build pooled Han ∪ BERT panels at dyad-year (same as h2 script) ──────────
reg1  <- fread(here::here("Econometrics", "Data", "reg1.csv"),  na.strings = c("", "NA"))
reg3  <- fread(here::here("Econometrics", "Data", "reg3.csv"),  na.strings = c("", "NA"))
reg1m <- fread(here::here("Econometrics", "Data", "reg1_mitigation.csv"), na.strings = c("", "NA"))
reg3m <- fread(here::here("Econometrics", "Data", "reg3_mitigation.csv"), na.strings = c("", "NA"))

agg_han_a  <- aggregate_dyad_year(reg1,  "AdaptAmount", BASELINE_RECIPIENT_HAN)
agg_bert_a <- aggregate_dyad_year(reg3,  "AdaptAmount", BASELINE_RECIPIENT_BERT)
agg_han_m  <- aggregate_dyad_year(reg1m, "MitiAmount",  BASELINE_RECIPIENT_HAN)
agg_bert_m <- aggregate_dyad_year(reg3m, "MitiAmount",  BASELINE_RECIPIENT_BERT)

build_pool <- function(han, bert) {
  han[,  IN_BERT := 0L]; bert[, IN_BERT := 1L]
  cols <- intersect(names(han), names(bert))
  rbindlist(list(han[, ..cols], bert[, ..cols]), use.names = TRUE)
}
pool_a <- build_pool(copy(agg_han_a),  copy(agg_bert_a))
pool_m <- build_pool(copy(agg_han_m),  copy(agg_bert_m))

# ── h1 AIPW: linear-probability OLS on any_flow with AIPW reweighting ────────
# Outcome is binary; do NOT filter to positive-flow cells (that's the h2 case).
aipw_h1_one_coef <- function(pooled, coef_name, covars,
                             trim_lo = 0.01, trim_hi = 0.99) {
  pooled <- as.data.table(pooled)
  if (!"any_flow" %in% names(pooled)) {
    return(list(estimate = NA_real_, se = NA_real_, n_used = 0L, n_trimmed = 0L))
  }
  pooled <- pooled[complete.cases(pooled[, c("any_flow", covars, coef_name, "IN_BERT"),
                                          with = FALSE])]
  n0 <- nrow(pooled)
  if (n0 < 200L) {
    return(list(estimate = NA_real_, se = NA_real_, n_used = n0, n_trimmed = 0L))
  }
  # Propensity model on full union sample (all cells, not just positive flows)
  rhs_ps <- paste(intersect(c(covars, coef_name), names(pooled)),
                  collapse = " + ")
  ps_fit <- suppressWarnings(glm(as.formula(paste0("IN_BERT ~ ", rhs_ps)),
                                 family = binomial(link = "logit"),
                                 data = pooled))
  phat <- predict(ps_fit, type = "response")
  phat <- pmin(pmax(phat, EPS_PROB_LOCAL), 1 - EPS_PROB_LOCAL)
  keep <- phat >= trim_lo & phat <= trim_hi
  n_trim <- sum(!keep)
  d <- pooled[keep]; phat <- phat[keep]

  # Outcome model on BERT subset (linear probability)
  rhs_y <- paste(intersect(c(covars, coef_name), names(d)), collapse = " + ")
  y_fit <- lm(as.formula(paste0("any_flow ~ ", rhs_y)),
              data = d[IN_BERT == 1L])
  mu_hat <- predict(y_fit, newdata = d)

  y <- as.numeric(d$any_flow); s <- d$IN_BERT
  y_dr <- mu_hat + (s / phat) * (y - mu_hat)
  d2 <- copy(d); d2[, y_dr := y_dr]
  rhs_y2 <- paste(intersect(c(covars, coef_name), names(d2)), collapse = " + ")
  dr_fit <- lm(as.formula(paste0("y_dr ~ ", rhs_y2)), data = d2)
  est <- unname(coef(dr_fit)[coef_name])
  Xmat <- model.matrix(dr_fit); res <- residuals(dr_fit)
  bread <- solve(crossprod(Xmat) / nrow(Xmat))
  meat  <- crossprod(Xmat * res) / nrow(Xmat)
  vcov_hc0 <- (bread %*% meat %*% bread) / nrow(Xmat)
  se <- sqrt(vcov_hc0[coef_name, coef_name])
  list(estimate = est, se = unname(se), n_used = nrow(d2), n_trimmed = n_trim)
}

# ── Bootstrap with cluster on recipient ──────────────────────────────────────
boot_aipw_h1 <- function(pool, headline_coefs, trim_lo, trim_hi,
                         n_boot = N_BOOT) {
  pool <- as.data.table(pool)
  est_pt <- numeric(length(headline_coefs)); names(est_pt) <- headline_coefs
  for (cv in headline_coefs) {
    others <- setdiff(DYAD_COVARS, cv)
    res <- tryCatch(
      aipw_h1_one_coef(pool, cv, covars = others,
                       trim_lo = trim_lo, trim_hi = trim_hi),
      error = function(e) list(estimate = NA_real_)
    )
    est_pt[cv] <- res$estimate
  }
  rec_levels <- unique(pool$recipient)
  boot_mat <- matrix(NA_real_, nrow = n_boot, ncol = length(headline_coefs),
                     dimnames = list(NULL, headline_coefs))
  for (b in seq_len(n_boot)) {
    set.seed(20260503L + b)
    samp <- sample(rec_levels, length(rec_levels), replace = TRUE)
    keys_b <- data.table(recipient = samp)
    p_b <- pool[keys_b, on = "recipient", nomatch = NULL, allow.cartesian = TRUE]
    for (cv in headline_coefs) {
      others <- setdiff(DYAD_COVARS, cv)
      res <- tryCatch(
        aipw_h1_one_coef(p_b, cv, covars = others,
                         trim_lo = trim_lo, trim_hi = trim_hi),
        error = function(e) list(estimate = NA_real_)
      )
      boot_mat[b, cv] <- res$estimate
    }
    if (b %% 25L == 0L) message("    boot ", b, "/", n_boot)
  }
  conv <- apply(boot_mat, 2L,
                function(v) sum(is.finite(v)) / length(v))
  list(point = est_pt,
       se    = apply(boot_mat, 2L, sd, na.rm = TRUE),
       convergence = conv,
       n_boot_used = n_boot)
}

trim_counts <- function(pool, trim_lo, trim_hi) {
  pool <- pool[complete.cases(pool[, c(DYAD_COVARS, "IN_BERT"), with = FALSE])]
  rhs <- paste(intersect(DYAD_COVARS, names(pool)), collapse = " + ")
  ps_fit <- suppressWarnings(glm(as.formula(paste0("IN_BERT ~ ", rhs)),
                                 family = binomial(link = "logit"),
                                 data = pool))
  phat <- predict(ps_fit, type = "response")
  sum(phat < trim_lo | phat > trim_hi)
}

run_trim_stability_h1 <- function(pool, label) {
  message("  >>> ", label, ": ", length(TRIM_GRID), " trim grids x ",
          length(APE_VARS), " coefs x ", N_BOOT, " reps")
  rows <- list(); conv_all <- list()
  for (tg in TRIM_GRID) {
    t_grid <- Sys.time()
    message("      trim = [", tg[1L], ", ", tg[2L], "]")
    n_trim <- trim_counts(pool, tg[1L], tg[2L])
    res <- boot_aipw_h1(pool, APE_VARS, tg[1L], tg[2L], N_BOOT)
    rows[[length(rows) + 1L]] <- data.table(
      trim_lo = tg[1L], trim_hi = tg[2L], n_trimmed = n_trim,
      covariate = APE_VARS,
      estimate = res$point, se = res$se,
      convergence = unname(res$convergence[APE_VARS])
    )
    conv_all[[paste(label, tg[1L], tg[2L], sep = "_")]] <- res$convergence
    message(sprintf("      done in %.1f s",
                    as.numeric(difftime(Sys.time(), t_grid, units = "secs"))))
  }
  out <- rbindlist(rows); out[, label := label]
  attr(out, "convergence") <- conv_all
  out
}

trim_a <- run_trim_stability_h1(pool_a, "adapt")
trim_m <- run_trim_stability_h1(pool_m, "miti")
trim_all <- rbind(trim_a, trim_m)
fwrite(trim_all, file.path(out_dir, "trim_stability.csv"))

build_headline <- function(trim_dt, lab) {
  d <- trim_dt[label == lab & trim_lo == 0.01 & trim_hi == 0.99]
  d[, .(covariate, estimate, se,
        ci_low  = estimate - 1.96 * se,
        ci_high = estimate + 1.96 * se,
        n_trimmed = n_trimmed)]
}
fwrite(build_headline(trim_a, "adapt"), file.path(out_dir, "coef_adapt.csv"))
fwrite(build_headline(trim_m, "miti"),  file.path(out_dir, "coef_miti.csv"))

conv_a <- attr(trim_a, "convergence")
conv_m <- attr(trim_m, "convergence")
saveRDS(list(
  trim_grid = TRIM_GRID, n_boot_target = N_BOOT,
  expanded_covariates = APE_VARS,
  convergence_per_covariate = list(
    adapt = sapply(APE_VARS, function(cv) mean(sapply(conv_a, function(v) v[cv]), na.rm = TRUE)),
    miti  = sapply(APE_VARS, function(cv) mean(sapply(conv_m, function(v) v[cv]), na.rm = TRUE))
  ),
  runtime_sec = as.numeric(difftime(Sys.time(), t0, units = "secs"))
), file.path(out_dir, "diagnostics.rds"))

message(sprintf("[%s] h1 AIPW done in %.1f min", format(Sys.time()),
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))
