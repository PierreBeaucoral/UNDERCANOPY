# =============================================================================
# scripts/marginal_effects_ape.R
# Task D — Average partial effects (APE) for the BERT main spec
#
# Implements §D of strategy_review_2026-05-03_identification.md: compute the
# unconditional, selection, and intensive APE on the BERT-only sample at the
# native sector level (i,j,k,t). Effects are standardised (1 SD for continuous,
# 0->1 for binary) and translated into "$ million per dyad-sector-year".
#
# ------------------------------------------------------------------------------
# 2026-05-04 RERUN — three fixes applied vs the prior production version:
#
#   FIX 1. Provider, recipient, and year fixed effects are now included as
#          additive controls in BOTH the selection and the outcome equations of
#          the Cragg double hurdle. The country-level aggregates that the FE
#          block absorbs (ProviderGDPtot, ProviderPop, Providerdebt, ProviderfisB
#          and the analogous Recipient-side variables) are dropped to avoid the
#          near-collinearity that destroyed the Hessian (see smoke test in
#          explorations/ape_rerun_2026-05). Provider/recipient/year columns that
#          are constant in the sample, all-zero on positives, or have fewer than
#          MIN_FE_CELLS observations in either margin are also dropped.
#
#   FIX 2. The conditional intensive APE is now the ANALYTICAL conditional
#          partial of the Cragg-tobit-normal hurdle, replacing the unstable
#          mhurdle::predict(..., what = "Ep") path. Under dist = "n",
#          corr = FALSE:
#              E[y | y > 0, x] = x'beta2 + sigma * lambda(z),  z = x'beta2 / sigma
#              lambda(z) = phi(z) / Phi(z)
#              d E[y | y > 0, x] / d x_k = beta2_k * { 1 - lambda(z) * (z + lambda(z)) }
#          For binary covariates we use the discrete-change form
#              E[y | y > 0, x_k = 1] - E[y | y > 0, x_k = 0]
#          with both expectations computed analytically and averaged over the
#          empirical distribution of the OTHER covariates restricted to
#          positive-flow rows (the right empirical distribution for a
#          conditional-on-positive APE).
#
#   FIX 3. Bootstrap replicate count raised from 100 to 300 (clustered on
#          recipient). Per-replicate seed = SEED_BASE + b. Parallelised across
#          recipient-cluster resamples via future.apply::future_lapply. If the
#          full 300 reps would exceed the 3-hour wall-clock budget we fall back
#          to 200 reps and document the fallback in the diagnostics RDS.
#
# Inputs : Econometrics/Data/reg3.csv, Econometrics/Data/reg3_mitigation.csv
# Outputs: regressions/ape/ape_adapt.csv, regressions/ape/ape_miti.csv
#          regressions/ape/ape_diagnostics.rds
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(mhurdle)
  library(future)
  library(future.apply)
  library(here)
})

set.seed(20260503L)
SEED_BASE <- 20260503L

ROOT <- here::here()
out_dir <- here::here("Econometrics", "regressions", "ape")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

t0 <- Sys.time()

# ── Constants ────────────────────────────────────────────────────────────────
N_BOOT <- 300L                   # FIX 3 — production target.
N_BOOT_FALLBACK <- 200L          # If 300-rep budget overruns we drop to this.
WALL_CLOCK_BUDGET_SEC <- 3 * 3600  # 3 hours total
MIN_FE_CELLS <- 30L              # Drop FE dummies with <30 obs in either margin
MIN_BOOT_OBS <- 200L             # Skip bootstrap reps with <200 complete rows
N_WORKERS <- max(1L, parallel::detectCores() - 2L)

# Headline coefficients reported in the APE table. The APE table is a
# magnitudes table — it speaks to allocation, not to entry. Row inclusion is
# therefore gated by the allocation (h2) margin: a variable enters APE_VARS
# iff at least one of its h2 cells is sustained, where sustained means either
# (a) §5.4 verdict "Sustained (classification)" or (b) stable_sig at h2 in
# regressions/var_selection/changing_audit.csv.
#
# 2026-05-12 refactor: previous APE_VARS included WRI, EIA, and CPIAbudget on
# substantive-policy-interest grounds. Under the new rule:
#   - WRI Vulnerability: stable_sig at h1 only (adapt_h1, miti_h1). Robustness
#     at the entry margin is real and is reported in §4 and §5.4; the h2
#     allocation gain is CHANGING:gain and not sustained in §5.4. WRI carries
#     no allocation-magnitude claim and is therefore not in the magnitudes
#     table.
#   - EIA: §5.4 sustained loss-of-significance at h1 only (adapt and miti);
#     status_*_h2 is CHANGING:gain (adapt) and stable_ns (miti). No allocation
#     magnitude claim.
#   - CPIAbudget: no stable_sig anywhere and not sustained at any §5.4 cell.
# One variable added because it carries a sustained h2 margin: InvestAgree
# (stable_sig at miti_h2 and adapt_h1; the relevant gating cell is miti_h2).
#
# 2026-05-12 carve-out (numerical stability): NDCActOnly and NDCnonGHG also
# satisfy the h2-sustained rule (stable_sig at miti_h2). However, they are
# constructed in HurdleRegHuei.R:63-65 as time-localised dummies set to 1 only
# in the NDC submission year, so year-FE absorption leaves very little
# within-cell variance. Including both in the FE-augmented APE specification
# dropped mitigation bootstrap convergence from 72% (2026-05-11 run, prior
# APE_VARS) to 27% (2026-05-12 first rerun) — well below the 80% gate. The
# dummies are therefore dropped from APE_VARS on numerical-stability grounds,
# parallel to how country-level aggregates (ProviderPop, ProviderfisB, etc.)
# are dropped on FE-collinearity grounds. The NDC story remains in §4 and §5.4.
#
# The h1-only-sustained findings remain visible in the §4 coefficient
# comparison and the §5.4 verdicts table by design — the APE table delivers
# dollar magnitudes only.
APE_VARS <- c("colony", "comlang", "comrelig",
              "CPIAPublicAdm", "distw",
              "MDBDummy", "InvestAgree",
              "RecipientGDPtot", "ProviderGDPtot", "Providerdebt")
BINARY_VARS <- c("colony", "comlang", "comrelig", "EIA", "MDBDummy",
                 "InvestAgree")

# Economic covariates retained alongside the FE block. The country-level
# aggregates (ProviderPop, ProviderfisB and analogous Recipient-side aggregates)
# remain dropped because they are nearly explained by the provider/recipient
# FE — keeping them produced a singular Hessian (see smoke test).
#
# NDCActOnly and NDCnonGHG are also dropped from the control set: even when
# they are not reported, including them in the FE-augmented regression killed
# mitigation bootstrap convergence (27% in the 2026-05-12 first rerun). WRI,
# EIA, CPIAbudget, and wto remain in the regression as controls — preserves §4
# main-spec parity, even though these variables are no longer reported in the
# magnitudes table.
ECON_COVARS <- c(
  "RecipientWRIVul", "CPIAPublicAdm", "CPIAbudget",
  "colony", "comlang", "MDBDummy", "distw",
  "comrelig", "wto", "EIA", "InvestAgree",
  "RecipientGDPtot", "ProviderGDPtot", "Providerdebt"
)

EPS_PROB <- 1e-12   # CDF clamping per project numerical-discipline rule

# ── Helpers ──────────────────────────────────────────────────────────────────

#' Prepare a BERT panel: rename outcome to y_h, force non-negative, attach
#' recipient label for cluster bootstrap.
#'
#' @param d  data.table loaded from reg3.csv or reg3_mitigation.csv (Econometrics/Data/)
#' @param outcome character ("AdaptAmount" or "MitiAmount")
#' @return data.table with y_h column and `recipient` factor
prep_panel <- function(d, outcome) {
  d <- as.data.table(d)
  setnames(d, outcome, "y_h", skip_absent = TRUE)
  d[, y_h := pmax(0, y_h)]
  rec_cols <- grep("^RecipientISO_", names(d), value = TRUE)
  if (length(rec_cols) > 0L) {
    M <- as.matrix(d[, rec_cols, with = FALSE])
    storage.mode(M) <- "integer"
    has_one <- rowSums(M) > 0L
    rec <- rep("BASELINE", nrow(M))
    if (any(has_one)) {
      idx <- max.col(M[has_one, , drop = FALSE], ties.method = "first")
      rec[has_one] <- sub("RecipientISO_", "", rec_cols[idx])
    }
    d[, recipient := rec]
  } else {
    d[, recipient := "BASELINE"]
  }
  d
}

#' Identify FE columns to drop from a candidate set (constant overall, all-zero
#' on positives, or fewer than MIN_FE_CELLS observations in either margin).
#'
#' @param d         data.table with y_h
#' @param fe_cols   character vector of candidate FE column names
#' @param min_cells integer, minimum cells in EACH margin (positives vs zeros)
#' @return character vector of columns to drop
identify_fe_drops <- function(d, fe_cols, min_cells = MIN_FE_CELLS) {
  if (length(fe_cols) == 0L) return(character(0))
  pos <- d$y_h > 0
  drops <- character(0)
  for (c in fe_cols) {
    v <- d[[c]]
    if (is.null(v)) next
    n_pos <- sum(v == 1L & pos)
    n_zer <- sum(v == 1L & !pos)
    n_one <- n_pos + n_zer
    if (n_one == 0L) {                          # constant 0 column
      drops <- c(drops, c); next
    }
    if (length(unique(v)) < 2L) {               # constant overall
      drops <- c(drops, c); next
    }
    if (n_pos < min_cells || n_zer < min_cells) # sparse
      drops <- c(drops, c)
  }
  unique(drops)
}

#' Build the full RHS for the Cragg double-hurdle on a (sub)sample.
#'
#' @param d data.table containing y_h plus all candidate covariates
#' @return list(rhs = char vector of column names actually used,
#'              fe_dropped = char vector of FE cols dropped)
build_rhs <- function(d) {
  econ <- intersect(ECON_COVARS, names(d))
  prov_fe <- grep("^ProviderISO_",  names(d), value = TRUE)
  rec_fe  <- grep("^RecipientISO_", names(d), value = TRUE)
  yr_fe   <- grep("^Year_",         names(d), value = TRUE)
  fe_cand <- c(prov_fe, rec_fe, yr_fe)
  fe_drop <- identify_fe_drops(d, fe_cand, MIN_FE_CELLS)
  fe_keep <- setdiff(fe_cand, fe_drop)
  list(rhs = c(econ, fe_keep), fe_dropped = fe_drop, n_fe = length(fe_keep))
}

#' Fit the Cragg double hurdle (dist = "n", corr = FALSE) on a (sub)sample.
#'
#' Recomputes the RHS each call so that bootstrap replicates handle their own
#' sample-specific sparsity. Returns NULL fit on convergence failure.
#'
#' @param d data.table with y_h and all candidate covariates
#' @return list(fit, rhs, n)  with fit = NULL on failure
fit_hurdle_ape <- function(d) {
  d <- as.data.table(d)
  econ <- intersect(ECON_COVARS, names(d))
  d <- d[complete.cases(d[, c("y_h", econ), with = FALSE])]
  if (nrow(d) < MIN_BOOT_OBS)
    return(list(fit = NULL, rhs = character(0), n = nrow(d), msg = "too few rows"))
  spec <- build_rhs(d)
  rhs_str <- paste(spec$rhs, collapse = " + ")
  f <- as.formula(paste0("y_h ~ ", rhs_str, " | ", rhs_str))
  fit <- tryCatch(
    suppressWarnings(
      mhurdle::mhurdle(f, data = as.data.frame(d), dist = "n",
                       method = "bfgs", corr = FALSE)
    ),
    error = function(e) e)
  if (inherits(fit, "error"))
    return(list(fit = NULL, rhs = spec$rhs, n = nrow(d),
                msg = conditionMessage(fit)))
  list(fit = fit, rhs = spec$rhs, n = nrow(d), msg = "ok",
       fe_dropped = spec$fe_dropped)
}

#' Build the design matrix used by the OUTCOME equation (h2) of an mhurdle fit.
#' Returns the columns in the order matching coef(fit, "h2") names so that
#' X %*% beta2 is well-defined.
build_h2_design <- function(d, rhs_cols, b2_names) {
  X <- cbind(`(Intercept)` = 1, as.matrix(d[, rhs_cols, with = FALSE]))
  storage.mode(X) <- "double"
  X[, b2_names, drop = FALSE]
}

#' Analytical conditional intensive APE for one covariate under the Cragg
#' double hurdle with normal errors and uncorrelated equations.
#'
#' For continuous covariates, returns
#'   beta2_k * E_pos[ 1 - lambda(z) * (z + lambda(z)) ] * sd(x_k)
#' i.e. the standardised per-1-SD analytical derivative averaged over the
#' positive-flow subsample.
#'
#' For binary covariates returns
#'   E_pos[ E[y | y > 0, x_k = 1] - E[y | y > 0, x_k = 0] ]
#' computed analytically (no mhurdle::predict call).
#'
#' @param fit  fitted mhurdle object (dist = "n", corr = FALSE)
#' @param d    data.table including all rhs_cols
#' @param rhs_cols character vector of RHS columns used in the fit
#' @param var  covariate name
#' @param is_binary logical
#' @param b2 numeric named vector = coef(fit, "h2"); pre-fetched for speed
#' @param sigma numeric scalar = coef(fit, "sd")
#' @return numeric scalar (the analytical conditional intensive APE)
ape_intensive_analytical <- function(d, rhs_cols, var, is_binary,
                                      b2, sigma) {
  X <- build_h2_design(d, rhs_cols, names(b2))
  pos <- d$y_h > 0
  if (!any(pos)) return(NA_real_)
  if (is_binary) {
    if (!var %in% colnames(X)) return(NA_real_)
    X1 <- X; X1[, var] <- 1
    X0 <- X; X0[, var] <- 0
    xb1 <- as.numeric(X1 %*% b2); xb0 <- as.numeric(X0 %*% b2)
    z1 <- xb1 / sigma;            z0 <- xb0 / sigma
    p1 <- pmin(pmax(pnorm(z1), EPS_PROB), 1 - EPS_PROB)
    p0 <- pmin(pmax(pnorm(z0), EPS_PROB), 1 - EPS_PROB)
    e1 <- xb1 + sigma * dnorm(z1) / p1
    e0 <- xb0 + sigma * dnorm(z0) / p0
    return(mean((e1 - e0)[pos], na.rm = TRUE))
  }
  # Continuous: per-1-SD derivative, averaged on positives
  if (!var %in% names(b2)) return(NA_real_)
  beta_k <- b2[[var]]
  sdv <- sd(d[[var]], na.rm = TRUE)
  if (!is.finite(sdv) || sdv <= 0) sdv <- 1
  xb <- as.numeric(X %*% b2)
  z  <- xb / sigma
  p  <- pmin(pmax(pnorm(z), EPS_PROB), 1 - EPS_PROB)
  lambda <- dnorm(z) / p
  d_e <- beta_k * (1 - lambda * (z + lambda))
  mean(d_e[pos], na.rm = TRUE) * sdv
}

#' Numerical APE for unconditional E[y] and selection P(y>0), using mhurdle's
#' built-in predict paths (which are well-behaved, unlike the "Ep" path).
#'
#' Returns the standardised effect (per 1-SD continuous or 0->1 binary).
ape_E_and_p_numerical <- function(fit, d, var, is_binary,
                                   perturb_frac = 0.01) {
  d <- as.data.frame(d)
  if (is_binary) {
    d0 <- d; d1 <- d
    d0[[var]] <- 0L; d1[[var]] <- 1L
    delta <- 1
    sdv <- 1
  } else {
    sdv <- sd(d[[var]], na.rm = TRUE)
    if (!is.finite(sdv) || sdv <= 0) sdv <- 1
    delta <- perturb_frac * sdv
    d0 <- d; d1 <- d
    d0[[var]] <- d[[var]] - delta / 2
    d1[[var]] <- d[[var]] + delta / 2
  }
  e0 <- tryCatch(predict(fit, newdata = d0, what = "E"), error = function(e) NULL)
  e1 <- tryCatch(predict(fit, newdata = d1, what = "E"), error = function(e) NULL)
  s0 <- tryCatch(predict(fit, newdata = d0, what = "p"), error = function(e) NULL)
  s1 <- tryCatch(predict(fit, newdata = d1, what = "p"), error = function(e) NULL)
  marg <- function(a, b) {
    if (is.null(a) || is.null(b)) NA_real_
    else mean((b - a) / delta, na.rm = TRUE)
  }
  unc <- marg(e0, e1)
  sel <- marg(s0, s1)
  # Standardise per 1-SD (continuous) or report as 0->1 (binary, where delta=1)
  if (is_binary) c(unc = unc, sel = sel)
  else c(unc = unc * sdv, sel = sel * sdv)
}

#' Compute all APEs (unconditional, selection, intensive) for one fit, returned
#' as a data.table indexed by covariate.
compute_ape_pack <- function(fit_obj, d) {
  fit <- fit_obj$fit
  if (is.null(fit)) {
    return(data.table(covariate = APE_VARS,
                      ape_unconditional = NA_real_,
                      ape_selection     = NA_real_,
                      ape_intensive     = NA_real_))
  }
  d <- as.data.table(d)
  d <- d[complete.cases(d[, c("y_h", fit_obj$rhs), with = FALSE])]
  b2    <- coef(fit, which = "h2")
  sigma <- coef(fit, which = "sd")
  if (length(sigma) != 1L) sigma <- sigma[1]
  rows <- vector("list", length(APE_VARS))
  for (i in seq_along(APE_VARS)) {
    v <- APE_VARS[i]
    is_b <- v %in% BINARY_VARS
    Ep_sel <- tryCatch(ape_E_and_p_numerical(fit, d, v, is_b),
                       error = function(e) c(unc = NA_real_, sel = NA_real_))
    inten <- tryCatch(ape_intensive_analytical(d, fit_obj$rhs, v, is_b,
                                                b2, sigma),
                      error = function(e) NA_real_)
    rows[[i]] <- data.table(covariate = v,
                            ape_unconditional = unname(Ep_sel["unc"]),
                            ape_selection     = unname(Ep_sel["sel"]),
                            ape_intensive     = inten)
  }
  rbindlist(rows)
}

# ── 1. Load BERT panels ──────────────────────────────────────────────────────
message("[", format(Sys.time()), "] Task D rerun: APEs on BERT main spec")
message("  N_BOOT (target) = ", N_BOOT, " | fallback = ", N_BOOT_FALLBACK,
        " | budget = ", WALL_CLOCK_BUDGET_SEC, "s | workers = ", N_WORKERS)

reg3  <- fread(here::here("Econometrics", "Data", "reg3.csv"),  na.strings = c("", "NA"))
reg3m <- fread(here::here("Econometrics", "Data", "reg3_mitigation.csv"),
               na.strings = c("", "NA"))
d_a <- prep_panel(reg3,  "AdaptAmount")
d_m <- prep_panel(reg3m, "MitiAmount")
message(sprintf("  adapt n=%d (positives=%.4f) | miti n=%d (positives=%.4f)",
                nrow(d_a), mean(d_a$y_h > 0),
                nrow(d_m), mean(d_m$y_h > 0)))

# ── 2. Fit main hurdle on full sample ────────────────────────────────────────
t_fit <- Sys.time()
main_a <- fit_hurdle_ape(d_a)
main_m <- fit_hurdle_ape(d_m)
fit_secs_total <- as.numeric(difftime(Sys.time(), t_fit, units = "secs"))
message(sprintf("  main fits done in %.1fs (adapt rhs=%d, msg=%s | miti rhs=%d, msg=%s)",
                fit_secs_total, length(main_a$rhs), main_a$msg,
                length(main_m$rhs), main_m$msg))

if (is.null(main_a$fit) || is.null(main_m$fit))
  stop("Main hurdle fit failed; cannot proceed with APE computation.")

pt_a <- compute_ape_pack(main_a, d_a)
pt_m <- compute_ape_pack(main_m, d_m)

# Sanity probe: the colony intensive APE on mitigation should NOT be in the
# tens-of-units range (the broken Ep predict path returned -43.79). Log it.
colony_int_m <- pt_m[covariate == "colony", ape_intensive]
colony_int_a <- pt_a[covariate == "colony", ape_intensive]
message(sprintf("  SANITY: analytical conditional intensive APE for colony — adapt=%.4f, miti=%.4f",
                colony_int_a, colony_int_m))

# ── 3. Cluster bootstrap on recipient ────────────────────────────────────────

#' Decide N_BOOT based on per-fit timing + remaining budget.
n_boot_decision <- function(seconds_per_fit, n_workers,
                            target = N_BOOT, fallback = N_BOOT_FALLBACK,
                            budget = WALL_CLOCK_BUDGET_SEC,
                            elapsed = 0) {
  remaining <- budget - elapsed
  per_rep_parallel <- seconds_per_fit / n_workers
  est_target  <- per_rep_parallel * target  + 60   # 60s overhead margin
  est_fallbk  <- per_rep_parallel * fallback + 60
  if (est_target <= remaining) {
    return(list(n = target, used_fallback = FALSE,
                est_seconds = est_target, remaining = remaining))
  }
  if (est_fallbk <= remaining) {
    return(list(n = fallback, used_fallback = TRUE,
                est_seconds = est_fallbk, remaining = remaining))
  }
  # In the very-tight case fall back even further but document it.
  feasible <- max(50L, as.integer((remaining - 60) / per_rep_parallel))
  list(n = feasible, used_fallback = TRUE,
       est_seconds = feasible * per_rep_parallel + 60, remaining = remaining)
}

#' One bootstrap replicate: draw recipient clusters with replacement, refit,
#' compute APEs.
#'
#' @return data.table with one row per APE_VARS covariate; NA on convergence
#'         failure.
boot_one <- function(d, b, rec_levels) {
  set.seed(SEED_BASE + b)
  samp <- sample(rec_levels, length(rec_levels), replace = TRUE)
  d_b <- d[data.table(recipient = samp), on = "recipient",
           nomatch = NULL, allow.cartesian = TRUE]
  fit_b <- fit_hurdle_ape(d_b)
  if (is.null(fit_b$fit)) {
    return(list(ape = data.table(covariate = APE_VARS,
                                 ape_unconditional = NA_real_,
                                 ape_selection     = NA_real_,
                                 ape_intensive     = NA_real_),
                ok = FALSE))
  }
  out <- compute_ape_pack(fit_b, d_b)
  list(ape = out, ok = TRUE)
}

run_bootstrap <- function(d, n_boot, label) {
  rec_levels <- unique(d$recipient)
  message(sprintf("  >>> bootstrap %s (n_boot=%d, %d clusters)",
                  label, n_boot, length(rec_levels)))
  t_b <- Sys.time()
  results <- future.apply::future_lapply(
    seq_len(n_boot),
    function(b) boot_one(d, b, rec_levels),
    future.seed = TRUE)
  secs <- as.numeric(difftime(Sys.time(), t_b, units = "secs"))
  ok_vec <- vapply(results, function(r) isTRUE(r$ok), logical(1L))
  message(sprintf("  ... %s done in %.1fs (convergence %d/%d = %.3f)",
                  label, secs, sum(ok_vec), n_boot, mean(ok_vec)))
  # Pre-allocate matrices
  ape_unc <- matrix(NA_real_, nrow = n_boot, ncol = length(APE_VARS),
                    dimnames = list(NULL, APE_VARS))
  ape_sel <- ape_unc; ape_int <- ape_unc
  for (b in seq_len(n_boot)) {
    r <- results[[b]]$ape
    setkey(r, covariate)
    ape_unc[b, ] <- r[APE_VARS, ape_unconditional]
    ape_sel[b, ] <- r[APE_VARS, ape_selection]
    ape_int[b, ] <- r[APE_VARS, ape_intensive]
  }
  list(unc = ape_unc, sel = ape_sel, int = ape_int,
       conv_rate = mean(ok_vec), n_boot = n_boot, secs = secs)
}

# Decide the actual N_BOOT given timing
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
# Use the longer of the two main fits as per-rep cost proxy
main_fit_cost <- fit_secs_total
# (We have 2 panels each requiring n_boot fits — so per-rep cost above is
# total per round. n_workers gives parallel speedup.)
decision <- n_boot_decision(seconds_per_fit = main_fit_cost,
                            n_workers = N_WORKERS,
                            elapsed = elapsed)
N_BOOT_ACTUAL <- decision$n
message(sprintf("  budget decision: n_boot=%d (used_fallback=%s) elapsed=%.0fs remaining=%.0fs",
                N_BOOT_ACTUAL, decision$used_fallback,
                elapsed, decision$remaining))

# Configure parallel plan. The mitigation panel (~82k rows) plus closure
# captures pushes the per-worker globals payload above the 500 MiB default
# cap; bump to 8 GiB to accommodate the data.table snapshot once per worker.
options(future.globals.maxSize = 8 * 1024^3)
plan(multisession, workers = N_WORKERS)
on.exit(plan(sequential), add = TRUE)

boot_a <- run_bootstrap(d_a, N_BOOT_ACTUAL, "adapt")
boot_m <- run_bootstrap(d_m, N_BOOT_ACTUAL, "miti")

# ── 4. Dollar translation ────────────────────────────────────────────────────
mean_pos_dollar <- function(d) {
  pos <- d$y_h[d$y_h > 0]
  if (!length(pos)) return(NA_real_)
  mean(exp(pos), na.rm = TRUE) / 1e6   # USD millions
}
m_dollar_a <- mean_pos_dollar(d_a)
m_dollar_m <- mean_pos_dollar(d_m)
message(sprintf("  mean positive flow (USD M): adapt = %.3f, miti = %.3f",
                m_dollar_a, m_dollar_m))

# ── 5. Build output tables ───────────────────────────────────────────────────

#' Combine point estimates with bootstrap SEs and the dollar translation.
build_ape_table <- function(pt, boot, mean_dollar_M) {
  d <- copy(pt)
  se_unc <- apply(boot$unc, 2L, sd, na.rm = TRUE)
  se_sel <- apply(boot$sel, 2L, sd, na.rm = TRUE)
  se_int <- apply(boot$int, 2L, sd, na.rm = TRUE)
  d[, ape_unconditional_se := se_unc[d$covariate]]
  d[, ape_selection_se     := se_sel[d$covariate]]
  d[, ape_intensive_se     := se_int[d$covariate]]
  d[, ape_dollar_per_dyad_sector_year := ape_unconditional * mean_dollar_M]
  setcolorder(d, c("covariate", "ape_unconditional", "ape_selection",
                   "ape_intensive",
                   "ape_unconditional_se", "ape_selection_se",
                   "ape_intensive_se",
                   "ape_dollar_per_dyad_sector_year"))
  d[]
}
ape_adapt <- build_ape_table(pt_a, boot_a, m_dollar_a)
ape_miti  <- build_ape_table(pt_m, boot_m, m_dollar_m)

fwrite(ape_adapt, file.path(out_dir, "ape_adapt.csv"))
fwrite(ape_miti,  file.path(out_dir, "ape_miti.csv"))

# ── 6. Diagnostics ───────────────────────────────────────────────────────────
runtime_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

extract_main_coefs <- function(fit_obj) {
  if (is.null(fit_obj$fit)) return(NULL)
  list(h1 = coef(fit_obj$fit, which = "h1"),
       h2 = coef(fit_obj$fit, which = "h2"),
       sigma = coef(fit_obj$fit, which = "sd"),
       n = fit_obj$n,
       rhs = fit_obj$rhs,
       fe_dropped = fit_obj$fe_dropped)
}

diag_obj <- list(
  fixes_applied = list(
    fix1_full_FE_in_APE_spec = TRUE,
    fix2_analytical_conditional_intensive = TRUE,
    fix3_n_boot_target = N_BOOT,
    fix3_n_boot_actual = N_BOOT_ACTUAL,
    fix3_used_fallback = decision$used_fallback
  ),
  main_coefs_adapt = extract_main_coefs(main_a),
  main_coefs_miti  = extract_main_coefs(main_m),
  conv_rate_adapt  = boot_a$conv_rate,
  conv_rate_miti   = boot_m$conv_rate,
  m_dollar_a = m_dollar_a, m_dollar_m = m_dollar_m,
  n_boot     = N_BOOT_ACTUAL,
  n_workers  = N_WORKERS,
  runtime_sec = runtime_sec,
  pt_adapt    = pt_a, pt_miti = pt_m,
  ape_adapt   = ape_adapt, ape_miti = ape_miti,
  boot_a_secs = boot_a$secs, boot_m_secs = boot_m$secs,
  seed_base   = SEED_BASE,
  notes       = paste(
    "Country-level aggregates (Provider/Recipient GDPtot, Pop, debt, fisB)",
    "dropped because of near-collinearity with provider/recipient FE that",
    "produced a singular Hessian; see header comment for the full FIX 1",
    "diagnosis. Sparse FE columns (<", MIN_FE_CELLS,
    "in either margin) also dropped per fit."))

saveRDS(diag_obj, file.path(out_dir, "ape_diagnostics.rds"))

message(sprintf("[%s] Task D rerun complete in %.1fs (n_boot=%d, conv adapt=%.3f, conv miti=%.3f)",
                format(Sys.time()), runtime_sec, N_BOOT_ACTUAL,
                boot_a$conv_rate, boot_m$conv_rate))
