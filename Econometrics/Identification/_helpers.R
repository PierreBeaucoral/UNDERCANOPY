# =============================================================================
# scripts/_helpers.R
# Shared helpers for the four identification-package scripts.
#
# Contents:
#   * recover_provider, recover_recipient, recover_year — recover labels from
#     one-hot dummies, handling the omitted baseline (USA, ARM, 2016).
#   * aggregate_dyad_year — collapse a sector-level (i,j,k,t) panel to dyad-year
#     (i,j,t) per the strategy memo Section B.1.
#   * dyad_covariates — vector of covariates used in every reweighting model.
#   * fit_hurdle_dyad — convenience wrapper around mhurdle for the dyad-year
#     specification with a single sample-restriction filter.
#   * safe_aipw_one_coef — hand-coded AIPW influence function for a single
#     headline coefficient (no `survey` dependency).
#   * lee_bound_one_coef — Lee (2009) trimming bound for one binary covariate.
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(here)
  library(mhurdle)
  library(boot)
  library(dplyr)
})

# ── Constants ────────────────────────────────────────────────────────────────
BASELINE_PROVIDER <- "USA"
BASELINE_RECIPIENT_HAN <- "ARM"
BASELINE_RECIPIENT_BERT <- "BASELINE"   # BERT panel never carries ARM
BASELINE_YEAR <- 2016L
EPS_PROB <- 1e-12                       # for CDF clamping

# Headline covariates used in dyad-year structural model (per strategy memo).
# Expanded 2026-05-04 to include the full set of variables that change sign or
# significance between Han / Rio / BERT specifications, so that AIPW and Lee
# checks can cover every §4 result (no §3.4 balance fallback needed).
DYAD_COVARS <- c(
  "RecipientWRIVul", "CPIAPublicAdm", "CPIAbudget",
  "distw", "colony", "comlang", "comrelig", "wto",
  "MDBDummy", "EIA", "InvestAgree",
  "ProviderGDPtot", "RecipientGDPtot", "ProviderPop", "RecipientPop",
  "Recipientdebt", "Providerdebt", "RecipientfisB", "ProviderfisB",
  "NDCActOnly", "NDCnonGHG", "NDC15", "NDC16", "NDC17", "NDC18"
)

# Headline coefficients reported in identification block (subset of DYAD_COVARS)
# kept for backward compatibility with the original 7-covariate run.
HEADLINE_COEFS <- c(
  "RecipientWRIVul", "CPIAPublicAdm", "CPIAbudget",
  "colony", "comlang", "MDBDummy", "distw"
)

# Expanded covariate list for AIPW + Lee (every variable that changes sign or
# significance across Han / Rio / BERT in §4 of the EARE Round-1 revision).
EXPANDED_COVARS_BINARY <- c(
  "colony", "comlang", "comrelig", "wto", "EIA", "InvestAgree", "MDBDummy",
  "NDCActOnly", "NDCnonGHG", "NDC15", "NDC16", "NDC17", "NDC18"
)
EXPANDED_COVARS_CONTINUOUS <- c(
  "RecipientWRIVul", "CPIAPublicAdm", "CPIAbudget",
  "RecipientGDPtot", "ProviderGDPtot",
  "RecipientPop", "ProviderPop",
  "Recipientdebt", "Providerdebt",
  "RecipientfisB", "ProviderfisB",
  "distw"
)
EXPANDED_COVARS <- c(EXPANDED_COVARS_BINARY, EXPANDED_COVARS_CONTINUOUS)

# ── Recovery from one-hot dummies ────────────────────────────────────────────

#' Recover a categorical label from a wide one-hot matrix.
#' Rows where all dummies equal zero are mapped to `baseline`.
#' @param X data.table containing the one-hot columns
#' @param prefix character prefix that identifies the columns (e.g. "ProviderISO_")
#' @param baseline label assigned to rows where every dummy equals zero
#' @return character vector of length nrow(X)
recover_label <- function(X, prefix, baseline) {
  cols <- grep(paste0("^", prefix), names(X), value = TRUE)
  if (length(cols) == 0L) {
    return(rep(baseline, nrow(X)))
  }
  M <- as.matrix(X[, cols, with = FALSE])
  storage.mode(M) <- "integer"
  out <- rep(baseline, nrow(M))
  has_one <- rowSums(M) > 0L
  if (any(has_one)) {
    idx <- max.col(M[has_one, , drop = FALSE], ties.method = "first")
    out[has_one] <- sub(prefix, "", cols[idx])
  }
  out
}

recover_provider  <- function(X) recover_label(X, "ProviderISO_",  BASELINE_PROVIDER)
recover_recipient <- function(X, baseline = BASELINE_RECIPIENT_HAN)
  recover_label(X, "RecipientISO_", baseline)

recover_year <- function(X) {
  cols <- grep("^Year_", names(X), value = TRUE)
  if (length(cols) == 0L) return(rep(BASELINE_YEAR, nrow(X)))
  M <- as.matrix(X[, cols, with = FALSE])
  storage.mode(M) <- "integer"
  out <- rep(BASELINE_YEAR, nrow(M))
  has_one <- rowSums(M) > 0L
  if (any(has_one)) {
    idx <- max.col(M[has_one, , drop = FALSE], ties.method = "first")
    out[has_one] <- as.integer(sub("Year_", "", cols[idx]))
  }
  out
}

# ── Aggregation (i,j,k,t) -> (i,j,t) ─────────────────────────────────────────

#' Collapse a sector-level adaptation/mitigation panel to dyad-year.
#'
#' @param dt data.table with columns from reg{1,2,3} schema
#' @param outcome character: "AdaptAmount" or "MitiAmount"
#' @param baseline_recipient character: "ARM" for reg1/reg2, "BASELINE" for reg3
#' @return data.table keyed by (provider, recipient, year)
aggregate_dyad_year <- function(dt, outcome, baseline_recipient = BASELINE_RECIPIENT_HAN) {
  stopifnot(outcome %in% names(dt))
  dt <- as.data.table(dt)
  dt[, provider  := recover_provider(.SD)]
  dt[, recipient := recover_recipient(.SD, baseline = baseline_recipient)]
  dt[, year      := recover_year(.SD)]
  setnames(dt, outcome, "y_raw")

  # Continuous time-varying covariates (mean within (i,j,t))
  mean_cols <- intersect(c(
    "RecipientWRIVul", "RecipientWRIExpo", "WRI",
    "RecipientGDPtot", "RecipientPop", "Recipientdebt", "RecipientfisB",
    "CPIAPublicAdm", "CPIAbudget",
    "ProviderGDPtot", "ProviderPop", "Providerdebt", "ProviderfisB"
  ), names(dt))
  # WRI is built only in the parallel script; recover it if missing.
  if (!"WRI" %in% names(dt) && all(c("RecipientWRIExpo", "RecipientWRIVul") %in% names(dt))) {
    dt[, WRI := RecipientWRIExpo * RecipientWRIVul]
    if (!"WRI" %in% mean_cols) mean_cols <- c(mean_cols, "WRI")
  }
  # Provider/Recipient GDP totals are likewise built downstream; reconstruct on the fly.
  if (!"ProviderGDPtot" %in% names(dt) && "ProviderGDPCur" %in% names(dt)) {
    dt[, ProviderGDPtot := ProviderGDPCur]   # already log-transformed in CSV
    if (!"ProviderGDPtot" %in% mean_cols) mean_cols <- c(mean_cols, "ProviderGDPtot")
  }
  if (!"RecipientGDPtot" %in% names(dt) && "RecipientGDPCur" %in% names(dt)) {
    dt[, RecipientGDPtot := RecipientGDPCur]
    if (!"RecipientGDPtot" %in% mean_cols) mean_cols <- c(mean_cols, "RecipientGDPtot")
  }

  # Time-invariant dyadic (any/first within (i,j))
  inv_cols <- intersect(c("distw", "colony", "comlang", "comrelig",
                          "wto", "EIA", "InvestAgree"), names(dt))
  # Max within dyad-year
  max_cols <- intersect(c("MDBDummy", "NDC15", "NDC16", "NDC17", "NDC18",
                          "NDCActOnly", "NDCnonGHG", "NDCGHG"), names(dt))

  agg <- dt[, c(
    list(any_flow = as.integer(any(y_raw > 0, na.rm = TRUE)),
         tot_amount = sum(y_raw, na.rm = TRUE),
         n_sector_cells = .N),
    lapply(.SD[, mean_cols, with = FALSE], function(v) mean(v, na.rm = TRUE)),
    lapply(.SD[, inv_cols,  with = FALSE], function(v) {
      v <- v[!is.na(v)]; if (length(v) == 0L) NA_real_ else v[1L]
    }),
    lapply(.SD[, max_cols,  with = FALSE], function(v) max(v, na.rm = TRUE))
  ), by = .(provider, recipient, year)]

  # Replace -Inf produced by max() over all-NA from older R behaviour
  for (col in max_cols) {
    set(agg, which(!is.finite(agg[[col]])), col, 0L)
  }
  for (col in mean_cols) {
    set(agg, which(!is.finite(agg[[col]])), col, NA_real_)
  }

  setkey(agg, provider, recipient, year)
  agg[]
}

# ── Hurdle wrapper ───────────────────────────────────────────────────────────

#' Fit the dyad-year Cragg double hurdle (mhurdle, dist="n", uncorrelated by
#' default for stability; corr=TRUE is attempted as a refinement).
#'
#' @param d data.table with columns: any_flow, tot_amount, DYAD_COVARS
#' @param verbose logical
#' @return list(model, ok, msg, coefs_outcome, coefs_selection)
fit_hurdle_dyad <- function(d, verbose = FALSE) {
  d <- as.data.table(d)
  d <- d[complete.cases(d[, c("tot_amount", DYAD_COVARS), with = FALSE])]
  if (nrow(d) < 200L) {
    return(list(ok = FALSE, msg = sprintf("Sample too small: n=%d", nrow(d))))
  }
  # mhurdle expects y as the dependent; positive cells should carry positive
  # values. The CSV stores log-amounts already, so positives are typically > 0
  # and negatives are absent. Guard against zero-but-positive-y_raw cases.
  d[, y_h := pmax(0, tot_amount)]
  rhs_sel <- paste(DYAD_COVARS, collapse = " + ")
  rhs_out <- rhs_sel
  f <- as.formula(paste0("y_h ~ ", rhs_sel, " | ", rhs_out))

  fit <- tryCatch(
    mhurdle::mhurdle(f, data = as.data.frame(d), dist = "n",
                     method = "bfgs", corr = FALSE),
    error = function(e) e)
  if (inherits(fit, "error")) {
    if (verbose) message("mhurdle failed: ", conditionMessage(fit))
    return(list(ok = FALSE, msg = conditionMessage(fit)))
  }
  cf <- coef(fit)
  # Extract SE from vcov diagonal; fall back to NA if vcov inversion failed
  se_vec <- tryCatch(
    sqrt(diag(vcov(fit))),
    error = function(e) setNames(rep(NA_real_, length(cf)), names(cf))
  )
  if (length(se_vec) != length(cf)) {
    se_vec <- setNames(rep(NA_real_, length(cf)), names(cf))
  } else if (is.null(names(se_vec))) {
    names(se_vec) <- names(cf)
  }
  # Native p-values from the model's own summary table (column 4 of
  # summary(fit)$coefficients). mhurdle uses asymptotic Wald inference;
  # column 4 is the model's preferred p-value.
  p_vec <- tryCatch({
    cm <- summary(fit)$coefficients
    if (is.null(cm) || ncol(cm) < 4L) {
      setNames(rep(NA_real_, length(cf)), names(cf))
    } else {
      pv <- cm[, 4L]
      pv[match(names(cf), rownames(cm))]
    }
  }, error = function(e) setNames(rep(NA_real_, length(cf)), names(cf)))
  if (length(p_vec) != length(cf)) {
    p_vec <- setNames(rep(NA_real_, length(cf)), names(cf))
  } else if (is.null(names(p_vec))) {
    names(p_vec) <- names(cf)
  }
  list(ok = TRUE, model = fit, coefs = cf, se = se_vec, pvals = p_vec,
       coefs_selection = cf[grep("^h1\\.", names(cf))],
       coefs_outcome   = cf[grep("^h2\\.", names(cf))],
       se_selection    = se_vec[grep("^h1\\.", names(se_vec))],
       se_outcome      = se_vec[grep("^h2\\.", names(se_vec))],
       pval_selection  = p_vec[grep("^h1\\.", names(p_vec))],
       pval_outcome    = p_vec[grep("^h2\\.", names(p_vec))],
       n = nrow(d))
}

# ── Lee bounds for one binary covariate ───────────────────────────────────────

#' Compute Lee (2009) trimming bound for one binary covariate when the
#' "treatment" is the BERT-vs-Han classification regime. We work on cells in the
#' union sample and identify the over-retained group via the empirical retention
#' ratio q = (r_T - r_C) / r_T, then trim the appropriate tail of y in the
#' over-retained group within the BERT sample to compute bounds.
#'
#' Returns a list with point estimate (BERT, OLS on log-positives), lower and
#' upper bounds (also OLS on log-positives, with trimming applied), q, and
#' n_trimmed.
#'
#' For continuous covariates, the caller dichotomises the variable at the
#' pooled-sample median and passes the binary indicator as `var`; the OLS
#' regression for the reported coefficient should still use the original
#' continuous regressor (handled by `lee_bound_one()` below).
lee_bound_binary <- function(bert, han, var, outcome = "tot_amount",
                             covars = DYAD_COVARS,
                             flow_col = "any_flow") {
  bert <- as.data.table(bert); han <- as.data.table(han)
  stopifnot(var %in% names(bert), var %in% names(han))
  stopifnot(flow_col %in% names(bert), flow_col %in% names(han))

  # Restrict to positive-amount cells for the OLS outcome model
  b_pos <- bert[get(flow_col) == 1L & is.finite(get(outcome)) & get(outcome) > 0]
  h_pos <- han [get(flow_col) == 1L & is.finite(get(outcome)) & get(outcome) > 0]

  retained_b1 <- nrow(b_pos[get(var) == 1L])
  retained_h1 <- nrow(h_pos[get(var) == 1L])
  retained_b0 <- nrow(b_pos[get(var) == 0L])
  retained_h0 <- nrow(h_pos[get(var) == 0L])
  r1 <- if (retained_h1 > 0L) retained_b1 / retained_h1 else NA_real_
  r0 <- if (retained_h0 > 0L) retained_b0 / retained_h0 else NA_real_

  if (!is.finite(r1) || !is.finite(r0) || r1 == r0) {
    # Compute the BERT OLS point estimate so the row is not entirely empty;
    # bound is undefined under degenerate retention but we still report point.
    rhs_vars_pt <- unique(c(var, intersect(covars, names(b_pos))))
    rhs_pt <- paste(rhs_vars_pt, collapse = " + ")
    f_pt_only <- as.formula(paste0(outcome, " ~ ", rhs_pt))
    fit_pt_only <- tryCatch(lm(f_pt_only, data = b_pos), error = function(e) NULL)
    pt_only <- if (is.null(fit_pt_only)) NA_real_ else unname(coef(fit_pt_only)[var])
    return(list(point = pt_only, lower = pt_only, upper = pt_only,
                q_trim = 0, n_trimmed = 0L,
                over_is_one = NA, r1 = r1, r0 = r0,
                msg = "tight: degenerate retention (one cell empty)"))
  }

  # Over-retained group is the one with larger r
  over_is_one <- r1 > r0
  q <- abs(r1 - r0) / max(r1, r0)

  # Point estimate on BERT — include the variable-of-interest in RHS
  rhs_vars <- unique(c(var, intersect(covars, names(b_pos))))
  rhs <- paste(rhs_vars, collapse = " + ")
  f_pt <- as.formula(paste0(outcome, " ~ ", rhs))
  fit_pt <- tryCatch(lm(f_pt, data = b_pos), error = function(e) NULL)
  point <- if (is.null(fit_pt)) NA_real_ else unname(coef(fit_pt)[var])

  # For lower bound: trim top q% of y in over-retained group (assume retained
  # extras are the highest)
  # For upper bound: trim bottom q% of y in over-retained group (assume retained
  # extras are the lowest)
  group_idx <- if (over_is_one) which(b_pos[[var]] == 1L) else which(b_pos[[var]] == 0L)
  y_grp <- b_pos[[outcome]][group_idx]
  # Cap n_trim so at least one obs survives in the over-retained group; without
  # this cap, q == 1 (one cell empty in BERT) trims the entire group and leaves
  # a singular regression. The cap caps the bound at the smallest non-degenerate
  # interval; the diagnostic msg flags it.
  n_trim <- min(floor(q * length(y_grp)), max(length(y_grp) - 2L, 0L))

  if (n_trim == 0L) {
    # No differential retention: Lee bounds collapse to the OLS point estimate.
    # Reported transparently — q ≈ 0 means selection on `var` is symmetric and
    # the sharp identified interval is a singleton at the BERT OLS coefficient.
    return(list(point = point,
                lower = point, upper = point,
                q_trim = q, n_trimmed = 0L,
                over_is_one = over_is_one,
                r1 = r1, r0 = r0,
                msg = "tight: q approx 0"))
  }

  # Lower bound: drop top n_trim
  ord <- order(y_grp, decreasing = TRUE)
  drop_lower <- group_idx[ord[seq_len(n_trim)]]
  d_lower <- b_pos[-drop_lower]
  fit_l <- tryCatch(lm(f_pt, data = d_lower), error = function(e) NULL)
  lower <- if (is.null(fit_l)) NA_real_ else unname(coef(fit_l)[var])

  # Upper bound: drop bottom n_trim
  ord <- order(y_grp, decreasing = FALSE)
  drop_upper <- group_idx[ord[seq_len(n_trim)]]
  d_upper <- b_pos[-drop_upper]
  fit_u <- tryCatch(lm(f_pt, data = d_upper), error = function(e) NULL)
  upper <- if (is.null(fit_u)) NA_real_ else unname(coef(fit_u)[var])

  list(point = point,
       lower = min(lower, upper, na.rm = TRUE),  # ensure ordering
       upper = max(lower, upper, na.rm = TRUE),
       q_trim = q,
       n_trimmed = n_trim,
       over_is_one = over_is_one,
       r1 = r1, r0 = r0,
       msg = "ok")
}

# ── Lee bounds for one covariate (binary or continuous) ──────────────────────

#' Generalised Lee (2009) bound that handles both binary and continuous
#' covariates. For continuous covariates, the differential-retention quantity
#' `q` is computed by dichotomising at the pooled-sample median (across BERT
#' and Han positive-flow cells), but the OLS coefficient reported in
#' `point` / `lower` / `upper` is the slope on the *original continuous*
#' regressor, not on the median-split indicator. This keeps the reported
#' coefficient comparable to the §4 OLS / hurdle estimate for the same variable.
#'
#' For binary covariates this collapses to `lee_bound_binary()`.
#'
#' @param bert,han  BERT and Han dyad-year panels (data.table)
#' @param var       character: covariate name
#' @param is_binary logical: TRUE if `var` is binary (0/1)
#' @param outcome   character: outcome column name (e.g., "tot_adapt_amount")
#' @param covars    character vector of other covariates
#' @param flow_col  character: column name flagging positive flows
#' @return list with point, lower, upper, q_trim, n_trimmed, median_value,
#'         r1, r0, msg
lee_bound_one <- function(bert, han, var, is_binary,
                          outcome = "tot_amount",
                          covars = DYAD_COVARS,
                          flow_col = "any_flow") {
  bert <- as.data.table(bert); han <- as.data.table(han)
  stopifnot(var %in% names(bert), var %in% names(han))
  stopifnot(flow_col %in% names(bert), flow_col %in% names(han))

  if (isTRUE(is_binary)) {
    res <- lee_bound_binary(bert, han, var = var, outcome = outcome,
                            covars = covars, flow_col = flow_col)
    res$median_value <- NA_real_
    return(res)
  }

  # Continuous case: median-split on pooled positive-flow values and reuse
  # lee_bound_binary's q-and-trim logic, but report the *continuous* OLS
  # coefficient on the original variable.
  b_pos <- bert[get(flow_col) == 1L & is.finite(get(outcome)) & get(outcome) > 0]
  h_pos <- han [get(flow_col) == 1L & is.finite(get(outcome)) & get(outcome) > 0]
  pooled_vals <- c(b_pos[[var]], h_pos[[var]])
  pooled_vals <- pooled_vals[is.finite(pooled_vals)]
  if (length(pooled_vals) < 10L) {
    return(list(point = NA_real_, lower = NA_real_, upper = NA_real_,
                q_trim = 0, n_trimmed = 0L, median_value = NA_real_,
                over_is_one = NA, r1 = NA_real_, r0 = NA_real_,
                msg = "too few finite values for median split"))
  }
  med <- median(pooled_vals, na.rm = TRUE)
  bin_name <- paste0(var, "_hi")
  b_pos[, (bin_name) := as.integer(get(var) > med)]
  h_pos[, (bin_name) := as.integer(get(var) > med)]

  # Retention computation on the binary partition
  retained_b1 <- nrow(b_pos[get(bin_name) == 1L])
  retained_h1 <- nrow(h_pos[get(bin_name) == 1L])
  retained_b0 <- nrow(b_pos[get(bin_name) == 0L])
  retained_h0 <- nrow(h_pos[get(bin_name) == 0L])
  r1 <- if (retained_h1 > 0L) retained_b1 / retained_h1 else NA_real_
  r0 <- if (retained_h0 > 0L) retained_b0 / retained_h0 else NA_real_
  if (!is.finite(r1) || !is.finite(r0) || abs(r1 - r0) < 1e-12) {
    # No differential retention — bound is tight on the BERT OLS slope.
    rhs_vars <- unique(c(var, intersect(covars, names(b_pos))))
    rhs <- paste(rhs_vars, collapse = " + ")
    f_pt <- as.formula(paste0(outcome, " ~ ", rhs))
    fit_pt <- tryCatch(lm(f_pt, data = b_pos), error = function(e) NULL)
    pt <- if (is.null(fit_pt)) NA_real_ else unname(coef(fit_pt)[var])
    return(list(point = pt, lower = pt, upper = pt,
                q_trim = 0, n_trimmed = 0L, median_value = med,
                over_is_one = NA, r1 = r1, r0 = r0,
                msg = "tight: q approx 0 (continuous; median split symmetric)"))
  }
  over_is_one <- r1 > r0
  q <- abs(r1 - r0) / max(r1, r0)

  # Point estimate: OLS slope on the *continuous* `var` (not the median split).
  rhs_vars <- unique(c(var, intersect(covars, names(b_pos))))
  rhs <- paste(rhs_vars, collapse = " + ")
  f_pt <- as.formula(paste0(outcome, " ~ ", rhs))
  fit_pt <- tryCatch(lm(f_pt, data = b_pos), error = function(e) NULL)
  point <- if (is.null(fit_pt)) NA_real_ else unname(coef(fit_pt)[var])

  group_idx <- if (over_is_one) which(b_pos[[bin_name]] == 1L) else which(b_pos[[bin_name]] == 0L)
  y_grp <- b_pos[[outcome]][group_idx]
  # Cap n_trim to keep at least two obs in the over-retained group (see binary
  # version: prevents singular regressions when q approaches 1).
  n_trim <- min(floor(q * length(y_grp)), max(length(y_grp) - 2L, 0L))

  if (n_trim == 0L) {
    return(list(point = point, lower = point, upper = point,
                q_trim = q, n_trimmed = 0L, median_value = med,
                over_is_one = over_is_one, r1 = r1, r0 = r0,
                msg = "tight: q approx 0 (continuous)"))
  }

  ord <- order(y_grp, decreasing = TRUE)
  drop_lower <- group_idx[ord[seq_len(n_trim)]]
  d_lower <- b_pos[-drop_lower]
  fit_l <- tryCatch(lm(f_pt, data = d_lower), error = function(e) NULL)
  lower <- if (is.null(fit_l)) NA_real_ else unname(coef(fit_l)[var])

  ord <- order(y_grp, decreasing = FALSE)
  drop_upper <- group_idx[ord[seq_len(n_trim)]]
  d_upper <- b_pos[-drop_upper]
  fit_u <- tryCatch(lm(f_pt, data = d_upper), error = function(e) NULL)
  upper <- if (is.null(fit_u)) NA_real_ else unname(coef(fit_u)[var])

  list(point = point,
       lower = min(lower, upper, na.rm = TRUE),
       upper = max(lower, upper, na.rm = TRUE),
       q_trim = q, n_trimmed = n_trim, median_value = med,
       over_is_one = over_is_one, r1 = r1, r0 = r0,
       msg = "ok")
}

# ── AIPW for one OLS coefficient (handcoded; no `survey` dependency) ─────────

#' Single-coefficient AIPW: estimate the partial association of `coef_name`
#' with the outcome on the BERT sample, transported to the Han baseline X
#' distribution. Uses the union (Han, BERT) for the propensity model and the
#' BERT-positives for the outcome model. Reweighted-OLS coefficient is the
#' AIPW point estimate; SE is the empirical IF SE.
#'
#' This implements the partially-linear AIPW used in Robins-Rotnitzky-Zhao
#' (1994) and Chernozhukov et al. (2018, ch. 3): score-based identification of
#' a single linear coefficient under a doubly-robust outer expectation.
#'
#' @param pooled data.table with IN_BERT, outcome, covars
#' @param coef_name character: covariate of interest (must be in pooled)
#' @param outcome   character: name of the outcome column (log-positives)
#' @param covars    character vector of covariates (excluding coef_name)
#' @param trim_lo,trim_hi propensity-score trimming bounds
#' @return list(estimate, se, n_used, n_trimmed)
aipw_one_coef <- function(pooled, coef_name, outcome, covars,
                          trim_lo = 0.01, trim_hi = 0.99,
                          flow_col = "any_flow") {
  pooled <- as.data.table(pooled)
  if (!flow_col %in% names(pooled)) {
    # Fall back: treat positive-outcome rows as flow==1
    pooled[, (flow_col) := as.integer(get(outcome) > 0)]
  }
  pooled <- pooled[get(flow_col) == 1L]                  # AIPW on the intensive margin
  pooled <- pooled[is.finite(get(outcome)) & get(outcome) > 0]
  pooled <- pooled[complete.cases(pooled[, c(outcome, covars, coef_name, "IN_BERT"),
                                          with = FALSE])]
  n0 <- nrow(pooled)
  if (n0 < 200L) {
    return(list(estimate = NA_real_, se = NA_real_, n_used = n0, n_trimmed = 0L))
  }

  # Propensity model (full union sample, not restricted to positive flows for
  # ps estimation -- but we need the same support for the outcome equation, so
  # we estimate ps on positive-flow cells consistent with the second-stage
  # OLS).
  rhs_ps <- paste(intersect(c(covars, coef_name), names(pooled)),
                  collapse = " + ")
  f_ps <- as.formula(paste0("IN_BERT ~ ", rhs_ps))
  ps_fit <- suppressWarnings(glm(f_ps, family = binomial(link = "logit"),
                                 data = pooled))
  phat <- predict(ps_fit, type = "response")
  phat <- pmin(pmax(phat, EPS_PROB), 1 - EPS_PROB)       # clamp

  # Trim
  keep <- phat >= trim_lo & phat <= trim_hi
  n_trim <- sum(!keep)
  d <- pooled[keep]; phat <- phat[keep]

  # Outcome model on BERT subset
  rhs_y <- paste(intersect(c(covars, coef_name), names(d)), collapse = " + ")
  f_y <- as.formula(paste0(outcome, " ~ ", rhs_y))
  y_fit <- lm(f_y, data = d[IN_BERT == 1L])
  mu_hat <- predict(y_fit, newdata = d)

  y <- d[[outcome]]; s <- d$IN_BERT
  # IPW outcome
  y_dr <- mu_hat + (s / phat) * (y - mu_hat)
  # Run the single-coefficient regression on the doubly-robust pseudo-outcome,
  # weighting all observations equally (the augmentation already handles the
  # selection bias).
  d2 <- copy(d); d2[, y_dr := y_dr]
  rhs_y2 <- paste(intersect(c(covars, coef_name), names(d2)), collapse = " + ")
  f_dr <- as.formula(paste0("y_dr ~ ", rhs_y2))
  dr_fit <- lm(f_dr, data = d2)
  est <- unname(coef(dr_fit)[coef_name])

  # IF-based SE: residuals of the DR pseudo-outcome regression projected onto
  # the partial residual of the coefficient of interest.
  # (Conservative: uses sandwich-style HC0 on the pseudo-outcome regression.)
  Xmat <- model.matrix(dr_fit)
  res  <- residuals(dr_fit)
  bread <- solve(crossprod(Xmat) / nrow(Xmat))
  meat  <- crossprod(Xmat * res) / nrow(Xmat)
  vcov_hc0 <- (bread %*% meat %*% bread) / nrow(Xmat)
  se <- sqrt(vcov_hc0[coef_name, coef_name])

  list(estimate = est, se = unname(se), n_used = nrow(d2), n_trimmed = n_trim)
}
