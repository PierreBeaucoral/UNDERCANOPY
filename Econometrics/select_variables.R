# =============================================================================
# select_variables.R
# -----------------------------------------------------------------------------
# Han -> BERT variable-selection rule for the climate-finance hurdle results.
#
# Rule (per (outcome in {adapt, miti}, stage in {h1, h2}, variable v)):
#   Using only Correlated_Estimate (beta) and Correlated_Pr(>|t|) (p):
#     sig(p) := !is.na(p) & p < alpha
#     Flag A (lost significance):  sig(p_han) & !sig(p_bert)
#     Flag B (sign change w/ sig): sig(p_han) & sig(p_bert) &
#                                  sign(beta_han) != sign(beta_bert)
#     Flag C (gained significance): !sig(p_han) & sig(p_bert)
#   Selected := Flag A | Flag B | Flag C.
#   Rationale: a (covariate, equation, finance) triple enters the §5
#   verification pipeline iff the classification choice (Han -> BERT)
#   moves an inferential conclusion in either direction.
#
# Skip set (regex): provider/recipient/year/sector dummies and auxiliary rows
# (intercept, corr12, sigma, Sd, Log-likelihood, McFadden, R2, N).
#
# Outputs (regressions/var_selection/):
#   - selection_diagnostic.csv  long table, one row per (outcome,stage,var)
#   - selected_variables.json   manifest consumed by HurdleRegHuei_selected.R
#   - selection_summary.txt     human-readable counts and lists
#
# Reproducible: pure function `select_variables()` takes paths + alpha; the
# wrapper at the bottom calls it with project defaults.
# =============================================================================

set.seed(20260505L)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(here)
})


# -----------------------------------------------------------------------------
# Display-name <-> data-column map.
# Mirrors `coef_labels` in HurdleRegHuei_parallel.R (line ~501). Applied as
# inverse here: CSV "Variable" column carries pretty names; we need the raw
# column names for the slim formula.
# -----------------------------------------------------------------------------
.coef_labels_pretty_to_raw <- c(
  "WRI (Vulnerability)"    = "WRI",
  "Colonial Ties"          = "colony",
  "Common Language"        = "comlang",
  "Common Religion"        = "comrelig",
  "Distance"               = "distw",
  "INDC15"                 = "NDC15",
  "INDC16"                 = "NDC16",
  "INDC17"                 = "NDC17",
  "INDC18"                 = "NDC18",
  "Investment Agreement"   = "InvestAgree",
  "Provider GDP"           = "ProviderGDPtot",
  "Recipient GDP"          = "RecipientGDPtot",
  "Provider Population"    = "ProviderPop",
  "Recipient Population"   = "RecipientPop",
  "Provider fiscal Bal"    = "ProviderfisB",
  "Provider debt to GDP"   = "Providerdebt",
  "Recipient Fiscal Bal"   = "RecipientfisB",
  "Recipient debt to GDP"  = "Recipientdebt"
)


# -----------------------------------------------------------------------------
# Default skip patterns. Regexes applied to the full CSV variable token,
# including any "h1." / "h2." prefix.
# -----------------------------------------------------------------------------
.default_skip_patterns <- c(
  "^h[12]\\.\\(Intercept\\)$",
  "^h[12]\\.ProviderISO_",
  "^h[12]\\.RecipientISO_",
  "^h[12]\\.Year_",
  "^h[12]\\.(Water|Transport|Agri|EnvProtect|MultiSec|Disaster|Energy)$",
  "^h[12]\\.climate_class_",
  "^(corr12|sigma|Sd|Log-likelihood|McFadden Pseudo-R2|Coefficient of Determination .*|N)$"
)


#' Parse a results CSV into a long table keyed by (variable, stage).
#'
#' @param path Path to a `combined_regression_results*.csv` file produced by
#'   `HurdleRegHuei_parallel.R`'s `.make_csv_df()`.
#' @return Tibble with columns `var_full`, `stage` ('h1'|'h2'|'aux'),
#'   `var_pretty` (no h-prefix), `estimate`, `pvalue`. The estimate and
#'   pvalue come from the `Correlated_*` columns.
.read_results <- function(path) {
  raw <- utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  needed <- c("Variable", "Correlated_Estimate", "Correlated_Pr(>|t|)")
  missing <- setdiff(needed, names(raw))
  if (length(missing) > 0L) {
    stop(sprintf("File '%s' is missing column(s): %s",
                 path, paste(missing, collapse = ", ")))
  }
  has_h1 <- grepl("^h1\\.", raw$Variable)
  has_h2 <- grepl("^h2\\.", raw$Variable)
  stage <- ifelse(has_h1, "h1",
           ifelse(has_h2, "h2", "aux"))
  var_pretty <- ifelse(has_h1 | has_h2,
                       sub("^h[12]\\.", "", raw$Variable),
                       raw$Variable)
  tibble::tibble(
    var_full   = raw$Variable,
    stage      = stage,
    var_pretty = var_pretty,
    estimate   = suppressWarnings(as.numeric(raw[["Correlated_Estimate"]])),
    pvalue     = suppressWarnings(as.numeric(raw[["Correlated_Pr(>|t|)"]]))
  )
}


#' Test whether a variable token matches any skip-pattern regex.
.is_skipped <- function(var_full, patterns = .default_skip_patterns) {
  Reduce(`|`, lapply(patterns, function(p) grepl(p, var_full)),
         init = rep(FALSE, length(var_full)))
}


#' Map a pretty variable name back to the raw data column.
#'
#' Names not in `.coef_labels_pretty_to_raw` pass through unchanged.
.pretty_to_raw <- function(pretty) {
  ifelse(pretty %in% names(.coef_labels_pretty_to_raw),
         unname(.coef_labels_pretty_to_raw[pretty]),
         pretty)
}


#' Apply the Han -> BERT selection rule for a single outcome.
#'
#' @param han_path Path to the Han results CSV (combined_regression_results.csv).
#' @param bert_path Path to the BERT results CSV (combined_regression_results3.csv).
#' @param outcome  Label inserted into the diagnostic table ('adapt' or 'miti').
#' @param alpha    Significance threshold.
#' @param skip_patterns Regex vector applied to the full variable token.
#' @return Tibble: one row per (outcome, stage, var_pretty) with han/bert
#'   estimates and p-values, skip flag, lost_sig flag, sign_change flag,
#'   selected flag, plus var_raw (mapped data-column name).
.compare_one_outcome <- function(han_path, bert_path, outcome,
                                 alpha = 0.05,
                                 skip_patterns = .default_skip_patterns) {
  han  <- .read_results(han_path)
  bert <- .read_results(bert_path)
  joined <- dplyr::full_join(
    han  %>% dplyr::rename(estimate_han = estimate, pvalue_han = pvalue),
    bert %>% dplyr::rename(estimate_bert = estimate, pvalue_bert = pvalue),
    by = c("var_full", "stage", "var_pretty")
  )
  joined <- joined %>%
    dplyr::mutate(
      outcome      = outcome,
      skipped      = .is_skipped(var_full, skip_patterns),
      sig_han      = !is.na(pvalue_han)  & pvalue_han  < alpha,
      sig_bert     = !is.na(pvalue_bert) & pvalue_bert < alpha,
      lost_sig     = sig_han & !sig_bert,
      sign_change  = sig_han & sig_bert &
                     !is.na(estimate_han) & !is.na(estimate_bert) &
                     sign(estimate_han) != sign(estimate_bert),
      gained_sig   = !sig_han & sig_bert,
      missing_han  = is.na(estimate_han),
      missing_bert = is.na(estimate_bert),
      selected     = !skipped & stage %in% c("h1", "h2") &
                     (lost_sig | sign_change | gained_sig),
      var_raw      = .pretty_to_raw(var_pretty)
    )
  joined %>%
    dplyr::select(outcome, stage, var_full, var_pretty, var_raw,
                  estimate_han, pvalue_han, sig_han,
                  estimate_bert, pvalue_bert, sig_bert,
                  missing_han, missing_bert,
                  skipped, lost_sig, sign_change, gained_sig, selected)
}


#' Run the Han -> BERT selection rule across both outcomes and return artifacts.
#'
#' Pure: no globals, no side effects unless `out_dir` is non-NULL.
#'
#' @param adapt_han_path  Path to adapt Han CSV.
#' @param adapt_bert_path Path to adapt BERT CSV.
#' @param miti_han_path   Path to miti Han CSV.
#' @param miti_bert_path  Path to miti BERT CSV.
#' @param alpha           Significance threshold (default 0.05).
#' @param skip_patterns   Regex vector applied to full variable tokens.
#' @param out_dir         If non-NULL, write the three artifacts there.
#' @return Named list: `diagnostic`, `selected`, `meta`.
select_variables <- function(adapt_han_path,
                             adapt_bert_path,
                             miti_han_path,
                             miti_bert_path,
                             alpha = 0.05,
                             skip_patterns = .default_skip_patterns,
                             out_dir = NULL) {
  paths <- c(adapt_han_path, adapt_bert_path, miti_han_path, miti_bert_path)
  for (p in paths) {
    if (!file.exists(p)) {
      stop(sprintf("Input file not found: %s", p))
    }
  }
  adapt <- .compare_one_outcome(adapt_han_path, adapt_bert_path,
                                outcome = "adapt", alpha = alpha,
                                skip_patterns = skip_patterns)
  miti  <- .compare_one_outcome(miti_han_path, miti_bert_path,
                                outcome = "miti",  alpha = alpha,
                                skip_patterns = skip_patterns)
  diag <- dplyr::bind_rows(adapt, miti)

  selected <- list(
    adapt_h1 = unique(diag$var_raw[diag$outcome == "adapt" &
                                   diag$stage   == "h1" & diag$selected]),
    adapt_h2 = unique(diag$var_raw[diag$outcome == "adapt" &
                                   diag$stage   == "h2" & diag$selected]),
    miti_h1  = unique(diag$var_raw[diag$outcome == "miti"  &
                                   diag$stage   == "h1" & diag$selected]),
    miti_h2  = unique(diag$var_raw[diag$outcome == "miti"  &
                                   diag$stage   == "h2" & diag$selected])
  )

  meta <- list(
    rule           = "han_to_bert_v1",
    alpha          = alpha,
    spec           = "Correlated",
    skip_patterns  = unname(skip_patterns),
    timestamp_utc  = format(Sys.time(), tz = "UTC", usetz = TRUE),
    inputs = list(
      adapt_han  = adapt_han_path,
      adapt_bert = adapt_bert_path,
      miti_han   = miti_han_path,
      miti_bert  = miti_bert_path
    ),
    counts = list(
      total_rows       = nrow(diag),
      total_skipped    = sum(diag$skipped),
      total_candidates = sum(!diag$skipped & diag$stage %in% c("h1", "h2")),
      total_selected   = sum(diag$selected),
      n_selected_per_stage = lapply(selected, length)
    )
  )

  if (!is.null(out_dir)) {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(diag,
                     file.path(out_dir, "selection_diagnostic.csv"),
                     row.names = FALSE)
    jsonlite::write_json(
      list(meta = meta, selected = selected),
      path = file.path(out_dir, "selected_variables.json"),
      auto_unbox = TRUE, pretty = TRUE
    )
    .write_summary(diag, selected, meta,
                   file.path(out_dir, "selection_summary.txt"))
  }

  list(diagnostic = diag, selected = selected, meta = meta)
}


#' Write a human-readable summary of the selection result.
.write_summary <- function(diag, selected, meta, path) {
  sink(path)
  on.exit(sink(NULL), add = TRUE)
  cat("Han -> BERT variable selection summary\n")
  cat("Generated: ", meta$timestamp_utc, "\n", sep = "")
  cat("Rule: ", meta$rule, "  alpha = ", meta$alpha,
      "  spec = ", meta$spec, "\n\n", sep = "")
  cat("Counts\n")
  cat("  total rows             : ", meta$counts$total_rows, "\n", sep = "")
  cat("  skipped (FE/aux)       : ", meta$counts$total_skipped, "\n", sep = "")
  cat("  candidates (h1+h2)     : ", meta$counts$total_candidates, "\n", sep = "")
  cat("  selected               : ", meta$counts$total_selected, "\n\n", sep = "")
  for (k in names(selected)) {
    v <- selected[[k]]
    cat(sprintf("Selected %s (%d)\n", k, length(v)))
    if (length(v) == 0L) {
      cat("  (none)\n")
    } else {
      for (vi in v) cat("  - ", vi, "\n", sep = "")
    }
    cat("\n")
  }
  flagged <- diag %>%
    dplyr::filter(selected) %>%
    dplyr::mutate(reason = dplyr::case_when(
      lost_sig & sign_change ~ "lost_sig+sign_change",
      lost_sig               ~ "lost_sig",
      sign_change            ~ "sign_change",
      gained_sig             ~ "gained_sig",
      TRUE                   ~ "other"
    )) %>%
    dplyr::select(outcome, stage, var_pretty, var_raw,
                  estimate_han, pvalue_han, estimate_bert, pvalue_bert, reason)
  cat("Per-row detail (selected only)\n")
  if (nrow(flagged) == 0L) {
    cat("  (none selected)\n")
  } else {
    print(as.data.frame(flagged), row.names = FALSE)
  }
  invisible(NULL)
}


# =============================================================================
# Default project wrapper. Sources should be present after running
# HurdleRegHuei_parallel.R.
# =============================================================================
if (sys.nframe() == 0L) {
  proj <- tryCatch(here::here(), error = function(e) getwd())

  default_inputs <- list(
    adapt_han  = file.path(proj, "Econometrics", "regressions", "main", "adaptation", "combined_regression_results.csv"),
    adapt_bert = file.path(proj, "Econometrics", "regressions", "main", "adaptation", "combined_regression_results3.csv"),
    miti_han   = file.path(proj, "Econometrics", "regressions", "main", "mitigation", "combined_regression_results.csv"),
    miti_bert  = file.path(proj, "Econometrics", "regressions", "main", "mitigation", "combined_regression_results3.csv")
  )

  out_dir <- file.path(proj, "Econometrics", "regressions", "var_selection")

  res <- select_variables(
    adapt_han_path  = default_inputs$adapt_han,
    adapt_bert_path = default_inputs$adapt_bert,
    miti_han_path   = default_inputs$miti_han,
    miti_bert_path  = default_inputs$miti_bert,
    alpha           = 0.05,
    skip_patterns   = .default_skip_patterns,
    out_dir         = out_dir
  )

  message("Variable selection complete.")
  message("  diagnostic : ", file.path(out_dir, "selection_diagnostic.csv"))
  message("  manifest   : ", file.path(out_dir, "selected_variables.json"))
  message("  summary    : ", file.path(out_dir, "selection_summary.txt"))
  message(sprintf("  selected   : adapt_h1=%d, adapt_h2=%d, miti_h1=%d, miti_h2=%d",
                  length(res$selected$adapt_h1),
                  length(res$selected$adapt_h2),
                  length(res$selected$miti_h1),
                  length(res$selected$miti_h2)))
}
