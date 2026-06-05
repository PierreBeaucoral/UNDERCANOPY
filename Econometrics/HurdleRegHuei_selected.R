# =============================================================================
# HurdleRegHuei_selected.R
# -----------------------------------------------------------------------------
# Slim re-estimation of the four core hurdle models (adapt-Han, adapt-BERT,
# miti-Han, miti-BERT) using only covariates flagged by the Han -> BERT
# selection rule (see select_variables.R).
#
# Inputs (must exist before running):
#   - Econometrics/regressions/var_selection/selected_variables.json
#       (produced by select_variables.R)
#   - reg1, reg3, reg4, reg6 data frames in the global environment
#       (produced by sourcing HurdleRegHuei_parallel.R sections 1-3)
#       OR  .RData at project root containing the same objects.
#
# Workflow:
#   Rscript -e "source('HurdleRegHuei_parallel.R')"   # main estimation + CSVs
#   Rscript -e "source('select_variables.R')"         # writes manifest
#   Rscript -e "source('HurdleRegHuei_selected.R')"   # this file
#
# Or, if .RData has reg1/reg3/reg4/reg6 from a prior parallel run:
#   Rscript HurdleRegHuei_selected.R
#
# Outputs:
#   Econometrics/regressions/selected/adaptation/combined_regression_results_slim.csv
#   Econometrics/regressions/selected/mitigation/combined_regression_results_slim.csv
#   Econometrics/regressions/selected/slim_models.rds   (named list of 4 mhurdle fits)
# =============================================================================

set.seed(20260505L)

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(jsonlite); library(here)
  library(data.table)
  library(mhurdle); library(future); library(future.apply); library(parallelly)
})


# -----------------------------------------------------------------------------
# Project root and inputs
# -----------------------------------------------------------------------------
.proj <- tryCatch(here::here(), error = function(e) getwd())

.manifest_path <- file.path(.proj, "Econometrics", "regressions", "var_selection", "selected_variables.json")
.out_dir       <- file.path(.proj, "Econometrics", "regressions", "selected")

if (!file.exists(.manifest_path)) {
  stop("Manifest not found: ", .manifest_path,
       "\nRun select_variables.R first.")
}

.manifest <- jsonlite::fromJSON(.manifest_path, simplifyVector = TRUE)

cat("Loaded manifest:\n")
cat("  rule        : ", .manifest$meta$rule, "\n", sep = "")
cat("  alpha       : ", .manifest$meta$alpha, "\n", sep = "")
cat("  spec        : ", .manifest$meta$spec, "\n", sep = "")
for (k in names(.manifest$selected)) {
  cat(sprintf("  selected %-10s: %d\n", k, length(.manifest$selected[[k]])))
}


# -----------------------------------------------------------------------------
# Load reg1, reg3, reg4, reg6 from (in order):
#   1) global env (sourced HurdleRegHuei_parallel.R in same session)
#   2) .RData at project root
#   3) Econometrics/Data/reg{1,3,1_mitigation,3_mitigation}.csv (written by
#      parallel.R) — preferred fallback, no estimation needed.
# -----------------------------------------------------------------------------
.required_objs <- c("reg1", "reg3", "reg4", "reg6")
.missing <- .required_objs[!sapply(.required_objs, exists, envir = .GlobalEnv)]

if (length(.missing) > 0L) {
  .rdata_path <- file.path(.proj, ".RData")
  if (file.exists(.rdata_path)) {
    message("Loading missing objects from .RData ...")
    load(.rdata_path, envir = .GlobalEnv)
    .missing <- .required_objs[!sapply(.required_objs, exists, envir = .GlobalEnv)]
  }
}
if (length(.missing) > 0L) {
  .csv_map <- c(
    reg1 = file.path(.proj, "Econometrics", "Data", "reg1.csv"),
    reg3 = file.path(.proj, "Econometrics", "Data", "reg3.csv"),
    reg4 = file.path(.proj, "Econometrics", "Data", "reg1_mitigation.csv"),
    reg6 = file.path(.proj, "Econometrics", "Data", "reg3_mitigation.csv")
  )
  if (all(file.exists(.csv_map[.missing]))) {
    message("Loading missing objects from Econometrics/Data/*.csv ...")
    for (nm in .missing) {
      assign(nm,
             data.table::fread(.csv_map[[nm]], data.table = FALSE),
             envir = .GlobalEnv)
    }
    .missing <- .required_objs[!sapply(.required_objs, exists, envir = .GlobalEnv)]
  }
}
if (length(.missing) > 0L) {
  stop("Missing objects: ", paste(.missing, collapse = ", "),
       "\nNeither global env, .RData, nor Econometrics/Data/reg*.csv supplied them.",
       "\nRun HurdleRegHuei_parallel.R first (sections 1-3) to materialise reg1/reg3/reg4/reg6.")
}

reg1 <- get("reg1", envir = .GlobalEnv)
reg3 <- get("reg3", envir = .GlobalEnv)
reg4 <- get("reg4", envir = .GlobalEnv)
reg6 <- get("reg6", envir = .GlobalEnv)
cat(sprintf("Data loaded: reg1=%d x %d, reg3=%d x %d, reg4=%d x %d, reg6=%d x %d\n",
            nrow(reg1), ncol(reg1), nrow(reg3), ncol(reg3),
            nrow(reg4), ncol(reg4), nrow(reg6), ncol(reg6)))


# -----------------------------------------------------------------------------
# FE block strings — Han uses hard-coded blocks (mirrors HurdleRegHuei_parallel.R
# lines 534-538). BERT uses data-driven country dummies (>=0.5% presence in the
# BERT sample), reproducing the logic at parallel.R lines 240-265 / 460-493.
# -----------------------------------------------------------------------------
.p_sel    <- "ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+ProviderISO_LUX+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE"
.p_alloc  <- "ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+ProviderISO_LTU+ProviderISO_LUX+ProviderISO_LVA+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE"
.r_alloc  <- "RecipientISO_AFG+RecipientISO_AGO+RecipientISO_BDI+RecipientISO_BEN+RecipientISO_BFA+RecipientISO_BGD+RecipientISO_BIH+RecipientISO_BTN+RecipientISO_CAF+RecipientISO_CIV+RecipientISO_CMR+RecipientISO_COM+RecipientISO_CPV+RecipientISO_DJI+RecipientISO_ERI+RecipientISO_ETH+RecipientISO_GEO+RecipientISO_GHA+RecipientISO_GIN+RecipientISO_GMB+RecipientISO_GNB+RecipientISO_GRD+RecipientISO_GUY+RecipientISO_HND+RecipientISO_HTI+RecipientISO_IND+RecipientISO_KEN+RecipientISO_KGZ+RecipientISO_KHM+RecipientISO_KIR+RecipientISO_LAO+RecipientISO_LBR+RecipientISO_LKA+RecipientISO_LSO+RecipientISO_MDA+RecipientISO_MDG+RecipientISO_MLI+RecipientISO_MMR+RecipientISO_MNG+RecipientISO_MOZ+RecipientISO_MRT+RecipientISO_MWI+RecipientISO_NER+RecipientISO_NGA+RecipientISO_NIC+RecipientISO_NPL+RecipientISO_PAK+RecipientISO_PNG+RecipientISO_RWA+RecipientISO_SDN+RecipientISO_SEN+RecipientISO_SLB+RecipientISO_SLE+RecipientISO_STP+RecipientISO_TCD+RecipientISO_TGO+RecipientISO_TJK+RecipientISO_TON+RecipientISO_TZA+RecipientISO_UGA+RecipientISO_UZB+RecipientISO_VNM+RecipientISO_VUT+RecipientISO_WSM+RecipientISO_YEM+RecipientISO_ZMB+RecipientISO_ZWE"
.yr_alloc <- "Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018"
.adapt_sector_tail <- "Water+Transport+Agri+EnvProtect+MultiSec+Disaster"
.miti_sector_tail  <- "Water+Transport+Agri+EnvProtect+MultiSec+Energy"

.bert_country_fe <- function(df, kinds = "ProviderISO_", min_share = 0.005) {
  patt <- paste(kinds, collapse = "|")
  iso_vars <- grep(patt, names(df), value = TRUE)
  if (length(iso_vars) == 0L) return(character(0))
  iso_long <- df %>%
    dplyr::select(dplyr::all_of(iso_vars)) %>%
    dplyr::mutate(.row = dplyr::row_number()) %>%
    tidyr::pivot_longer(cols = dplyr::all_of(iso_vars),
                        names_to = "iso", values_to = "presence",
                        values_drop_na = TRUE) %>%
    dplyr::filter(.data$presence == 1)
  shares <- iso_long %>%
    dplyr::group_by(.data$iso) %>%
    dplyr::summarize(share = dplyr::n() / nrow(df), .groups = "drop")
  shares %>%
    dplyr::filter(.data$share > min_share) %>%
    dplyr::pull(.data$iso)
}

.adapt_bert_p1 <- .bert_country_fe(reg3, kinds = "ProviderISO_")
.adapt_bert_p2 <- .bert_country_fe(reg3, kinds = c("ProviderISO_", "RecipientISO_"))
.miti_bert_m1  <- .bert_country_fe(reg6, kinds = "ProviderISO_")
.miti_bert_m2  <- .bert_country_fe(reg6, kinds = c("ProviderISO_", "RecipientISO_"))


# -----------------------------------------------------------------------------
# Slim formula construction.
# Returns a one-string formula:
#   "<LHS> ~ <h1 covariates + provider FE> | <h2 covariates + h2 FE>"
# Guard: if a stage has zero selected vars, the slim formula keeps FE only and
# emits a warning so the user is aware.
# -----------------------------------------------------------------------------
.compose <- function(parts) {
  parts <- parts[nzchar(parts) & !is.na(parts)]
  paste(parts, collapse = " + ")
}

.build_slim_formula <- function(lhs, h1_vars, h2_vars, h1_fe, h2_fe, label) {
  if (length(h1_vars) == 0L) {
    warning("[", label, "] no h1 vars selected — slim h1 keeps FE only.")
  }
  if (length(h2_vars) == 0L) {
    warning("[", label, "] no h2 vars selected — slim h2 keeps FE only.")
  }
  rhs_h1 <- .compose(c(h1_vars, h1_fe))
  rhs_h2 <- .compose(c(h2_vars, h2_fe))
  paste0(lhs, " ~ ", rhs_h1, " | ", rhs_h2)
}

f_adapt_han_slim <- .build_slim_formula(
  lhs     = "AdaptAmount",
  h1_vars = .manifest$selected$adapt_h1,
  h2_vars = .manifest$selected$adapt_h2,
  h1_fe   = .p_sel,
  h2_fe   = paste(.r_alloc, .p_alloc, .yr_alloc, .adapt_sector_tail, sep = "+"),
  label   = "adapt_han_slim"
)

f_miti_han_slim <- .build_slim_formula(
  lhs     = "MitiAmount",
  h1_vars = .manifest$selected$miti_h1,
  h2_vars = .manifest$selected$miti_h2,
  h1_fe   = .p_sel,
  h2_fe   = paste(.r_alloc, .p_alloc, .yr_alloc, .miti_sector_tail, sep = "+"),
  label   = "miti_han_slim"
)

f_adapt_bert_slim <- .build_slim_formula(
  lhs     = "AdaptAmount",
  h1_vars = .manifest$selected$adapt_h1,
  h2_vars = .manifest$selected$adapt_h2,
  h1_fe   = paste(.adapt_bert_p1, collapse = "+"),
  h2_fe   = paste(c(.adapt_bert_p2, .yr_alloc,
                    "climate_class_Climate_Adaptation"),
                  collapse = "+"),
  label   = "adapt_bert_slim"
)

f_miti_bert_slim <- .build_slim_formula(
  lhs     = "MitiAmount",
  h1_vars = .manifest$selected$miti_h1,
  h2_vars = .manifest$selected$miti_h2,
  h1_fe   = paste(.miti_bert_m1, collapse = "+"),
  h2_fe   = paste(c(.miti_bert_m2, .yr_alloc,
                    "climate_class_Air_Pollution_Mitigation",
                    "climate_class_Hydro_Power_Plants",
                    "climate_class_Renewable_energy",
                    "climate_class_Solar_Energy",
                    "climate_class_Wind_power_farms",
                    "climate_class_Green_Growth_Strategies"),
                  collapse = "+"),
  label   = "miti_bert_slim"
)

cat("\nSlim formulas built. First 200 chars of each:\n")
for (nm in c("f_adapt_han_slim", "f_adapt_bert_slim",
             "f_miti_han_slim",  "f_miti_bert_slim")) {
  cat(sprintf("  %-20s : %s ...\n", nm, substr(get(nm), 1L, 200L)))
}


# -----------------------------------------------------------------------------
# Fit four slim hurdles in parallel — uncorrelated then correlated, mirroring
# HurdleRegHuei_parallel.R's two-batch pattern. We fit corr=FALSE first as a
# numerically stable warm-start, then corr=TRUE.
# -----------------------------------------------------------------------------
slim_jobs <- list(
  adapt_han  = list(f = f_adapt_han_slim,  d = reg1),
  adapt_bert = list(f = f_adapt_bert_slim, d = reg3),
  miti_han   = list(f = f_miti_han_slim,   d = reg4),
  miti_bert  = list(f = f_miti_bert_slim,  d = reg6)
)

future::plan(future::multisession,
             workers = min(4L, parallelly::availableCores()))

dir.create(.out_dir, recursive = TRUE, showWarnings = FALSE)
.uncorr_cache <- file.path(.out_dir, "slim_uncorr.rds")
.corr_cache   <- file.path(.out_dir, "slim_corr.rds")

if (file.exists(.uncorr_cache)) {
  cat("\nLoading cached uncorrelated fits from ", .uncorr_cache, " ...\n", sep = "")
  slim_uncorr <- readRDS(.uncorr_cache)
} else {
  cat("\nFitting 4 slim mhurdles (corr = FALSE) in parallel ...\n")
  t0 <- proc.time()
  slim_uncorr <- future.apply::future_lapply(slim_jobs, function(job) {
    library(mhurdle)
    do.call(mhurdle, list(
      formula      = as.formula(job$f),
      data         = job$d,
      dist         = "n",
      h2           = TRUE,
      corr         = FALSE,
      method       = "Bhhh",
      print.level  = 0,
      finalHessian = TRUE
    ))
  }, future.seed = NULL)
  cat(sprintf("  uncorrelated batch done in %.1f s\n", (proc.time() - t0)[3]))
  saveRDS(slim_uncorr, .uncorr_cache)
  cat("  cached to ", .uncorr_cache, "\n", sep = "")
}

if (file.exists(.corr_cache)) {
  cat("\nLoading cached correlated fits from ", .corr_cache, " ...\n", sep = "")
  slim_corr <- readRDS(.corr_cache)
} else {
  cat("\nFitting 4 slim mhurdles (corr = TRUE) in parallel ...\n")
  t0 <- proc.time()
  slim_corr <- future.apply::future_lapply(slim_jobs, function(job) {
    library(mhurdle)
    do.call(mhurdle, list(
      formula      = as.formula(job$f),
      data         = job$d,
      dist         = "n",
      h2           = TRUE,
      corr         = TRUE,
      method       = "Bhhh",
      print.level  = 0,
      finalHessian = TRUE
    ))
  }, future.seed = NULL)
  cat(sprintf("  correlated batch done in %.1f s\n", (proc.time() - t0)[3]))
  saveRDS(slim_corr, .corr_cache)
  cat("  cached to ", .corr_cache, "\n", sep = "")
}

future::plan(future::sequential)


# -----------------------------------------------------------------------------
# Merge uncorrelated + correlated coefficient tables into the same long format
# used by HurdleRegHuei_parallel.R's .make_csv_df(). Replicated locally to avoid
# sourcing the parallel script.
# -----------------------------------------------------------------------------
#' Safely build the slim coefficient table.
#'
#' Resilient to:
#'   - singular Hessian / OPG inversion failures inside summary.mhurdle ->
#'     vcov.mhurdle -> solve. Falls back to coef() with NA SEs/t/p.
#'   - missing r.squared components.
.coef_table_safe <- function(model, prefix) {
  s <- tryCatch(summary(model), error = function(e) {
    message("  summary(", prefix, ") failed: ", conditionMessage(e),
            " — falling back to coef() with NA SEs.")
    NULL
  })
  if (!is.null(s) && !is.null(s$coefficients)) {
    df <- as.data.frame(s$coefficients)
  } else {
    cf <- tryCatch(coef(model), error = function(e) numeric(0))
    df <- data.frame(
      Estimate     = unname(cf),
      `Std. Error` = NA_real_,
      `t-value`    = NA_real_,
      `Pr(>|t|)`   = NA_real_,
      check.names  = FALSE,
      row.names    = names(cf)
    )
  }
  names(df) <- paste0(prefix, "_", names(df))
  df$Variable <- rownames(df)
  list(df = df, summary = s)
}

.make_csv_df_slim <- function(stn, slnd) {
  u <- .coef_table_safe(stn,  "Uncorrelated")
  c <- .coef_table_safe(slnd, "Correlated")
  out <- merge(u$df, c$df, by = "Variable", all = TRUE)
  .pull <- function(s, key) {
    if (is.null(s) || is.null(s$r.squared) || !(key %in% names(s$r.squared))) {
      NA_real_
    } else {
      unname(s$r.squared[key])
    }
  }
  extra <- data.frame(
    Variable = c("Log-likelihood",
                 "McFadden Pseudo-R2",
                 "Coefficient of Determination (R²)"),
    Uncorrelated_Estimate = c(
      tryCatch(as.numeric(logLik(stn)),  error = function(e) NA_real_),
      .pull(u$summary, "lratio"),
      .pull(u$summary, "coefdet")
    ),
    Correlated_Estimate = c(
      tryCatch(as.numeric(logLik(slnd)), error = function(e) NA_real_),
      .pull(c$summary, "lratio"),
      .pull(c$summary, "coefdet")
    )
  )
  out <- dplyr::bind_rows(out, extra) %>%
    dplyr::select(Variable, dplyr::everything())
  out
}

#' Wrap a single CSV-write so one model's failure doesn't poison the others.
.write_one <- function(stn, slnd, path, label) {
  ok <- tryCatch({
    df <- .make_csv_df_slim(stn, slnd)
    write.csv(df, path, row.names = FALSE)
    cat("  wrote ", label, " -> ", path, " (", nrow(df), " rows)\n", sep = "")
    TRUE
  }, error = function(e) {
    message("  FAILED to write ", label, ": ", conditionMessage(e))
    FALSE
  })
  invisible(ok)
}


# -----------------------------------------------------------------------------
# Write outputs
# -----------------------------------------------------------------------------
dir.create(file.path(.out_dir, "adaptation"),
           recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(.out_dir, "mitigation"),
           recursive = TRUE, showWarnings = FALSE)

cat("\nWriting slim coefficient tables ...\n")
.write_one(slim_uncorr$adapt_han,  slim_corr$adapt_han,
           file.path(.out_dir, "adaptation/combined_regression_results_slim_han.csv"),
           "adapt_han")
.write_one(slim_uncorr$adapt_bert, slim_corr$adapt_bert,
           file.path(.out_dir, "adaptation/combined_regression_results_slim_bert.csv"),
           "adapt_bert")
.write_one(slim_uncorr$miti_han,   slim_corr$miti_han,
           file.path(.out_dir, "mitigation/combined_regression_results_slim_han.csv"),
           "miti_han")
.write_one(slim_uncorr$miti_bert,  slim_corr$miti_bert,
           file.path(.out_dir, "mitigation/combined_regression_results_slim_bert.csv"),
           "miti_bert")

saveRDS(
  list(uncorr = slim_uncorr, corr = slim_corr,
       formulas = list(
         adapt_han  = f_adapt_han_slim,
         adapt_bert = f_adapt_bert_slim,
         miti_han   = f_miti_han_slim,
         miti_bert  = f_miti_bert_slim
       ),
       manifest = .manifest),
  file.path(.out_dir, "slim_models.rds")
)

cat("\nSlim re-estimation complete. Outputs:\n")
cat("  ", file.path(.out_dir, "adaptation/combined_regression_results_slim_han.csv"), "\n", sep = "")
cat("  ", file.path(.out_dir, "adaptation/combined_regression_results_slim_bert.csv"), "\n", sep = "")
cat("  ", file.path(.out_dir, "mitigation/combined_regression_results_slim_han.csv"), "\n", sep = "")
cat("  ", file.path(.out_dir, "mitigation/combined_regression_results_slim_bert.csv"), "\n", sep = "")
cat("  ", file.path(.out_dir, "slim_models.rds"), "\n", sep = "")
