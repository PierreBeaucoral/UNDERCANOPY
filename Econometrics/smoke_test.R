# =============================================================================
# Econometrics/smoke_test.R
# -----------------------------------------------------------------------------
# Purpose
#   Clone-runnable replication smoke test for the UNDERCANOPY package. Confirms
#   that a fresh checkout can: (a) unpack the shipped data, (b) run the variable
#   selection step end-to-end, and (c) run the lightest identification script
#   (common-support hurdle) end-to-end, producing the expected output files.
#
# What it runs (and ONLY this — it is deliberately light)
#   1. Unzip Econometrics/Data/Data.zip (skipped if reg1.csv already present).
#   2. select_variables.R  -> regressions/var_selection/{...}
#   3. identification_common_support.R -> regressions/common_support/{...}
#
# Data
#   Uses ONLY data shipped inside the repo (Econometrics/Data/Data.zip). No
#   external drive, no network, no GPU. The six top-level CSVs in the zip
#   (reg1/2/3 and their _mitigation counterparts) are everything these two
#   steps need.
#
# Bootstrap override
#   Sys.setenv(UC_N_BOOT = "5") is set so that any script honouring the
#   UC_N_BOOT env-var (the AIPW / Lee-bounds / APE scripts) would bootstrap
#   tiny. The common-support step itself does NOT bootstrap; the override is
#   set here for completeness and to document the hook for anyone extending
#   this smoke test to the heavier identification scripts.
#
# Exit status
#   Prints a PASS/FAIL table over the steps. Any failed step or missing/empty
#   output file -> quit(status = 1L). Full success -> "SMOKE TEST: PASS" and
#   quit(status = 0L). Suitable as a CI gate.
#
# Side effect (local runs)
#   To make the assertions an HONEST fresh-generation check, the test deletes
#   the 8 expected outputs before regenerating them. Running it locally will
#   therefore show the committed regressions/var_selection/* and
#   regressions/common_support/* tables as modified afterwards (cosmetic: a UTC
#   timestamp in var_selection; mhurdle is mildly run-sensitive in
#   common_support). Discard these with `git checkout -- Econometrics/regressions`.
#   In CI the checkout is ephemeral, so this is a non-issue there.
# =============================================================================

suppressPackageStartupMessages({
  library(here)
  library(data.table)
  library(dplyr)
  library(jsonlite)
})

# .git at the repo root lets here() resolve to the repo root from anywhere.
ROOT <- here::here()

# Honour the bootstrap override hook (tiny). See header note.
Sys.setenv(UC_N_BOOT = "5")

# -----------------------------------------------------------------------------
# Status accumulation helpers.
# -----------------------------------------------------------------------------
step_names  <- character(0)
step_status <- character(0)   # "OK" or "FAIL"

record_step <- function(name, ok) {
  step_names  <<- c(step_names, name)
  step_status <<- c(step_status, if (isTRUE(ok)) "OK" else "FAIL")
  invisible(ok)
}

# A step wrapper that reports the step name + condition message rather than
# aborting silently. Warnings are handled with withCallingHandlers (which does
# NOT unwind the call stack) so a benign warning fired mid-script does not abort
# the rest of `expr` — we log it and invokeRestart("muffleWarning") to resume
# exactly where the warning fired. Only ERRORS (caught by the surrounding
# tryCatch) mark a step FAIL. This mirrors a plain source(), where warnings do
# not unwind; a `warning = ...` handler in tryCatch would unwind and produce a
# false PASS by skipping the remainder of the step.
run_step <- function(name, expr) {
  ok <- tryCatch(
    withCallingHandlers(
      {
        force(expr)
        TRUE
      },
      warning = function(w) {
        message(sprintf("[WARN] %s: %s", name, conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      message(sprintf("[FAIL] %s: %s", name, conditionMessage(e)))
      FALSE
    }
  )
  record_step(name, ok)
}

# Non-empty-file checker.
file_ok <- function(p) file.exists(p) && isTRUE(file.info(p)$size > 0)

# -----------------------------------------------------------------------------
# Expected GENERATED outputs (defined once, reused by cleanup + assertions).
#
# These are all regenerated artifacts of steps 2-3 — three from var_selection
# and five from common_support. They are committed in the repo, so unless we
# delete them first, a step that silently fails to (re)write its output would
# still pass the assertions: a stale-file false PASS. We therefore unlink them
# BEFORE the steps run, forcing the smoke test to verify FRESH generation.
#
# NOTE: the unzipped reg*.csv files are INPUTS, not outputs — they are NOT in
# this vector and must NOT be deleted.
# -----------------------------------------------------------------------------
EXPECTED <- c(
  file.path(ROOT, "Econometrics", "regressions", "var_selection",
            "selection_diagnostic.csv"),
  file.path(ROOT, "Econometrics", "regressions", "var_selection",
            "selected_variables.json"),
  file.path(ROOT, "Econometrics", "regressions", "var_selection",
            "selection_summary.txt"),
  file.path(ROOT, "Econometrics", "regressions", "common_support",
            "coef_adapt.csv"),
  file.path(ROOT, "Econometrics", "regressions", "common_support",
            "coef_miti.csv"),
  file.path(ROOT, "Econometrics", "regressions", "common_support",
            "cells_C_adapt.rds"),
  file.path(ROOT, "Econometrics", "regressions", "common_support",
            "cells_C_miti.rds"),
  file.path(ROOT, "Econometrics", "regressions", "common_support",
            "summary.rds")
)

# Delete any pre-existing generated outputs so the assertions step verifies
# fresh generation rather than committed stale files. Done BEFORE steps 2-3
# (and after the unzip step is defined; the actual unzip runs below). The
# unzipped CSV inputs are deliberately left untouched.
unlink(EXPECTED[file.exists(EXPECTED)])

# =============================================================================
# Step 1 — Unzip shipped data (only if reg1.csv is absent).
# =============================================================================
run_step("unzip_data", {
  data_dir <- file.path(ROOT, "Econometrics", "Data")
  reg1     <- file.path(data_dir, "reg1.csv")
  if (!file.exists(reg1)) {
    zip <- file.path(data_dir, "Data.zip")
    if (!file.exists(zip)) {
      stop(sprintf("Data archive not found: %s", zip))
    }
    utils::unzip(zip, exdir = data_dir)
  }
  if (!file.exists(reg1)) {
    stop("reg1.csv still absent after unzip attempt")
  }
})

# =============================================================================
# Step 2 — Variable selection, end-to-end.
#
# select_variables.R guards its wrapper with `if (sys.nframe() == 0L)`, which is
# FALSE when sourced. We therefore source the file (to define the functions)
# and then CALL select_variables() ourselves with the project default inputs.
# =============================================================================
run_step("select_variables", {
  source(file.path(ROOT, "Econometrics", "select_variables.R"))

  main_dir <- file.path(ROOT, "Econometrics", "regressions", "main")
  out_dir  <- file.path(ROOT, "Econometrics", "regressions", "var_selection")

  select_variables(
    adapt_han_path  = file.path(main_dir, "adaptation", "combined_regression_results.csv"),
    adapt_bert_path = file.path(main_dir, "adaptation", "combined_regression_results3.csv"),
    miti_han_path   = file.path(main_dir, "mitigation", "combined_regression_results.csv"),
    miti_bert_path  = file.path(main_dir, "mitigation", "combined_regression_results3.csv"),
    alpha           = 0.05,
    out_dir         = out_dir
  )
  invisible(NULL)
})

# =============================================================================
# Step 3 — Common-support identification script, end-to-end.
#
# This script has no sys.nframe guard: sourcing executes it fully. It reads the
# six reg CSVs, fits dyad-year hurdles (mhurdle, with a probit/OLS fallback),
# and writes its outputs. It is the lightest identification script — no
# bootstrap.
# =============================================================================
run_step("common_support", {
  source(file.path(ROOT, "Econometrics", "Identification",
                   "identification_common_support.R"))
  invisible(NULL)
})

# =============================================================================
# Step 4 — Assert expected outputs exist and are non-empty.
# =============================================================================
run_step("assertions", {
  # Reuse the EXPECTED vector defined near the top; these were unlinked before
  # the steps ran, so their presence here proves fresh generation.
  missing_or_empty <- EXPECTED[!vapply(EXPECTED, file_ok, logical(1))]
  if (length(missing_or_empty) > 0L) {
    stop(sprintf("Missing or empty output file(s):\n  - %s",
                 paste(missing_or_empty, collapse = "\n  - ")))
  }
})

# =============================================================================
# Summary table + exit status.
# =============================================================================
cat("\n", strrep("=", 50), "\n", sep = "")
cat("REPLICATION SMOKE TEST — SUMMARY\n")
cat(strrep("=", 50), "\n", sep = "")
width <- max(nchar(step_names))
for (i in seq_along(step_names)) {
  cat(sprintf("  %-*s  %s\n", width, step_names[i], step_status[i]))
}
cat(strrep("=", 50), "\n", sep = "")

if (any(step_status == "FAIL")) {
  cat("SMOKE TEST: FAIL\n")
  quit(status = 1L)
} else {
  cat("SMOKE TEST: PASS\n")
  quit(status = 0L)
}
