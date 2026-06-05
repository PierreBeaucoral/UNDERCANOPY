# Build Report — Econometrics revision layer

**Date:** 2026-06-05
**Branch:** `revision-econometrics-2026-06`
**Scope:** Add the revised R estimation + identification/robustness/APE layer to
the public replication package, refactored for repo-relative reproducibility.
No git operations, no heavy analysis run. `Estimations.R` left untouched.

---

## 1. File inventory added

### Scripts — Group A (`Econometrics/`, flat) — 4 files
- `HurdleRegHuei_parallel.R`
- `HurdleRegHuei_parallel_robustness.R`
- `HurdleRegHuei_selected.R`
- `select_variables.R`

### Scripts — Group B (`Econometrics/Identification/`) — 7 files
- `_helpers.R`
- `identification_aipw.R`
- `identification_aipw_h1.R`
- `identification_lee_bounds.R`
- `identification_lee_bounds_h1.R`
- `identification_common_support.R`
- `marginal_effects_ape.R`

No `*.bak*` files copied (the source `marginal_effects_ape.R.bak.2026-05-12` was
excluded).

### Result tables (`Econometrics/regressions/`) — 34 files
- `main/adaptation/`: `combined_regression_results.csv`, `combined_regression_results3.csv`
- `main/mitigation/`: `combined_regression_results.csv`, `combined_regression_results3.csv`
- `robustness/adaptation/`: `combined_regression_results{,2,3}.csv`
- `robustness/mitigation/`: `combined_regression_results{,2,3}.csv`
- `common_support/`: `coef_adapt.csv`, `coef_miti.csv`, `summary.rds` (253 B — read by the paper)
- `aipw/`: `coef_adapt.csv`, `coef_miti.csv`, `trim_stability.csv`
- `aipw_h1/`: `coef_adapt.csv`, `coef_miti.csv`, `trim_stability.csv`
- `lee_bounds/`: `bounds_adapt.csv`, `bounds_miti.csv`
- `lee_bounds_h1/`: `bounds_adapt.csv`, `bounds_miti.csv`
- `ape/`: `ape_adapt.csv`, `ape_miti.csv`
- `selected/adaptation/`: `combined_regression_results_slim_han.csv`, `combined_regression_results_slim_bert.csv`
- `selected/mitigation/`: `combined_regression_results_slim_han.csv`, `combined_regression_results_slim_bert.csv`
- `var_selection/`: `selected_variables.json`, `selection_diagnostic.csv`, `selection_summary.txt`, `changing_audit.csv`, `prose_vs_audit.csv`

All 34 verified present and non-empty. No excluded heavy objects copied
(confirmed zero matches for `diagnostics.rds`, `bounds_diagnostics.rds`,
`ape_diagnostics.rds`, `slim_models.rds`, `slim_uncorr.rds`, `slim_corr.rds`,
`cells_C_*.rds`, `ape/ARCHIVE/`).

### Documentation — 4 created, 2 patched
- CREATED `Econometrics/regressions/readme.md`
- CREATED `Econometrics/Identification/readme.md`
- CREATED `Econometrics/external-data.md`
- PATCHED `Econometrics/readme.md` (appended "Revised analysis (2025 revision)" section; existing content intact)
- PATCHED `README.md` (Econometrics block of the repo-tree code fence only)
- CREATED `.here` sentinel at repo root (empty, for ZIP-download `here::here()` resolution)

**Counts:** 11 scripts, 34 result files, 4 docs created, 2 docs patched, 1 sentinel.

---

## 2. Path-rewrite table (old -> new)

### `HurdleRegHuei_parallel.R`
| Old | New |
|-----|-----|
| `setwd("/Users/...Determinant of climate finance")` | (deleted) |
| `rm(list = ls())` | (deleted) |
| (no `library(here)`) | `library(here)` added to lib block |
| `fread('/Users/.../Travaux CRS/Data/DataPB.csv')` | `fread(here::here("Econometrics","Data","DataPB.csv"))` + external comment |
| `fread("Data/climate_finance_total.csv")` | `fread(here::here("Econometrics","Data","climate_finance_total.csv"))` + external comment |
| `read.csv("Data/Adaptation with gravity vars amended FULL Feb 14 2023.csv")` | `read.csv(here::here("Climate finance estimation","Raw Data","Adaptation with gravity vars amended FULL Feb 14 2023.csv"))` |
| `read.csv("Data/Mitigation ... 2023.csv")` | `read.csv(here::here("Climate finance estimation","Raw Data","Mitigation ... 2023.csv"))` |
| `dir.create("./regressions/main/{adaptation,mitigation}")` | `dir.create(here::here("Econometrics","regressions","main","{adaptation,mitigation}"))` |
| `"./regressions/main/adaptation/combined_regression_results{,2,3}.csv"` | `here::here("Econometrics","regressions","main","adaptation","combined_regression_results{,2,3}.csv")` |
| `"./regressions/main/mitigation/combined_regression_results{,2,3}.csv"` | `here::here("Econometrics","regressions","main","mitigation","combined_regression_results{,2,3}.csv")` |
| `"./Redaction/vif_adapt_main.csv"` / `"./Redaction/vif_miti_main.csv"` | `here::here("Econometrics","regressions","main","vif_{adapt,miti}_main.csv")` |
| `"./Redaction/reg1.csv"` … `reg3.csv` | `here::here("Econometrics","Data","reg{1,2,3}.csv")` |
| `"./Redaction/reg{1,2,3}_mitigation.csv"` | `here::here("Econometrics","Data","reg{1,2,3}_mitigation.csv")` |
| `"./Redaction/summary_sample.csv"` | `here::here("Econometrics","regressions","main","summary_sample.csv")` |
| `dir.create("./regressions/ipw/{adaptation,mitigation}")` | `dir.create(here::here("Econometrics","regressions","ipw","{adaptation,mitigation}"))` |
| `"./regressions/ipw/{adaptation,mitigation}/combined_regression_results_cs.csv"` | `here::here("Econometrics","regressions","ipw","{adaptation,mitigation}","combined_regression_results_cs.csv")` |
| `"./regressions/ipw/smd_balance.csv"` | `here::here("Econometrics","regressions","ipw","smd_balance.csv")` |
| message strings `./regressions/ipw/...` | `Econometrics/regressions/ipw/...` |

### `HurdleRegHuei_parallel_robustness.R`
Identical pattern to `parallel.R`, except output target is `robustness/` not
`main/`, and the VIF files are `vif_{adapt,miti}_robustness.csv` (now written to
`regressions/robustness/`). `summary_sample.csv` -> `regressions/robustness/`.
This script has no `ipw`/`smd` block. `setwd`/`rm(list = ls())` deleted;
`library(here)` added; `DataPB.csv` / `climate_finance_total.csv` / gravity CSVs
rewritten exactly as above.

### `select_variables.R`
Already used `here::here()` + `library(here)`. Prefixed the four input paths and
the `out_dir` with `Econometrics`:
| Old | New |
|-----|-----|
| `file.path(proj,"regressions/main/adaptation/combined_regression_results.csv")` | `file.path(proj,"Econometrics","regressions","main","adaptation","combined_regression_results.csv")` |
| `file.path(proj,"regressions/main/adaptation/combined_regression_results3.csv")` | `…"Econometrics","regressions","main","adaptation","combined_regression_results3.csv"` |
| `…/mitigation/combined_regression_results.csv` | `…"Econometrics",…,"mitigation","combined_regression_results.csv"` |
| `…/mitigation/combined_regression_results3.csv` | `…"Econometrics",…,"mitigation","combined_regression_results3.csv"` |
| `file.path(proj,"regressions/var_selection")` | `file.path(proj,"Econometrics","regressions","var_selection")` |
(Inner writes use the passed `out_dir`, so no further change needed.)

### `HurdleRegHuei_selected.R`
Already used `here::here()` + `library(here)`. Rewrites:
| Old | New |
|-----|-----|
| `file.path(.proj,"regressions/var_selection/selected_variables.json")` | `file.path(.proj,"Econometrics","regressions","var_selection","selected_variables.json")` |
| `file.path(.proj,"regressions/selected")` (`.out_dir`) | `file.path(.proj,"Econometrics","regressions","selected")` |
| `.csv_map` fallbacks `Redaction/reg{1,3,1_mitigation,3_mitigation}.csv` | `Econometrics/Data/reg{1,3,1_mitigation,3_mitigation}.csv` |
| message + 3 header comments referencing `Redaction/` and `regressions/` | updated to `Econometrics/Data/` and `Econometrics/regressions/` |
(All output writes use `.out_dir`, now correctly under `Econometrics`.)

### `_helpers.R`
No paths, no `setwd`/`rm`, only `library(here)`. **No changes required.**

### `identification_aipw.R`, `identification_aipw_h1.R`, `identification_lee_bounds.R`, `identification_lee_bounds_h1.R`, `identification_common_support.R`
Common pattern, all rewritten:
| Old | New |
|-----|-----|
| `ROOT <- "/Users/...Determinant of climate finance"` | `ROOT <- here::here()` |
| `source(file.path(ROOT,"scripts","_helpers.R"))` | `source(here::here("Econometrics","Identification","_helpers.R"))` |
| `out_dir <- file.path(ROOT,"regressions","<method>")` | `out_dir <- here::here("Econometrics","regressions","<method>")` |
| `fread(file.path(ROOT,"reg{1,2,3}.csv"))` | `fread(here::here("Econometrics","Data","reg{1,2,3}.csv"))` |
| `fread(file.path(ROOT,"Redaction","reg{1,2,3}_mitigation.csv"))` | `fread(here::here("Econometrics","Data","reg{1,2,3}_mitigation.csv"))` |
| header `# Inputs : ... Redaction/...` comments | updated to `Econometrics/Data/...` |
`<method>` is `aipw`, `aipw_h1`, `lee_bounds`, `lee_bounds_h1`, `common_support`
respectively. Output `saveRDS`/`fwrite` calls use `out_dir`, so all land under
`Econometrics/regressions/<method>/`. `library(here)` was already present.

### `marginal_effects_ape.R`
Self-contained (does **not** source `_helpers.R`); had no `library(here)`.
| Old | New |
|-----|-----|
| (no `library(here)`) | `library(here)` added to the `suppressPackageStartupMessages({...})` block |
| `ROOT <- "/Users/...Determinant of climate finance"` | `ROOT <- here::here()` |
| `out_dir <- file.path(ROOT,"regressions","ape")` | `out_dir <- here::here("Econometrics","regressions","ape")` |
| `fread(file.path(ROOT,"reg3.csv"))` | `fread(here::here("Econometrics","Data","reg3.csv"))` |
| `fread(file.path(ROOT,"Redaction","reg3_mitigation.csv"))` | `fread(here::here("Econometrics","Data","reg3_mitigation.csv"))` |
| header `# Inputs :` and a roxygen `#'` comment referencing `Redaction/` | updated to `Econometrics/Data/` |

`set.seed()` values were preserved byte-for-byte in every script
(`20260503L`, `20260505L`, etc.). No model formulas, `mhurdle()` calls, or
post-processing logic were altered.

---

## 3. Parse-check results (`Rscript -e 'invisible(parse(<file>))'`)

| File | Result |
|------|--------|
| `HurdleRegHuei_parallel.R` | PASS |
| `HurdleRegHuei_parallel_robustness.R` | PASS |
| `HurdleRegHuei_selected.R` | PASS |
| `select_variables.R` | PASS |
| `Identification/_helpers.R` | PASS |
| `Identification/identification_aipw.R` | PASS |
| `Identification/identification_aipw_h1.R` | PASS |
| `Identification/identification_lee_bounds.R` | PASS |
| `Identification/identification_lee_bounds_h1.R` | PASS |
| `Identification/identification_common_support.R` | PASS |
| `Identification/marginal_effects_ape.R` | PASS |

All 11 parse with no syntax errors. (Parse only — no script was sourced/run.)

---

## 4. Grep clean-check results

Scoped to the 11 copied/edited A/B files only:

```
setwd(            -> 0 matches
rm(list           -> 0 matches
/Users/           -> 0 matches
[A-Z]:[/\\] (drive paths) -> 0 matches
Thèse             -> 0 matches
Travaux CRS       -> 0 matches
Redaction/        -> 0 matches
ALL CLEAN (no matches)
```

`here::here` or `library(here)` present in all 11 files.

Note on a false positive: a naive `grep 'C:'` matches the prose string
`"Task C: Lee (2009) bounds"` in `identification_lee_bounds.R`. A proper
Windows-path regex (`[A-Z]:[/\\]`) returns zero, confirming no real drive path.
The only `setwd(`/`rm(list` hits anywhere under `Econometrics/` are in the
pre-existing `Estimations.R`, which is out of scope and was not modified.

---

## 5. Suspected-unused libraries flagged in `HurdleRegHuei_parallel*.R`

Per instruction, the large library block was left intact (functionality
preserved) and a `# NOTE: several libraries below are not required for
estimation; see build report` comment was added above each block.

A call-site scan (stripping comments and the `library()` declarations
themselves, counting representative functions/namespaces) found **zero** uses of
the following in either parallel script — these are the suspected-unused
libraries a reviewer may decide to prune:

`tmap`, `leaflet`, `treemap`, `ggalluvial`, `ggforce`, `flextable`, `skimr`,
`png`, `gtable`, `gridExtra`, `pander`, `kableExtra`, `rlist`, `hms`, `readxl`,
`knitr`, `purrr`.

Libraries with confirmed usage (do NOT prune): `readr` (`write_csv`/`read_csv`),
`ggplot2` (the `AdaptAmount` density plot), `dplyr`/`tidyverse`, `data.table`,
`fastDummies`, `mhurdle`, `texreg`, `car`, `future`, `future.apply`, `here`.

Caveat: this is a static scan. `tidyverse` re-exports `purrr`/`readr`, so a
zero count for `purrr` does not guarantee no transitive use; the recommendation
is advisory only and intentionally conservative (nothing was deleted).

---

## 6. Anomalies and judgment calls

1. **Gravity CSVs are zipped in the repo, not loose.** The task says to point
   scripts at `Climate finance estimation/Raw Data/<file>.csv`, but those files
   currently live inside
   `Climate finance estimation/Raw Data/adaptation and mitigation with gravity vars.zip`.
   I pointed the scripts at the loose `<file>.csv` paths (matching the repo
   README's stated tree, which already lists them loose) and documented the
   unzip step in `external-data.md`. **Action for the user:** the gravity zip
   must be unzipped into `Raw Data/` before `HurdleRegHuei_parallel*.R` will run.

2. **Six `reg*.csv` panels routed to `Econometrics/Data/`.** The original
   scripts wrote/read them under `Redaction/` (parallel) and read fallbacks from
   `Redaction/` (selected, identification). Per the spec, all six live in
   `Data.zip` -> `Econometrics/Data/`, so every read and write of these panels
   now targets `Econometrics/Data/`. This makes `HurdleRegHuei_parallel.R`
   re-emit them there, and the `selected` / identification scripts read them
   from there — consistent and self-contained.

3. **VIF and `summary_sample.csv` outputs re-homed.** These previously went to
   `Redaction/` (the paper-source folder, which is not part of this public
   package). Since `Redaction/` does not exist in the repo, I redirected them to
   the matching `regressions/main/` and `regressions/robustness/` folders so the
   scripts do not fail on a missing directory and outputs stay inside the
   shipped tree. These are not in the STEP-2 ship list, so they are generated on
   run (their parent dirs already exist).

4. **`ipw/` subfolder.** `HurdleRegHuei_parallel.R` Section 16 writes a
   common-support/IPW re-estimation and an SMD balance table to
   `regressions/ipw/`. This subfolder was not in the STEP-2 ship list, so it is
   not pre-populated; the script creates it (`dir.create(..., recursive=TRUE)`)
   on run. Documented in `regressions/readme.md`.

5. **`.here` sentinel created at repo root.** Empty file so `here::here()`
   resolves to the repo root for users who download a ZIP (no `.git`). With
   `.git` present it is redundant but harmless.

6. **`marginal_effects_ape.R` is self-contained.** Unlike the five
   `identification_*` scripts, it does not `source(_helpers.R)`; it had no
   `library(here)`. I added `library(here)` inside its existing
   `suppressPackageStartupMessages({...})` block and converted `ROOT`/`out_dir`
   /reg reads. Documented in `Identification/readme.md`.

7. **README tree used non-breaking spaces.** The repo-tree code fence in
   `README.md` indents with U+00A0 (nbsp) after each `│`. The Econometrics block
   was patched preserving that exact convention; only the Econometrics block was
   touched.

8. **Header comments updated for accuracy.** A few in-script header/roxygen
   comments referenced the old `Redaction/` and bare `regressions/` paths. I
   updated those comment lines so the grep clean-check passes and the docs match
   reality. No executable logic was changed by these comment edits.

9. **No live `install.packages()` introduced or left in A/B scripts.** The only
   `install.packages` references in the copied scripts are inside comments
   (documentation). The one live `install.packages(pkg)` in the repo is in
   `Estimations.R`, which is out of scope and untouched.

---

## Self-assessment

All STEP 1–5 requirements met: 11 scripts copied + refactored (parse-clean,
grep-clean), 34 result files shipped (present + non-empty, no heavy objects),
4 docs created + 2 patched, sentinel created, `Estimations.R` untouched, no git
or heavy-analysis run. Quality score: **93/100** (the -7 reflects items the user
must complete out-of-band: unzip the gravity CSVs and download the two external
files before the scripts can execute end-to-end).

---

## Round 2 fixes (2026-06-05)

Targeted follow-up addressing the one reproducibility advisory plus one trivial
INV item flagged by coder-critic (91/100, no blockers). Scope limited strictly
to these two fixes; no other code touched, no git, no analysis run.

### Fix 1 — texreg `write.table()` outputs re-routed into the `here()` tree

The six `write.table()` calls in each script wrote bare relative filenames (no
directory), so reruns landed in R's CWD instead of the shipped
`Econometrics/Results/` tree. Each is now wrapped in
`here::here("Econometrics", "Results", <same filename>)`. A single
`dir.create(here::here("Econometrics", "Results"), recursive = TRUE, showWarnings = FALSE)`
was added immediately before the first such write in each script. Filenames were
preserved byte-for-byte as the scripts currently emit them (model/texreg content
unchanged). CSV outputs already routed under `regressions/...` were left as-is.

Note: the script emits `"Baseline Result for Mitigation 103950 obs"` (single
spaces), whereas the shipped tree file is `Baseline Result for  Mitigation 103950 obs`
(double space after "for"). Per instruction ("do not change the filenames; match
whatever the script emits"), the single-space string was preserved. This
pre-existing emit-vs-shipped discrepancy is out of scope for this round.

`HurdleRegHuei_parallel.R` (post-edit line numbers):

| Line | Old | New |
|------|-----|-----|
| 690 (new) | — | `dir.create(here::here("Econometrics", "Results"), recursive = TRUE, showWarnings = FALSE)` |
| 691 | `write.table(result1, "Baseline Result for Adaptation")` | `write.table(result1, here::here("Econometrics", "Results", "Baseline Result for Adaptation"))` |
| 692 | `write.table(result2, "Rio Result for Adaptation")` | `write.table(result2, here::here("Econometrics", "Results", "Rio Result for Adaptation"))` |
| 693 | `write.table(result3, "ClimateFinanceBERT Result for Adaptation")` | `write.table(result3, here::here("Econometrics", "Results", "ClimateFinanceBERT Result for Adaptation"))` |
| 698 | `write.table(result_m1, "Baseline Result for Mitigation 103950 obs")` | `write.table(result_m1, here::here("Econometrics", "Results", "Baseline Result for Mitigation 103950 obs"))` |
| 699 | `write.table(result_m2, "Rio Result for Mitigation")` | `write.table(result_m2, here::here("Econometrics", "Results", "Rio Result for Mitigation"))` |
| 700 | `write.table(result_m3, "ClimateFinanceBERT Result for Mitigation")` | `write.table(result_m3, here::here("Econometrics", "Results", "ClimateFinanceBERT Result for Mitigation"))` |

`HurdleRegHuei_parallel_robustness.R` (post-edit line numbers):

| Line | Old | New |
|------|-----|-----|
| 692 (new) | — | `dir.create(here::here("Econometrics", "Results"), recursive = TRUE, showWarnings = FALSE)` |
| 693 | `write.table(result1, "Baseline Result for Adaptation")` | `write.table(result1, here::here("Econometrics", "Results", "Baseline Result for Adaptation"))` |
| 694 | `write.table(result2, "Rio Result for Adaptation")` | `write.table(result2, here::here("Econometrics", "Results", "Rio Result for Adaptation"))` |
| 695 | `write.table(result3, "ClimateFinanceBERT Result for Adaptation")` | `write.table(result3, here::here("Econometrics", "Results", "ClimateFinanceBERT Result for Adaptation"))` |
| 700 | `write.table(result_m1, "Baseline Result for Mitigation 103950 obs")` | `write.table(result_m1, here::here("Econometrics", "Results", "Baseline Result for Mitigation 103950 obs"))` |
| 701 | `write.table(result_m2, "Rio Result for Mitigation")` | `write.table(result_m2, here::here("Econometrics", "Results", "Rio Result for Mitigation"))` |
| 702 | `write.table(result_m3, "ClimateFinanceBERT Result for Mitigation")` | `write.table(result_m3, here::here("Econometrics", "Results", "ClimateFinanceBERT Result for Mitigation"))` |

Grep confirmation that no bare relative output filenames remain in the two scripts:

```
$ grep -nE 'write\.table\([^,]+, *"[^/)]' HurdleRegHuei_parallel.R HurdleRegHuei_parallel_robustness.R
NONE FOUND (good)
```

All 12 `write.table()` calls (6 per script) now use `here::here(...)`. No
`texreg(file=)`, `writeLines()`, or `sink()` file targets exist in either script.

### Fix 2 — INV-12 in-figure title removed

The distribution density plot carried an in-figure title (INV-12: titles belong
in LaTeX `\caption{}`, not the figure). The `title = "Distribution of AdaptAmount"`
argument was removed from `labs()`; axis labels (`x`, `y`) preserved. Plot is not
exported, so this is cosmetic compliance only.

- `HurdleRegHuei_parallel.R` line 822: `labs(title = "Distribution of AdaptAmount", x = "AdaptAmount", y = "Density")` -> `labs(x = "AdaptAmount", y = "Density")`
- `HurdleRegHuei_parallel_robustness.R` line 824: same change

Grep confirmation: `grep -n 'title = "Distribution'` returns NONE in both files.

### Verification

Both edited scripts re-parsed clean:

```
$ Rscript -e 'invisible(parse("HurdleRegHuei_parallel.R"))'             # PARSE OK
$ Rscript -e 'invisible(parse("HurdleRegHuei_parallel_robustness.R"))'  # PARSE OK
```
