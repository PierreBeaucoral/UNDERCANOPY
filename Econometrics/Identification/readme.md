# Identification and robustness layer (`Econometrics/Identification/`)

This folder implements the identification / robustness checks added in the 2025
revision. All scripts resolve paths with `here::here()` relative to the
repository root, read the six regression panels from `Econometrics/Data/`
(written by `HurdleRegHuei_parallel.R`, packaged in `Data.zip`), and write their
result tables to `Econometrics/regressions/<method>/`.

## Shared helper

- **`_helpers.R`** — common utilities (sample construction, propensity / outcome
  model fitting, trimming, table builders) sourced by the identification scripts
  via `source(here::here("Econometrics", "Identification", "_helpers.R"))`. It
  performs no I/O of its own and creates no output. `marginal_effects_ape.R` is
  self-contained and does **not** source `_helpers.R`.

## Methods, inputs, and outputs

| Script | Method | Inputs (`Econometrics/Data/`) | Outputs (`Econometrics/regressions/`) |
|--------|--------|-------------------------------|----------------------------------------|
| `identification_aipw.R` | AIPW (doubly-robust) trim-stability, **h2** allocation margin | `reg1.csv`, `reg3.csv`, `reg1_mitigation.csv`, `reg3_mitigation.csv` | `aipw/coef_adapt.csv`, `aipw/coef_miti.csv`, `aipw/trim_stability.csv`, `aipw/diagnostics.rds` |
| `identification_aipw_h1.R` | AIPW trim-stability, **h1** entry margin | same four reg panels | `aipw_h1/coef_adapt.csv`, `aipw_h1/coef_miti.csv`, `aipw_h1/trim_stability.csv`, `aipw_h1/diagnostics.rds` |
| `identification_lee_bounds.R` | Lee (2009) bounds, **h2** margin | same four reg panels | `lee_bounds/bounds_adapt.csv`, `lee_bounds/bounds_miti.csv`, `lee_bounds/bounds_diagnostics.rds` |
| `identification_lee_bounds_h1.R` | Lee (2009) bounds, **h1** margin | same four reg panels | `lee_bounds_h1/bounds_adapt.csv`, `lee_bounds_h1/bounds_miti.csv`, `lee_bounds_h1/bounds_diagnostics.rds` |
| `identification_common_support.R` | Common-support reweighting of the hurdle estimates | `reg{1,2,3}.csv`, `reg{1,2,3}_mitigation.csv` | `common_support/coef_adapt.csv`, `common_support/coef_miti.csv`, `common_support/summary.rds`, `common_support/cells_C_*.rds` |
| `marginal_effects_ape.R` | Average partial effects (APE) for the BERT main spec | `reg3.csv`, `reg3_mitigation.csv` | `ape/ape_adapt.csv`, `ape/ape_miti.csv`, `ape/ape_diagnostics.rds` |

Only the small CSV tables (and `common_support/summary.rds`, which the paper
reads) are shipped; the heavy `*_diagnostics.rds` / `cells_C_*.rds` objects are
regenerated on run.

## Run order

1. Run `HurdleRegHuei_parallel.R` first — it materialises the six reg panels in
   `Econometrics/Data/` (`reg1.csv`, `reg2.csv`, `reg3.csv`,
   `reg1_mitigation.csv`, `reg2_mitigation.csv`, `reg3_mitigation.csv`).
2. Then run any identification script independently. Each one re-reads the
   panels from disk, so they do not depend on each other and can be run in any
   order. Each sources `_helpers.R` (except `marginal_effects_ape.R`, which is
   self-contained).

## Notes

- The four AIPW / Lee-bounds scripts come in **h1** (entry) and **h2**
  (allocation) variants to bound and doubly-robust-check both stages of the
  double-hurdle decision separately.
- `set.seed()` is fixed at the top of every stochastic script (bootstraps use
  per-replicate seeds `SEED_BASE + b`); seeds are preserved exactly as in the
  original analysis.
- The APE script is computationally heavy (300 clustered bootstrap replicates,
  3-hour wall-clock budget). Do not run it as part of a quick parse/structure
  check.
