# Regression outputs (`Econometrics/regressions/`)

This folder holds the result tables for the 2025 revision of the climate-finance
determinants paper. Each subfolder is produced by a specific script in
`Econometrics/` or `Econometrics/Identification/`. Only the small, paper-facing
tables are shipped; heavy diagnostic objects (`*_diagnostics.rds`,
`slim_models.rds`, `cells_C_*.rds`, etc.) are intentionally **not** committed and
are regenerated when the corresponding script is run.

All paths in the scripts are resolved with `here::here()` relative to the
repository root (the repo ships a `.here` sentinel so this works from a ZIP
download without `.git`). Outputs therefore always land back in this tree.

| Subfolder | Produced by | What it feeds in the paper |
|-----------|-------------|----------------------------|
| `main/` | `HurdleRegHuei_parallel.R` | Main double-hurdle estimates (Han, Rio, BERT specs) for adaptation and mitigation — the headline coefficient tables. |
| `robustness/` | `HurdleRegHuei_parallel_robustness.R` | Robustness re-estimation of the main specs (alternative sample/controls) — robustness section tables. |
| `var_selection/` | `select_variables.R` | Han→BERT variable-selection manifest and diagnostics — defines the slim covariate set and documents the selection audit. |
| `selected/` | `HurdleRegHuei_selected.R` | Slim re-estimation of the four core models on the selected covariates — parsimony/selected-spec tables. |
| `common_support/` | `Identification/identification_common_support.R` | Common-support reweighted hurdle coefficients and SMD balance summary — identification robustness (overlap). |
| `aipw/` | `Identification/identification_aipw.R` | AIPW headline coefficients and trimming-stability table (h2 allocation margin) — doubly-robust identification check. |
| `aipw_h1/` | `Identification/identification_aipw_h1.R` | AIPW headline coefficients and trimming stability for the entry (h1) margin. |
| `lee_bounds/` | `Identification/identification_lee_bounds.R` | Lee (2009) treatment-effect bounds (h2 margin) — selection-bias bounding. |
| `lee_bounds_h1/` | `Identification/identification_lee_bounds_h1.R` | Lee (2009) bounds for the entry (h1) margin. |
| `ape/` | `Identification/marginal_effects_ape.R` | Average partial effects ($ million per dyad-sector-year) for the BERT main spec — economic-magnitudes table. |
| `ipw/` | `HurdleRegHuei_parallel.R` (Section 16) | Common-support / IPW re-estimation and SMD balance written during the main run. Not pre-shipped; created on run. |

## Files shipped per subfolder

- `main/adaptation`, `main/mitigation`: `combined_regression_results.csv` (Han),
  `combined_regression_results3.csv` (BERT). The `*2.csv` (Rio) main tables are
  not shipped to keep the package light; they regenerate on run.
- `robustness/adaptation`, `robustness/mitigation`:
  `combined_regression_results{,2,3}.csv` (Han, Rio, BERT).
- `selected/adaptation`, `selected/mitigation`:
  `combined_regression_results_slim_han.csv`,
  `combined_regression_results_slim_bert.csv`.
- `common_support`: `coef_adapt.csv`, `coef_miti.csv`, `summary.rds`
  (the paper reads `summary.rds`).
- `aipw`, `aipw_h1`: `coef_adapt.csv`, `coef_miti.csv`, `trim_stability.csv`.
- `lee_bounds`, `lee_bounds_h1`: `bounds_adapt.csv`, `bounds_miti.csv`.
- `ape`: `ape_adapt.csv`, `ape_miti.csv`.
- `var_selection`: `selected_variables.json`, `selection_diagnostic.csv`,
  `selection_summary.txt`, `changing_audit.csv`, `prose_vs_audit.csv`.

## Not shipped (regenerate on run)

`aipw*/diagnostics.rds`, `lee_bounds*/bounds_diagnostics.rds`,
`ape/ape_diagnostics.rds`, `selected/slim_models.rds`,
`selected/slim_uncorr.rds`, `selected/slim_corr.rds`,
`common_support/cells_C_*.rds`, and any `ape/ARCHIVE/`. These are large
intermediate objects; the scripts recreate them when run.
