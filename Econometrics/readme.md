# Determinants Analysis of Bilateral Climate Finance

This script performs an in‐depth regression analysis to investigate the determinants of bilateral mitigation finance. Using newly created datasets derived from both Rio markers and ClimateFinanceBERT data, it applies double hurdle (two-part) models to examine both the decision to commit finance and the amount committed. The analysis leverages advanced data processing, variable transformation, and model comparison techniques to produce robust, replicable results.

---

## Overview

The script is organized into the following major steps:

1. **Data Preparation and Cleaning:**
   - **Loading Data:**  
     Raw data for climate finance are loaded from preprocessed sources. Three distinct datasets are used:
     - **Data from Han et al. (2023):**  
       Coming from the replication package of Han et al. (2023)
     - **Rio Data for Adaptation/Mitigation:**  
       Extracted from the broader CRS dataset, this data subset focuses on projects identified as mitigation-related based on the `ClimateMitigation` and `ClimateAdaptation` variable.
     - **ClimateFinanceBERT Data for Adaptation/Mitigation:**  
       This dataset is derived from the ClimateFinanceBERT classification, filtered to include only projects with a meta-category of "Mitigation."
   - **Data Transformation:**  
     Key variables (e.g., GDP, population, fiscal indicators, and distance) are transformed (e.g., logarithmically scaled, normalized) to improve model fit. Missing values for variables such as MDB and finance amounts are set to zero, and extremely small deals (e.g., less than 100 USD) are removed.

2. **Regression Modeling:**
   - **Double Hurdle Model (mhurdle):**  
     The script fits double hurdle regression models to account for both the selection (whether finance is committed) and the intensity (the amount of finance) decisions:
     - **Uncorrelated Model:**  
       The first model is estimated without allowing correlation between the hurdle parts.
     - **Correlated Model:**  
       The model is then updated to incorporate correlations between the selection and intensity equations.
   - **Model Comparison and Reporting:**  
     Coefficient estimates, standard errors, and goodness-of-fit statistics (log-likelihood, McFadden pseudo-R², and coefficient of determination) are extracted for both models.  
     These results are merged into a single summary table and exported as a CSV file. In addition, formatted regression tables are produced using the `texreg` package for reporting.

3. **Exporting Results:**
   - All outputs go to `Econometrics/Results/`:
     - Combined regression result tables in `Results/adaptation/` and `Results/mitigation/` (e.g., `combined_regression_results2_mitigation.csv`, `combined_regression_results3_mitigation.csv`)
     - Formatted regression tables (exported via `texreg` to text files at the root of `Results/`, e.g. `Rio Result for Mitigation`)
     - Individual regression datasets (`reg1.csv` … `reg3.csv`, `reg1_mitigation.csv` … `reg3_mitigation.csv`) at the root of `Results/`
     - A summary CSV (`Results/summary_sample.csv`) that merges attributes from different model outputs.
   - The `reg*.csv` files in `Results/` are regenerated intermediates: `Estimations.R` rewrites them on every run and they are not part of the shipped outputs.

---

## Detailed Workflow

1. **Data Loading and Preprocessing:**
   - The script begins by reading in the preprocessed mitigation data from the raw data file (using `fread()` and `read.csv()`).
   - It then creates separate datasets for Rio markers and ClimateFinanceBERT data.
   - Grouping, filtering, and variable transformations are applied:
     - Grouping by key identifiers such as Year, ProviderISO, and RecipientISO.
     - Summing commitment amounts and converting these figures (multiplied by 1000) to a logarithmic scale.
     - Adjustments are made to remove deals with very low financial amounts and to set NA values to zero.

2. **Regression Analysis with mhurdle:**
   - The `mhurdle` package is used to estimate a double hurdle (two-part) model:
     - The first part models the likelihood of committing finance.
     - The second part models the amount committed (conditional on a positive commitment).
   - Two versions of the model are fitted:
     - An uncorrelated model.
     - A correlated model updated using the `update()` function.
   - Coefficients from both models are extracted and merged into a comprehensive results table.
   - Additional model parameters such as log-likelihood and pseudo-R² statistics are computed and appended to the results.

3. **Processing and Exporting Regression Outputs:**
   - The script uses the `texreg` package to create publication-ready regression tables.
   - Regression outputs are saved as CSV files and text files in the Results folder.
   - A separate segment of the script handles the assembly of regression results for both Rio data and ClimateFinanceBERT data.

4. **Final Combination of Regression Data:**
   - The key dependent variable (`MitiAmount`) is extracted from multiple regression datasets (reg4, reg5, and reg6).
   - These subsets are labeled with their source and combined into a single data frame for further analysis.

---

## Replication Instructions

1. **Data Requirements:**
   - Ensure that the preprocessed data files (for analysis) are available in the specified directories.
   - Verify that all necessary raw data have been processed (via earlier scripts) and that variables such as `ProviderISO`, `RecipientISO`, and commitment amounts are correctly formatted.

2. **Install Required Packages:**
   - This script requires several R packages, including:
     ```r
     install.packages(c("here", "readr", "data.table", "dplyr", "ggplot2", "fastDummies", "mhurdle", "texreg", "tidyverse", "countrycode"))
     ```
   - Load additional packages as needed.

3. **Set Up and Run the Script:**
   - Open R or RStudio.
   - No working directory needs to be set: all paths resolve from the repository root with `here::here()`. Inputs are read from `Econometrics/Data/` (`DataPB.csv`, `climate_finance_total.csv`, see `external-data.md`) and `Climate finance estimation/Raw Data/` (gravity files).
   - Run the script:
     ```r
     source(here::here("Econometrics", "Estimations.R"))
     ```
   - The script will execute the regression analyses, generate output tables, and export results to the `Econometrics/Results/` folder.

4. **Review Outputs:**
   - Check the CSV files and regression tables to ensure that the results are as expected.
   - Review the merged coefficient tables, model diagnostics, and summary statistics provided in the output.

---

## Two result folders: original paper vs. revision

| Folder | Version of the paper | Produced by |
|--------|----------------------|-------------|
| `Results/` | **Original submission** (outputs as shipped in April 2025). | `Estimations.R` |
| `regressions/` | **Revision (2026, first-round R&R) — the version currently under review.** | `HurdleRegHuei_parallel.R`, `HurdleRegHuei_parallel_robustness.R`, `select_variables.R`, `HurdleRegHuei_selected.R`, `Identification/*.R` |

To reproduce the paper under review, use `regressions/` and the run order below.
The revision scripts never write to `Results/`: their `texreg` text tables
(`Baseline/Rio/ClimateFinanceBERT Result for ...`) go to `regressions/main/texreg/`
and `regressions/robustness/texreg/`.

## Revised analysis (2026 revision, first-round R&R)

The paper revision adds a larger estimation, identification, and average-partial-
effects (APE) layer on top of the original `Estimations.R`. All revised scripts
are repo-relative: paths resolve with `here::here()` from the repository root
(a `.here` sentinel ships at the root so this also works from a ZIP download with
no `.git`). Run them in this order:

1. **Prepare data.** Unzip `Data/Data.zip` into `Econometrics/Data/` (gives the
   six `reg*.csv` panels), unzip the gravity files in
   `Climate finance estimation/Raw Data/`, and put the two external inputs in `Econometrics/Data/`:
   `climate_finance_total.csv` (download from the drive) and `DataPB.csv`
   (rebuilt with `Climate finance estimation/Raw Data/Treatment.R`; it is not on
   the drive) — see `external-data.md`.
2. **`HurdleRegHuei_parallel.R`** — main double-hurdle estimation. Writes the
   headline tables to `regressions/main/` (and re-writes the six `reg*.csv`
   panels to `Data/` and the `texreg` text tables to `regressions/main/texreg/`).
3. **`HurdleRegHuei_parallel_robustness.R`** — robustness re-estimation. Writes
   to `regressions/robustness/`.
4. **`select_variables.R`** — Han→BERT variable selection. Writes the manifest
   and diagnostics to `regressions/var_selection/`.
5. **`HurdleRegHuei_selected.R`** — slim re-estimation on the selected
   covariates. Writes to `regressions/selected/`.
6. **`Identification/` scripts** — each reads the panels from `Data/` and sources
   `_helpers.R` (except `marginal_effects_ape.R`, which is self-contained). They
   are independent of one another and write to
   `regressions/{aipw,aipw_h1,lee_bounds,lee_bounds_h1,common_support,ape}/`.

See `regressions/readme.md` and `Identification/readme.md` for the full mapping
of scripts to outputs. Heavy diagnostic `.rds` objects are not shipped and
regenerate on run. The original `Estimations.R` is unchanged apart from its
file paths, which now also resolve with `here::here()` (its `texreg` text tables
now land at the root of `Results/`, where the shipped copies sit), and its
`library()` calls, which are all at the top of the script.

---

## Contact

For further questions or additional information regarding this analysis, please contact:

**Pierre Beaucoral**  
Email: [pierre.beaucoral@uca.fr]
