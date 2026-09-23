# Climate Finance Data Analysis and Visualization

This repository contains a Python script designed to analyze climate finance data and produce a suite of high-quality figures for academic research. The code aggregates and processes data related to funding and commitments for climate change adaptation, mitigation, and environmental projects. In addition, it compares classifications between two methods (Rio markers and ClimateFinanceBERT) and generates time series forecasts using SARIMA models.

---

## Overview

The main objectives of the code are to:

- **Aggregate and Visualize Timelines:**  
  Generate annual timelines for both disbursements (funding) and commitments, segmented by climate change categories (Adaptation, Mitigation, Environment).

- **Prepare Rio Marker Data:**  
  Process the original dataset to obtain climate finance data as classified by Rio markers, separated into adaptation and mitigation funding/commitments.

- **Generate Graphical Outputs:**  
  Produce a set of figures including:
  - Stacked area charts for mitigation and adaptation (both disbursement and commitment).
  - Stackplots for overall disbursements and commitments.
  - Combined adaptation and mitigation plots (global view).
  - Combined climate finance analysis plots showing disbursements, commitments, trends, and seasonal patterns.
  - SARIMA forecast plots for projecting future climate finance disbursements.
  - Donor-based comparison plots (both a combined comparison and a ratio comparison) for Rio markers versus ClimateFinanceBERT.

All generated figures are saved in `Climate finance estimation/Figures/Graphs/` (the SARIMA forecast goes to its `forecasts/` subfolder).

---

## Data Requirements

The code requires two primary input datasets:
1. **DataPB.csv**  
   This file contains the original data, including columns such as `Year`, `DonorCode`, `DonorName`, `USD_Disbursement_Defl`, `USD_Commitment_Defl`, and classification markers (e.g., `ClimateAdaptation`, `ClimateMitigation`). This dataset is used to prepare the Rio markers.

2. **climate_finance_total.csv**  
   This dataset contains climate finance information processed by the ClimateFinanceBERT classification. It must include columns for `Year`, `USD_Disbursement_Defl`, `USD_Commitment_Defl`, and a `meta_category` that differentiates between Adaptation, Mitigation, and Environment.

Place these files in the `Data` folder (or update the file paths accordingly in the code).

---

## Dependencies

The code is implemented in Python and relies on the following libraries:
- **Pandas:** For data manipulation and aggregation.
- **NumPy:** For numerical operations.
- **Matplotlib:** For generating the visualizations.
- **Seaborn:** (Optional) for advanced plotting options.
- **Statsmodels:** For time series decomposition and SARIMA forecasting.
- **Scikit-Learn:** For machine learning functions (if extended forecasting is required).

Make sure these packages are installed in your Python environment. You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn

```

---

## Paper figures and tables → producing script → output file

All output paths are relative to `Climate finance estimation/Figures/Graphs/`. The paper's `Figure_*.png` files are these outputs renamed.

| Paper figure | Content | Producing script | Output file |
|---|---|---|---|
| Figure 1 | Two-stage ClimateFinanceBERT classification flowchart | `Figures/pipeline_diagram.R` (R: DiagrammeR) | `pipeline_diagram.png` |
| Figure 2 | Stacked BERT-estimated commitments by macro-category, 2000–2022 | `Figures/graph_final.py`, `stacked_area()` | `stackplot_commitment.png` |
| Figure 3 | ClimateFinanceBERT vs Rio markers (principal / significant), adaptation and mitigation disbursements | `Figures/graph_final.py`, `combined_adaptation_mitigation_plot()` | `combined_adaptation_mitigation_plot.png` |
| Figure 4 | SARIMA extrapolation of BERT-classified disbursements | `Figures/graph_final.py`, `forecast_climate_finance_sarima()` | `forecasts/climate_finance_forecast_sarima.png` |
| Figure A1 | PVCCI vulnerability map (country level) | `Figures/PVCCImap.R` (R) | `vulnerability_map.png` |
| Figure A2 | Disbursements, commitments, trend and seasonal decomposition by macro-category | `Figures/graph_final.py`, `analyze_combined_climate_finance()` | `combined_climate_finance_analysis.png` |
| Figure A3 | Rio markers vs ClimateFinanceBERT by donor | `Figures/graph_final.py`, `create_comparison_timeline_by_donor()` | `combined_comparison_by_donor.png` |
| Figure A4 | BERT / Rio-marker ratio by donor | `Figures/graph_final.py`, `create_ratio_comparison_timeline_by_donor()` | `ratio_comparison_by_donor.png` |

The classifier-performance and BERTopic tables are documented in `Climate finance estimation/Data/readme.md`.

### How to run

```bash
# From the repository root
python3 "Climate finance estimation/Figures/graph_final.py"    # Figures 2, 3, 4, A2, A3, A4
Rscript "Climate finance estimation/Figures/pipeline_diagram.R" # Figure 1
Rscript "Climate finance estimation/Figures/PVCCImap.R"         # Figure A1
```

- `graph_final.py` needs `Data/DataPB.csv` and `Data/climate_finance_total.csv` (both under `Climate finance estimation/`). Neither is shipped. `DataPB.csv` is rebuilt from the raw OECD CRS files with `Raw Data/Treatment.R` and then copied into `Data/`; `climate_finance_total.csv` is on the external drive as `climate_finance_total.csv.zip`. See `Data/readme.md`.
- `pipeline_diagram.R` has no data input. The counts in the boxes are hard-coded; they come from the classification of the full CRS corpus.
- `PVCCImap.R` needs R packages `here`, `sf`, `dplyr`, `data.table`, `ggplot2`, `viridis`, `rnaturalearth`, `rnaturalearthdata`, `countrycode`. Its input, `Data/pvcciNational.csv`, is shipped: country-level scores from the Physical Vulnerability to Climate Change Index published by FERDI (https://ferdi.fr). Country boundaries come from Natural Earth through `rnaturalearth`, so the paper figure needs no download.
- `PVCCImap.R` can also draw an ADM2 (sub-national) map, which is not in the paper. It does this only when both of the following files, which are not shipped, are present:
  - `Climate finance estimation/Data/external/gadm_410-levels.gpkg`: GADM 4.1 "levels" GeoPackage from https://gadm.org (license does not allow redistribution).
  - `Climate finance estimation/Data/external/pvcciSubNational.csv`: FERDI sub-national PVCCI (columns `country`, `adm2`, `PVCCI`).

### Known limitations

- `graph_final.py` targets pandas < 3. It uses `freq='Y'` in `pd.date_range` and `fillna(method='ffill')`, which pandas 3 removed. Run it with the pandas version pinned in `Climate finance estimation/requirements.txt`.
- The year range 2000–2022 is hard-coded in `graph_final.py` (sample filters, axis limits and x ticks). Extending the data beyond 2022 requires editing these values.
- `PVCCImap.R` keeps the titles inside the plot so that it reproduces the published Figure A1 as it appears in the paper.
