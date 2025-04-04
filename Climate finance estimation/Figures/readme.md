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

All generated figures are saved in a centralized folder (`Figures`), which facilitates subsequent review and publication.

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

