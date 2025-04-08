# UNDERCANOPY: Consistent and Replicable Estimation of Bilateral Climate Finance and its determinant

This repository hosts a comprehensive, multi-method framework for the estimation and classification of bilateral climate finance and its determinant. Developed to meet high academic standards of transparency and reproducibility, UNDERCANOPY integrates data preprocessing, advanced classification, and forecasting techniques to generate robust insights into climate finance flows.

The project builds on the methodology of Toetzke et al. (2022) (see their repository [here](https://github.com/MalteToetzke/consistent-and-replicable-estimation-of-bilateral-climate-finance/tree/main)) and has been designed to be fully replicable by other researchers.

---

## Repository Overview

UNDERCANOPY is organized into several interrelated components:

### 1. Data Preprocessing and Consolidation (R Scripts)
- **UploadBase.R:**  
  - Loads raw OECD CRS data (annual and multi-year text files).
  - Cleans, merges, and consolidates project details by creating a unified text field (`raw_text`).
  - Produces both a full dataset (`DataPB.csv`) and a beta-test subsample (`DataPBsample.csv`).

### 2. Classification and Estimation (Python Scripts)
- **Classification Pipeline:**  
  - **Classify.py, Relevance_classifier.py, multi-classifier.py:**  
    Utilize transformer-based models (e.g., ClimateBERT) to:
      - Filter projects by relevance (binary classification).
      - Assign detailed climate finance categories (multiclass classification).
  - **EstimationClimateFinance.py:**  
    Applies time series forecasting (e.g., SARIMA) to estimate future climate finance flows based on historical disbursement and commitment data.
  - **Meta-Categorization Script:**  
    - Processes the output from UploadBase.R (e.g., `ClassifiedCRS.csv`).
    - Assigns high-level meta-categories (Adaptation, Mitigation, Environment) based on detailed classification numbers.
    - Performs data integrity checks and saves the final cleaned dataset (`climate_finance_total.csv`).

### 3. Graphical Outputs (Graphs Folder)
This folder contains all publication-ready figures generated by the pipeline, including:
- **Climate Finance Forecast (SARIMA)**
- **Combined Adaptation & Mitigation Plots**
- **Combined Climate Finance Analysis**
- **Donor-Level Comparison Graphs (Combined & Ratio)**
- **Stacked Area and Stackplot Figures for Disbursements and Commitments**

Each graph is accompanied by detailed descriptions in its own README to explain its content and academic relevance.

### 4. Data Sources and Resources
The project relies on several data sources:
- **Raw Project Data:**  
  Clustering outputs from OECD CRS aid activities (e.g., `Data.csv`).
- **Preprocessed Training Data:**  
  A balanced dataset (`train_set.csv`) for model training, derived via extensive filtering and sampling.
- **Auxiliary Files:**  
  JSON label dictionaries (e.g., `reverse_dictionary_classes.json`) for mapping classifier outputs to human-readable labels.
- **Model Weights:**  
  Saved weights for the relevance and multiclass classifiers (`saved_weights_relevance.pt` and `saved_weights_multiclass.pt`).

> **Note:** Due to file size, key datasets and model weights are hosted externally. You can access these files at:  
> [https://drive.uca.fr/d/6058b184ba134a02a708/](https://drive.uca.fr/d/6058b184ba134a02a708/)

### 5. Econometrics Analysis of Aid Determinants (R)

This section of the project focuses on the econometric analysis of the determinants of bilateral aid flows, with a special emphasis on climate finance. The analysis is performed using R and leverages advanced regression techniques to account for both the decision to allocate aid and the magnitude of the allocation.

#### Key Components

- **Data Preparation and Variable Transformation:**  
  The analysis begins with cleaned datasets derived from earlier preprocessing steps. The script further refines these datasets by:
  - Grouping the data by key identifiers such as Year, ProviderISO, and RecipientISO.
  - Aggregating commitment values (converted to USD using appropriate multipliers) and applying logarithmic transformations to variables (e.g., GDP, population, fiscal indicators, and distance).
  - Normalizing or scaling financial and demographic variables to ensure comparability.
  - Creating dummy variables for categorical predictors (e.g., ProviderISO, RecipientISO, Year) using the `fastDummies` package.

- **Double Hurdle Regression Models:**  
  To address the dual decision process in aid allocation—first, the decision whether to commit finance (selection) and second, the decision on the amount of finance committed (intensity)—the analysis employs double hurdle models using the `mhurdle` package. Two models are estimated:
  - **Uncorrelated Model:**  
    This model estimates the two stages (selection and intensity) separately without assuming any correlation between them.
  - **Correlated Model:**  
    This model is updated from the uncorrelated version to allow for correlation between the selection and intensity processes, thus better capturing the underlying decision-making dynamics.

- **Model Comparison and Output Generation:**  
  After fitting the models:
  - Coefficient estimates, standard errors, and significance levels are extracted from both the uncorrelated and correlated models.
  - Key goodness-of-fit metrics such as log-likelihood, McFadden’s pseudo-R², and the coefficient of determination are calculated.
  - The coefficients from both models are merged into combined data frames for side-by-side comparison.
  - Publication-ready regression tables are generated using the `texreg` package, and both CSV and text outputs are saved to the `./Results/` directory.

- **Separate Analyses for Adaptation and Mitigation:**  
  The econometric analysis is performed separately for different facets of climate finance (e.g., adaptation vs. mitigation). For each, the script:
  - Prepares datasets specific to the category (e.g., Rio markers data vs. ClimateFinanceBERT data).
  - Estimates regression models and compares the results.
  - Exports detailed regression results, allowing for the assessment of differences between the classification methods and understanding of the underlying determinants.
---

## How to Replicate

### Data Preparation (R Scripts)
1. **Download Raw Data:**  
   Obtain the raw OECD CRS text files from the OECD website and place them in the appropriate folders (e.g., `./Data/CRS/`).
2. **Run Treatment.R:**  
   This script will load (by calling `UploadBase.R`), clean, and merge the raw data to produce `DataPB.csv` and `DataPBsample.csv`.

### Model Training and Estimation (Python Scripts)
1. **Install Dependencies:**  
   Ensure you have Python 3.7+ installed along with:
   ```bash
   pip install pandas numpy torch transformers scikit-learn tqdm

### Econometric analysis (R) 

You will find the related data and script in the `Econometrics` Folder.

[Click here to view the interactive organization of repository](https://PierreBeaucoral.github.io/UNDERCANOPY/output.html)
