# Climate Finance Classification Codebase

This repository contains a comprehensive set of Python modules developed for the classification and estimation of climate finance data. The codebase integrates state-of-the-art machine learning, natural language processing, and statistical estimation techniques to (i) classify climate finance projects by category (e.g., adaptation, mitigation, environment), (ii) filter for relevance, and (iii) estimate the financial flows associated with these projects. The tools provided here support rigorous academic research by enabling reproducible, transparent, and extensible analyses of climate finance.

---

## Overview

This codebase is organized into several modules, each responsible for a distinct step in the overall classification and estimation pipeline:

- **Classify.py**  
  Serves as the main entry point for the classification process. This script orchestrates the workflow by loading raw data, invoking the multi-class and relevance classifiers, and outputting the classified results. It is designed to integrate the outputs of different models into a coherent classification of climate finance projects.

- **EstimationClimateFinance.py**  
  Implements methods for estimating the monetary values associated with classified climate finance projects. In addition to aggregating the results of the classification, it applies statistical and time series models (such as SARIMA) to forecast and quantify financial flows. This module is critical for translating classification outputs into actionable financial estimates.

- **meta.py**  
  Contains utility functions and routines for handling metadata related to the climate finance datasets. These functions may include feature extraction, data cleaning, and preparation of auxiliary variables that support both the classification and estimation tasks. By centralizing meta-information handling, this module ensures consistency across the analytical pipeline.

- **multi-classifier.py**  
  Implements a multi-class classification model specifically designed to assign each climate finance project to one or more categories (e.g., adaptation, mitigation, or environment). This module typically leverages ensemble methods or combines several base classifiers to enhance prediction accuracy and robustness. Its design reflects the complex nature of climate finance, where projects may span multiple categories.

- **Relevance_classifier.py**  
  Focuses on filtering the data to retain only projects that are truly relevant to the field of climate finance. This binary classification model helps to exclude noise and non-relevant entries, ensuring that subsequent analyses are performed on a curated subset of data. The module is essential for maintaining the quality and precision of the classification results.

---

## Folder Structure and Module Details


### Classify.py
- **Purpose:**  
  Acts as the central script to execute the entire classification workflow.  
- **Functionality:**  
  - Loads and preprocesses the raw climate finance data.
  - Calls the multi-classifier and relevance classifier functions to assign project categories.
  - Consolidates and writes the classification results to output files.  
- **Academic Note:**  
  The design emphasizes reproducibility and modularity, allowing researchers to trace the transformation from raw input to final classified output.

### EstimationClimateFinance.py
- **Purpose:**  
  Provides methods to estimate and forecast financial flows associated with climate finance projects.  
- **Functionality:**  
  - Aggregates classification outputs to compute total funding and commitment values.
  - Implements forecasting routines (e.g., using SARIMA models) to predict future financial trends.
  - Outputs visualizations and numerical summaries of estimated values.
- **Academic Note:**  
  This module bridges the gap between qualitative classification and quantitative financial analysis, supporting policy-relevant forecasting.

### meta.py
- **Purpose:**  
  Houses utility functions for metadata management.  
- **Functionality:**  
  - Cleans and preprocesses metadata fields.
  - Extracts or computes additional features required for classification and estimation.
  - Ensures that ancillary data is consistently formatted and integrated into the analytical pipeline.
- **Academic Note:**  
  Accurate metadata processing is crucial for ensuring that downstream machine learning models operate on high-quality, standardized inputs.

### multi-classifier.py
- **Purpose:**  
  Implements an ensemble-based or multi-model classification system for categorizing climate finance projects.  
- **Functionality:**  
  - Combines multiple classification algorithms to improve accuracy.
  - Optimizes hyperparameters and aggregates predictions across models.
  - Outputs final class labels for each project.
- **Academic Note:**  
  The multi-classifier approach addresses the complexity inherent in climate finance classification by leveraging the strengths of different models. It is designed for robustness in heterogeneous datasets.

### Relevance_classifier.py
- **Purpose:**  
  Determines whether a given project or document is relevant to the climate finance domain.  
- **Functionality:**  
  - Implements a binary classification model (relevant vs. not relevant).
  - Filters out non-relevant data to focus subsequent analysis on pertinent projects.
  - May include confidence scoring to assess classification certainty.
- **Academic Note:**  
  This filtering step is critical in high-stakes policy research, ensuring that analyses are based on data that directly inform climate finance discussions.

---

## Dependencies

The codebase requires the following Python libraries:
- **Pandas** – Data manipulation and aggregation.
- **NumPy** – Numerical computing.
- **Matplotlib** – High-quality figure generation.
- **Scikit-learn** – Machine learning algorithms.
- **Statsmodels** – Time series analysis and forecasting.

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels

