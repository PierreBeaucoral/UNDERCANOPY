# Data Sources for the Climate Finance Classification Pipeline

This document details all data sources used in the climate finance classification and post‐processing pipeline. The sources span raw project data from aid activities, preprocessed training sets for transformer‐based classification, auxiliary JSON files for label mapping, and saved model weights. These datasets underpin our rigorous, reproducible analyses in climate finance research.

---

## Overview

The pipeline integrates multiple data sources:
- **Raw Project Data:** Contains unprocessed text and metadata derived from aid activity clustering.
- **Training Data for Classification:** A balanced dataset created from the raw data via extensive filtering, sampling, and deduplication.
- **Label Mapping Files:** JSON files that map numeric classifier outputs to descriptive class names.
- **Pre-trained Model Weights:** Saved weights from fine-tuned relevance and multiclass classifiers, ensuring consistency in inference.

Each of these sources is described in detail below.

---

## Data Sources

### 1. Raw Project Data
- **File Path:**  
  `/Users/pierrebeaucoral/Documents/Pro/Thèse CERDI/Recherche/ML clustering aid activities/outputbig/Data.csv`
- **Description:**  
  This file is the output of a clustering analysis performed on aid activities. It contains:
  - **`raw_text`**: Unstructured textual descriptions of projects.
  - **`DonorCode`**: Numerical codes identifying donor entities.
  - Additional metadata fields used for further processing.
- **Role in Pipeline:**  
  The raw project data is grouped by unique text and donor code. It serves as the foundation for extracting, filtering, and ultimately classifying projects as climate finance–relevant or not.

### 2. Training Set for Climate Finance Classification
- **File Path:**  
  `Data/Estimation of Climate Finance/train_set.csv`
- **Description:**  
  This CSV file contains a preprocessed, balanced dataset used for training the transformer classifiers. It includes:
  - **`text`**: The processed project description.
  - **`label`**: Fine-grained class labels assigned during data preparation.
  - **`relevance`**: A binary flag (1 for climate-relevant, 0 for non-relevant).
- **Role in Pipeline:**  
  The training set is constructed by merging a representative sample of climate-related projects (filtered and labeled from the raw data) with non-climate projects, ensuring balanced class distributions for model training.

### 3. Reverse Label Dictionary
- **File Path:**  
  `reverse_dictionary_classes.json`
- **Description:**  
  This JSON file stores a dictionary that maps numeric class labels (produced by the multiclass classifier) to human-readable, descriptive labels.  
- **Role in Pipeline:**  
  During inference, the multiclass predictions (numeric) are converted into interpretable labels using this mapping, which is crucial for generating understandable classification reports and summaries.

### 4. Model Weights for Relevance Classification
- **File Path:**  
  `Data/Estimation of Climate Finance/saved_weights_relevance.pt`
- **Description:**  
  These model weights represent the state of the relevance classifier after fine-tuning. The classifier distinguishes between projects that are relevant to climate finance and those that are not.
- **Role in Pipeline:**  
  The relevance classifier is loaded with these weights to perform binary classification on incoming project texts, ensuring consistent predictions during data processing.

### 5. Model Weights for Multiclass Classification
- **File Path:**  
  `Data/Estimation of Climate Finance/saved_weights_multiclass.pt`
- **Description:**  
  This file contains the weights for the multiclass classifier trained to assign detailed climate finance categories to relevant projects.
- **Role in Pipeline:**  
  Once a project is deemed relevant by the relevance classifier, the multiclass classifier (loaded with these weights) assigns a fine-grained climate category. The numeric predictions are then mapped to descriptive labels using the reverse label dictionary.

---

## Additional Notes

- **Data Preparation and Processing:**  
  The raw project data is first grouped by `raw_text` and aggregated by `DonorCode` to ensure each project is uniquely represented. The grouped data is then processed in parallel (using a thread pool) to predict relevance and class labels efficiently.
  
- **Integration:**  
  After processing, the predictions are merged back with the original raw data to produce a final classified dataset (`ClassifiedCRS.csv`). Additionally, the pipeline extracts and aggregates funding information for specific climate categories (e.g., adaptation, environment, mitigation) into separate CSV files.

- **Reproducibility:**  
  All data sources are stored in well-defined directories. The training set and model weights have been generated using standardized methods (with fixed random seeds and documented preprocessing steps), ensuring reproducibility and transparency in research findings.

---
