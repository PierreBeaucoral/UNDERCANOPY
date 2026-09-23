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
  `Climate finance estimation/Data/Data.csv`
- **Description:**  
  This file is the output of a clustering analysis performed on aid activities. It contains:
  - **`raw_text`**: Unstructured textual descriptions of projects.
  - **`DonorCode`**: Numerical codes identifying donor entities.
  - Additional metadata fields used for further processing.
- **Role in Pipeline:**  
  The raw project data is grouped by unique text and donor code. It serves as the foundation for extracting, filtering, and ultimately classifying projects as climate finance–relevant or not.

### 2. Training Data for Climate Finance Classification
- **File Path:**  
  `Climate finance estimation/Data/train_set.csv`
- **Description:**  
  This CSV file contains a preprocessed, balanced dataset used for training the transformer classifiers. It includes:
  - **`text`**: The processed project description.
  - **`label`**: Fine-grained class labels assigned during data preparation.
  - **`relevance`**: A binary flag (1 for climate-relevant, 0 for non-relevant).
- **Role in Pipeline:**  
  The training set is constructed by merging a representative sample of climate-related projects (filtered and labeled from the raw data) with non-climate projects, ensuring balanced class distributions for model training.

### 3. Reverse Label Dictionary
- **File Path:**  
  `Climate finance estimation/Data/reverse_dictionary_classes.json`
- **Description:**  
  This JSON file stores a dictionary that maps numeric class labels (produced by the multiclass classifier) to human-readable, descriptive labels.
- **Role in Pipeline:**  
  During inference, the multiclass predictions (numeric) are converted into interpretable labels using this mapping, which is crucial for generating understandable classification reports and summaries.

### 4. Model Weights for Relevance Classification
- **File Path:**  
  `Climate finance estimation/Data/saved_weights_relevance.pt`
- **Description:**  
  These model weights represent the state of the relevance classifier after fine-tuning. The classifier distinguishes between projects that are relevant to climate finance and those that are not.
- **Role in Pipeline:**  
  The relevance classifier is loaded with these weights to perform binary classification on incoming project texts, ensuring consistent predictions during data processing.

### 5. Model Weights for Multiclass Classification
- **File Path:**  
  `Climate finance estimation/Data/saved_weights_multiclass.pt`
- **Description:**  
  This file contains the weights for the multiclass classifier trained to assign detailed climate finance categories to relevant projects.
- **Role in Pipeline:**  
  Once a project is deemed relevant by the relevance classifier, the multiclass classifier (loaded with these weights) assigns a fine-grained climate category. The numeric predictions are then mapped to descriptive labels using the reverse label dictionary.

---

## Additional Data Storage

Due to the large size of some intermediate and final datasets used in this research, several key data files are not stored directly in this GitHub repository. Where to obtain each one is given below (external drive, OECD website, or rebuilt by a script in this package):
- **`projects_clusters.csv`** — required input for `EstimationClimateFinance.py`. Produced by the upstream BERTopic clustering in the companion repository [`ML-clustering-of-development-activities`](https://github.com/PierreBeaucoral/ML-clustering-of-development-activities) (`Machine learning/Topic modelling.py`); it carries the `Topic`, `raw_text`, and `CustomName` fields used to build the training set. **Not shipped in this repository.** Expected at `Climate finance estimation/Data/projects_clusters.csv`. Download from the external drive below.
- **`train_set.csv`** — balanced training set read by `Relevance_classifier.py`, `multi-classifier.py`, and produced by `EstimationClimateFinance.py`. Expected at `Climate finance estimation/Data/train_set.csv`.
- **`Data.csv`** — topic-merged CRS project file classified by `Classify.py`: the `merged_projects.csv` output of `Machine learning/Topic modelling.py` in [`ML-clustering-of-development-activities`](https://github.com/PierreBeaucoral/ML-clustering-of-development-activities), renamed. Expected at `Climate finance estimation/Data/Data.csv`. The copy in `Archive.zip` on the external drive already carries classification columns from earlier runs.
- **`DataPB.csv`** — preprocessed CRS panel (with Rio markers) read by `Figures/graph_final.py`. **Not on the external drive:** rebuild it from the raw OECD CRS files with `Raw Data/Treatment.R`, which writes `Climate finance estimation/Raw Data/DataPB.csv`, then copy that file to `Climate finance estimation/Data/DataPB.csv` (for `graph_final.py`) and to `Econometrics/Data/DataPB.csv` (for the econometric scripts). See `Econometrics/external-data.md`.
- **`ClassifiedCRS.csv`** — full classified dataset written by `Classify.py` and read by `meta.py`. Expected at `Climate finance estimation/Data/ClassifiedCRS.csv`.
- **`climate_finance_total.csv`** (referred to elsewhere as ClimateFinanceTotal) — meta-categorized dataset written by `meta.py` and read by `Figures/graph_final.py`. Expected at `Climate finance estimation/Data/climate_finance_total.csv`. On the external drive as `climate_finance_total.csv.zip` (about 21.5 GB once unzipped).
- **`reverse_dictionary_classes.json`** and **`dictionary_classes.json`** — label-mapping files written by `multi-classifier.py` and read by `Classify.py`. Expected at `Climate finance estimation/Data/`.
- **Saved Model Weights:** Both `saved_weights_relevance.pt` and `saved_weights_multiclass.pt`, expected at `Climate finance estimation/Data/`.
- **Raw OECD CRS `.txt` files** — the annual and multi-year CRS extracts read by `Raw Data/UploadBase.R` (e.g. `CRS 2006 Data.txt`, `CRS 1973-94 data.txt`). Obtain from the OECD CRS website and place in `Climate finance estimation/Raw Data/CRS/`.

The external drive is at [this link](https://drive.uca.fr/d/6058b184ba134a02a708/). It holds `Archive.zip`, `ClassifiedCRS.csv.zip`, `climate_finance_total.csv.zip`, and the two model-weight files (`.pt`). `DataPB.csv` is not on the drive.

---

## Additional Notes

- **Data Preparation and Processing:**  
  The raw project data is first grouped by `raw_text` and aggregated by `DonorCode` to ensure each project is uniquely represented. The grouped data is then processed in parallel (using a thread pool) to predict relevance and class labels efficiently.
  
- **Integration:**  
  After processing, the predictions are merged back with the original raw data to produce a final classified dataset (`ClassifiedCRS.csv`). Additionally, the pipeline extracts and aggregates funding information for specific climate categories (e.g., adaptation, environment, mitigation) into separate CSV files.

- **Reproducibility:**  
  All data sources are stored in well-defined directories. The training set and model weights have been generated using standardized methods (with fixed random seeds and documented preprocessing steps), ensuring reproducibility and transparency in research findings.

---

## Paper figures and tables → producing script → output file

This section covers the paper tables built from files in this folder. The paper figures are listed in `Climate finance estimation/Figures/readme.md`.

| Paper table | File(s) read by the manuscript | Producing script | Shipped here? |
|---|---|---|---|
| Classifier performance (appendix table `modelmetrics`), relevance rows | `classification_report.csv` | `Training and Classifying/Relevance_classifier.py` (written after test-set evaluation) | Yes |
| Classifier performance, Adaptation / Environment / Mitigation rows | `classification_reportmulticlassifier_gen.csv` | `Training and Classifying/multi-classifier.py` (macro-category report) | Yes |
| Climate topics used to build the training sample (appendix table built from BERTopic output) | `topic_info.csv` | `Machine learning/Topic modelling.py` in the companion repository (see below) | Yes |
| Training-sample composition | `train_set.csv` | `Training and Classifying/EstimationClimateFinance.py` | Yes |
| Figure A1 input (PVCCI) | `pvcciNational.csv` | Input data from FERDI (not produced here), used by `Figures/PVCCImap.R` | Yes |

**Classifier reports.** Each report CSV has the layout of scikit-learn's `classification_report(..., output_dict=True)` written as a table. The header is `,precision,recall,f1-score,support`. There is one row per class, followed by `accuracy`, `macro avg` and `weighted avg`. The scripts write these files to this folder when they run. The shipped `classification_report.csv` (relevance) and `classification_reportmulticlassifier_gen.csv` (macro-categories) are the files the submitted manuscript's performance table reads, so that table can be rebuilt without retraining the transformers (GPU). Note that the macro-category report comes from an earlier 18-class training run (2024-09-17), not from the 17-class production model shipped as `saved_weights_multiclass.pt` (2024-09-25), and was computed with an index-based grouping that `multi-classifier.py` no longer uses. Running `multi-classifier.py` now writes both reports for the production model, with the macro-categories grouped by class name exactly as in `meta.py`.

**`topic_info.csv` (BERTopic topic table).** This is the topic-information table from the unsupervised BERTopic clustering of OECD CRS project descriptions (columns `Topic, Count, Name, CustomName, Representation, Zephyr, Representative_Docs`). The clustering is the upstream step that also produced `projects_clusters.csv`, and it comes from a separate study cited in the paper. It is produced by `Machine learning/Topic modelling.py` in the companion repository [`ML-clustering-of-development-activities`](https://github.com/PierreBeaucoral/ML-clustering-of-development-activities); it is shipped here so the appendix table can be rebuilt without rerunning the topic model. The manuscript keeps only the climate topics it lists and truncates long text before building the appendix table.

**`pvcciNational.csv`.** Country-level Physical Vulnerability to Climate Change Index (PVCCI) with its components (Flood, Aridity, Rainfall, Temperature, Storms), from FERDI (https://ferdi.fr). It is redistributed here only to reproduce Figure A1. Please cite FERDI when you reuse it.
