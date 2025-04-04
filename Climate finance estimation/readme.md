# Climate Finance Estimation

This repository, part of the UNDERCANOPY project, provides a comprehensive, multi-method pipeline for analyzing and estimating climate finance. The project integrates data preprocessing, classification, and estimation components to process raw OECD CRS aid activities data, classify projects into climate-relevant categories, and quantify financial flows. The tools and scripts in this repository have been developed under rigorous academic standards to ensure transparency, reproducibility, and high data integrity.

---

## Overview

The Climate Finance Estimation pipeline is organized into several interrelated components:

1. **Data Preprocessing and Consolidation (R Scripts):**  
   - **UploadBase.R:**  
     Loads, cleans, and merges raw OECD CRS data (annual and multi-year files) into unified CSV files.  
     - Combines project titles and descriptions into a single text field.
     - Removes duplicates and unwanted entries.
     - Generates both a full dataset and a smaller beta-test subsample.

2. **Classification and Estimation (Python Scripts):**  
   - **Climate Finance Classification Pipeline:**  
     Contains modules for transformer-based relevance and multiclass classification.
     - **Climate Finance Estimation:**  
       Creating a training dataset. 
     - **Relevance Classifier:**  
       A deep learning model (using a ClimateBERT variant) that distinguishes between climate-relevant and non-relevant projects.
     - **Multiclass Classifier:**  
       Fine-tunes a pre-trained transformer to assign detailed climate finance categories to relevant projects.
     - **Data Post-Processing:**  
       Merges classification outputs with the original data and aggregates funding information by category.

   - **Inference Utilities:**  
     Includes functions to load model weights, tokenize text, and predict labels on new data in batches (with parallel processing where applicable).

3. **Data Sources and Resources:**  
   The project uses multiple data sources:
   - **Raw Project Data:**  
     Obtained from OECD CRS aid activities (files such as `Data.csv` from ML clustering outputs).
   - **Preprocessed Training Data:**  
     A dataset (`train_set.csv`) derived from extensive filtering and sampling.
   - **Auxiliary Files:**  
     JSON label dictionaries (e.g., `reverse_dictionary_classes.json`) and pre-trained model weights (`saved_weights_relevance.pt`, `saved_weights_multiclass.pt`).

   > **Note:** Due to their size, key intermediate and final data files (including DataPB, Data, ClassifiedCRS, ClimateFinanceTotal, and the model weight files) are not stored directly in this repository. They are available at:  
   > [This drive](https://drive.uca.fr/d/6058b184ba134a02a708/)

---


## How to Replicate

### Data Preparation
1. **Download Raw Data:**  
   Obtain the original OECD CRS raw text files (annual and multi-year) from the OECD website. Place them in the `Data/CRS/` folder.
2. **Run R Preprocessing:**  
   - First, execute `Treatment.R` to load and merge the raw CRS files.
  
### Model Training and Estimation
1. **Ensure Dependencies:**  
   - For Python scripts, install required packages:
     ```bash
     pip install pandas numpy torch transformers scikit-learn tqdm
     ```
   - For R scripts, install required libraries (`data.table`, `dplyr`, `ggplot2`).
2. **Download Pre-trained Weights and Label Files:**  
   The model weights (`saved_weights_relevance.pt` and `saved_weights_multiclass.pt`) and JSON label mapping files (e.g., `reverse_dictionary_classes.json`) are available from the external drive:
   [https://drive.uca.fr/d/6058b184ba134a02a708/](https://drive.uca.fr/d/6058b184ba134a02a708/)
3. **Run Python Pipelines:**  
   - Execute `EstimationClimateFinance.py` (if applicable) to perform financial estimation and forecasting.
   - Execute `Relevance_classifier.py` or related scripts to fine-tune and evaluate the models.
   - Execute `Multi-classifier.py` or related scripts to fine-tune and evaluate the models.
   - Execute `Classify.py` to run the classification pipeline.
   - Execute `Meta.py` to run the meta categories pipeline.
   - Execute `Graph_Final.py` to run the classification pipeline.


### Overall Workflow
- **Data Preprocessing (R):**  
  Convert raw OECD data into a unified, clean dataset.
- **Classification (Python):**  
  Use transformer-based models to classify project relevance and assign detailed climate finance categories.
- **Graphs (Python):**  
  Forecast future climate finance flows based on historical data.
- **Post-Processing:**  
  Merge outputs and aggregate funding information by category.
---

## Academic Context

This project has been developed as part of high-standard academic research into climate finance. By combining rigorous data preprocessing, advanced transformer-based classification, and robust time series forecasting, the pipeline provides reproducible and transparent analyses of aid activities. The modular design and detailed documentation support peer review, reproducibility, and potential extension by other researchers.

---

## Contact

For further questions or details regarding this project, please contact:

**Pierre Beaucoral**  
Email: [pierre.beaucoral@uca.fr]
