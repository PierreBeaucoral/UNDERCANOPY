# OECD CRS Data Preprocessing and Meta-Categorization Pipeline

This repository contains a two-stage R processing pipeline that prepares raw OECD CRS data for subsequent climate finance analyses. The pipeline consists of:

1. **UploadBase.R:**  
   The first script in the pipeline that loads, cleans, and merges the raw text files from the OECD CRS database. It produces a unified dataset (`DataPB.csv`) as well as a smaller subsample (`DataPBsample.csv`) for beta-testing purposes.

2. **Meta-Categorization Script (this script):**  
   This script loads the previously generated classified dataset (`ClassifiedCRS.csv`), assigns meta-categories (Adaptation, Mitigation, Environment) based on detailed classification numbers, and performs data integrity checks before saving the final cleaned dataset.

> **Important:**  
> The raw data files (text files) must be downloaded directly from the OECD website. Place these files in the appropriate directories (e.g., `./Data/CRS/`) as expected by the pipeline.

---

## UploadBase.R

**Purpose:**  
The `UploadBase.R` script is responsible for:
- Setting the working environment.
- Loading multiple raw data files (both annual and multi-year) from the OECD CRS database using the `fread()` function.
- Combining the annual and multi-year datasets into a single, comprehensive data frame.
- Cleaning and standardizing text data:
  - Concatenating project titles and descriptions into a single `raw_text` field.
  - Converting all text to lowercase and removing extraneous characters (e.g., underscores).
  - Removing unwanted entries based on specific patterns (e.g., "semi-aggregates", "sectors not specified").
- Generating a full dataset (`DataPB.csv`) and a smaller random subsample (`DataPBsample.csv`) for testing.

**How It Works:**  
- **Working Directory Setup:**  
  The script clears the current environment and sets the working directory.
- **Data Loading:**  
  It reads annual datasets (e.g., "CRS 2006 Data.txt", …, "CRS 2023 Data.txt") and multi-year datasets (e.g., "CRS 1973-94 data.txt", "CRS 1995-99 data.txt", etc.).
- **Data Merging:**  
  Datasets are grouped by year and merged into a single data frame.
- **Text Processing:**  
  The script concatenates key text fields (ProjectTitle, ShortDescription, LongDescription) into a unified `raw_text` field, converts text to lowercase, and removes duplicates.
- **Output:**  
  Finally, it writes the cleaned and merged data to CSV files (`DataPB.csv` and `DataPBsample.csv`).

---

## Meta-Categorization Script

**Purpose:**  
This script refines the dataset produced by the earlier steps by assigning high-level meta-categories to each project and verifying data integrity.

**Key Steps:**

1. **Data Loading:**  
   - Uses a custom function `csv_import` to load the classified dataset (`ClassifiedCRS.csv`) from the specified directory.
   - Drops rows with missing values in the `climate_class_number` column.
   - Performs an initial check on the dataset’s size to ensure adequate volume.

2. **Meta-Category Assignment:**  
   - Three sets of numeric category codes are defined:
     - **Adaptation:** `[10, 13]`
     - **Environment:** `[0, 1, 2, 5, 9, 12, 14, 15]`
     - **Mitigation:** `[3, 4, 6, 7, 8, 11, 16]`
   - A new column, `meta_category`, is created and initialized to `"None"`.
   - Rows are assigned to the appropriate meta-category based on their `climate_class_number`.
   - The script prints the count of rows for each meta-category.

3. **Data Integrity Checks:**  
   - It identifies rows with missing `climate_class_number` and groups data by `raw_text` to check for inconsistencies (e.g., mixed NA and non-NA values).
   - A warning is printed if there are any missing values in `climate_class_number`.
   - The script verifies that all rows with `meta_category` remaining as `"None"` correspond exactly to non-relevant projects (`climate_relevance == 0`).
   - It ensures that the total number of rows remains unchanged during processing, and exits if not.

4. **Saving the Final Dataset:**  
   - After successful verification, the processed dataset (now including the `meta_category` column) is saved as `climate_finance_total.csv` using a pipe (`|`) as the delimiter.

---

## Replication Instructions

1. **Download and Place Data Files:**  
   - Download the raw OECD CRS text files from the OECD website and place them in the expected folder (e.g., `./Data/CRS/`).
   - Run `UploadBase.R` to generate the initial cleaned datasets (`DataPB.csv` and `DataPBsample.csv`).

2. **Set Up Your Environment:**  
   - Ensure that R and the required packages (`data.table`, `dplyr`, and `ggplot2`) are installed.
   - Set the working directory to `/Users/pierrebeaucoral/Documents/Pro/Thèse CERDI/Recherche/Travaux CRS` or modify the script accordingly.

3. **Run the Meta-Categorization Script:**  
   - Execute the meta-categorization script:
     ```r
     source("YourMetaCategorizationScript.R")
     ```
   - The script will load the `ClassifiedCRS.csv` file, assign meta-categories, perform integrity checks, and save the final dataset to `Data/climate_finance_total.csv`.

4. **Verify Outputs:**  
   - Monitor console messages to confirm that the working directory is correctly set, data processing steps are executed, and integrity checks pass.
   - Confirm that the output CSV files are correctly generated in the `Data` folder.

---
