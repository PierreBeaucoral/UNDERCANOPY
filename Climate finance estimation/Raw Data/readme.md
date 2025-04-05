# Data Preprocessing Script – Raw Text Consolidation for CRS Data

This R script is the second stage in the overall data preparation pipeline for climate finance analysis. It builds upon the output of the first script (UploadBase.R) by further cleaning, filtering, and consolidating the raw OECD CRS data. The goal is to generate a unified dataset with a single text field (`raw_text`) that combines project titles and descriptions. This dataset is then saved for subsequent analyses and model development.

---

## Overview

This script performs the following key tasks:

1. **Environment and Working Directory Setup:**  
   - Clears the current workspace.
   - Sets the working directory to the desired location.
   - Verifies that the current working directory is correct, and if not, changes it accordingly.

2. **Data Import from UploadBase.R:**  
   - Sources the `UploadBase.R` script, which loads multiple raw data files into a list object (`BDD`).  
   - The script then calls garbage collection to free up memory.

3. **Data Filtering:**  
   - For each dataset in the list `BDD`, rows that lack essential identifiers (i.e., missing values in `Year`, `DonorCode`, or `RecipientCode`) are dropped.
   - The environment is then cleaned to retain only key objects (`BDD`, `Bound`, and `Period`).

4. **Creation of a Unified Text Field (`raw_text`):**  
   - Each dataset is processed to select columns of interest (e.g., project titles, short and long descriptions, donor information, funding data, and climate indicators).
   - Rows with missing text fields (i.e., `ProjectTitle`, `ShortDescription`, or `LongDescription`) are filtered out.
   - The `unite()` function is applied twice to concatenate these columns into a single column called `raw_text`.
   - All text is converted to lowercase for consistency.

5. **Data Consolidation and Cleaning:**  
   - The processed datasets from the list are merged (using `rbind`) into a single data frame (`Data`).
   - Duplicate records are removed using the `distinct()` function.
   - The dataset is further filtered to remove rows containing unwanted phrases such as "semi-aggregates" and "sectors not specified".
   - Underscores in the `raw_text` field are removed to enhance text quality.

6. **Subsampling for Beta Testing:**  
   - A random subsample of 10,000 records is created (for rapid beta testing of downstream analyses).
   - Both the full consolidated dataset and the beta sample are saved as CSV files using a pipe (`|`) as the delimiter.

---

## Detailed Workflow

1. **Clearing and Setting the Environment:**  
   - The script begins by clearing all objects from the R environment.
   - It then defines and checks the desired working directory, changing it if necessary.

2. **Importing Data:**  
   - The `UploadBase.R` script is sourced, which loads raw CRS data into the variable `BDD` (a list of data frames) along with additional objects (`Bound`, `Period`).
   - Memory is freed using `gc()` after loading the data.

3. **Filtering Data of Interest:**  
   - The script applies a function to each element of `BDD` to drop rows where `Year`, `DonorCode`, or `RecipientCode` are missing.
   - All other objects, except for `BDD`, `Bound`, and `Period`, are removed from the environment to conserve memory.

4. **Creating the `raw_text` Variable:**  
   - For every dataset in `BDD`, a subset of columns (including project titles, short descriptions, long descriptions, and funding details) is selected.
   - Rows missing any critical text fields are excluded.
   - The `unite()` function is used twice:
     - First, to combine `ProjectTitle` and `ShortDescription`.
     - Then, to combine the result with `LongDescription` into a single column named `raw_text`.
   - The text is transformed to lowercase to standardize the dataset.

5. **Consolidating the Processed Data:**  
   - The individual data frames in the processed list are combined into one data frame (`Data`) using `rbind`.
   - Duplicate rows are removed.
   - Additional filtering is performed to remove rows containing certain phrases (e.g., "semi-aggregates", "sectors not specified").
   - Any underscores in the `raw_text` field are removed.

6. **Creating a Beta Subsample and Saving Data:**  
   - A random sample of 10,000 rows is selected from `Data` for beta testing.
   - The full dataset is saved as `DataPB.csv` and the beta subsample as `DataPBsample.csv` in the `Data` folder, using the pipe (`|`) as the field delimiter.

---

## Replication Instructions

1. **Download Required Data Files:**  
   - Before running this script, download the original CRS data files from the OECD website as instructed in the documentation for `UploadBase.R`. Ensure these files are placed in the expected directory structure.

2. **Run the UploadBase.R Script:**  
   - First, run `UploadBase.R` to load and combine the raw data files. This script initializes the list `BDD` and other necessary objects.

3. **Set Up Your R Environment:**  
   - Open R or RStudio and install the required packages (e.g., `data.table`, `dplyr`, `ggplot2`) if they are not already installed.
   - Set the working directory to:
     ```r
     ".../Raw Data"
     ```

4. **Execute the Script:**  
   - Run this script (source it) by:
     ```r
     source("YourScriptName.R")
     ```
   - The script will process the data, create the unified `raw_text` variable, remove unwanted rows, and save the processed datasets as CSV files.

5. **Verify the Output:**  
   - Check the console for messages indicating the working directory status and memory clean-up.
   - Verify that `DataPB.csv` and `DataPBsample.csv` are created in the `Data` folder.

---
