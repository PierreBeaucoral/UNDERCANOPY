# Climate Finance Classification Codebase

This repository contains a comprehensive set of Python modules developed for the classification and estimation of climate finance data. The codebase integrates state-of-the-art machine learning, natural language processing, and statistical estimation techniques to (i) classify climate finance projects by category (e.g., adaptation, mitigation, environment), (ii) filter for relevance, and (iii) estimate the financial flows associated with these projects. The tools provided here support rigorous academic research by enabling reproducible, transparent, and extensible analyses of climate finance.

---

## Overview

This codebase is organized into several modules, each responsible for a distinct step in the overall classification and estimation pipeline:

### **EstimationClimateFinance.py**  
The script is designed to replicate data preprocessing pipeline for climate finance classification. It:

 - Loads and preprocesses an initial projects clustering dataset.
 - Filters projects based on topic codes to isolate climate-related projects.
 - Renames project labels based on topic values and manually updates a subset of indices.
 - Samples projects proportionally from each climate topic group.
 - Merges these climate projects with a set of non-climate projects from another data source.
 - Adjusts the merged dataset by removing duplicates and ensuring balanced class representation.
 - Saves the final, balanced dataset for subsequent model training and analysis.

### **Relevance_classifier.py**  
This script orchestrates the following steps:

1. **Data Loading and Preprocessing:**  
   The script loads a training dataset from a CSV file, preprocesses the text data, and splits it into training, validation, and test sets. Tokenization is performed using a Hugging Face tokenizer associated with the chosen pre-trained model.

2. **Model and Tokenizer Setup:**  
   It loads a pre-trained transformer model (specified by the `base_model` in the configuration) and its corresponding tokenizer. The model is then adapted for sequence classification with a binary output.

3. **Custom Model Architecture:**  
   A custom wrapper (`BERT_Arch`) is defined around the pre-trained model. This class adds a dropout layer to the logits output for regularization, which is crucial for preventing overfitting.

4. **Training with Mixed Precision and Warm Restarts:**  
   The training routine leverages PyTorch’s mixed precision capabilities (via `torch.cuda.amp`) and uses a CosineAnnealingWarmRestarts learning rate scheduler. Gradient clipping is applied to stabilize training.

5. **Evaluation and Reporting:**  
   After training, the script evaluates the model on a held-out test set and generates a detailed classification report (precision, recall, F1-score) using scikit-learn’s metrics.

6. **Reproducibility and Logging:**  
   Logging is set up to provide informative messages during execution, ensuring transparency and reproducibility. The script also saves the best model weights for future use.
   
### **multi-classifier.py**  
This script implements a complete deep learning workflow that includes:

- **Data Loading and Preprocessing:**  
  Reads a CSV dataset, optionally filters the data to include only relevant (climate) projects, and splits the data into training, validation, and test sets using stratified sampling.

- **Tokenization and Encoding:**  
  Uses a Hugging Face tokenizer to convert text inputs into token IDs and attention masks with a fixed maximum length.

- **Model Initialization:**  
  Loads a pre-trained transformer model (from the `climatebert/distilroberta-base-climate-f` checkpoint) and adapts it for sequence classification. A custom wrapper (`BERT_Arch`) applies a LogSoftmax layer to the model’s logits.

- **Training with Advanced Techniques:**  
  The training loop incorporates:
  - Class weight computation for imbalanced data.
  - A linear learning rate scheduler with warmup.
  - Cosine annealing warm restarts.
  - Gradient clipping for training stability.
  - Detailed logging of progress and loss metrics.

- **Evaluation and Reporting:**  
  The script evaluates the model on a held-out test set, computes loss, and produces a detailed classification report. Additionally, it maps fine-grained label predictions to more generic categories and prints a secondary classification report for these aggregated classes.

- **Reproducibility:**  
  Fixed random seeds, detailed logging, and model checkpointing (saving the best model based on validation loss) ensure that the experiments are reproducible and transparent.
   
### **Classify.py**  

1. **Environment Setup and Logging:**  
   Configures logging to display key steps and warnings. It also sets the device (using MPS when available, otherwise CPU) for model inference.

2. **Utility Functions:**  
   - **JSON Loader:**  
     Loads JSON files (e.g., for label dictionaries).
   - **Device Initialization:**  
     Chooses the computation device.
   - **Model and Tokenizer Loader:**  
     Loads a pre-trained transformer model and tokenizer from Hugging Face, wraps the model with a custom architecture (`BERT_Arch`), and loads saved weights.
   - **Prediction Function:**  
     Processes a list of texts in batches and outputs class predictions.
   - **Chunk Processing:**  
     Processes subsets (chunks) of the input dataset by filtering and applying both the relevance classifier (binary) and the multiclass classifier. It assigns a final climate class based on model outputs and a provided label mapping.
   - **Category Data Saving:**  
     Filters the final merged data by specific climate category codes, aggregates funding data per year, and saves the results to CSV files.

3. **Data Loading and Preprocessing:**  
   - The script loads a main dataset (from a CSV file produced by a clustering procedure) and groups it by unique project text.
   - It then processes the grouped data in parallel using a `ThreadPoolExecutor` to apply the classification models on each chunk.

4. **Merging and Saving:**  
   - After processing, the predicted classification results are merged back with the original dataset based on `raw_text` and `DonorCode`.
   - The final merged dataset is saved as a classified CSV file.
   - Additionally, the script filters the merged data to generate category-specific CSV summaries for adaptation, environment, and mitigation funding (based on a defined mapping).

### **meta.py**  

The script processes a CSV file containing previously classified climate finance projects. It:

1. **Loads the Classified Data:**  
   Reads the dataset from a CSV file and drops rows missing the `climate_class_number`.

2. **Applies Meta-Categorization:**  
   - Defines three lists of numeric category codes corresponding to the three meta-categories:
     - *Adaptation*: `[10, 13]`
     - *Environment*: `[0, 1, 2, 5, 9, 12, 14, 15]`
     - *Mitigation*: `[3, 4, 6, 7, 8, 11, 16]`
   - Creates a new column `meta_category` initialized to `"None"`.
   - Updates `meta_category` based on the value of `climate_class_number`.

3. **Data Integrity and Consistency Checks:**  
   - Reports rows with missing `climate_class_number` values.
   - Identifies groups of rows (by `raw_text`) where some rows have missing values and others do not.
   - Prints warnings if any relevant projects (i.e. those with `climate_relevance` equal to 1) remain unclassified.
   - Verifies that all projects with `meta_category` still set to `"None"` are exactly those with `climate_relevance` equal to 0.
   - Checks that the total number of rows in the dataset remains unchanged after processing.

4. **Saves the Processed Data:**  
   The final DataFrame, now enriched with a `meta_category` column, is saved back to a CSV file (using a pipe `|` as the delimiter).

---

## Folder Structure and Module Details

### EstimationClimateFinance.py

#### 1. Setting Up and Data Loading

- **Working Directory:**  
  The script sets the working directory to a specified path where the project files are stored.

- **Loading the Projects Cluster Data:**  
  The file `projects_clusters.csv` is loaded into a DataFrame (`df1`). If the file is not found, the script raises an error.  
  This file contains various topics (represented by numeric codes) that have been determined by a clustering procedure on aid activities.

#### 2. Filtering and Saving Climate Adaptation Projects

- **Climate Adaptation Extraction:**  
  The script extracts a subset of `df1` where the `Topic` column equals 5. These projects are assumed to pertain to climate adaptation and are saved as `climate_adaptation_projects.csv` for reference.

#### 3. Filtering by Climate Topics and Non-Climate Projects

- **Defining Climate Topics:**  
  A list of topic codes (e.g., 2, 5, 11, 26, etc.) is defined to represent climate-relevant projects.
  
- **Filtering DataFrames:**  
  - `filtered_df1` contains projects with a `Topic` value in the specified climate topics.
  - `non_climate_df1` is created as the complement (i.e., projects not in the climate topics list) with duplicates (based on `raw_text`) removed. It is saved as `non_climate_projects.csv`.

#### 4. Relabeling and Adding Relevance Flags

- **Renaming Projects:**  
  A dictionary (`rename_dict`) maps certain topic codes to more descriptive labels (e.g., topics 11 and 312 become "Renewable energy").  
  The script uses this dictionary to update the `CustomName` column in the filtered climate projects.  
- **Adding Relevance:**  
  A new column `relevance` is added and set to 1 for these projects.  
  The updated DataFrame is saved as `climate_projects.csv`.

#### 5. Sampling Climate Projects

- **Removing Duplicates:**  
  Duplicate projects (based on `raw_text`) are removed.
- **Determining Sample Size:**  
  A sample size corresponding to 5% of the total number of climate projects is calculated.
- **Proportional Sampling:**  
  For each `Topic` group within the filtered climate projects, a proportionate number of projects is sampled to form a representative subset.  
- **Column Selection and Renaming:**  
  The resulting sample is restricted to the columns `raw_text`, `relevance`, and `CustomName`, which are then renamed to `text`, `relevance`, and `label`, respectively.

#### 6. Merging with Non-Climate Projects

- **Loading the Non-Climate Data:**  
  A second CSV file (`train_set.csv`) is loaded, and rows with `relevance == 0` are selected.
- **Merging Datasets:**  
  The sampled climate projects (with `relevance == 1`) are concatenated with the non-climate projects to form a merged DataFrame.
- **Sorting and Resetting Index:**  
  The merged DataFrame is sorted by the `text` column and the index is reset.

#### 7. Manual Replacement and Label Updates

- **Index-Based Updates:**  
  A list of indices (`indices_to_remove`) is specified to remove certain projects from the merged dataset, and a corresponding list of indices (`indices_to_add`) is used to add projects from the climate adaptation subset.  
- **Label Updates:**  
  A dictionary (`label_updates`) defines new label values for selected indices. The script iterates over this dictionary to update the `label` and set `relevance` to 1 for those projects.

#### 8. Deduplication and Balancing

- **Deduplication:**  
  After merging and manual updates, the script removes any duplicate entries based on the `text` column.
- **Balancing the Dataset:**  
  The number of projects with `relevance == 1` is counted.  
  The script calculates how many additional non-climate projects (with `relevance == 0`) are needed to balance the dataset.  
- **Proportional Sampling from Non-Climate Projects:**  
  Duplicates are removed from the non-climate projects that already exist in the merged DataFrame.  
  Sampling is performed within each non-climate `Topic` group according to its proportion in the overall non-climate subset.  
  The sampled non-climate projects are then merged into the main DataFrame.
- **Final Deduplication and Reporting:**  
  Duplicates are dropped again, and final counts for projects with relevance 1 and 0 are printed.

#### 9. Saving the Final Dataset

- **Output:**  
  The balanced and merged dataset, containing only the columns `text`, `label`, and `relevance`, is saved to `train_set.csv` in the `Data/Estimation of Climate Finance` folder.
  
- **Replication:**  
  This final dataset is intended for use in downstream machine learning tasks such as classification and regression analyses on climate finance.

---

#### How to Replicate

1. **Ensure File Paths Are Correct:**  
   Verify that your data files (e.g., `projects_clusters.csv` and the initial `train_set.csv`) are stored in the paths specified in the script. Adjust the paths if necessary.

2. **Run the Script:**  
   Execute the script using Python 3:
   ```bash
   python your_script_name.py

---

### Relevance_classifier.py

#### 1. **Device and Logging Setup**
- **Device Configuration:**  
  The code automatically selects the GPU (CUDA) if available, or falls back to Apple’s MPS (if available) or CPU.
  
- **Logging:**  
  A logging configuration is established to output messages with timestamps and log levels, facilitating debugging and progress tracking.

#### 2. **Data Loading and Preprocessing**
- **`load_dataset(path)`:**  
  Reads a CSV file from the specified path, renames columns for consistency, and sets the label column using the `relevance` field.
  
- **`prepare_data(df, tokenizer, n_words, random_state)`:**  
  Splits the dataset into training (80%), validation (10%), and test (10%) subsets using stratified sampling. It tokenizes the text data using the provided tokenizer with a maximum length of `n_words`.

- **`convert_labels_to_tensors(...)`:**  
  Converts label data into PyTorch tensors required for model training.

#### 3. **Model and Tokenizer Initialization**
- **`load_model_and_tokenizer(base_model, class_number)`:**  
  Loads the pre-trained transformer model and tokenizer from Hugging Face. The model is configured for a binary classification task with `num_labels` set to 2.

- **Custom Model Architecture – `BERT_Arch`:**  
  A PyTorch module that wraps the pre-trained model and adds a dropout layer (with 30% drop rate) on the logits to improve generalization.

#### 4. **Training and Evaluation**
- **`train(...)`:**  
  Implements the training loop with mixed precision training (using `autocast` and `GradScaler`), gradient clipping, and a learning rate scheduler based on CosineAnnealingWarmRestarts. The model’s performance is monitored on the validation set, and early stopping is applied based on a patience parameter.

- **`evaluate(...)`:**  
  Evaluates the model on the validation or test set, computes the average loss, and collects model predictions.

- **Learning Rate Scheduling:**  
  Uses a CosineAnnealingWarmRestarts scheduler to adjust the learning rate dynamically during training.

#### 5. **DataLoader and Batch Collation**
- **`create_dataloader(...)` and `collate_batch(...)`:**  
  Create PyTorch DataLoader objects to feed batches of data into the model. The custom collate function stacks input IDs, attention masks, and labels into a single batch dictionary.

#### 6. **Main Execution Flow**
- The `main()` function coordinates:
  - Loading the dataset.
  - Initializing the model and tokenizer.
  - Preparing the data and converting labels.
  - Creating DataLoader objects.
  - Training the model while monitoring validation loss.
  - Loading the best saved model weights.
  - Evaluating on the test set and generating a detailed classification report.

---


### multi-classifier.py

#### 1. Environment Setup

- **Device Configuration:**  
  The script first checks for available GPUs (CUDA), then Apple’s MPS, and falls back to CPU if neither is available. This guarantees optimal computation resources are utilized.

- **Logging:**  
  Logging is configured to include timestamps and severity levels, enabling detailed monitoring of training progress and debugging.

#### 2. Hyperparameter and Training Configuration

- **Hyperparameters:**  
  The script defines key hyperparameters such as:
  - `base_model`: Pre-trained transformer model identifier.
  - `n_words`: Maximum number of tokens per text (set to 150).
  - `batch_size`: Batch size for training (set to 64).
  - `learning_rate`: Initial learning rate (set to 2e-5).
  - `num_train_epochs`: Maximum number of training epochs (set to 75).
  - `patience`: Early stopping patience (set to 5 epochs).
  - `only_relevant_data`: A boolean flag to optionally filter the dataset to only include relevant projects.

- **Training Arguments:**  
  These are encapsulated in a dictionary (`training_args`) to organize settings such as learning rate, batch sizes, number of epochs, weight decay, and token length.

#### 3. Data Loading and Preparation

- **Dataset Loading:**  
  The dataset is loaded from `Data/Estimation of Climate Finance/train_set.csv` using a semicolon (`;`) delimiter. If the file is not found, the script logs an error and raises an exception.

- **Data Preparation Function (`prepare_data`):**  
  - **Filtering:** If `only_relevant_data` is set to `True`, the DataFrame is filtered to keep only rows with `relevance == 1`.
  - **Label Mapping:** The function creates a mapping from unique label names to numerical indices and a reverse mapping.
  - **Splitting:** The data is split into training (70%), validation (15%), and test (15%) sets using stratified sampling to maintain label proportions.
  - **Return Values:** The function returns lists of text samples, corresponding numeric labels, and the label dictionaries.

#### 4. Tokenization

- **Batch Encoding:**  
  The script uses the Hugging Face tokenizer’s `batch_encode_plus` method to tokenize the training, validation, and test texts. The texts are padded and truncated to the fixed length (`n_words`), and tensors are returned for model input.

#### 5. Model and Architecture

- **Model Loading:**  
  The pre-trained model is loaded using `AutoModelForSequenceClassification` with the number of labels determined from the label dictionary.

- **Custom Model Wrapper (`BERT_Arch`):**  
  This wrapper class takes the loaded transformer model and applies a `LogSoftmax` function over its output logits. The LogSoftmax layer converts raw scores into log-probabilities, which are used by the loss function.

#### 6. Data Preparation for Training

- **TensorDataset Creation:**  
  Training, validation, and test datasets are wrapped into PyTorch `TensorDataset` objects.
- **DataLoader Creation:**  
  Custom collate function (`collate_batch`) is defined to stack the inputs and labels. DataLoaders for training and validation use random and sequential samplers, respectively.

#### 7. Training Pipeline

- **Optimizer and Scheduler:**  
  The `AdamW` optimizer is used with weight decay. A linear scheduler with warmup (10% of total training steps) is employed using `get_linear_schedule_with_warmup`.  
  In addition, a CosineAnnealingWarmRestarts scheduler is initialized (although later the linear scheduler is used).

- **Class Weights:**  
  To handle class imbalance, class weights are computed and applied to the cross-entropy loss.

- **Training Loop:**  
  For each epoch:
  - The training function iterates over batches, computes predictions, evaluates loss, backpropagates gradients (with clipping), and updates the learning rate.
  - The average training loss is logged.
  - The validation function is called to compute the average validation loss.
  - Early stopping is triggered if validation loss does not improve for a specified number of epochs, and the best model is saved.

#### 8. Evaluation and Reporting

- **Test Set Evaluation:**  
  After training, the best model is loaded, and predictions are computed on the test set.
- **Metrics:**  
  The script uses scikit-learn’s `classification_report` to generate performance metrics (precision, recall, F1-score) for the test set.
- **Label Dictionary Export:**  
  The label mapping and reverse mapping are saved as JSON files for future reference.
- **Generic Prediction Mapping:**  
  The script maps the fine-grained predicted labels to three generic categories (“Adaptation”, “Environment”, “Mitigation”) based on preset conditions and prints a separate classification report for these generic classes.

---

### Classify.py

#### Environment and Logging Setup

- **Logging and Warnings:**  
  Logging is configured with `INFO` level to track progress and report errors. Warnings from pandas (e.g., `SettingWithCopyWarning`) are suppressed for clarity.
  
- **Device Initialization:**  
  The function `initialize_device()` selects the device:
  - Uses Apple’s MPS if available.
  - Otherwise, falls back to CPU.
  
  This ensures the code runs optimally in different hardware environments.

#### Model Loading

- **Loading JSON Configuration:**  
  The function `load_json_file(filename)` loads JSON files (e.g., reverse label mappings) into a Python dictionary.
  
- **Model and Tokenizer:**  
  The function `load_model_and_tokenizer(base_model, device, num_labels, weight_path)`:
  - Loads a pre-trained model for sequence classification from Hugging Face.
  - Wraps the model using the custom class `BERT_Arch`, which simply passes inputs through the model and returns logits.
  - Loads model weights from a specified file.
  - Loads the corresponding tokenizer.
  
- **Prediction Function:**  
  `predict_labels(text_list, model, tokenizer, device, batch_size=64)` tokenizes text in batches, runs model inference, and returns the predicted class labels.

#### Data Processing and Parallelization

- **Chunk Processing:**  
  The core function `process_chunk(chunk, relevance_classifier, multiclass_classifier, tokenizer, device, label_dict)`:
  - Adds a new column `DonorType` based on `DonorCode` to filter projects from donor countries.
  - Computes a simple criterion (`prediction_criterium`) based on the word count of `raw_text` (only texts with at least three words are processed).
  - For texts meeting the criterion, it applies the relevance classifier to predict a binary relevance flag.
  - For rows predicted as relevant (i.e., `climate_relevance` equal to 1), it uses the multiclass classifier to generate a detailed climate class.
  - The numeric class predictions are then mapped to string labels using the provided `label_dict`.
  - Non-relevant rows are assigned a default class of "500".
  
- **Parallel Processing:**  
  The script splits the grouped dataset (grouped by `raw_text`) into chunks and processes these chunks in parallel using a `ThreadPoolExecutor`. The results are then concatenated to form the complete classified dataset.

#### Merging and Final Output

- **Dataset Merging:**  
  The processed predictions are merged back with the original dataset (loaded from `Data.csv`) on matching columns (`raw_text` and `DonorCode`).
  
- **Saving the Classified Data:**  
  The merged DataFrame is saved as `ClassifiedCRS.csv` in the designated folder.
  
- **Category-Specific Saving:**  
  The function `save_category_data(df, category_list, filename, wd)` filters the merged dataset for specific climate category codes (defined in a mapping) and aggregates funding (disbursement and commitment) by year. It then saves each category’s aggregated data to its own CSV file.

#### Custom Model Architecture

- **BERT_Arch:**  
  This is a simple wrapper around the loaded transformer model. It directly returns the output logits from the model, which are used in the prediction functions.

---

#### Replication Instructions

1. **Data Preparation:**  
   Ensure that the following files exist in the specified paths:
   - The raw data file (e.g., `Data.csv` from the ML clustering aid activities folder).
   - The JSON file containing the reverse label dictionary (`reverse_dictionary_classes.json`).
   - The pre-trained model weight files:
     - `saved_weights_relevance.pt` for the relevance classifier.
     - `saved_weights_multiclass.pt` for the multiclass classifier.
     
2. **Environment Setup:**  
   - Install required packages:
     ```bash
     pip install torch transformers pandas numpy scikit-learn tqdm
     ```
   - Confirm that your device supports MPS or falls back to CPU.

3. **Running the Script:**  
   Execute the script via the command line:
   ```bash
   python <script_name>.py

---

### meta.py

1. **Setting Up the Environment:**  
   - The working directory is set using `os.chdir()`, ensuring that all file paths are relative to the specified root.
   - The CSV file is loaded with a custom `csv_import` function that sets proper encoding and data types for key columns.

2. **Initial Data Filtering:**  
   - The dataset is filtered to remove any rows with missing `climate_class_number` values.
   - A check on the total number of rows is performed (with an output if the count is below 2,700,000).

3. **Meta-Category Assignment:**  
   - Three lists of numeric identifiers define which detailed classes map to the three overarching meta-categories.
   - The script initializes a new column `meta_category` to `"None"` for all rows.
   - Using `DataFrame.loc`, rows whose `climate_class_number` values fall in the respective lists are assigned the meta-categories "Adaptation", "Mitigation", or "Environment".
   - The script prints the number of rows assigned to each category for verification.

4. **Integrity Checks:**  
   - The script reports the number of rows with `NaN` values in `climate_class_number`.
   - It groups data by `raw_text` to detect cases where a project might have mixed missing and non-missing `climate_class_number` values.
   - A warning is issued if any `NaN` values are found in `climate_class_number`.
   - It then checks if all rows with a meta-category of `"None"` are exactly those with `climate_relevance` equal to 0. If this condition is not met, the script prints the conflicting counts and terminates.

5. **Final Consistency Test and Saving:**  
   - A final test confirms that the total number of rows in the DataFrame remains unchanged after processing.
   - The processed DataFrame is then saved to a CSV file (`climate_finance_total.csv`) using a pipe (`|`) delimiter.

---

#### Replication Instructions

1. **Data Preparation:**  
   Ensure that the input file:
   - `ClassifiedCRS.csv` is located in the folder `data/Estimation of Climate Finance/` relative to the working directory.
   - This file should contain at least the columns `raw_text`, `climate_class_number`, and `climate_relevance`.

2. **Environment Setup:**  
   - Install the necessary Python packages (e.g., Pandas, NumPy).
   - Confirm that your working directory (`wd`) is correctly set at the beginning of the script.

3. **Execution:**  
   Run the script using:
   ```bash
   python <script_name>.py
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

