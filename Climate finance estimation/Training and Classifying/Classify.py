import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings

# Set up logging to show warnings and errors, with INFO for key steps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress SettingWithCopyWarning
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')


def load_json_file(filename):
    """Loads a JSON file and returns it as a Python dictionary."""
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File {filename} not found.")
        raise


def initialize_device():
    """Sets the device to MPS if available, otherwise CPU."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


def load_model_and_tokenizer(base_model, device, num_labels, weight_path):
    """Loads a pre-trained model, tokenizer, and assigns it to the specified device."""
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
    model = BERT_Arch(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    logging.info(f"Model loaded from {weight_path}")
    
    return model, tokenizer


def predict_labels(text_list, model, tokenizer, device, batch_size=64):
    """Predicts labels for a list of texts in batches."""
    all_preds = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        tokenized = tokenizer.batch_encode_plus(
            batch_texts,
            max_length=150,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)  # Directly get logits
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            all_preds.extend(preds)
    
    return all_preds


def process_chunk(chunk, relevance_classifier, multiclass_classifier, tokenizer, device, label_dict):
    """Processes a single chunk of data, predicting relevance and class labels."""
    chunk['DonorType'] = np.select(
        [
            chunk.DonorCode < 807,
            chunk.DonorCode > 1600,
            chunk.DonorCode == 104,
            chunk.DonorCode == 820
        ],
        ['Donor Country', 'Private Donor', 'Multilateral Donor Organization', 'Donor Country'],
        default='Multilateral Donor Organization'
    )

    chunk = chunk[chunk.DonorType == 'Donor Country'].copy()  # Filter to relevant donor types

    chunk['prediction_criterium'] = (chunk.raw_text.str.split().str.len() >= 3).astype(int)
    relevant_rows = chunk[chunk['prediction_criterium'] == 1]

    if not relevant_rows.empty:
        chunk['climate_relevance'] = 0
        chunk.loc[chunk['prediction_criterium'] == 1, 'climate_relevance'] = predict_labels(
            relevant_rows.raw_text.to_list(),
            relevance_classifier,
            tokenizer,
            device
        )

        relevant_climate_rows = chunk[chunk.climate_relevance == 1]
        chunk['climate_class_number'] = 500  # Default class for non-relevant documents
        if not relevant_climate_rows.empty:
            chunk.loc[chunk.climate_relevance == 1, 'climate_class_number'] = predict_labels(
                relevant_climate_rows.raw_text.to_list(),
                multiclass_classifier,
                tokenizer,
                device
            )
        
        chunk['climate_class'] = chunk['climate_class_number'].astype(str).replace(label_dict)
        chunk.loc[chunk.climate_relevance == 0, 'climate_class'] = '500'  # Default class for non-relevant
    else:
        logging.info("No relevant rows found in chunk, skipping predictions.")

    return chunk


def save_category_data(df, category_list, filename, wd):
    """Filters the dataframe for specific climate categories and saves to a CSV."""
    filtered_df = df[df.climate_class.isin(category_list)]
    grouped_df = filtered_df.groupby('Year')[['USD_Disbursement_Defl', 'USD_Commitment_Defl']].sum().reset_index()
    output_path = os.path.join(wd, filename)
    grouped_df.to_csv(output_path, encoding='utf8', index=False, header=True)
    logging.info(f"Saved {filename} with {grouped_df.shape[0]} rows.")


class BERT_Arch(nn.Module):
    """Custom BERT-based model architecture."""
    def __init__(self, model):
        super().__init__()
        self.bert = model

    def forward(self, sent_id, attention_mask):
        output = self.bert(sent_id, attention_mask=attention_mask)
        return output.logits  # Return only the logits


# Start of the main execution flow
logging.info("Starting main execution")

# Initialize device
device = initialize_device()

# Set working directory
wd = ".../Climate finance estimation/Data/"

# Load label dictionary
label_dict_path = os.path.join(wd, 'reverse_dictionary_classes.json')
label_dict = load_json_file(label_dict_path)

# Base model for BERT
base_model = 'climatebert/distilroberta-base-climate-f'

# Load relevance and multiclass classifiers
relevance_classifier, tokenizer = load_model_and_tokenizer(
    base_model,
    device,
    num_labels=2,
    weight_path=os.path.join(wd, 'saved_weights_relevance.pt')
)

multiclass_classifier, _ = load_model_and_tokenizer(
    base_model,
    device,
    num_labels=len(label_dict),
    weight_path=os.path.join(wd, 'saved_weights_multiclass.pt')
)

# Load the dataset
aid_data_path = '.../Data/Data.csv'
data_df = pd.read_csv(aid_data_path, encoding='utf8')

# Group by raw_text and aggregate donorcode
grouped_data = data_df.groupby('raw_text').agg({
    'DonorCode': 'first',  # Use 'first' to keep DonorCode; could use other aggregations if needed
    'raw_text': 'first'  # Keep raw_text
}).reset_index(drop=True)

# Process chunks of grouped data
chunk_size = 10**3
total_chunks = (len(grouped_data) + chunk_size - 1) // chunk_size
logging.info(f"Total number of chunks: {total_chunks}")

# Process chunks in parallel
processed_chunks = []
with ThreadPoolExecutor(max_workers=12) as executor:
    for processed_chunk in tqdm(
        executor.map(
            lambda chunk: process_chunk(
                chunk, relevance_classifier, multiclass_classifier, tokenizer, device, label_dict
            ),
            np.array_split(grouped_data, total_chunks)
        ),
        total=total_chunks,
        desc="Processing chunks"
    ):
        processed_chunks.append(processed_chunk)

# Concatenate processed chunks
aid_data = pd.concat(processed_chunks, axis=0, ignore_index=True)

# Merge classification results back to the main dataset
columns_to_drop = ['climate_relevance', 'climate_class_number', 'climate_class']
data_df = data_df.drop(columns=columns_to_drop, errors='ignore')

logging.info("Merging processed data with Data.csv based on raw_text and DonorCode")
merged_df = pd.merge(data_df, aid_data, on=['raw_text', 'DonorCode'], how='left')

# Save the final merged dataset
final_csv_path = os.path.join(wd, 'ClassifiedCRS.csv')
merged_df.to_csv(final_csv_path, encoding='utf8', index=False)
logging.info(f"Saved final merged climate finance data to {final_csv_path}")

# Filter and save category-specific data
category_mappings = {
    'adaptation': [10, 13],
    'environment': [0, 1, 2, 5, 9, 12, 14, 15],
    'mitigation': [3, 4, 6, 7, 8, 11, 16]
}

for category_name, categories in category_mappings.items():
    save_category_data(merged_df, categories, f'{category_name}_disbursement.csv', wd)

logging.info("Main execution completed")
