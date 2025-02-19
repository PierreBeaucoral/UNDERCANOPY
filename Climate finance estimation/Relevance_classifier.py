#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:37:13 2024

@author: pierrebeaucoral
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Working directory
wd = "/Users/pierrebeaucoral/Documents/Pro/Thèse CERDI/Recherche/Determinant of climate finance/"
os.chdir(wd)

# Hyperparameters configuration
config = {
    "class_number": 2,
    "base_model": 'climatebert/distilroberta-base-climate-f',
    "n_words": 150,
    "batch_size": 32,
    "random_state": 2022,
    "learning_rate": 1e-5,
    "num_epochs": 50,
    "weight_decay": 0.1
}

# Load dataset
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        df = df.rename(columns={'label': 'class'})
        df['label'] = df.relevance
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Prepare data
def prepare_data(df, tokenizer, n_words, random_state):
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        df['text'], df['label'], random_state=random_state, test_size=0.2, stratify=df['label']
    )
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, random_state=random_state, test_size=0.5, stratify=temp_labels
    )
    
    tokenize = lambda texts: tokenizer.batch_encode_plus(texts.tolist(), max_length=n_words, padding='max_length', truncation=True, return_tensors='pt')
    
    tokens_train = tokenize(train_text)
    tokens_val = tokenize(val_text)
    tokens_test = tokenize(test_text)

    logging.info("Data preparation complete.")
    return tokens_train, tokens_val, tokens_test, train_labels, val_labels, test_labels

# Set up autocast context
def get_autocast_context():
    if device.type == 'mps':
        return nullcontext()
    elif device.type == 'cuda':
        return autocast(device_type='cuda')
    else:
        return nullcontext()

# Load model and tokenizer
def load_model_and_tokenizer(base_model, class_number):
    tokenizer = AutoTokenizer.from_pretrained(base_model, clean_up_tokenization_spaces=True)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=class_number)
    logging.info("Model and tokenizer loaded successfully.")
    return tokenizer, model

# Convert labels to tensors
def convert_labels_to_tensors(train_labels, val_labels, test_labels):
    return (torch.tensor(train_labels.tolist(), dtype=torch.long),
            torch.tensor(val_labels.tolist(), dtype=torch.long),
            torch.tensor(test_labels.tolist(), dtype=torch.long))

# Create DataLoader
def create_dataloader(data, batch_size):
    return DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size, collate_fn=collate_batch)

# Collate function for DataLoader
def collate_batch(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Define model architecture
class BERT_Arch(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.3)  # Add dropout layer with 30% drop rate

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.logits
        pooled_output = self.dropout(pooled_output)  # Apply dropout before final layer
        return pooled_output

# Training function with mixed precision and warm restarts
def train(model, train_dataloader, val_dataloader, optimizer, cross_entropy, scaler, num_epochs, patience=10):
    model.train()
    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    
    # Define CosineAnnealingWarmRestarts scheduler
    T_0 = 10  # The number of epochs after which a warm restart occurs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

            model.zero_grad()
            with get_autocast_context():
                preds = model(input_ids, attention_mask)
                loss = cross_entropy(preds, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Step the scheduler after each batch
            scheduler.step(epoch + len(train_dataloader) / len(train_dataloader))

        avg_loss = total_loss / len(train_dataloader)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}')
        
        valid_loss, _ = evaluate(model, val_dataloader, cross_entropy)
        
        if valid_loss < best_valid_loss:
            logging.info('Validation loss decreased. Saving model...')
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(wd, 'Data/Estimation of Climate Finance/saved_weights_relevance.pt'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info('Early stopping...')
                break



# Evaluation function with mixed precision
def evaluate(model, val_dataloader, cross_entropy):
    logging.info("\nEvaluating...")
    model.eval()
    total_loss, total_preds = 0, []

    for step, batch in enumerate(val_dataloader):
        if step % 50 == 0 and step > 0:
            logging.info(f'  Batch {step:>5,} of {len(val_dataloader):>5,}.')

        batch = {key: value.to(device) for key, value in batch.items()}
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        with get_autocast_context():
            with torch.no_grad():
                preds = model(input_ids, attention_mask)
                loss = cross_entropy(preds, labels)

        total_loss += loss.item()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader)
    logging.info(f'Validation Loss: {avg_loss:.4f}')  # Log validation loss once
    
    # Concatenate all predictions to form a 2D array
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# Main execution flow
def main():
    try:
        # Load dataset
        df = load_dataset(os.path.join(wd, "Data/Estimation of Climate Finance/train_set.csv"))

        # Load model and tokenizer
        tokenizer, auto_model = load_model_and_tokenizer(config["base_model"], config["class_number"])

        # Prepare data
        tokens_train, tokens_val, tokens_test, train_labels, val_labels, test_labels = prepare_data(df, tokenizer, config["n_words"], config["random_state"])

        # Convert labels to tensors
        train_y, val_y, test_y = convert_labels_to_tensors(train_labels, val_labels, test_labels)

        # Create TensorDataset objects
        train_data = TensorDataset(tokens_train['input_ids'], tokens_train['attention_mask'], train_y)
        val_data = TensorDataset(tokens_val['input_ids'], tokens_val['attention_mask'], val_y)
        test_data = TensorDataset(tokens_test['input_ids'], tokens_test['attention_mask'], test_y)

        # Define DataLoader
        train_dataloader = create_dataloader(train_data, config["batch_size"])
        val_dataloader = create_dataloader(val_data, config["batch_size"])  # Ensure val_dataloader is created

        # Initialize model
        model = BERT_Arch(auto_model).to(device)

        # Define optimizer and loss function
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        cross_entropy = nn.CrossEntropyLoss()

        # Initialize GradScaler for mixed precision
        scaler = GradScaler()

        # Train the model
        train(model, train_dataloader, val_dataloader, optimizer, cross_entropy, scaler, config["num_epochs"])

        # Evaluate the model on the test set
        model.load_state_dict(torch.load(os.path.join(wd, 'Data/Estimation of Climate Finance/saved_weights_relevance.pt')))
        test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=config["batch_size"], collate_fn=collate_batch)

        # Get predictions and labels for the test set
        test_loss, test_preds = evaluate(model, test_dataloader, cross_entropy)

        # Generate classification report
        test_y_labels = test_y.numpy()
        test_preds_labels = np.argmax(test_preds, axis=1)
        report = classification_report(test_y_labels, test_preds_labels, target_names=['Class 0', 'Class 1'])
        logging.info("\n" + report)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
