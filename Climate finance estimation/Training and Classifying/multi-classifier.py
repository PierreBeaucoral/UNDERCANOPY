#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:37:13 2024

@author: pierrebeaucoral
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# Set up logging for better debugging and progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration: check for CUDA first, then MPS, then CPU fallback
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logging.info(f"Using device: {device}")

# Working directory
wd = "./UNDERCANOPY/Climate finance estimation/Data/"
os.chdir(wd)

# Hyperparameters
base_model = 'climatebert/distilroberta-base-climate-f'
n_words = 150
batch_size = 64  # Increased batch size
random_states = 2022
learning_rate = 2e-5  # Slightly increased learning rate
only_relevant_data = True
num_train_epochs = 75  # Reduced number of epochs
patience = 5  # Early stopping patience

# Training arguments
training_args = {
    "learning_rate": learning_rate,
    "per_device_train_batch_size": batch_size,
    "per_device_eval_batch_size": batch_size,
    "num_train_epochs": num_train_epochs,
    "weight_decay": 0.01,
    "n_words": n_words}

# Load dataset
path = os.path.join(wd, "train_set.csv")
df = pd.read_csv(path)

# Data preparation function
def prepare_data(df, tokenizer, n_words, random_states, only_relevant_data):
    if only_relevant_data:
        df = df[df.relevance == 1]
    else:
        df.loc[df.relevance == 0, 'label'] = 'Not relevant'

    possible_labels = df['label'].unique()
    label_dict = {label: idx for idx, label in enumerate(possible_labels)}
    reverse_label_dict = {idx: label for label, idx in label_dict.items()}

    train_text, temp_text, train_labels, temp_labels = train_test_split(
        df['text'], df['label'],
        random_state=random_states,
        test_size=0.3,
        stratify=df['label']
    )
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels,
        random_state=random_states,
        test_size=0.5,
        stratify=temp_labels
    )

    # Convert labels to numeric values using label_dict
    train_labels = [label_dict[label] for label in train_labels.tolist()]
    val_labels = [label_dict[label] for label in val_labels.tolist()]
    test_labels = [label_dict[label] for label in test_labels.tolist()]

    return train_text.tolist(), val_text.tolist(), test_text.tolist(), train_labels, val_labels, test_labels, label_dict, reverse_label_dict

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Call prepare_data and capture the test_text
train_text, val_text, test_text, train_labels, val_labels, test_labels, label_dict, reverse_label_dict = prepare_data(df, tokenizer, n_words, random_states, only_relevant_data)

# Tokenize the texts
tokens_train = tokenizer.batch_encode_plus(
    train_text,
    max_length=n_words,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
tokens_val = tokenizer.batch_encode_plus(
    val_text,
    max_length=n_words,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
tokens_test = tokenizer.batch_encode_plus(
    test_text,
    max_length=n_words,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Define test_seq and test_mask
test_seq = tokens_test['input_ids']
test_mask = tokens_test['attention_mask']

auto_model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=len(label_dict))

# Convert labels to tensors after mapping
train_y = torch.tensor(train_labels, dtype=torch.long)
val_y = torch.tensor(val_labels, dtype=torch.long)
test_y = torch.tensor(test_labels, dtype=torch.long)

# Create TensorDataset objects
train_data = TensorDataset(tokens_train['input_ids'], tokens_train['attention_mask'], train_y)
val_data = TensorDataset(tokens_val['input_ids'], tokens_val['attention_mask'], val_y)
test_data = TensorDataset(tokens_test['input_ids'], tokens_test['attention_mask'], test_y)

# Define the collate function to handle batching properly
def collate_batch(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_dataloader = DataLoader(
    train_data,
    sampler=RandomSampler(train_data),
    batch_size=batch_size,
    collate_fn=collate_batch
)

val_dataloader = DataLoader(
    val_data,
    sampler=SequentialSampler(val_data),
    batch_size=batch_size,
    collate_fn=collate_batch
)

# Define the model architecture
class BERT_Arch(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert = model
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.softmax(output[0])
        return x

# Initialize model and move to device
model = BERT_Arch(auto_model)
model = model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=training_args["weight_decay"])

# Initialize the warm restart scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

# Compute class weights for imbalanced data
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Define loss function with class weights
cross_entropy = nn.CrossEntropyLoss(weight=weights)

# Initialize learning rate scheduler
num_training_steps = len(train_dataloader) * num_train_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),  # 10% of total steps
    num_training_steps=num_training_steps
)

# Training function with detailed logging
def train():
    model.train()
    total_loss, total_preds = 0, []
    num_batches = len(train_dataloader)

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and step > 0:
            logging.info(f'  Batch {step:>5,} of {num_batches:>5,}.')

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        
        loss.backward()  # Backpropagate the loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Optional gradient clipping
        optimizer.step()  # Update parameters
        scheduler.step()  # Update the learning rate

        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / num_batches
    total_preds = np.concatenate(total_preds, axis=0)

    # Log average loss
    logging.info(f'Average Training Loss: {avg_loss:.3f}')
    return avg_loss, total_preds



# Evaluation function with detailed logging and generic predictions
def evaluate(val_dataloader):
    logging.info("\nEvaluating...")
    model.eval()
    total_loss, total_preds = 0, []
    num_batches = len(val_dataloader)

    for step, batch in enumerate(val_dataloader):
        if step % 50 == 0 and step > 0:
            logging.info(f'  Batch {step:>5,} of {num_batches:>5,}.')

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            preds = model(input_ids, attention_mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / num_batches
    total_preds = np.concatenate(total_preds, axis=0)

    # Log average loss
    logging.info(f'Average Validation Loss: {avg_loss:.3f}')
    return avg_loss, total_preds

# Early stopping logic
best_val_loss = float('inf')
patience_counter = 0

# Main training loop
for epoch in range(num_train_epochs):
    logging.info(f'\nEpoch {epoch + 1}/{num_train_epochs}')
    train_loss, _ = train()
    val_loss, val_preds = evaluate(val_dataloader)

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the model
        torch.save(model.state_dict(), os.path.join(wd, 'saved_weights_multiclass.pt'))
        logging.info("Model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load(os.path.join(wd, 'saved_weights_multiclass.pt')))

# Get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

# Convert logits to predicted classes
preds = np.argmax(preds, axis=1)
prints_true = list(test_y)
prints_pred = list(preds)
_preds = []
underlying_texts = []
true_y = []

for i, prediction in enumerate(preds):
    _preds.append(prediction)
    true_y.append(test_y[i])
    underlying_texts.append(test_text[i]) 

# Print the classification report on the test set
print(label_dict)
print(classification_report(test_y, preds))

# Save the label dictionary and reverse mapping to JSON files
with open('dictionary_classes.json', 'w') as f:
    f.write(json.dumps(label_dict))

with open('reverse_dictionary_classes.json', 'w') as f:
    f.write(json.dumps(reverse_label_dict))

# Get more generic predictions
test_y_generic = ['Adaptation' if y in [10, 13] else 'Environment' if y in [0, 1, 2, 4, 5, 6, 7, 8, 9, 12] else 'Mitigation' for y in test_y]

preds_generic = []
for pred in preds:
    if pred in [10, 13]:  # Adaptation
        preds_generic.append('Adaptation')
    elif pred in [0, 1, 2, 4, 5, 6, 7, 8, 9, 12]:  # Environment
        preds_generic.append('Environment')
    else:  # Any other predictions are Mitigation
        preds_generic.append('Mitigation')

# Print the classification report on the test set for more generic categories
print("Generic Classification Report:")
print(classification_report(test_y_generic, preds_generic))


