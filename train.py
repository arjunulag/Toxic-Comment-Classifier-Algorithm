"""
Toxic Comment Classification using Transfer Learning with DistilBERT
=====================================================================
This script fine-tunes a pre-trained DistilBERT model for binary classification
of toxic vs non-toxic comments.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Model settings - Using DistilBERT for best quality
    MODEL_NAME = 'distilbert-base-uncased'  # 66M params - best language understanding
    MAX_LENGTH = 256
    
    # Training settings
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    
    # Data settings
    DATA_PATH = 'data.csv'
    TOXICITY_THRESHOLD = 0.5  # Comments with target >= 0.5 are toxic
    SAMPLE_SIZE = None  # Full dataset (~175K samples)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Output
    MODEL_SAVE_PATH = 'toxic_classifier_model'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Dataset Class
# ============================================================================

class ToxicCommentsDataset(Dataset):
    """Custom Dataset for toxic comments classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_and_preprocess_data(config):
    """Load and preprocess the toxic comments dataset."""
    
    print("=" * 60)
    print("Loading and Preprocessing Data")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(config.DATA_PATH)
    print(f"✓ Loaded {len(df):,} comments")
    
    # Sample if specified (useful for testing)
    if config.SAMPLE_SIZE:
        df = df.sample(n=min(config.SAMPLE_SIZE, len(df)), random_state=config.RANDOM_STATE)
        print(f"✓ Sampled {len(df):,} comments for training")
    
    # Extract text and create binary labels
    texts = df['comment_text'].fillna('').values
    labels = (df['target'] >= config.TOXICITY_THRESHOLD).astype(int).values
    
    # Class distribution
    toxic_count = labels.sum()
    non_toxic_count = len(labels) - toxic_count
    print(f"✓ Class distribution:")
    print(f"  - Non-toxic (0): {non_toxic_count:,} ({non_toxic_count/len(labels)*100:.1f}%)")
    print(f"  - Toxic (1):     {toxic_count:,} ({toxic_count/len(labels)*100:.1f}%)")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels
    )
    
    print(f"✓ Train size: {len(X_train):,}")
    print(f"✓ Validation size: {len(X_val):,}")
    
    return X_train, X_val, y_train, y_val


def create_data_loaders(X_train, X_val, y_train, y_val, tokenizer, config):
    """Create PyTorch DataLoaders for training and validation."""
    
    train_dataset = ToxicCommentsDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
    val_dataset = ToxicCommentsDataset(X_val, y_val, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


# ============================================================================
# Model Setup
# ============================================================================

def setup_model(config):
    """Initialize tokenizer and model with pre-trained weights."""
    
    print("\n" + "=" * 60)
    print("Setting Up Model")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print(f"✓ Loaded tokenizer: {config.MODEL_NAME}")
    
    # Load pre-trained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    
    # Move model to device
    model = model.to(config.DEVICE)
    print(f"✓ Loaded model: {config.MODEL_NAME}")
    print(f"✓ Device: {config.DEVICE}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    return tokenizer, model


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate(model, data_loader, device):
    """Evaluate the model on validation data."""
    
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    try:
        auc_roc = roc_auc_score(true_labels, probabilities)
    except:
        auc_roc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }


def train_model(model, train_loader, val_loader, config):
    """Complete training loop with validation."""
    
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, eps=1e-8)
    
    # Setup scheduler
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"✓ Total training steps: {total_steps:,}")
    print(f"✓ Warmup steps: {warmup_steps:,}")
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(config.EPOCHS):
        print(f"\n{'─' * 60}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print('─' * 60)
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, config.DEVICE
        )
        
        # Validation
        val_metrics = evaluate(model, val_loader, config.DEVICE)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"Precision:  {val_metrics['precision']:.4f} | Recall:    {val_metrics['recall']:.4f}")
        print(f"F1 Score:   {val_metrics['f1']:.4f} | AUC-ROC:   {val_metrics['auc_roc']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            print(f"★ New best model! F1: {best_val_f1:.4f}")
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_f1


# ============================================================================
# Save Model
# ============================================================================

def save_model(model, tokenizer, config):
    """Save the trained model and tokenizer."""
    
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    
    # Create directory if it doesn't exist
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    # Save model
    model.save_pretrained(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
    
    print(f"✓ Model saved to: {config.MODEL_SAVE_PATH}/")
    print(f"  - config.json")
    print(f"  - model.safetensors")
    print(f"  - tokenizer files")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "█" * 60)
    print("  TOXIC COMMENT CLASSIFIER - Transfer Learning")
    print("  Using DistilBERT Pre-trained Model")
    print("█" * 60)
    
    # Initialize config
    config = Config()
    
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data(config)
    
    # Setup model and tokenizer
    tokenizer, model = setup_model(config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, X_val, y_train, y_val, tokenizer, config
    )
    
    # Train model
    model, best_f1 = train_model(model, train_loader, val_loader, config)
    
    # Save model
    save_model(model, tokenizer, config)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"✓ Best Validation F1: {best_f1:.4f}")
    print(f"✓ Model saved to: {config.MODEL_SAVE_PATH}/")
    print(f"\nTo make predictions, run: python predict.py")


if __name__ == "__main__":
    main()
