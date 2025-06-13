import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# Training parameters
TRAINING_PARAMS = {
    'model_name': 'bert-base-uncased',  # or other pre-trained models
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'logging_steps': 100,
    'eval_steps': 100,
    'save_steps': 1000,
}

def load_data():
    """Load and prepare data"""
    df = pd.read_csv('../news_dataset.csv')
    return df

def prepare_dataset(df, tokenizer):
    """Prepare dataset for transformer model"""
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['category']
    )
    
    # Get unique categories
    categories = df['category'].unique()
    num_labels = len(categories)
    
    # Create label to id mapping
    label2id = {label: i for i, label in enumerate(categories)}
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=TRAINING_PARAMS['max_length']
        )
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    return train_dataset, test_dataset, num_labels, label2id

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_PARAMS['model_name'])
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, test_dataset, num_labels, label2id = prepare_dataset(df, tokenizer)
    
    # Initialize model
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        TRAINING_PARAMS['model_name'],
        num_labels=num_labels
    )
    
    # Set up training arguments
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_args = TrainingArguments(
        output_dir=f'../results/transformer_model/checkpoints_{timestamp}',
        num_train_epochs=TRAINING_PARAMS['num_epochs'],
        per_device_train_batch_size=TRAINING_PARAMS['batch_size'],
        per_device_eval_batch_size=TRAINING_PARAMS['batch_size'],
        learning_rate=TRAINING_PARAMS['learning_rate'],
        weight_decay=TRAINING_PARAMS['weight_decay'],
        warmup_steps=TRAINING_PARAMS['warmup_steps'],
        logging_steps=TRAINING_PARAMS['logging_steps'],
        eval_steps=TRAINING_PARAMS['eval_steps'],
        save_steps=TRAINING_PARAMS['save_steps'],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    model_save_path = f'../results/transformer_model/model_{timestamp}'
    trainer.save_model(model_save_path)
    
    # Save training parameters and results
    results = {
        'parameters': TRAINING_PARAMS,
        'metrics': trainer.evaluate(),
        'timestamp': timestamp
    }
    
    with open(f'../results/transformer_model/results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
