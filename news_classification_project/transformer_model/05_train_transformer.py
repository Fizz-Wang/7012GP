
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
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

# --- Training Configuration ---
TRAINING_PARAMS = {
    'model_name': 'bert-base-uncased',  # You can specify other pre-trained models
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'logging_steps': 100,  # Log and evaluate every 100 steps
    'save_steps': 1000,  # Save a model checkpoint every 1000 steps
}


def load_data():
    """
    Loads data from a CSV file.
    If the file doesn't exist, it creates a placeholder CSV to prevent errors.
    """
    data_path = Path('../news_dataset.csv')
    # Ensure the parent directory exists
    if not data_path.parent.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a dummy file if the target does not exist
    if not data_path.exists():
        print(f"Warning: Data file not found at {data_path}. Creating a placeholder.")
        dummy_df = pd.DataFrame({
            'text': ['This is a sample text for politics.', 'This is a sample for sports news.',
                     'Tech news is exciting.', 'Entertainment news is fun.'],
            'category': ['politics', 'sports', 'technology', 'entertainment']
        })
        dummy_df.to_csv(data_path, index=False)

    df = pd.read_csv(data_path)
    return df


def prepare_dataset(df, tokenizer):
    """
    Prepares the dataset for the transformer model.
    This includes splitting data, creating label mappings, and tokenizing.
    """
    # Ensure the required columns exist
    if 'category' not in df.columns or 'text' not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'category' columns.")

    # Split data into training and testing sets, stratifying by category
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['category']
    )

    # Get unique categories and create mappings
    # Sorting ensures that label-to-ID mapping is consistent across runs
    categories = sorted(df['category'].unique())
    num_labels = len(categories)
    label2id = {label: i for i, label in enumerate(categories)}
    id2label = {i: label for i, label in enumerate(categories)}

    # Add a 'labels' column to the dataframes for the Trainer
    train_df['labels'] = train_df['category'].map(label2id)
    test_df['labels'] = test_df['category'].map(label2id)

    def tokenize_function(examples):
        """Tokenization function to be applied to the dataset."""
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=TRAINING_PARAMS['max_length']
        )

    # Convert pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

    # Apply tokenization to the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    return train_dataset, test_dataset, num_labels, label2id, id2label


def compute_metrics(eval_pred):
    """
    Computes and returns evaluation metrics.
    """
    predictions, labels = eval_pred
    # Get the index of the highest probability for each prediction
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics, using 'weighted' average to account for label imbalance
    # set zero_division=0 to avoid warnings when a class has no predictions
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """
    Main function to run the training pipeline.
    """
    # Load data
    print("Loading data...")
    df = load_data()

    # Initialize tokenizer from the pre-trained model
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_PARAMS['model_name'])

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, test_dataset, num_labels, label2id, id2label = prepare_dataset(df, tokenizer)

    # Initialize model for sequence classification
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        TRAINING_PARAMS['model_name'],
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # --- Set up Training Arguments ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(f'../results/transformer_model')
    checkpoints_dir = output_base / f'checkpoints_{timestamp}'

    # These arguments are compatible with the latest transformers versions.
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=TRAINING_PARAMS['num_epochs'],
        per_device_train_batch_size=TRAINING_PARAMS['batch_size'],
        per_device_eval_batch_size=TRAINING_PARAMS['batch_size'],
        learning_rate=TRAINING_PARAMS['learning_rate'],
        weight_decay=TRAINING_PARAMS['weight_decay'],
        warmup_steps=TRAINING_PARAMS['warmup_steps'],
        logging_steps=TRAINING_PARAMS['logging_steps'],
        save_steps=TRAINING_PARAMS['save_steps'],

        # 【CRITICAL CHANGE】 Use 'eval_strategy' instead of the deprecated 'evaluation_strategy'
        eval_strategy="steps",  # Evaluate at each `logging_steps`
        save_strategy="steps",  # Save checkpoint at each `save_steps`

        # The 'eval_steps' argument is also deprecated. When not provided,
        # evaluation frequency defaults to 'logging_steps'. This code relies on that behavior for compatibility.

        load_best_model_at_end=True,  # Load the best model found during training at the end
        metric_for_best_model="f1",  # Use F1-score to determine the "best" model
        greater_is_better=True,  # A higher F1-score is better

        report_to="none",  # Disable integration with external services like W&B or MLflow
        dataloader_pin_memory = False

    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer  # Passing the tokenizer is a good practice
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the final best model
    print("Saving final best model...")
    model_save_path = output_base / f'model_{timestamp}'
    trainer.save_model(str(model_save_path))

    # Evaluate the final model on the test set and save results
    print("Evaluating final model on the test set...")
    final_metrics = trainer.evaluate(eval_dataset=test_dataset)

    results = {
        'training_parameters': TRAINING_PARAMS,
        'final_metrics': final_metrics,
        'timestamp': timestamp,
        'model_path': str(model_save_path)
    }

    results_path = output_base / f'results_{timestamp}.json'
    print(f"Saving results to {results_path}")
    results_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\nTraining completed!")
    print(f"Final Metrics: {final_metrics}")


if __name__ == "__main__":
    main()
