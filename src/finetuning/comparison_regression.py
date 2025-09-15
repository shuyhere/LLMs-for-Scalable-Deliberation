#!/usr/bin/env python3
"""
Regression training script for comparison data.
Predicts comparison scores (1 or 2) for each of the 4 dimensions (perspective, informativeness, neutrality, policy).
Treats each dimension as a separate regression task with values 1.0 or 2.0.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from scipy.stats import spearmanr as _scipy_spearmanr, pearsonr as _scipy_pearsonr
except Exception:
    _scipy_spearmanr = None
    _scipy_pearsonr = None


class ComparisonRegressionModel(nn.Module):
    """Regression model for comparison tasks - predicts comparison scores for each dimension."""

    def __init__(self, base_model_name: str, num_dimensions: int = 4):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.config = self.base_model.config
        self.dropout = nn.Dropout(0.1)
        self.num_dimensions = num_dimensions
        
        # Regression head for each dimension
        self.regression_head = nn.Linear(self.base_model.config.hidden_size, num_dimensions)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each dimension
        logits = self.regression_head(pooled_output)
        
        loss = None
        if labels is not None:
            # Use MSE loss for regression
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels.float())
        
        return {
            "loss": loss,
            "logits": logits,
        }


def load_comparison_dataset(dataset_path: str) -> Dataset:
    """Load and preprocess the comparison dataset."""
    
    # Load the dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Extract labels first before tokenization
    def extract_labels(example):
        scores = example["metadata"]["comparison_scores"]
        # Keep original scores as regression targets (1.0 or 2.0)
        regression_labels = [float(score) for score in scores]
        return {"labels": regression_labels}
    
    # Add labels to dataset
    dataset_with_labels = dataset.map(extract_labels)
    
    # Tokenize the prompts
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            truncation=True,
            padding=False,
            max_length=4096,
            return_tensors=None  # Don't return tensors during mapping
        )
    
    # Tokenize the dataset and remove unnecessary columns
    tokenized_dataset = dataset_with_labels.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "completion", "metadata"]
    )
    
    return tokenized_dataset


def compute_metrics(eval_pred):
    """Compute metrics for regression."""
    predictions, labels = eval_pred
    
    # Calculate overall metrics
    mse = mean_squared_error(labels.flatten(), predictions.flatten())
    mae = mean_absolute_error(labels.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    
    # Per-dimension metrics
    dimension_names = ["perspective", "informativeness", "neutrality", "policy"]
    dimension_metrics = {}
    
    for i, dim_name in enumerate(dimension_names):
        dim_pred = predictions[:, i]
        dim_labels = labels[:, i]
        dim_mse = mean_squared_error(dim_labels, dim_pred)
        dim_mae = mean_absolute_error(dim_labels, dim_pred)
        dim_rmse = np.sqrt(dim_mse)
        
        dimension_metrics[f"{dim_name}_mse"] = dim_mse
        dimension_metrics[f"{dim_name}_mae"] = dim_mae
        dimension_metrics[f"{dim_name}_rmse"] = dim_rmse
    
    # Calculate correlations if scipy is available
    correlations = {}
    if _scipy_spearmanr is not None and _scipy_pearsonr is not None:
        try:
            # Overall correlations
            spearman_corr, _ = _scipy_spearmanr(predictions.flatten(), labels.flatten())
            pearson_corr, _ = _scipy_pearsonr(predictions.flatten(), labels.flatten())
            
            correlations["spearman_correlation"] = spearman_corr if not np.isnan(spearman_corr) else 0.0
            correlations["pearson_correlation"] = pearson_corr if not np.isnan(pearson_corr) else 0.0
            
            # Per-dimension correlations
            for i, dim_name in enumerate(dimension_names):
                dim_pred = predictions[:, i]
                dim_labels = labels[:, i]
                
                spearman_corr, _ = _scipy_spearmanr(dim_pred, dim_labels)
                pearson_corr, _ = _scipy_pearsonr(dim_pred, dim_labels)
                
                dimension_metrics[f"{dim_name}_spearman"] = spearman_corr if not np.isnan(spearman_corr) else 0.0
                dimension_metrics[f"{dim_name}_pearson"] = pearson_corr if not np.isnan(pearson_corr) else 0.0
                
        except Exception as e:
            print(f"Warning: Could not calculate correlations: {e}")
            correlations["spearman_correlation"] = 0.0
            correlations["pearson_correlation"] = 0.0
    
    # Calculate accuracy (treating as binary classification)
    # Convert predictions to binary (closest to 1 or 2)
    binary_pred = (predictions > 1.5).astype(int) + 1
    binary_labels = labels.astype(int)
    
    # Overall accuracy
    accuracy = np.mean(binary_pred.flatten() == binary_labels.flatten())
    dimension_metrics["accuracy"] = accuracy
    
    # Per-dimension accuracy
    for i, dim_name in enumerate(dimension_names):
        dim_accuracy = np.mean(binary_pred[:, i] == binary_labels[:, i])
        dimension_metrics[f"{dim_name}_accuracy"] = dim_accuracy
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        **dimension_metrics,
        **correlations
    }


def main():
    parser = argparse.ArgumentParser(description="Train regression model for comparison data")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the comparison dataset JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for the trained model")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                       help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Logging steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16")
    parser.add_argument("--bf16", action="store_true",
                       help="Use BF16")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="Maximum number of checkpoints to save")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_comparison_dataset(args.dataset_path)
    
    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = ComparisonRegressionModel(args.model_name)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        save_total_limit=args.save_total_limit,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="spearman_correlation",
        greater_is_better=True,
        report_to=None,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()
