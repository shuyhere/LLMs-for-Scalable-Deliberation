#!/usr/bin/env python3
"""
Binary classification training script for comparison data.
Predicts which summary is better for each of the 4 dimensions (perspective, informativeness, neutrality, policy).
Each dimension is treated as a separate binary classification task.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import random
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:
    _scipy_spearmanr = None


class ComparisonBinaryClassifier(nn.Module):
    """Binary classifier for comparison tasks - predicts which summary is better for each dimension."""

    def __init__(self, base_model_name: str, num_dimensions: int = 4, label_smoothing: float = 0.0):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.config = self.base_model.config
        self.num_dimensions = num_dimensions
        self.label_smoothing = label_smoothing
        
        # Freeze base model parameters initially
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Simple but effective classifier
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_dimensions)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Get logits for each dimension (binary classification)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Convert labels to binary format (1 -> 1, 2 -> 0)
            binary_labels = (labels == 1).float()
            
            # Simple BCE loss without label smoothing for now
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, binary_labels)
        
        return {
            "loss": loss,
            "logits": logits,
        }


def hard_negative_sampling(dataset: Dataset, hard_ratio: float = 0.3) -> Dataset:
    """Implement hard negative sampling for better training."""
    # Group examples by similarity (simplified approach)
    # In practice, you might want to use more sophisticated similarity metrics
    
    examples = []
    for i in range(len(dataset)):
        example = dataset[i]
        examples.append({
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'labels': example['labels']
        })
    
    # Simple hard negative sampling: select examples with more balanced labels
    hard_examples = []
    easy_examples = []
    
    for example in examples:
        labels = example['labels']
        # Check if labels are balanced (not all 0s or all 1s)
        if 0.3 <= np.mean(labels) <= 0.7:  # More balanced labels are "harder"
            hard_examples.append(example)
        else:
            easy_examples.append(example)
    
    # Sample hard examples more frequently
    num_hard = int(len(examples) * hard_ratio)
    num_easy = len(examples) - num_hard
    
    if len(hard_examples) > num_hard:
        selected_hard = random.sample(hard_examples, num_hard)
    else:
        selected_hard = hard_examples
    
    if len(easy_examples) > num_easy:
        selected_easy = random.sample(easy_examples, num_easy)
    else:
        selected_easy = easy_examples
    
    # Combine and shuffle
    final_examples = selected_hard + selected_easy
    random.shuffle(final_examples)
    
    # Convert back to dataset format
    return Dataset.from_list(final_examples)


def augment_dataset(dataset: Dataset, augment_ratio: float = 0.2) -> Dataset:
    """Add data augmentation to increase dataset diversity."""
    augmented_examples = []
    
    for i in range(len(dataset)):
        example = dataset[i]
        augmented_examples.append(example)
        
        # Randomly augment some examples
        if random.random() < augment_ratio:
            # Simple augmentation: randomly mask some tokens
            input_ids = example['input_ids'].copy()
            attention_mask = example['attention_mask'].copy()
            
            # Randomly mask 5% of non-special tokens
            seq_len = len(input_ids)
            mask_indices = random.sample(
                range(1, seq_len - 1),  # Skip first and last tokens
                max(1, int(seq_len * 0.05))
            )
            
            # Use tokenizer's mask token (or a random token)
            for idx in mask_indices:
                if attention_mask[idx] == 1:  # Only mask attended tokens
                    input_ids[idx] = 103  # [MASK] token for BERT-like models
            
            augmented_examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': example['labels']
            })
    
    return Dataset.from_list(augmented_examples)


def load_comparison_dataset(dataset_path: str, use_hard_sampling: bool = True, use_augmentation: bool = True) -> Dataset:
    """Load and preprocess the comparison dataset."""
    
    # Load the dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Analyze and balance the dataset
    print("Analyzing dataset balance...")
    examples_a_better = []
    examples_b_better = []
    
    for i, example in enumerate(dataset):
        scores = example["metadata"]["comparison_scores"]
        # Count how many dimensions favor A vs B
        a_count = sum(1 for s in scores if s == 1.0)
        b_count = sum(1 for s in scores if s == 2.0)
        
        if a_count > b_count:
            examples_a_better.append(example)
        elif b_count > a_count:
            examples_b_better.append(example)
    
    print(f"Examples favoring A: {len(examples_a_better)}")
    print(f"Examples favoring B: {len(examples_b_better)}")
    
    # Balance the dataset
    min_size = min(len(examples_a_better), len(examples_b_better))
    balanced_examples = examples_a_better[:min_size] + examples_b_better[:min_size]
    random.shuffle(balanced_examples)
    
    print(f"Balanced dataset size: {len(balanced_examples)}")
    
    # Extract labels
    def extract_labels(example):
        scores = example["metadata"]["comparison_scores"]
        # Convert to binary labels (1 = Summary A better, 2 = Summary B better)
        # For binary classification: 1 -> 1, 2 -> 0
        binary_labels = [1.0 if score == 1.0 else 0.0 for score in scores]
        return {"labels": binary_labels}
    
    # Create balanced dataset
    balanced_dataset = Dataset.from_list(balanced_examples)
    dataset_with_labels = balanced_dataset.map(extract_labels)
    
    # Tokenize the prompts
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
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
    
    # Apply data augmentation if requested
    if use_augmentation:
        print("Applying data augmentation...")
        tokenized_dataset = augment_dataset(tokenized_dataset, augment_ratio=0.2)
    
    # Apply hard negative sampling if requested
    if use_hard_sampling:
        print("Applying hard negative sampling...")
        tokenized_dataset = hard_negative_sampling(tokenized_dataset, hard_ratio=0.3)
    
    return tokenized_dataset


def compute_metrics(eval_pred):
    """Compute metrics for binary classification."""
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(torch.tensor(predictions)).numpy()
    pred_labels = (probs > 0.5).astype(int)
    
    # Flatten for overall metrics
    pred_labels_flat = pred_labels.flatten()
    labels_flat = labels.flatten()
    
    # Overall metrics
    accuracy = accuracy_score(labels_flat, pred_labels_flat)
    f1 = f1_score(labels_flat, pred_labels_flat, average='weighted')
    
    # Per-dimension metrics
    dimension_names = ["perspective", "informativeness", "neutrality", "policy"]
    dimension_metrics = {}
    
    for i, dim_name in enumerate(dimension_names):
        dim_pred = pred_labels[:, i]
        dim_labels = labels[:, i]
        dim_accuracy = accuracy_score(dim_labels, dim_pred)
        dim_f1 = f1_score(dim_labels, dim_pred, average='weighted', zero_division=0)
        dimension_metrics[f"{dim_name}_accuracy"] = dim_accuracy
        dimension_metrics[f"{dim_name}_f1"] = dim_f1
    
    # Calculate correlations if scipy is available
    correlations = {}
    if _scipy_spearmanr is not None:
        try:
            # Convert binary predictions back to original scale (0->2, 1->1)
            pred_original = 2 - pred_labels_flat
            labels_original = 2 - labels_flat
            spearman_corr, _ = _scipy_spearmanr(pred_original, labels_original)
            correlations["spearman_correlation"] = spearman_corr if not np.isnan(spearman_corr) else 0.0
        except Exception:
            correlations["spearman_correlation"] = 0.0
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        **dimension_metrics,
        **correlations
    }


def run_cross_validation(dataset: Dataset, model_name: str, output_base_dir: str, 
                        cv_folds: int = 5, **training_args) -> Dict[str, float]:
    """Run k-fold cross validation."""
    print(f"Running {cv_folds}-fold cross validation...")
    
    # Create labels for stratification (use the first dimension as primary label)
    labels_for_cv = []
    for i in range(len(dataset)):
        labels_for_cv.append(int(dataset[i]['labels'][0]))  # Use first dimension for stratification
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels_for_cv)):
        print(f"\n=== Fold {fold + 1}/{cv_folds} ===")
        
        # Create fold datasets
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)
        
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        # Create fold output directory
        fold_output_dir = os.path.join(output_base_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Initialize model for this fold
        model = ComparisonBinaryClassifier(model_name)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Training arguments for this fold
        fold_training_args = TrainingArguments(
            output_dir=fold_output_dir,
            **training_args,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=fold_training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        cv_results.append(eval_results)
        
        # Save fold results
        with open(os.path.join(fold_output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Fold {fold + 1} results: {eval_results}")
    
    # Calculate average results
    avg_results = {}
    for key in cv_results[0].keys():
        if key.startswith('eval_'):
            values = [result[key] for result in cv_results]
            avg_results[key.replace('eval_', '')] = np.mean(values)
            avg_results[f"{key.replace('eval_', '')}_std"] = np.std(values)
    
    # Save cross-validation summary
    cv_summary = {
        "fold_results": cv_results,
        "average_results": avg_results
    }
    
    with open(os.path.join(output_base_dir, "cv_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\n=== Cross Validation Summary ===")
    for key, value in avg_results.items():
        if not key.endswith('_std'):
            std_key = f"{key}_std"
            std_value = avg_results.get(std_key, 0)
            print(f"{key}: {value:.4f} ± {std_value:.4f}")
    
    return avg_results


def main():
    parser = argparse.ArgumentParser(description="Train binary classifier for comparison data")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the comparison dataset JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for the trained model")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                       help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=10,
                       help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging steps")
    parser.add_argument("--seed", type=int, default=6666,
                       help="Random seed")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16")
    parser.add_argument("--bf16", action="store_true",
                       help="Use BF16")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="Maximum number of checkpoints to save")
    
    # New arguments for improved training
    parser.add_argument("--cross_validation", action="store_true",
                       help="Run cross validation instead of single train/test split")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="Number of folds for cross validation")
    parser.add_argument("--use_hard_sampling", action="store_true", default=True,
                       help="Use hard negative sampling")
    parser.add_argument("--use_augmentation", action="store_true", default=True,
                       help="Use data augmentation")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                       help="Label smoothing factor")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                       help="Gradient clipping value")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=["linear", "cosine", "constant"],
                       help="Learning rate scheduler type")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_comparison_dataset(
        args.dataset_path, 
        use_hard_sampling=args.use_hard_sampling,
        use_augmentation=args.use_augmentation
    )
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Prepare training arguments
    training_args = {
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "eval_strategy": args.eval_strategy,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "save_total_limit": args.save_total_limit,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.gradient_clip_val,
        "fp16": True,  # Use mixed precision
        "report_to": None,
        "remove_unused_columns": False,
    }
    
    # Run cross validation or single training
    if args.cross_validation:
        print("Running cross validation...")
        cv_results = run_cross_validation(
            dataset, 
            args.model_name, 
            args.output_dir,
            args.cv_folds,
            **training_args
        )
        
        print("Cross validation completed!")
        print("Final results:", cv_results)
        
    else:
        # Single train/test split
        print("Running single train/test split...")
        train_test_split = dataset.train_test_split(test_size=0.2, seed=args.seed)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize model with label smoothing
        model = ComparisonBinaryClassifier(args.model_name, label_smoothing=args.label_smoothing)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Training arguments
        training_args_full = TrainingArguments(
            output_dir=args.output_dir,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            **training_args
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args_full,
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
