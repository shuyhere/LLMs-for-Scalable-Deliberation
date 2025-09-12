#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Any
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import mean_squared_error
from transformers import DataCollatorWithPadding
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:
    _scipy_spearmanr = None


class DirectRegressionModel(nn.Module):
    """
    Regression model that directly predicts 0-1 scale ratings.
    """
    
    def __init__(self, base_model_name: str, num_dimensions: int = 4):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Direct regression head - outputs values in 0-1 range after Sigmoid
        self.regression_head = nn.Linear(self.base_model.config.hidden_size, num_dimensions)
        
        # Use sigmoid to map to 0-1 range
        self.activation = nn.Sigmoid()
        
        # Expose config for Trainer compatibility
        self.config = self.base_model.config
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get base model outputs
        # Remove Trainer housekeeping args not accepted by base model
        kwargs.pop("num_items_in_batch", None)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get regression predictions
        logits = self.regression_head(pooled_output)  # [batch_size, 4]
        
        # Apply sigmoid to get 0-1 scale predictions
        predictions = self.activation(logits)
        
        loss = None
        if labels is not None:
            # Direct MSE loss between predictions and 0-1 scale labels
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels.float())
        
        return {
            "loss": loss,
            "logits": predictions,  # Direct 0-1 scale predictions
        }


class DirectRegressionTrainer(Trainer):
    """Custom trainer for direct 0-1 scale regression"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation for direct regression
        """
        labels = inputs.get("labels")
        sample_weights = inputs.pop("weights", None)
        outputs = model(**inputs)
        
        if "loss" in outputs and outputs["loss"] is not None:
            loss = outputs["loss"]
        else:
            # Compute MSE loss manually
            predictions = outputs.get("logits")
            if predictions is None:
                raise ValueError("Model outputs must contain 'logits'")
            
            if labels is None:
                raise ValueError("Labels are required for loss computation")
            
            # Direct loss computation - no normalization needed
            # Ensure model output shape matches label shape
            predictions = predictions.view(-1, 4)
            labels = labels.view(-1, 4)

            # Compute per-element MSE then aggregate with optional sample weights
            per_element = nn.functional.mse_loss(predictions, labels.float(), reduction="none")  # [B, 4]
            per_sample = per_element.mean(dim=1)  # [B]
            if sample_weights is not None:
                # Expect shape [B] or [B, 1]
                if sample_weights.dim() == 2 and sample_weights.size(1) == 1:
                    sample_weights = sample_weights.squeeze(1)
                per_sample = per_sample * sample_weights.to(per_sample.device)
            loss = per_sample.mean()
            
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune model for 4-dimensional rating regression (direct 0-1 scale)")
    
    # Model and data arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="FacebookAI/roberta-base",
        help="Pre-trained model name or path",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the training dataset (JSONL file)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Optional path to JSONL test file to evaluate and save predictions at the end",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the trained model",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,  # Slightly higher for direct regression
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Proportion of data for validation split",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    # Imbalance handling
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Use stratified split by binned average label",
    )
    parser.add_argument(
        "--stratify-bins",
        type=int,
        default=5,
        help="Number of bins for stratification based on average label",
    )
    parser.add_argument(
        "--stratify-on",
        type=str,
        default="avg",
        choices=["avg", "dim0", "dim1", "dim2", "dim3"],
        help="Which target to bin for stratification",
    )
    parser.add_argument(
        "--sample-weighting",
        action="store_true",
        help="Enable sample weighting by inverse bin frequency (train only)",
    )
    parser.add_argument(
        "--weighting-bins",
        type=int,
        default=5,
        help="Number of bins for computing sample weights (based on average label)",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="tags",
        choices=["tags", "sep"],
        help="Input format: 'tags' for <QUESTION>...</QUESTION> format, 'sep' for <sep> separated format",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for logging",
    )
    
    return parser.parse_args()


def compute_mse_corr_metrics(eval_pred):
    """Compute MSE, Pearson, Spearman (overall and per-dimension) on 0-1 scale."""
    predictions, labels = eval_pred
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Ensure values are within [0, 1]
    predictions = np.clip(predictions, 0.0, 1.0)
    labels = np.clip(labels, 0.0, 1.0)

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a = a.reshape(-1)
        b = b.reshape(-1)
        a_std = a.std()
        b_std = b.std()
        if a_std == 0 or b_std == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        a = a.reshape(-1)
        b = b.reshape(-1)
        if _scipy_spearmanr is not None:
            try:
                val, _ = _scipy_spearmanr(a, b)
                if np.isnan(val):
                    return 0.0
                return float(val)
            except Exception:
                pass
        # Fallback: rank via argsort twice (ties broken arbitrarily)
        a_ranks = np.argsort(np.argsort(a))
        b_ranks = np.argsort(np.argsort(b))
        a_std = a_ranks.std()
        b_std = b_ranks.std()
        if a_std == 0 or b_std == 0:
            return 0.0
        return float(np.corrcoef(a_ranks, b_ranks)[0, 1])

    # Overall metrics
    mse = mean_squared_error(labels, predictions)
    pearson = _pearson(labels, predictions)
    spearman = _spearman(labels, predictions)

    # Per-dimension metrics
    dim_metrics = {}
    dimension_names = ["perspective", "informativeness", "neutrality", "policy"]
    for i, dim_name in enumerate(dimension_names):
        dim_mse = mean_squared_error(labels[:, i], predictions[:, i])
        dim_pearson = _pearson(labels[:, i], predictions[:, i])
        dim_spearman = _spearman(labels[:, i], predictions[:, i])
        dim_metrics.update({
            f"{dim_name}_mse": dim_mse,
            f"{dim_name}_pearson": dim_pearson,
            f"{dim_name}_spearman": dim_spearman,
        })

    return {
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
        **dim_metrics,
    }


def print_regression_report(trainer, test_dataset=None):
    """Print detailed regression evaluation report (MSE, Pearson)."""
    print("\n" + "="*80)
    print("DIRECT 0-1 REGRESSION REPORT")
    print("="*80)

    # Evaluate on validation set
    if trainer.eval_dataset is not None:
        print("\n📊 VALIDATION SET RESULTS:")
        eval_results = trainer.evaluate()

        print("  Overall Metrics:")
        print(f"    MSE: {eval_results.get('eval_mse', 0):.6f}")
        print(f"    Pearson: {eval_results.get('eval_pearson', 0):.4f}")
        print(f"    Spearman: {eval_results.get('eval_spearman', 0):.4f}")

        print("\n  Per-Dimension Metrics:")
        dimensions = ["perspective", "informativeness", "neutrality", "policy"]
        for dim in dimensions:
            print(f"    {dim.capitalize()}:")
            print(f"      MSE: {eval_results.get(f'eval_{dim}_mse', 0):.6f}")
            print(f"      Pearson: {eval_results.get(f'eval_{dim}_pearson', 0):.4f}")
            print(f"      Spearman: {eval_results.get(f'eval_{dim}_spearman', 0):.4f}")

    # Evaluate on test set if provided
    if test_dataset is not None:
        print("\n📊 TEST SET RESULTS:")
        test_results = trainer.evaluate(test_dataset)

        print("  Overall Metrics:")
        print(f"    MSE: {test_results.get('eval_mse', 0):.6f}")
        print(f"    Pearson: {test_results.get('eval_pearson', 0):.4f}")
        print(f"    Spearman: {test_results.get('eval_spearman', 0):.4f}")

        print("\n  Per-Dimension Metrics:")
        dimensions = ["perspective", "informativeness", "neutrality", "policy"]
        for dim in dimensions:
            print(f"    {dim.capitalize()}:")
            print(f"      MSE: {test_results.get(f'eval_{dim}_mse', 0):.6f}")
            print(f"      Pearson: {test_results.get(f'eval_{dim}_pearson', 0):.4f}")
            print(f"      Spearman: {test_results.get(f'eval_{dim}_spearman', 0):.4f}")


def print_classification_report(trainer, test_dataset=None):
    """Print detailed classification-style evaluation report (Accuracy, F1)"""
    print("\n" + "="*80)
    print("DIRECT REGRESSION (ROUNDED) CLASSIFICATION REPORT")
    print("="*80)
    
    # Evaluate on validation set
    if trainer.eval_dataset is not None:
        print("\n📊 VALIDATION SET RESULTS:")
        eval_results = trainer.evaluate()
        
        print("  Overall Metrics:")
        print(f"    Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
        print(f"    F1-Macro: {eval_results.get('eval_f1_macro', 0):.4f}")
        print(f"    F1-Weighted: {eval_results.get('eval_f1_weighted', 0):.4f}")
        
        print("\n  Per-Dimension Metrics:")
        dimensions = ["perspective", "informativeness", "neutrality", "policy"]
        for dim in dimensions:
            print(f"    {dim.capitalize()}:")
            print(f"      Accuracy: {eval_results.get(f'eval_{dim}_accuracy', 0):.4f}")
            print(f"      F1-Macro: {eval_results.get(f'eval_{dim}_f1_macro', 0):.4f}")
            print(f"      F1-Weighted: {eval_results.get(f'eval_{dim}_f1_weighted', 0):.4f}")
    
    # Evaluate on test set if provided
    if test_dataset is not None:
        print("\n TEST SET RESULTS:")
        test_results = trainer.evaluate(test_dataset)
        
        print("  Overall Metrics:")
        print(f"    Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
        print(f"    F1-Macro: {test_results.get('eval_f1_macro', 0):.4f}")
        print(f"    F1-Weighted: {test_results.get('eval_f1_weighted', 0):.4f}")


def create_data_collator(tokenizer):
    """Create data collator for padding"""
    return DataCollatorWithPadding(tokenizer=tokenizer)


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Starting Direct 0-1 Scale Regression Training")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Add special tokens based on input format
    if args.input_format == "tags":
        special_tokens = {
            "additional_special_tokens": [
                "<QUESTION>", "</QUESTION>", 
                "<ANSWER>", "</ANSWER>", 
                "<SUMMARY>", "</SUMMARY>"
            ]
        }
    else:  # sep format
        special_tokens = {
            "additional_special_tokens": ["<sep>"]
        }
    
    tokenizer.add_special_tokens(special_tokens)
    print(f"Added {len(special_tokens['additional_special_tokens'])} special tokens")
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # Load dataset
    print(f"\n📊 Loading dataset from {args.dataset_path}...")
    dataset = load_dataset('json', data_files=args.dataset_path, split='train')
    print(f"Total samples: {len(dataset)}")
    
    # Display label distribution
    def _dist(dataset, field_name="rating_scores"):
        """Display distribution of rating scores"""
        valid_scores = [scores for scores in dataset[field_name] if scores and len(scores) == 4]
        if not valid_scores:
            print(f"No valid {field_name} found!")
            return
        
        scores_array = np.array(valid_scores)
        print(f"\n📈 {field_name.replace('_', ' ').title()} Distribution:")
        print(f"  Valid samples: {len(valid_scores)}")
        
        dimension_names = ["Perspective", "Informativeness", "Neutrality", "Policy"]
        for i, dim_name in enumerate(dimension_names):
            dim_scores = scores_array[:, i]
            print(f"  {dim_name}: mean={dim_scores.mean():.2f}, std={dim_scores.std():.2f}, range=[{dim_scores.min()}-{dim_scores.max()}]")
    
    _dist(dataset)
    
    # Helpers for binning labels
    def _score_value(scores, mode: str) -> float:
        # scores are original 1-5; convert to 0-1 before using
        if mode == "avg":
            val = float(np.mean(scores))
        elif mode.startswith("dim"):
            idx = int(mode[-1])
            val = float(scores[idx])
        else:
            val = float(np.mean(scores))
        return (val - 1.0) / 4.0

    def _bin_index(val: float, bins: int) -> int:
        v = min(1.0, max(0.0, float(val)))
        # Ensure rightmost edge falls into last bin
        idx = int(np.floor(v * bins))
        return min(bins - 1, max(0, idx))

    # Split dataset (optionally stratified)
    if args.stratify:
        def _add_strata(examples):
            strata = []
            for scores in examples["rating_scores"]:
                val01 = _score_value(scores, args.stratify_on)
                strata.append(_bin_index(val01, args.stratify_bins))
            return {"__strata__": strata}

        dataset_with_strata = dataset.map(_add_strata, batched=True)
        split = dataset_with_strata.train_test_split(
            test_size=args.eval_split,
            seed=args.seed,
            stratify_by_column="__strata__",
        )
        train_dataset = split["train"].remove_columns(["__strata__"])  
        eval_dataset = split["test"].remove_columns(["__strata__"])  
    else:
        def safe_split(dataset, test_size=0.1, seed=42):
            """Split dataset without stratification (for regression)"""
            dataset = dataset.shuffle(seed=seed)
            train_size = int(len(dataset) * (1 - test_size))
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, len(dataset)))
            return train_dataset, eval_dataset
        train_dataset, eval_dataset = safe_split(dataset, test_size=args.eval_split, seed=args.seed)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Process datasets
    print("\n🔄 Preprocessing datasets...")
    effective_max_len = min(args.max_length, getattr(tokenizer, "model_max_length", args.max_length))

    # Optional sample weighting on train set
    weight_map = None
    if args.sample_weighting:
        def _add_wbin(examples):
            wbin = []
            for scores in examples["rating_scores"]:
                val01 = _score_value(scores, "avg")
                wbin.append(_bin_index(val01, args.weighting_bins))
            return {"__wbin__": wbin}

        train_with_bins = train_dataset.map(_add_wbin, batched=True)
        bins_arr = np.array(train_with_bins["__wbin__"]) if len(train_with_bins) > 0 else np.array([])
        if bins_arr.size > 0:
            counts = np.bincount(bins_arr, minlength=args.weighting_bins).astype(np.float64)
            counts[counts == 0] = 1.0
            inv = 1.0 / counts
            # Normalize to mean 1
            inv = inv * (counts.sum() / (args.weighting_bins * counts.size))
            weight_map = {int(i): float(inv[i]) for i in range(args.weighting_bins)}
        else:
            weight_map = {int(i): 1.0 for i in range(args.weighting_bins)}

    def preprocess_train(examples):
        inputs = []
        labels = []
        weights = []
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            answer = examples['answer_text'][i]
            summary = examples['displayed_text'][i]
            rating_scores = examples['rating_scores'][i]

            if not rating_scores or len(rating_scores) != 4:
                continue
                
            if question.startswith('[Question] '):
                question = question[11:]

            # Format input based on chosen format
            if args.input_format == "tags":
                input_text = f"<QUESTION>{question}</QUESTION><ANSWER>{answer}</ANSWER><SUMMARY>{summary}</SUMMARY>"
            else:  # sep format
                input_text = f"{question}<sep>{answer}<sep>{summary}"
            
            inputs.append(input_text)
            # Scale labels from 1-5 to 0-1
            scaled = [(float(s) - 1.0) / 4.0 for s in rating_scores]
            scaled = [min(1.0, max(0.0, s)) for s in scaled]
            labels.append(scaled)

            if args.sample_weighting and weight_map is not None:
                val01 = _score_value(rating_scores, "avg")
                wbin = _bin_index(val01, args.weighting_bins)
                weights.append(weight_map.get(int(wbin), 1.0))

        tokenized = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=effective_max_len,
            return_tensors="pt"
        )
        tokenized['labels'] = torch.tensor(labels, dtype=torch.float)
        if args.sample_weighting and len(weights) == len(labels):
            tokenized['weights'] = torch.tensor(weights, dtype=torch.float)
        return tokenized

    def preprocess_eval(examples):
        inputs = []
        labels = []
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            answer = examples['answer_text'][i]
            summary = examples['displayed_text'][i]
            rating_scores = examples['rating_scores'][i]

            if not rating_scores or len(rating_scores) != 4:
                continue
                
            if question.startswith('[Question] '):
                question = question[11:]

            # Format input based on chosen format
            if args.input_format == "tags":
                input_text = f"<QUESTION>{question}</QUESTION><ANSWER>{answer}</ANSWER><SUMMARY>{summary}</SUMMARY>"
            else:  # sep format
                input_text = f"{question}<sep>{answer}<sep>{summary}"
            
            inputs.append(input_text)
            scaled = [(float(s) - 1.0) / 4.0 for s in rating_scores]
            scaled = [min(1.0, max(0.0, s)) for s in scaled]
            labels.append(scaled)
        
        tokenized = tokenizer(
            inputs,
            padding="max_length",
            truncation=True,
            max_length=effective_max_len,
            return_tensors="pt"
        )
        tokenized['labels'] = torch.tensor(labels, dtype=torch.float)
        return tokenized

    train_dataset = train_dataset.map(
        preprocess_train,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        preprocess_eval, 
        batched=True, 
        remove_columns=eval_dataset.column_names
    )
    
    print(f"Processed train samples: {len(train_dataset)}")
    print(f"Processed eval samples: {len(eval_dataset)}")
    
    # Create model
    print(f"\n🤖 Creating model...")
    model = DirectRegressionModel(args.model_name, num_dimensions=4)
    
    # Resize token embeddings to match tokenizer (model-agnostic)
    input_emb = model.base_model.get_input_embeddings()
    old_vocab = input_emb.weight.shape[0] if input_emb is not None else len(tokenizer)
    print(f"Resizing token embeddings: {old_vocab} -> {len(tokenizer)}")
    model.base_model.resize_token_embeddings(len(tokenizer))

    # Training arguments
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        
        # Training setup
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        
        # Learning rate and optimization
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        
        # Regularization
        label_smoothing_factor=0.0,  # Not applicable for regression
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=1,
        
        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        
        # Other settings
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        fp16=True,
        remove_unused_columns=False,
        push_to_hub=False,
        save_total_limit=3,
        report_to=["wandb"],
        run_name=args.run_name,
    )
    
    # Create data collator
    data_collator = create_data_collator(tokenizer)
    
    # Create trainer (no EarlyStoppingCallback)
    trainer = DirectRegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_mse_corr_metrics,
        data_collator=data_collator,
    )

    # Train model
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to {output_dir}/final_model...")
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    # Prepare optional external test set
    test_dataset_proc = None
    test_raw = None
    if args.test_file is not None and os.path.isfile(args.test_file):
        print("\n📦 Loading external test set for final evaluation and predictions...")
        test_raw = load_dataset('json', data_files=args.test_file, split='train')
        test_dataset_proc = test_raw.map(
            preprocess_eval,
            batched=True,
            remove_columns=test_raw.column_names
        )

    # Print evaluation report (and on test set if provided)
    print_regression_report(trainer, test_dataset=test_dataset_proc)

    # Save predictions on external test set if provided
    if test_dataset_proc is not None and test_raw is not None and len(test_dataset_proc) == len(test_raw):
        print("\n📝 Saving per-sample predictions on external test set...")
        preds_output = trainer.predict(test_dataset_proc)
        preds = preds_output.predictions  # expected shape [N, 4] in 0-1
        preds = np.clip(np.asarray(preds), 0.0, 1.0)

        out_path = output_dir / "test_predictions.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for i in range(len(test_raw)):
                src = test_raw[i]
                pred01 = preds[i].tolist()
                pred15_round = [int(round(x * 4.0) + 1) for x in pred01]
                row = {
                    "triplet_key": src.get("triplet_key"),
                    "question": src.get("question"),
                    "answer_text": src.get("answer_text"),
                    "displayed_text": src.get("displayed_text"),
                    "rating_scores": src.get("rating_scores"),
                    "pred_scores_01": pred01,
                    "pred_scores_1_5_rounded": pred15_round,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved predictions to {out_path}")
    
    print(f"\n✅ Training completed! Model saved to {output_dir}/final_model")
    print("🎯 This model directly predicts 0-1 scale ratings.")


if __name__ == "__main__":
    main()


