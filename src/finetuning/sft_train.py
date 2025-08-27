#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, Any
import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 5-class RoBERTa classifier on constructed.jsonl")
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to constructed.jsonl (with fields: input, output, metadata)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="FacebookAI/roberta-base",
        help="HF model checkpoint to fine-tune (e.g., roberta-base)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints/roberta-5class-eva",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512, 
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=64,
        help="Train batch size per device",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=64,
        help="Eval batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10, 
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6, 
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
        default=0.1,  # Add warmup
        help="Warmup ratio",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Proportion of data for validation split",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Proportion of data for test split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights for imbalanced data",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
    )
    return parser.parse_args()


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            # Move class weights to the same device as logits
            device_weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=device_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> Dict[str, Any]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Basic metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    
    # Per-class F1 scores
    f1_per_class = f1_score(labels, preds, average=None)
    per_class_metrics = {f"f1_class_{i}": f1_per_class[i] for i in range(len(f1_per_class))}
    
    return {
        "accuracy": acc, 
        "f1_macro": f1_macro, 
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision, 
        "recall_macro": recall,
        **per_class_metrics
    }


def print_classification_report(trainer, dataset, split_name="test"):
    """Print detailed classification report"""
    predictions = trainer.predict(dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    
    print(f"\n=== {split_name.upper()} SET CLASSIFICATION REPORT ===")
    print(classification_report(
        y_true, y_pred, 
        target_names=[f"Class {i+1}" for i in range(5)],
        digits=4
    ))


def main() -> None:
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    data_path = Path(args.data_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset from JSONL
    print("Loading dataset...")
    dataset_all = load_dataset("json", data_files=str(data_path))

    # Report overall label distribution before splitting
    def _dist(ds, name):
        try:
            vc = {}
            for v in ds["output"]:
                vc[str(v)] = vc.get(str(v), 0) + 1
            total = len(ds)
            print(f"[{name}] size={total} label distribution:")
            for k in sorted(vc.keys()):
                print(f"  Class {k}: {vc[k]} ({vc[k]/total:.2%})")
        except Exception as e:
            print(f"Error computing distribution for {name}: {e}")
    
    _dist(dataset_all["train"], "all")

    # Split into train/val/test with stratification
    test_size = args.test_split
    eval_size = args.eval_split / (1.0 - test_size) if args.eval_split > 0 else 0.0

    def safe_split(ds, test_size, seed, label_col="output"):
        try:
            return ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column=label_col)
        except Exception as e:
            print(f"Stratified split failed ({e}); falling back to random split.")
            return ds.train_test_split(test_size=test_size, seed=seed)

    print("Splitting dataset...")
    split1 = safe_split(dataset_all["train"], test_size=test_size, seed=args.seed)
    split2 = safe_split(split1["train"], test_size=eval_size, seed=args.seed)

    dataset = DatasetDict({
        "train": split2["train"],
        "validation": split2["test"],
        "test": split1["test"],
    })

    # Shuffle training data
    dataset["train"] = dataset["train"].shuffle(seed=args.seed)

    # Report per-split label distributions
    _dist(dataset["train"], "train")
    _dist(dataset["validation"], "validation")
    _dist(dataset["test"], "test")

    # Prepare labels mapping
    id2label = {i: str(i + 1) for i in range(5)}
    label2id = {v: k for k, v in id2label.items()}

    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        print("Computing class weights...")
        train_labels = [label2id[str(x)] for x in dataset["train"]["output"]]
        class_weights_np = compute_class_weight(
            'balanced', 
            classes=np.arange(5), 
            y=train_labels
        )
        class_weights = torch.FloatTensor(class_weights_np)
        print(f"Class weights: {class_weights}")

    # Tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use proper max length
    effective_max_len = min(args.max_length, 512)
    print(f"Using max_length: {effective_max_len}")

    def preprocess(batch):
        labels = [label2id[str(x)] for x in batch["output"]]
        enc = tokenizer(
            batch["input"],
            padding=False,  # We'll pad in the data collator
            truncation=True,
            max_length=effective_max_len,
            return_token_type_ids=False,
        )
        enc["labels"] = labels
        return enc

    print("Preprocessing dataset...")
    dataset = dataset.map(
        preprocess, 
        batched=True, 
        remove_columns=[c for c in dataset["train"].column_names if c not in ("input", "output", "metadata")]
    )

    # Model configuration
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=5,
        id2label=id2label,
        label2id=label2id,
        # hidden_dropout_prob=0.1,  # Add dropout
        # attention_probs_dropout_prob=0.1,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Training arguments
    total_steps = len(dataset["train"]) * args.num_epochs // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=100,
        
        # Learning rate and optimization
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        
        # Regularization
        label_smoothing_factor=args.label_smoothing,
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=25,  # More frequent logging
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=3,
        
        # Other settings
        seed=args.seed,
        data_seed=args.seed,
        report_to=["wandb"],
        dataloader_num_workers=4,
        # fp16=False,  # Disabled mixed precision as requested
        max_grad_norm=1.0,
    )

    # Data collator for dynamic padding
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize trainer
    if args.use_class_weights:
        trainer_class = WeightedTrainer
        trainer_kwargs = {"class_weights": class_weights}
    else:
        trainer_class = Trainer
        trainer_kwargs = {}

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        **trainer_kwargs
    )

    # Train
    print("Starting training...")
    trainer.train()
    
    # Final evaluation
    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate()
    print(f"Validation metrics: {val_metrics}")
    
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(dataset["test"])
    print(f"Test metrics: {test_metrics}")
    
    # Print detailed classification reports
    print_classification_report(trainer, dataset["validation"], "validation")
    print_classification_report(trainer, dataset["test"], "test")
    
    # Save test metrics
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        import json as _json
        _json.dump(test_metrics, f, indent=2)
    
    # Save final model
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    
    print(f"Training completed! Models and logs saved to {output_dir}")


if __name__ == "__main__":
    main()