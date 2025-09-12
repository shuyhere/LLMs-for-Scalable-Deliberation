#!/usr/bin/env python3
"""
Multiclass training script (1..5 per dimension) with CLI aligned to regression script.
Adds Pearson and Spearman correlations (overall and per-dimension) using expected value
over class probabilities mapped to 0..1 for comparability with regression.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any
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
from transformers.models.deberta_v2 import DebertaV2Tokenizer
from sklearn.metrics import accuracy_score, f1_score

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:
    _scipy_spearmanr = None


class MultiClassRatingModel(nn.Module):
    """Base encoder with a classification head producing [batch, 4, 5] logits."""

    def __init__(self, base_model_name: str, num_dimensions: int = 4, num_classes: int = 5):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.config = self.base_model.config
        self.dropout = nn.Dropout(0.1)
        self.num_dimensions = num_dimensions
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_dimensions * num_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits_flat = self.classifier(cls)  # [batch, 4*5]
        logits = logits_flat.view(-1, self.num_dimensions, self.num_classes)  # [batch, 4, 5]

        loss = None
        if labels is not None:
            # labels: shape [batch, 4] with class indices in 0..4
            loss_fct = nn.CrossEntropyLoss()
            per_dim_loss = []
            for d in range(self.num_dimensions):
                per_dim_loss.append(loss_fct(logits[:, d, :], labels[:, d]))
            loss = torch.stack(per_dim_loss).mean()

        return {"loss": loss, "logits": logits}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 4-dim multiclass model with regression-style CLI")
    # Aligned with regression script
    p.add_argument("--dataset-path", type=str, required=True, help="Path to JSONL train file")
    p.add_argument("--test-file", type=str, default=None, help="Optional path to JSONL test file to evaluate after training")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.add_argument("--model-name", type=str, default="allenai/longformer-base-4096")
    p.add_argument("--batch-size", type=int, default=16, help="Train batch size")
    p.add_argument("--eval-batch-size", type=int, default=16, help="Eval batch size")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--eval-split", type=float, default=0.1, help="Validation split from train")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-format", type=str, default="tags", choices=["tags", "sep"], 
                   help="Input format: 'tags' for <QUESTION>...</QUESTION> format, 'sep' for <sep> separated format")
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    p.add_argument("--run_name", type=str, default=None, help="Run name for logging")
    return p.parse_args()


def compute_metrics(eval_pred) -> Dict[str, Any]:
    logits, labels = eval_pred
    logits = np.asarray(logits)  # [B, 4, 5]
    labels = np.asarray(labels)  # [B, 4]
    preds = logits.argmax(axis=2)  # [B, 4] for accuracy/F1

    overall_acc = accuracy_score(labels.flatten(), preds.flatten())
    f1_macro = f1_score(labels.flatten(), preds.flatten(), average="macro")
    f1_weighted = f1_score(labels.flatten(), preds.flatten(), average="weighted")

    # Correlations using Expected Value over class probabilities (0..4)
    # This avoids variance collapse when argmax is constant within a dimension.
    # probs shape [B, 4, 5]
    # ev shape [B, 4]
    with np.errstate(over='ignore'):
        # Stabilize softmax by subtracting max per dim
        max_logits = logits.max(axis=2, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.clip(exp_logits.sum(axis=2, keepdims=True), 1e-12, None)
    class_values = np.arange(5, dtype=np.float64).reshape(1, 1, 5)
    ev = (probs * class_values).sum(axis=2)  # [B, 4]

    true_scores = labels.astype(np.float64)
    pred_scores = ev.astype(np.float64)

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
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        ar_std = ar.std()
        br_std = br.std()
        if ar_std == 0 or br_std == 0:
            return 0.0
        return float(np.corrcoef(ar, br)[0, 1])

    pearson_all = _pearson(true_scores, pred_scores)
    spearman_all = _spearman(true_scores, pred_scores)

    dims = ["perspective", "informativeness", "neutrality", "policy"]
    out = {
        "accuracy": overall_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "pearson": pearson_all,
        "spearman": spearman_all,
    }
    for i, name in enumerate(dims):
        out[f"acc_{name}"] = accuracy_score(labels[:, i], preds[:, i])
        out[f"f1_{name}"] = f1_score(labels[:, i], preds[:, i], average="macro")
        out[f"pearson_{name}"] = _pearson(true_scores[:, i], pred_scores[:, i])
        out[f"spearman_{name}"] = _spearman(true_scores[:, i], pred_scores[:, i])
    return out


def print_multiclass_report(trainer):
    """Pretty-print consolidated metrics for validation set (overall + per-dimension)."""
    print("\n" + "="*80)
    print("MULTICLASS 1..5 REPORT (with Pearson/Spearman via EV on 0-1)")
    print("="*80)

    results = trainer.evaluate()
    print("\n📊 VALIDATION SET RESULTS:")
    print("  Overall Metrics:")
    print(f"    Accuracy: {results.get('eval_accuracy', 0):.4f}")
    print(f"    F1-Macro: {results.get('eval_f1_macro', 0):.4f}")
    print(f"    F1-Weighted: {results.get('eval_f1_weighted', 0):.4f}")
    print(f"    Pearson: {results.get('eval_pearson', 0):.4f}")
    print(f"    Spearman: {results.get('eval_spearman', 0):.4f}")

    print("\n  Per-Dimension Metrics:")
    dims = ["perspective", "informativeness", "neutrality", "policy"]
    for name in dims:
        print(f"    {name.capitalize()}:")
        print(f"      Accuracy: {results.get(f'eval_acc_{name}', 0):.4f}")
        print(f"      F1-Macro: {results.get(f'eval_f1_{name}', 0):.4f}")
        print(f"      Pearson: {results.get(f'eval_pearson_{name}', 0):.4f}")
        print(f"      Spearman: {results.get(f'eval_spearman_{name}', 0):.4f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.dataset_path).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset: Dataset = load_dataset("json", data_files=str(data_path), split="train")

    # Train/validation split similar to regression script
    def safe_split(ds: Dataset, test_size: float, seed: int):
        ds = ds.shuffle(seed=seed)
        train_size = int(len(ds) * (1 - test_size))
        train_ds = ds.select(range(train_size))
        eval_ds = ds.select(range(train_size, len(ds)))
        return train_ds, eval_ds

    train_ds, eval_ds = safe_split(dataset, test_size=args.eval_split, seed=args.seed)

    # Tokenizer - use specific tokenizer class for DeBERTa models to avoid conversion issues
    if "deberta" in args.model_name.lower():
        print(f"Using DebertaV2Tokenizer for DeBERTa model: {args.model_name}")
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        except Exception as e:
            print(f"Fast tokenizer failed for {args.model_name}, falling back to slow tokenizer: {e}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a pad token if none exists
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Add special tokens based on input format
    if args.input_format == "tags":
        special_tokens = {"additional_special_tokens": ["<QUESTION>", "</QUESTION>", "<ANSWER>", "</ANSWER>", "<SUMMARY>", "</SUMMARY>"]}
    else:  # sep format
        special_tokens = {"additional_special_tokens": ["<sep>"]}
    
    tokenizer.add_special_tokens(special_tokens)

    effective_max_len = min(args.max_length, getattr(tokenizer, "model_max_length", args.max_length))

    def preprocess(batch):
        texts = []
        labels = []
        for i in range(len(batch.get("question", []))):
            q = batch["question"][i] or ""
            a = batch["answer_text"][i] or ""
            s = batch["displayed_text"][i] or ""
            rs = batch["rating_scores"][i]
            if not rs or len(rs) != 4 or any(v is None for v in rs):
                continue
            if isinstance(q, str) and q.startswith("[Question] "):
                q = q[11:]
            
            # Format input based on chosen format
            if args.input_format == "tags":
                input_text = f"<QUESTION>{q}</QUESTION><ANSWER>{a}</ANSWER><SUMMARY>{s}</SUMMARY>"
            else:  # sep format
                input_text = f"{q}<sep>{a}<sep>{s}"
            
            texts.append(input_text)
            labels.append([int(float(v)) - 1 for v in rs])
        if not texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        enc = tokenizer(texts, padding=False, truncation=True, max_length=effective_max_len, return_token_type_ids=False)
        enc["labels"] = labels
        return enc

    remove_cols = [c for c in train_ds.column_names if c not in ("question", "answer_text", "displayed_text", "rating_scores")]
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=remove_cols)
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=remove_cols)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Loading model: {args.model_name}")
    model = MultiClassRatingModel(args.model_name, num_dimensions=4, num_classes=5)
    
    # Resize token embeddings if we added special tokens
    if len(tokenizer) != model.base_model.config.vocab_size:
        print(f"Resizing token embeddings from {model.base_model.config.vocab_size} to {len(tokenizer)}")
        model.base_model.resize_token_embeddings(len(tokenizer))

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=str(out_dir / "logs"),
        logging_strategy="steps",
        logging_steps=1,
        save_total_limit=3,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Starting training (no early stopping)...")
    trainer.train()

    # Consolidated validation report (mirrors regression style)
    print_multiclass_report(trainer)

    print("Saving final model...")
    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved to {final_dir}")

    # Optional: evaluate on provided external test set
    if args.test_file is not None and os.path.isfile(args.test_file):
        print("\n" + "="*80)
        print("EVALUATING ON EXTERNAL TEST SET")
        print("="*80)
        test_ds_raw: Dataset = load_dataset("json", data_files=str(Path(args.test_file).resolve()), split="train")
        remove_cols_test = [c for c in test_ds_raw.column_names if c not in ("question", "answer_text", "displayed_text", "rating_scores")]
        test_ds = test_ds_raw.map(preprocess, batched=True, remove_columns=remove_cols_test)
        test_results = trainer.evaluate(eval_dataset=test_ds)
        print("\n📊 TEST SET RESULTS:")
        print(f"  Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
        print(f"  F1-Macro: {test_results.get('eval_f1_macro', 0):.4f}")
        print(f"  F1-Weighted: {test_results.get('eval_f1_weighted', 0):.4f}")
        print(f"  Pearson: {test_results.get('eval_pearson', 0):.4f}")
        print(f"  Spearman: {test_results.get('eval_spearman', 0):.4f}")
        dims = ["perspective", "informativeness", "neutrality", "policy"]
        for name in dims:
            print(f"  {name.capitalize()}:")
            print(f"    Accuracy: {test_results.get(f'eval_acc_{name}', 0):.4f}")
            print(f"    F1-Macro: {test_results.get(f'eval_f1_{name}', 0):.4f}")
            print(f"    Pearson: {test_results.get(f'eval_pearson_{name}', 0):.4f}")
            print(f"    Spearman: {test_results.get(f'eval_spearman_{name}', 0):.4f}")


if __name__ == "__main__":
    main()


