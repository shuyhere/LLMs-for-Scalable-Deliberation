#!/usr/bin/env python3
"""
MSE + Direct Ordering Loss Training for Multi-output Regression
Replaces pairwise ranking loss with direct ordering accuracy optimization.

KEY FEATURES:
1. Dual Loss Function:
   - MSE loss for absolute value accuracy
   - Direct ordering loss that converts ordering accuracy to differentiable loss
   
2. Uses BCE loss to directly optimize ordering probability
3. Maintains all features from sft_train_mse_ranking.py (CustomTrainer, etc.)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')


TARGET_KEYS = [
    "perspective_representation",
    "informativeness", 
    "neutrality_balance",
    "policy_approval",
]


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
    """Normalize score from [min_val, max_val] to [0, 1]"""
    return (score - min_val) / (max_val - min_val)


def denormalize_score(normalized: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
    """Denormalize score from [0, 1] back to [min_val, max_val]"""
    return normalized * (max_val - min_val) + min_val


class RankingOptimizedDataset(Dataset):
    """Dataset optimized for ranking loss training"""
    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, 
                 max_length: int, augment: bool = False):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        # Group samples by question+opinion for better pair creation
        self.groups = defaultdict(list)
        for idx, rec in enumerate(records):
            key = f"{rec.get('question', '')}|||{rec.get('comment', '')}"
            self.groups[key].append(idx)
        
        print(f"Dataset: {len(records)} samples, {len(self.groups)} unique question-opinion groups")

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        question = rec.get("question", "")
        comment = rec.get("comment", "")
        summary = rec.get("summary", "")

        sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]"
        text = f"Question: {question} {sep_token} Annotator opinion: {comment} {sep_token} Summary: {summary}"

        # Simple augmentation
        if self.augment and np.random.random() < 0.3:
            # Random template variation
            templates = [
                f"Q: {question} {sep_token} Opinion: {comment} {sep_token} Summary: {summary}",
                f"Topic: {question} {sep_token} Comment: {comment} {sep_token} Overview: {summary}",
            ]
            text = np.random.choice(templates)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        scores = rec.get("scores", {})
        labels = [normalize_score(float(scores.get(k, 0.0))) for k in TARGET_KEYS]
        
        # Add small noise during training
        if self.augment and np.random.random() < 0.1:
            noise = np.random.normal(0, 0.01, len(labels))
            labels = np.clip(np.array(labels) + noise, 0, 1).tolist()
        
        enc["labels"] = torch.tensor(labels, dtype=torch.float32)
        enc["group_key"] = f"{question}|||{comment}"  # For pair creation
        return enc


class DirectOrderingModel(nn.Module):
    """Model with MSE + Direct Ordering loss (replaces ranking loss)"""
    def __init__(self, base_model_name: str, num_dims: int = 4, 
                 dropout_rate: float = 0.2, hidden_dropout: float = 0.3,
                 mse_weight: float = 0.5, ordering_weight: float = 0.5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        
        # Moderate dropout to prevent overfitting
        if hasattr(self.encoder.config, 'hidden_dropout_prob'):
            self.encoder.config.hidden_dropout_prob = dropout_rate
        if hasattr(self.encoder.config, 'attention_probs_dropout_prob'):
            self.encoder.config.attention_probs_dropout_prob = 0.1
            
        hidden = self.encoder.config.hidden_size
        
        # Two-layer architecture
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.hidden1 = nn.Linear(hidden, hidden // 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation1 = nn.GELU()
        
        self.layer_norm2 = nn.LayerNorm(hidden // 2)
        self.hidden2 = nn.Linear(hidden // 2, hidden // 4)
        self.dropout2 = nn.Dropout(hidden_dropout)
        self.activation2 = nn.GELU()
        
        # Output head
        self.head = nn.Linear(hidden // 4, num_dims)
        
        self.num_dims = num_dims
        self.config = self.encoder.config
        
        # Loss weights (MSE + Direct Ordering)
        self.mse_weight = mse_weight
        self.ordering_weight = ordering_weight
        
        print(f"Model initialized with MSE weight: {mse_weight}, Ordering weight: {ordering_weight}")
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("group_key", None)  # Remove group_key from kwargs
        
        # Encoder
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Use CLS token
        pooled = out.last_hidden_state[:, 0, :]
        
        # First layer
        h1 = self.layer_norm1(pooled)
        h1 = self.activation1(self.hidden1(h1))
        h1 = self.dropout1(h1)
        
        # Second layer
        h2 = self.layer_norm2(h1)
        h2 = self.activation2(self.hidden2(h2))
        h2 = self.dropout2(h2)
        
        # Output with sigmoid for [0, 1] range
        logits = torch.sigmoid(self.head(h2))
        
        loss = None
        if labels is not None:
            batch_size = logits.shape[0]
            
            # 1. MSE Loss (for absolute value accuracy)
            mse_loss = F.mse_loss(logits, labels, reduction='mean')
            
            # 2. Direct Ordering Accuracy Loss
            ordering_losses = []
            
            # Create all pairs within batch
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    for dim in range(self.num_dims):
                        pred_i = logits[i, dim]
                        pred_j = logits[j, dim]
                        label_i = labels[i, dim]
                        label_j = labels[j, dim]
                        
                        # Compute ordering loss for all pairs
                        label_diff = label_i - label_j
                        pred_diff = pred_i - pred_j
                        
                        if abs(label_diff) > 0.01:  # Minimal threshold for numerical stability
                            # Convert ordering relationship to probability using sigmoid
                            # If label_i > label_j, target_prob approaches 1
                            # If label_i < label_j, target_prob approaches 0
                            # Scale factor controls sharpness of transition
                            scale_factor = 20.0
                            target_prob = torch.sigmoid(label_diff * scale_factor)
                            pred_prob = torch.sigmoid(pred_diff * scale_factor)
                            
                            # Binary cross-entropy for ordering accuracy
                            # This directly optimizes the same criterion as ordering accuracy metric
                            eps = 1e-7  # For numerical stability
                            ordering_loss = -target_prob * torch.log(pred_prob + eps) \
                                          - (1 - target_prob) * torch.log(1 - pred_prob + eps)
                            ordering_losses.append(ordering_loss)
            
            # Average ordering loss
            if ordering_losses:
                ordering_loss_mean = torch.stack(ordering_losses).mean()
            else:
                ordering_loss_mean = torch.tensor(0.0, device=logits.device)
            
            # Combined loss
            loss = self.mse_weight * mse_loss + self.ordering_weight * ordering_loss_mean
            
            # Log for monitoring
            if self.training:
                self.last_mse = mse_loss.item()
                self.last_ordering = ordering_loss_mean.item()
        
        return {"loss": loss, "logits": logits}


class ImprovedEarlyStoppingCallback(TrainerCallback):
    """Early stopping based on combined metrics (correlation + ordering accuracy)"""
    def __init__(self, patience: int = 10, min_delta: float = 0.0005,
                 corr_weight: float = 0.5, acc_weight: float = 0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.corr_weight = corr_weight
        self.acc_weight = acc_weight
        self.best_score = -float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Combined score from correlation and ordering accuracy
            current_corr = metrics.get('eval_corr_mean_spearman', 0.0)
            current_acc = metrics.get('eval_ordering_acc_overall', 0.0)
            
            # Weighted combination
            current_score = (self.corr_weight * current_corr + 
                           self.acc_weight * current_acc)
            
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.patience_counter = 0
                
                if 'model' in kwargs:
                    self.best_model_state = copy.deepcopy(kwargs['model'].state_dict())
                print(f"\n✅ New best score: {current_score:.4f} "
                      f"(Corr={current_corr:.4f}, Acc={current_acc:.4f})")
            else:
                self.patience_counter += 1
                print(f"\n⚠️ No improvement. Patience: {self.patience_counter}/{self.patience} "
                      f"(Score={current_score:.4f}, Corr={current_corr:.4f}, Acc={current_acc:.4f})")
                
                if self.patience_counter >= self.patience:
                    print("\n🛑 Early stopping triggered after sufficient training")
                    control.should_training_stop = True
                    
                    if self.best_model_state is not None and 'model' in kwargs:
                        kwargs['model'].load_state_dict(self.best_model_state)
                        print("✅ Restored best model weights")
        
        return control


class LossMonitorCallback(TrainerCallback):
    """Monitor individual loss components"""
    def __init__(self):
        self.loss_history = {'mse': [], 'ordering': []}
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is not None and hasattr(model, 'module'):
            base_model = model.module if hasattr(model, 'module') else model
            
            if hasattr(base_model, 'last_mse'):
                self.loss_history['mse'].append(base_model.last_mse)
                self.loss_history['ordering'].append(base_model.last_ordering)
                
                if len(self.loss_history['mse']) % 10 == 0:
                    avg_mse = np.mean(self.loss_history['mse'][-10:])
                    avg_ordering = np.mean(self.loss_history['ordering'][-10:])
                    print(f"\n📊 Loss components (last 10 steps): MSE={avg_mse:.4f}, Ordering={avg_ordering:.4f}")
        
        return control


def compute_metrics(eval_pred):
    """Compute metrics including correlation (for monitoring, not optimization)"""
    preds, labels = eval_pred
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    
    preds = np.clip(preds, 0.0, 1.0)
    
    # Denormalize
    preds_denorm = denormalize_score(preds)
    labels_denorm = denormalize_score(labels)
    
    metrics = {
        "mse_overall": float(((preds - labels) ** 2).mean()),
        "mae_overall": float(np.abs(preds_denorm - labels_denorm).mean()),
    }
    
    # Compute correlations for monitoring
    spearmans = []
    pearsons = []
    
    for i, key in enumerate(TARGET_KEYS):
        try:
            sp = spearmanr(labels_denorm[:, i], preds_denorm[:, i]).correlation
            if not np.isnan(sp):
                spearmans.append(sp)
        except:
            sp = 0.0
        
        try:
            pr = pearsonr(labels_denorm[:, i], preds_denorm[:, i])[0]
            if not np.isnan(pr):
                pearsons.append(pr)
        except:
            pr = 0.0
            
        metrics[f"corr_{key}_spearman"] = float(sp) if not np.isnan(sp) else 0.0
        metrics[f"corr_{key}_pearson"] = float(pr) if not np.isnan(pr) else 0.0
    
    metrics["corr_mean_spearman"] = float(np.mean(spearmans)) if spearmans else 0.0
    metrics["corr_mean_pearson"] = float(np.mean(pearsons)) if pearsons else 0.0
    
    # Add placeholder for ordering accuracy (will be computed in CustomTrainer)
    metrics["ordering_acc_overall"] = 0.0
    for key in TARGET_KEYS:
        metrics[f"ordering_acc_{key}"] = 0.0
    
    return metrics


def compute_ordering_accuracy(eval_dataset: Dataset, model, device='cuda') -> Dict[str, float]:
    """Compute ordering accuracy for evaluation"""
    model.eval()
    
    # Group samples by question + opinion
    question_opinion_groups = {}
    all_samples = []
    
    for i in range(len(eval_dataset)):
        sample = eval_dataset[i]
        rec = eval_dataset.records[i]
        key = f"{rec.get('question', '')}|||{rec.get('comment', '')}"
        
        if key not in question_opinion_groups:
            question_opinion_groups[key] = []
        question_opinion_groups[key].append((i, sample))
        all_samples.append((i, sample))
    
    # Find groups with multiple samples
    valid_groups = {k: v for k, v in question_opinion_groups.items() if len(v) >= 2}
    
    if not valid_groups:
        return {"ordering_acc_overall": 0.0}
    
    print(f"Found {len(valid_groups)} groups for ordering evaluation")
    
    # Get predictions
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for i, sample in all_samples:
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            labels = sample["labels"]
            
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask)
            if isinstance(labels, list):
                labels = torch.tensor(labels)
            
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs["logits"].cpu().numpy()[0]
            label = labels.numpy()
            
            all_preds.append(pred)
            all_labels.append(label)
            all_indices.append(i)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate ordering accuracy
    ordering_accs = {}
    
    for dim_idx, key in enumerate(TARGET_KEYS):
        correct = 0
        total = 0
        
        for samples in valid_groups.values():
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    idx1, _ = samples[i]
                    idx2, _ = samples[j]
                    
                    pred1_idx = all_indices.index(idx1)
                    pred2_idx = all_indices.index(idx2)
                    
                    pred1 = all_preds[pred1_idx, dim_idx]
                    pred2 = all_preds[pred2_idx, dim_idx]
                    label1 = all_labels[pred1_idx, dim_idx]
                    label2 = all_labels[pred2_idx, dim_idx]
                    
                    if label1 > label2:
                        if pred1 > pred2:
                            correct += 1
                    elif label1 < label2:
                        if pred1 < pred2:
                            correct += 1
                    else:
                        if abs(pred1 - pred2) < 0.01:
                            correct += 1
                    
                    total += 1
        
        if total > 0:
            acc = correct / total
            ordering_accs[f"ordering_acc_{key}"] = float(acc)
            print(f"  {key}: {acc:.4f}")
    
    overall_acc = np.mean(list(ordering_accs.values())) if ordering_accs else 0.0
    ordering_accs["ordering_acc_overall"] = float(overall_acc)
    print(f"  Overall: {overall_acc:.4f}")
    
    return ordering_accs


class CustomTrainer(Trainer):
    """Custom trainer that computes ordering accuracy during evaluation"""
    
    def __init__(self, eval_dataset_for_ordering=None, **kwargs):
        super().__init__(**kwargs)
        self.eval_dataset_for_ordering = eval_dataset_for_ordering
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Call parent evaluate method
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add ordering accuracy if eval_dataset is available
        eval_ds = self.eval_dataset_for_ordering if self.eval_dataset_for_ordering else eval_dataset
        if eval_ds is not None:
            try:
                device = next(self.model.parameters()).device
                ordering_results = compute_ordering_accuracy(eval_ds, self.model, device)
                
                # Add ordering accuracy to metrics with appropriate prefix
                for key, value in ordering_results.items():
                    metric_key = f"{metric_key_prefix}_{key}"
                    metrics[metric_key] = value
                    
                print(f"\n📊 Ordering Accuracy:")
                print(f"  Overall: {ordering_results.get('ordering_acc_overall', 0.0):.4f}")
                for key in TARGET_KEYS:
                    acc_key = f"ordering_acc_{key}"
                    if acc_key in ordering_results:
                        print(f"  {key}: {ordering_results[acc_key]:.4f}")
                
            except Exception as e:
                print(f"Error computing ordering accuracy: {e}")
                # Add zero values to maintain consistent metrics structure
                metrics[f"{metric_key_prefix}_ordering_acc_overall"] = 0.0
                for key in TARGET_KEYS:
                    metrics[f"{metric_key_prefix}_ordering_acc_{key}"] = 0.0
        
        return metrics


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "scores" in data and isinstance(data["scores"], dict):
                    out.append(data)
            except json.JSONDecodeError:
                continue
    return out


def split_train_eval(data: List[Dict[str, Any]], eval_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_eval = int(len(data) * eval_ratio)
    eval_idx = set(idx[:n_eval].tolist())
    train, eval_ = [], []
    for i, rec in enumerate(data):
        (eval_.append if i in eval_idx else train.append)(rec)
    return train, eval_


class RegressionDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features]
        # Remove group_key from features
        for f in features:
            f.pop("group_key", None)
        
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.stack(labels)
        return batch


def main():
    parser = argparse.ArgumentParser(description="MSE + Direct Ordering loss training")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--ordering-weight", type=float, default=0.5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llm_direct_ordering")
    parser.add_argument("--wandb-run-name", type=str, default="direct-ordering-deberta")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    data_path = Path(args.data)
    print(f"Loading data from {data_path}")
    data = read_jsonl(data_path)
    print(f"Loaded {len(data)} records")
    
    # Split data
    train_recs, eval_recs = split_train_eval(data, args.eval_ratio, args.seed)
    print(f"Train: {len(train_recs)}, Eval: {len(eval_recs)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_ds = RankingOptimizedDataset(
        train_recs, tokenizer, args.max_len, augment=True
    )
    eval_ds = RankingOptimizedDataset(
        eval_recs, tokenizer, args.max_len, augment=False
    )

    # Initialize model
    model = DirectOrderingModel(
        args.model,
        dropout_rate=args.dropout,
        hidden_dropout=args.dropout + 0.1,
        mse_weight=args.mse_weight,
        ordering_weight=args.ordering_weight
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    collator = RegressionDataCollator(tokenizer=tokenizer)

    # WANDB setup
    report_backends = ["none"]
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)
        report_backends = ["wandb"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        logging_steps=10,
        seed=args.seed,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps * 2,
        report_to=report_backends,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="corr_mean_spearman",
        greater_is_better=True,
        gradient_accumulation_steps=args.accumulation_steps,
        fp16=True,
        optim="adamw_torch",
        max_grad_norm=args.gradient_clip,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_safetensors=True,
    )

    # Initialize custom trainer with ordering accuracy
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        eval_dataset_for_ordering=eval_ds,  # Pass eval dataset for ordering accuracy
    )

    # Add callbacks
    trainer.add_callback(
        ImprovedEarlyStoppingCallback(
            patience=args.patience,
            min_delta=0.0005,
            corr_weight=0.5,  # Balance correlation and ordering accuracy
            acc_weight=0.5
        )
    )
    
    trainer.add_callback(LossMonitorCallback())

    print("\n" + "="*60)
    print("MSE + DIRECT ORDERING LOSS TRAINING")
    print("="*60)
    print(f"Using BCE loss to directly optimize ordering accuracy")
    print(f"MSE weight: {args.mse_weight}, Ordering weight: {args.ordering_weight}")
    print(f"Longer patience ({args.patience}) to prevent early stopping")
    print("="*60 + "\n")
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Final metrics
    metrics = trainer.evaluate()
    
    print("\n📊 Final Metrics:")
    print(f"  Mean Spearman: {metrics.get('eval_corr_mean_spearman', 0):.4f}")
    print(f"  Mean Pearson: {metrics.get('eval_corr_mean_pearson', 0):.4f}")
    print(f"  MAE: {metrics.get('eval_mae_overall', 0):.4f}")
    print(f"  Ordering Accuracy: {metrics.get('eval_ordering_acc_overall', 0):.4f}")
    
    # Save metrics
    metrics_path = Path(args.out) / "final_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()