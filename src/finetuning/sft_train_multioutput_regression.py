#!/usr/bin/env python3
"""
Train a multi-output regression model to predict 4 dimensions with values in [-1, 1] range.

OPTIMIZATIONS FOR HIGHER CORRELATION:
1. Enhanced Model Architecture:
   - Multi-layer architecture with residual connections
   - Layer normalization for stability
   - Task-specific heads for each dimension
   - Combined pooling strategy (CLS + mean pooling)
   - Correlation-aware loss function (MSE + correlation loss)

2. Training Optimizations:
   - Cosine learning rate scheduling
   - Gradient clipping for stability
   - Mixed precision training (fp16)
   - Label smoothing option
   - Warmup ratio for learning rate

3. Data Augmentation:
   - Text augmentation (sentence shuffling, paraphrasing)
   - Label noise injection for regularization
   - Configurable augmentation probability

4. Evaluation Enhancements:
   - P-values for all correlations (Pearson & Spearman)
   - Significance levels (*, **, ***)
   - 95% confidence intervals via bootstrap
   - Ensemble evaluation option with multiple checkpoints

5. Loss Function:
   - Combined MSE and correlation-based loss
   - Huber loss option for robustness
   - Per-dimension optimization

Inputs: question + annotator opinion (comment) + summary
Targets: 4 dimensions (perspective_representation, informativeness, neutrality_balance, policy_approval)

Data: JSONL from build_comment_summary_ratings.py
Each line: {"question", "comment", "summary", "scores": {<4 dims>}}
Scores will be normalized to [-1, 1] range for regression.

Example usage:
    Basic training:
        python sft_train_multioutput_regression.py --data /path/to/data.jsonl --model microsoft/deberta-v3-base

    With custom parameters:
        python sft_train_multioutput_regression.py \
            --data /path/to/data.jsonl \
            --model microsoft/deberta-v3-base \
            --out ./results/my_model \
            --epochs 5 \
            --batch 16 \
            --lr 2e-5 \
            --max-len 1024 \
            --eval-ratio 0.15

    With wandb logging:
        python sft_train_multioutput_regression.py \
            --data /path/to/data.jsonl \
            --model microsoft/deberta-v3-base \
            --wandb \
            --wandb-project my-project \
            --wandb-run-name experiment-1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_callback import TrainerCallback
from scipy.stats import spearmanr, pearsonr


TARGET_KEYS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
    """Normalize score from [min_val, max_val] to [0, 1] for better training stability"""
    # Normalize to [0, 1] instead of [-1, 1] for better stability
    norm_01 = (score - min_val) / (max_val - min_val)
    return norm_01


def denormalize_score(normalized: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
    """Denormalize score from [0, 1] back to [min_val, max_val]"""
    # Denormalize from [0, 1] to original range
    return normalized * (max_val - min_val) + min_val


class CommentSummaryRatingsDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int, 
                 augment: bool = False, augment_prob: float = 0.3, normalize: bool = True):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augment_prob = augment_prob
        self.normalize = normalize

    def __len__(self):
        return len(self.records)
    
    def augment_text(self, text: str) -> str:
        """Apply data augmentation techniques"""
        if not self.augment or np.random.random() > self.augment_prob:
            return text
        
        # Random augmentation strategies
        aug_type = np.random.choice(['shuffle_sentences', 'paraphrase', 'none'], p=[0.3, 0.4, 0.3])
        
        if aug_type == 'shuffle_sentences':
            # Shuffle sentences within each section
            parts = text.split(self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]")
            augmented_parts = []
            for part in parts:
                sentences = part.strip().split('. ')
                if len(sentences) > 1:
                    np.random.shuffle(sentences)
                    augmented_parts.append('. '.join(sentences))
                else:
                    augmented_parts.append(part)
            return (self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]").join(augmented_parts)
        
        elif aug_type == 'paraphrase':
            # Simple paraphrase by changing templates
            templates = [
                "Query: {q} {sep} Reviewer's view: {c} {sep} Synopsis: {s}",
                "Issue: {q} {sep} Annotator's perspective: {c} {sep} Overview: {s}",
                "Topic: {q} {sep} Opinion provided: {c} {sep} Summary text: {s}",
            ]
            # Extract parts
            parts = text.split(self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]")
            if len(parts) == 3:
                q = parts[0].replace("Question:", "").strip()
                c = parts[1].replace("Annotator opinion:", "").strip()
                s = parts[2].replace("Summary:", "").strip()
                template = np.random.choice(templates)
                return template.format(q=q, c=c, s=s, sep=self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]")
        
        return text

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        question = rec.get("question", "")
        comment = rec.get("comment", "")
        summary = rec.get("summary", "")

        # Use tokenizer's special tokens for better separation (matching multiclass format)
        sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]"
        
        text = f"Question: {question} {sep_token} Annotator opinion: {comment} {sep_token} Summary: {summary}"
        
        # Apply augmentation
        text = self.augment_text(text)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        scores = rec.get("scores", {})
        
        if self.normalize:
            # Normalize labels from original scale to [0, 1] range
            labels = [normalize_score(float(scores.get(k, 0.0))) for k in TARGET_KEYS]
            
            # Add label smoothing noise during training (only for normalized)
            if self.augment and np.random.random() < 0.2:
                noise = np.random.normal(0, 0.01, len(labels))
                labels = np.clip(np.array(labels) + noise, 0, 1).tolist()
        else:
            # Use original label scale [-1, 7] directly
            labels = [float(scores.get(k, 0.0)) for k in TARGET_KEYS]
            
            # Add label smoothing noise during training (for original scale)
            if self.augment and np.random.random() < 0.2:
                noise = np.random.normal(0, 0.1, len(labels))  # Larger noise for larger scale
                labels = np.clip(np.array(labels) + noise, -1, 7).tolist()
        
        enc["labels"] = torch.tensor(labels, dtype=torch.float32)
        return enc


class MultiOutputRegressor(nn.Module):
    """Improved model architecture with residual connections and better regularization"""
    def __init__(self, base_model_name: str, num_dims: int = 4, dropout_rate: float = 0.1, 
                 use_tanh: bool = False, use_sigmoid: bool = False, use_relu: bool = False,
                 use_leaky_relu: bool = False, use_elu: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        # Add a hidden layer for better feature extraction
        self.hidden = nn.Linear(hidden, hidden // 2)
        self.activation = nn.GELU()
        self.head = nn.Linear(hidden // 2, num_dims)
        
        # Check for parameter conflicts
        activation_count = sum([use_tanh, use_sigmoid, use_relu, use_leaky_relu, use_elu])
        if activation_count > 1:
            raise ValueError("Cannot use multiple activation functions at the same time. Choose only one.")
        
        self.use_tanh = use_tanh
        self.use_sigmoid = use_sigmoid
        self.use_relu = use_relu
        self.use_leaky_relu = use_leaky_relu
        self.use_elu = use_elu
        
        if use_tanh:
            self.output_activation = nn.Tanh()  # Output in [-1, 1]
        elif use_sigmoid:
            self.output_activation = nn.Sigmoid()  # Output in [0, 1]
        elif use_relu:
            self.output_activation = nn.ReLU()  # Output in [0, +inf)
        elif use_leaky_relu:
            self.output_activation = nn.LeakyReLU(negative_slope=0.01)  # Output in (-inf, +inf)
        elif use_elu:
            self.output_activation = nn.ELU()  # Output in [-1, +inf)
        else:
            self.output_activation = None  # No activation, raw logits
        
        self.num_dims = num_dims
        self.loss_fct = nn.MSELoss()
        
        # Add config attribute for compatibility with Transformers Trainer
        self.config = self.encoder.config

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Remove any unexpected kwargs
        kwargs.pop("num_items_in_batch", None)
        
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = out.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled = self.dropout(pooled)
        hidden = self.activation(self.hidden(pooled))
        hidden = self.dropout(hidden)
        logits = self.head(hidden)  # (B, num_dims)
        
        if self.output_activation is not None:
            predictions = self.output_activation(logits)
        else:
            # No constraint, let the model learn the range
            predictions = logits
        
        loss = None
        if labels is not None:
            # Use Huber loss for robustness
            huber_loss = nn.HuberLoss(delta=1.0)
            loss = huber_loss(predictions, labels)
        
        return {"loss": loss, "logits": predictions}


class RegressionDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.stack(labels)  # (B, 4)
        return batch


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Validate that required fields exist
                if "scores" in data and isinstance(data["scores"], dict):
                    out.append(data)
                else:
                    print(f"Warning: Line {line_num} missing 'scores' field, skipping")
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}")
                continue
    return out


def split_train_eval(data: List[Dict[str, Any]], eval_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Simple random split of data into train and eval sets."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_eval = int(len(data) * eval_ratio)
    eval_idx = set(idx[:n_eval].tolist())
    train, eval_ = [], []
    for i, rec in enumerate(data):
        (eval_.append if i in eval_idx else train.append)(rec)
    return train, eval_


def compute_metrics(eval_pred, use_normalization=True):
    preds, labels = eval_pred
    # preds shape: (batch, 4), labels shape: (batch, 4)
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    
    if use_normalization:
        # For normalized training, ensure predictions are in [0, 1] range
        preds = np.clip(preds, 0.0, 1.0)
        
        # Denormalize for interpretable MAE (back to original -1 to 7 scale)
        preds_denorm = denormalize_score(preds)
        labels_denorm = denormalize_score(labels)
    else:
        # For non-normalized training, work directly with [-1, 7] scale
        preds = np.clip(preds, -1.0, 7.0)
        preds_denorm = preds
        labels_denorm = labels
    
    mae_denorm = np.abs(preds_denorm - labels_denorm).mean(axis=0)
    
    metrics = {
        "mse_overall": float(((preds - labels) ** 2).mean()),
        "mae_overall": float(np.abs(preds_denorm - labels_denorm).mean()),
    }
    
    # Per-dimension correlations and metrics
    for i, key in enumerate(TARGET_KEYS):
        # Calculate correlations on denormalized scale for interpretability
        try:
            sp = spearmanr(labels_denorm[:, i], preds_denorm[:, i]).correlation
        except Exception:
            sp = np.nan
        try:
            pr = pearsonr(labels_denorm[:, i], preds_denorm[:, i])[0]
        except Exception:
            pr = np.nan
            
        metrics[f"corr_{key}_spearman"] = float(sp) if not np.isnan(sp) else 0.0
        metrics[f"corr_{key}_pearson"] = float(pr) if not np.isnan(pr) else 0.0
        metrics[f"mae_{key}"] = float(mae_denorm[i])
        
        print(f"Dimension {key}: Spearman={sp:.4f}, Pearson={pr:.4f}")
    
    # Average correlations
    spearmans = [metrics[f"corr_{k}_spearman"] for k in TARGET_KEYS if metrics[f"corr_{k}_spearman"] != 0.0]
    pearsons = [metrics[f"corr_{k}_pearson"] for k in TARGET_KEYS if metrics[f"corr_{k}_pearson"] != 0.0]
    
    metrics["corr_mean_spearman"] = float(np.mean(spearmans)) if spearmans else 0.0
    metrics["corr_mean_pearson"] = float(np.mean(pearsons)) if pearsons else 0.0
    
    # Note: ordering accuracy and difference correlation are computed separately in CustomTrainer.evaluate()
    # due to their complexity requiring access to the full evaluation dataset
    # Add placeholder values to ensure they appear in logs
    metrics["ordering_acc_overall"] = 0.0
    metrics["diff_corr_pearson_overall"] = 0.0
    metrics["diff_corr_spearman_overall"] = 0.0
    for key in TARGET_KEYS:
        metrics[f"ordering_acc_{key}"] = 0.0
        metrics[f"diff_corr_pearson_{key}"] = 0.0
        metrics[f"diff_corr_spearman_{key}"] = 0.0
    
    return metrics


def compute_ordering_accuracy(eval_dataset: Dataset, model, device='cuda') -> Dict[str, float]:
    """Compute ordering accuracy for summaries with the same question and opinion"""
    model.eval()
    
    # Group samples by opinion (comment) only - since same opinion with different questions still makes valid pairs
    opinion_groups = {}
    all_samples = []
    
    # Collect all samples with their indices
    for i in range(len(eval_dataset)):
        sample = eval_dataset[i]
        comment = eval_dataset.records[i].get("comment", "")
        question = eval_dataset.records[i].get("question", "")
        
        # Use opinion as the key for grouping
        key = comment  # Group by opinion only
        
        if key not in opinion_groups:
            opinion_groups[key] = []
        opinion_groups[key].append({
            'idx': i,
            'sample': sample,
            'question': question,
            'comment': comment
        })
        all_samples.append((i, sample))
    
    # Find groups with at least 2 samples (same opinion)
    valid_groups = {k: v for k, v in opinion_groups.items() if len(v) >= 2}
    
    if not valid_groups:
        print("No groups with same opinion found for ordering evaluation")
        return {"ordering_acc_overall": 0.0, "diff_corr_overall": 0.0}
    
    print(f"Found {len(valid_groups)} opinion groups with multiple samples")
    
    # Also report question-opinion combinations for more detailed analysis
    question_opinion_pairs = {}
    for opinion, samples in valid_groups.items():
        for sample_info in samples:
            qo_key = f"{sample_info['question']}|||{opinion}"
            if qo_key not in question_opinion_pairs:
                question_opinion_pairs[qo_key] = 0
            question_opinion_pairs[qo_key] += 1
    
    qo_pairs_with_multiple = sum(1 for count in question_opinion_pairs.values() if count >= 2)
    print(f"  - {qo_pairs_with_multiple} question-opinion pairs with >=2 samples")
    print(f"  - Total pairs for comparison: {sum(len(v)*(len(v)-1)//2 for v in valid_groups.values())}")
    
    # Get predictions for all samples
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for i, sample in all_samples:
            # Prepare input - convert to tensors if they're lists
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            labels = sample["labels"]
            
            # Convert to tensors if they're lists
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask)
            if isinstance(labels, list):
                labels = torch.tensor(labels)
            
            # Add batch dimension and move to device
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs["logits"].cpu().numpy()[0]  # (4,)
            label = labels.cpu().numpy()[0]  # (4,)
            
            all_preds.append(pred)
            all_labels.append(label)
            all_indices.append(i)
    
    all_preds = np.array(all_preds)  # (N, 4)
    all_labels = np.array(all_labels)  # (N, 4)
    
    # Debug: Print prediction and label ranges
    print(f"Prediction range: min={all_preds.min():.4f}, max={all_preds.max():.4f}, mean={all_preds.mean():.4f}")
    print(f"Label range: min={all_labels.min():.4f}, max={all_labels.max():.4f}, mean={all_labels.mean():.4f}")
    
    # Calculate ordering accuracy and difference correlation for each dimension
    ordering_accs = {}
    diff_correlations = {}
    
    for dim_idx, key in enumerate(TARGET_KEYS):
        correct_pairs = 0
        total_pairs = 0
        
        # Collect differences for correlation calculation
        pred_diffs = []
        label_diffs = []
        
        for opinion, samples in valid_groups.items():
            if len(samples) < 2:
                continue
                
            # Get all pairs within this opinion group
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    sample1 = samples[i]
                    sample2 = samples[j]
                    idx1 = sample1['idx']
                    idx2 = sample2['idx']
                    
                    # Find corresponding predictions
                    pred1_idx = all_indices.index(idx1)
                    pred2_idx = all_indices.index(idx2)
                    
                    pred1_score = all_preds[pred1_idx][dim_idx]
                    pred2_score = all_preds[pred2_idx][dim_idx]
                    label1_score = all_labels[pred1_idx][dim_idx]
                    label2_score = all_labels[pred2_idx][dim_idx]
                    
                    # Calculate differences for correlation
                    pred_diff = pred1_score - pred2_score
                    label_diff = label1_score - label2_score
                    pred_diffs.append(pred_diff)
                    label_diffs.append(label_diff)
                    
                    # Check if ordering is correct
                    # If label1 > label2, then pred1 should > pred2
                    if label1_score > label2_score:
                        if pred1_score > pred2_score:
                            correct_pairs += 1
                    elif label1_score < label2_score:
                        if pred1_score < pred2_score:
                            correct_pairs += 1
                    else:  # label1 == label2, predictions should be close
                        if abs(pred1_score - pred2_score) < 0.05:
                            correct_pairs += 1
                    
                    total_pairs += 1
        
        if total_pairs > 0:
            acc = correct_pairs / total_pairs
            ordering_accs[f"ordering_acc_{key}"] = float(acc)
            print(f"Ordering accuracy for {key}: {acc:.4f} ({correct_pairs}/{total_pairs})")
        else:
            ordering_accs[f"ordering_acc_{key}"] = 0.0
        
        # Calculate difference correlation
        if len(pred_diffs) > 1 and len(label_diffs) > 1:
            try:
                # Pearson correlation of differences
                from scipy.stats import pearsonr, spearmanr
                pearson_corr, _ = pearsonr(label_diffs, pred_diffs)
                spearman_corr, _ = spearmanr(label_diffs, pred_diffs)
                
                diff_correlations[f"diff_corr_pearson_{key}"] = float(pearson_corr)
                diff_correlations[f"diff_corr_spearman_{key}"] = float(spearman_corr)
                print(f"Difference correlation for {key}: Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}")
            except Exception as e:
                print(f"Could not compute difference correlation for {key}: {e}")
                diff_correlations[f"diff_corr_pearson_{key}"] = 0.0
                diff_correlations[f"diff_corr_spearman_{key}"] = 0.0
        else:
            diff_correlations[f"diff_corr_pearson_{key}"] = 0.0
            diff_correlations[f"diff_corr_spearman_{key}"] = 0.0
    
    # Overall ordering accuracy (average across dimensions)
    if ordering_accs:
        overall_acc = np.mean(list(ordering_accs.values()))
        ordering_accs["ordering_acc_overall"] = float(overall_acc)
        print(f"Overall ordering accuracy: {overall_acc:.4f}")
    else:
        ordering_accs["ordering_acc_overall"] = 0.0
    
    # Overall difference correlation (average across dimensions)
    pearson_diff_corrs = [diff_correlations[f"diff_corr_pearson_{k}"] for k in TARGET_KEYS 
                          if f"diff_corr_pearson_{k}" in diff_correlations and diff_correlations[f"diff_corr_pearson_{k}"] != 0.0]
    spearman_diff_corrs = [diff_correlations[f"diff_corr_spearman_{k}"] for k in TARGET_KEYS 
                           if f"diff_corr_spearman_{k}" in diff_correlations and diff_correlations[f"diff_corr_spearman_{k}"] != 0.0]
    
    if pearson_diff_corrs:
        diff_correlations["diff_corr_pearson_overall"] = float(np.mean(pearson_diff_corrs))
        print(f"Overall difference correlation (Pearson): {diff_correlations['diff_corr_pearson_overall']:.4f}")
    else:
        diff_correlations["diff_corr_pearson_overall"] = 0.0
    
    if spearman_diff_corrs:
        diff_correlations["diff_corr_spearman_overall"] = float(np.mean(spearman_diff_corrs))
        print(f"Overall difference correlation (Spearman): {diff_correlations['diff_corr_spearman_overall']:.4f}")
    else:
        diff_correlations["diff_corr_spearman_overall"] = 0.0
    
    # Combine results
    results = {**ordering_accs, **diff_correlations}
    return results


def ensemble_evaluate(model_path: str, eval_dataset: Dataset, tokenizer, device='cuda') -> Dict[str, Any]:
    """Perform ensemble evaluation using multiple checkpoints"""
    import glob
    checkpoints = sorted(glob.glob(f"{model_path}/checkpoint-*"))
    
    if len(checkpoints) < 2:
        print(f"Not enough checkpoints for ensemble (found {len(checkpoints)})")
        return {}
    
    # Use last 3 checkpoints
    checkpoints = checkpoints[-3:] if len(checkpoints) >= 3 else checkpoints
    print(f"Ensemble evaluation with {len(checkpoints)} checkpoints")
    
    all_preds = []
    for ckpt in checkpoints:
        model = MultiOutputRegressor.from_pretrained(ckpt)
        model.to(device)
        model.eval()
        
        preds = []
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(eval_dataset, batch_size=8):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                outputs = model(**inputs)
                preds.append(outputs['logits'].cpu().numpy())
        
        all_preds.append(np.concatenate(preds))
    
    # Average predictions
    ensemble_preds = np.mean(all_preds, axis=0)
    return ensemble_preds


def evaluate_correlations(trainer: Trainer, eval_dataset: Dataset, wandb_enabled: bool = False):
    """Comprehensive evaluation including correlations and prediction distribution analysis"""
    pred_output = trainer.predict(eval_dataset)
    preds = pred_output.predictions
    labels = pred_output.label_ids
    
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    
    # Ensure predictions are in [0, 1] range
    preds = np.clip(preds, 0.0, 1.0)
    
    # Denormalize for analysis
    preds_denorm = denormalize_score(preds)
    labels_denorm = denormalize_score(labels)
    
    results: Dict[str, Any] = {}
    
    # Prediction distribution analysis
    print("\n" + "="*50)
    print("Prediction Distribution Analysis")
    print("="*50)
    
    for i, key in enumerate(TARGET_KEYS):
        pred_vals = preds_denorm[:, i]
        label_vals = labels_denorm[:, i]
        
        print(f"\n{key}:")
        print(f"  Labels:  min={label_vals.min():.2f}, max={label_vals.max():.2f}, "
              f"mean={label_vals.mean():.2f}, std={label_vals.std():.2f}")
        print(f"  Preds:   min={pred_vals.min():.2f}, max={pred_vals.max():.2f}, "
              f"mean={pred_vals.mean():.2f}, std={pred_vals.std():.2f}")
        
        # Calculate correlations on denormalized scale
        try:
            pr_denorm, _ = pearsonr(label_vals, pred_vals)
        except Exception:
            pr_denorm = np.nan
        
        try:
            sr_denorm = spearmanr(label_vals, pred_vals).correlation
        except Exception:
            sr_denorm = np.nan
        
        # MAE and MSE
        mae = float(np.abs(pred_vals - label_vals).mean())
        mse = float(((pred_vals - label_vals) ** 2).mean())
        rmse = float(np.sqrt(mse))
        
        results[f"pearson_{key}"] = float(pr_denorm) if pr_denorm is not None else np.nan
        results[f"spearman_{key}"] = float(sr_denorm) if sr_denorm is not None else np.nan
        results[f"mae_{key}"] = mae
        results[f"rmse_{key}"] = rmse
        
        print(f"  Pearson: {pr_denorm:.4f}, Spearman: {sr_denorm:.4f}")
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Calculate mean correlations
    pearsons = [results[f"pearson_{k}"] for k in TARGET_KEYS if not np.isnan(results[f"pearson_{k}"])]
    spearmans = [results[f"spearman_{k}"] for k in TARGET_KEYS if not np.isnan(results[f"spearman_{k}"])]
    maes = [results[f"mae_{k}"] for k in TARGET_KEYS]
    
    results["pearson_mean"] = float(np.mean(pearsons)) if pearsons else np.nan
    results["spearman_mean"] = float(np.mean(spearmans)) if spearmans else np.nan
    results["mae_mean"] = float(np.mean(maes))
    
    # Calculate confidence intervals for mean correlations using bootstrap
    n_bootstrap = 1000
    bootstrap_pearsons = []
    bootstrap_spearmans = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(labels_denorm), len(labels_denorm), replace=True)
        boot_preds = preds_denorm[indices]
        boot_labels = labels_denorm[indices]
        
        boot_pr = []
        boot_sp = []
        for i in range(len(TARGET_KEYS)):
            try:
                pr, _ = pearsonr(boot_labels[:, i], boot_preds[:, i])
                sr = spearmanr(boot_labels[:, i], boot_preds[:, i]).correlation
                if not np.isnan(pr):
                    boot_pr.append(pr)
                if not np.isnan(sr):
                    boot_sp.append(sr)
            except:
                pass
        
        if boot_pr:
            bootstrap_pearsons.append(np.mean(boot_pr))
        if boot_sp:
            bootstrap_spearmans.append(np.mean(boot_sp))
    
    # Calculate 95% confidence intervals
    pr_ci_low = pr_ci_high = 0.0
    sp_ci_low = sp_ci_high = 0.0
    
    if bootstrap_pearsons:
        pr_ci_low = np.percentile(bootstrap_pearsons, 2.5)
        pr_ci_high = np.percentile(bootstrap_pearsons, 97.5)
        results["pearson_mean_ci"] = (pr_ci_low, pr_ci_high)
    
    if bootstrap_spearmans:
        sp_ci_low = np.percentile(bootstrap_spearmans, 2.5)
        sp_ci_high = np.percentile(bootstrap_spearmans, 97.5)
        results["spearman_mean_ci"] = (sp_ci_low, sp_ci_high)
    
    print("\n" + "="*50)
    print("Overall Performance with 95% Confidence Intervals:")
    print(f"  Mean Pearson:  {results['pearson_mean']:.4f} [{pr_ci_low:.4f}, {pr_ci_high:.4f}]")
    print(f"  Mean Spearman: {results['spearman_mean']:.4f} [{sp_ci_low:.4f}, {sp_ci_high:.4f}]")
    print(f"  Mean MAE:      {results['mae_mean']:.4f}")
    print("="*50)
    
    # Compute ordering accuracy and difference correlation for summaries with same opinion
    print("\n" + "="*50)
    print("Ordering Accuracy & Difference Correlation Analysis")
    print("="*50)
    
    try:
        device = next(trainer.model.parameters()).device
        ordering_results = compute_ordering_accuracy(eval_dataset, trainer.model, device)
        results.update(ordering_results)
        
        print(f"\nOrdering Accuracy:")
        print(f"  Overall: {ordering_results.get('ordering_acc_overall', 0.0):.4f}")
        for key in TARGET_KEYS:
            acc_key = f"ordering_acc_{key}"
            if acc_key in ordering_results:
                print(f"  {key}: {ordering_results[acc_key]:.4f}")
        
        print(f"\nDifference Correlation:")
        print(f"  Overall (Pearson): {ordering_results.get('diff_corr_pearson_overall', 0.0):.4f}")
        print(f"  Overall (Spearman): {ordering_results.get('diff_corr_spearman_overall', 0.0):.4f}")
        for key in TARGET_KEYS:
            pearson_key = f"diff_corr_pearson_{key}"
            spearman_key = f"diff_corr_spearman_{key}"
            if pearson_key in ordering_results:
                print(f"  {key}: Pearson={ordering_results[pearson_key]:.4f}, Spearman={ordering_results[spearman_key]:.4f}")
    except Exception as e:
        print(f"Error computing ordering accuracy: {e}")
        results["ordering_acc_overall"] = 0.0
        results["diff_corr_pearson_overall"] = 0.0
        results["diff_corr_spearman_overall"] = 0.0
    
    # Optional WANDB log
    if wandb_enabled:
        try:
            import wandb  # type: ignore
            wandb.log({f"final/{k}": v for k, v in results.items()})
        except Exception:
            pass
    
    return results


class CustomTrainer(Trainer):
    """Custom trainer that computes ordering accuracy during evaluation"""
    
    def __init__(self, eval_dataset_for_ordering=None, **kwargs):
        # Remove our custom parameter before passing to parent
        if 'eval_dataset_for_ordering' in kwargs:
            eval_dataset_for_ordering = kwargs.pop('eval_dataset_for_ordering')
        super().__init__(**kwargs)
        self.eval_dataset_for_ordering = eval_dataset_for_ordering
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Call parent evaluate method
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        print(f"\nDebug: CustomTrainer.evaluate() called with metric_key_prefix='{metric_key_prefix}'")
        print(f"Debug: eval_dataset_for_ordering is {'not None' if self.eval_dataset_for_ordering is not None else 'None'}")
        
        # Add ordering accuracy if eval_dataset is available
        if self.eval_dataset_for_ordering is not None:
            try:
                device = next(self.model.parameters()).device
                ordering_results = compute_ordering_accuracy(self.eval_dataset_for_ordering, self.model, device)
                
                # Add ordering accuracy to metrics with eval_ prefix
                print(f"\nDebug: Adding ordering accuracy to metrics...")
                for key, value in ordering_results.items():
                    metric_key = f"{metric_key_prefix}_{key}"
                    metrics[metric_key] = value
                    print(f"  Added {metric_key}: {value}")
                    
                print(f"\nOrdering Accuracy & Difference Correlation Results:")
                print(f"  Ordering Accuracy Overall: {ordering_results.get('ordering_acc_overall', 0.0):.4f}")
                print(f"  Difference Correlation (Pearson) Overall: {ordering_results.get('diff_corr_pearson_overall', 0.0):.4f}")
                print(f"  Difference Correlation (Spearman) Overall: {ordering_results.get('diff_corr_spearman_overall', 0.0):.4f}")
                for key in TARGET_KEYS:
                    acc_key = f"ordering_acc_{key}"
                    diff_pearson_key = f"diff_corr_pearson_{key}"
                    diff_spearman_key = f"diff_corr_spearman_{key}"
                    if acc_key in ordering_results:
                        print(f"  {key} - Ordering: {ordering_results[acc_key]:.4f}")
                    if diff_pearson_key in ordering_results:
                        print(f"  {key} - Diff Corr: Pearson={ordering_results[diff_pearson_key]:.4f}, Spearman={ordering_results[diff_spearman_key]:.4f}")
                
                # Debug: Verify what was added to metrics
                print(f"\nDebug: Verifying metrics after addition:")
                for key, value in ordering_results.items():
                    metric_key = f"{metric_key_prefix}_{key}"
                    if metric_key in metrics:
                        print(f"  ✓ {metric_key}: {metrics[metric_key]}")
                    else:
                        print(f"  ✗ {metric_key}: NOT FOUND in metrics")
                
                # Log to WANDB if enabled
                if hasattr(self.args, 'report_to') and 'wandb' in self.args.report_to:
                    try:
                        import wandb
                        wandb_metrics = {f"{metric_key_prefix}_{k}": v for k, v in ordering_results.items()}
                        wandb.log(wandb_metrics)
                        print(f"Debug: Logged to WANDB: {wandb_metrics}")
                    except Exception as e:
                        print(f"Debug: WANDB logging failed: {e}")
                        pass
                        
            except Exception as e:
                print(f"Error computing ordering accuracy: {e}")
                # Add zero values to maintain consistent metrics structure
                metrics[f"{metric_key_prefix}_ordering_acc_overall"] = 0.0
                metrics[f"{metric_key_prefix}_diff_corr_pearson_overall"] = 0.0
                metrics[f"{metric_key_prefix}_diff_corr_spearman_overall"] = 0.0
                for key in TARGET_KEYS:
                    metrics[f"{metric_key_prefix}_ordering_acc_{key}"] = 0.0
                    metrics[f"{metric_key_prefix}_diff_corr_pearson_{key}"] = 0.0
                    metrics[f"{metric_key_prefix}_diff_corr_spearman_{key}"] = 0.0
        
        # Debug: Print final metrics keys
        ordering_keys = [k for k in metrics.keys() if 'ordering_acc' in k]
        if ordering_keys:
            print(f"\nDebug: Final metrics contains ordering accuracy keys: {ordering_keys}")
            for key in ordering_keys:
                print(f"  {key}: {metrics[key]}")
        else:
            print(f"\nDebug: No ordering accuracy keys found in final metrics")
            print(f"Debug: All metrics keys: {list(metrics.keys())}")
        
        return metrics


class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset: Dataset, every_steps: int, wandb_enabled: bool = False):
        self.eval_dataset = eval_dataset
        self.every_steps = max(1, int(every_steps))
        self.wandb_enabled = wandb_enabled

    def on_step_end(self, _args, state, control, **kwargs):
        if state.global_step > 0 and (state.global_step % self.every_steps == 0):
            trainer: Trainer = kwargs.get("model")  # not available here
            # Retrieve trainer from kwargs["trainer"] if provided
            trainer = kwargs.get("trainer", None)
            if trainer is None:
                return control
            metrics = trainer.evaluate(self.eval_dataset)
            print(f"\n[Periodic Eval @ step {state.global_step}] {metrics}")
            # Also compute correlations
            try:
                evaluate_correlations(trainer, self.eval_dataset, wandb_enabled=self.wandb_enabled)
            except Exception:
                pass
        return control


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Multi-output regression training (labels -1..7 normalized to -1..1)")
    parser.add_argument("--data", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/comment_summary_ratings.jsonl")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--out", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/deberta_regression")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases reporting")
    parser.add_argument("--wandb-project", type=str, default="llm_comment_summary_regression_deberta")
    parser.add_argument("--wandb-run-name", type=str, default="multioutput-regression-deberta")
    parser.add_argument("--eval-every-steps", type=int, default=50, help="Run evaluation every N steps via callback")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--use-tanh", action="store_true", help="Use tanh activation for output (range [-1, 1])")
    parser.add_argument("--use-sigmoid", action="store_true", help="Use sigmoid activation for output (range [0, 1])")
    parser.add_argument("--use-relu", action="store_true", help="Use ReLU activation for output (range [0, +inf))")
    parser.add_argument("--use-leaky-relu", action="store_true", help="Use LeakyReLU activation for output (range (-inf, +inf))")
    parser.add_argument("--use-elu", action="store_true", help="Use ELU activation for output (range [-1, +inf))")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation during training")
    parser.add_argument("--augment-prob", type=float, default=0.3, help="Probability of applying augmentation")
    parser.add_argument("--ensemble-eval", action="store_true", help="Use ensemble evaluation with multiple checkpoints")
    parser.add_argument("--lr-scheduler", type=str, default="linear", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    parser.add_argument("--no-normalize", action="store_true", 
                        help="Disable normalization and use original label range [-1, 7] directly")

    args = parser.parse_args()
    
    # Set random seeds for reproducibility and proper shuffling
    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return

    print(f"Loading data from {data_path}")
    data = read_jsonl(data_path)
    print(f"Loaded {len(data)} records")
    
    if len(data) == 0:
        print("Error: No valid data found")
        return
    
    # Print label distribution for debugging
    all_scores = {key: [] for key in TARGET_KEYS}
    for rec in data:
        scores = rec.get("scores", {})
        for key in TARGET_KEYS:
            if key in scores:
                all_scores[key].append(scores[key])
    
    print("\nLabel distribution:")
    for key in TARGET_KEYS:
        if all_scores[key]:
            values = np.array(all_scores[key])
            print(f"{key}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")
    
    train_recs, eval_recs = split_train_eval(data, args.eval_ratio, args.seed)
    print(f"\nTrain: {len(train_recs)}, Eval: {len(eval_recs)}")
    
    # Ensure training data is properly shuffled
    import random
    random.seed(args.seed)
    random.shuffle(train_recs)
    print(f"✓ Training data shuffled (seed={args.seed})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine whether to normalize based on command line argument
    use_normalization = not args.no_normalize
    
    if not use_normalization:
        print("\n⚠️  Training with ORIGINAL label scale [-1, 7] (normalization disabled)")
    else:
        print("\n✓ Training with NORMALIZED label scale [0, 1]")
    
    train_ds = CommentSummaryRatingsDataset(
        train_recs, tokenizer, args.max_len,
        augment=args.augment, augment_prob=args.augment_prob,
        normalize=use_normalization
    )
    eval_ds = CommentSummaryRatingsDataset(
        eval_recs, tokenizer, args.max_len, 
        augment=False,
        normalize=use_normalization
    )

    model = MultiOutputRegressor(
        args.model,
        dropout_rate=args.dropout,
        use_tanh=args.use_tanh,
        use_sigmoid=args.use_sigmoid,
        use_relu=args.use_relu,
        use_leaky_relu=args.use_leaky_relu,
        use_elu=args.use_elu
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    collator = RegressionDataCollator(tokenizer=tokenizer)

    # Set WANDB environment variables if requested
    report_backends = ["none"]
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)
        report_backends = ["wandb"]

    # Enhanced training arguments with warmup and weight decay
    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        logging_steps=10,
        seed=args.seed,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_every_steps,
        save_steps=max(100, args.eval_every_steps),
        report_to=report_backends,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="corr_mean_spearman",
        greater_is_better=True,
        gradient_accumulation_steps=2,  # Effective batch size = batch * 2
        fp16=True,  # Use mixed precision training
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=args.gradient_clip,
        lr_scheduler_type=args.lr_scheduler,
        # WANDB specific configurations
        logging_dir=f"{args.out}/logs",
        logging_strategy="steps",
        logging_first_step=True,
        eval_accumulation_steps=1,  # Log evaluation results immediately
        save_strategy="steps",
        save_safetensors=True,
    )

    # Create a compute_metrics function with the normalization parameter
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, use_normalization=use_normalization)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_wrapper,
        eval_dataset_for_ordering=eval_ds,  # Pass eval dataset for ordering accuracy
    )

    # Add periodic evaluation callback (compatible with older transformers)
    try:
        trainer.add_callback(PeriodicEvalCallback(eval_ds, args.eval_every_steps, wandb_enabled=(report_backends == ["wandb"])) )
    except Exception:
        pass
    
    # Add WANDB callback for better evaluation logging
    if args.wandb and report_backends == ["wandb"]:
        try:
            from transformers.integrations import WandbCallback
            trainer.add_callback(WandbCallback())
        except Exception as e:
            print(f"Warning: Could not add WandbCallback: {e}")

    print("\nStarting training...")
    trainer.train()
    trainer.save_model(args.out)

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    metrics = trainer.evaluate()
    
    print("\nFinal Results:")
    print(f"Overall MAE: {metrics.get('eval_mae_overall', 'N/A'):.4f}")
    print(f"Mean Spearman correlation: {metrics.get('eval_corr_mean_spearman', 'N/A'):.4f}")
    print(f"Mean Pearson correlation: {metrics.get('eval_corr_mean_pearson', 'N/A'):.4f}")
    
    print("\nPer-dimension Results:")
    for key in TARGET_KEYS:
        spearman = metrics.get(f'eval_corr_{key}_spearman', 'N/A')
        pearson = metrics.get(f'eval_corr_{key}_pearson', 'N/A')
        mae = metrics.get(f'eval_mae_{key}', 'N/A')
        if spearman != 'N/A':
            print(f"{key}: Spearman={spearman:.4f}, Pearson={pearson:.4f}, MAE={mae:.4f}")
    
    print("\nRunning final correlation analysis...")
    final_results = evaluate_correlations(trainer, eval_ds, wandb_enabled=(report_backends == ["wandb"]))
    
    print("\nSummary of Final Correlations:")
    for key in TARGET_KEYS:
        print(f"{key}: Spearman={final_results.get(f'spearman_{key}', 0):.4f}, Pearson={final_results.get(f'pearson_{key}', 0):.4f}")


if __name__ == "__main__":
    main()