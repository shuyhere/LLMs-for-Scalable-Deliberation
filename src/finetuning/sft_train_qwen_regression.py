#!/usr/bin/env python3
"""
Train Qwen3-0.6B model for multi-output regression task.
Adapts the DeBERTa regression approach to use a Large Language Model architecture.

Key differences from DeBERTa version:
1. Uses Qwen3-0.6B (decoder-only architecture) instead of DeBERTa (encoder-only)
2. Uses last token's hidden state for pooling (GPT-style) instead of CLS token
3. Supports both full fine-tuning and parameter-efficient fine-tuning (LoRA)
4. Adjusted tokenization for causal LM
5. Option to freeze base model layers for initial training stability

Inputs: question + annotator opinion (comment) + summary
Targets: 4 dimensions (perspective_representation, informativeness, neutrality_balance, policy_approval)

Example usage:
    Basic training:
        python sft_train_qwen_regression.py --data /path/to/data.jsonl --model Qwen/Qwen2.5-0.5B

    With LoRA (parameter-efficient):
        python sft_train_qwen_regression.py \
            --data /path/to/data.jsonl \
            --model Qwen/Qwen2.5-0.5B \
            --use-lora \
            --lora-r 16 \
            --lora-alpha 32
            
    With frozen base model:
        python sft_train_qwen_regression.py \
            --data /path/to/data.jsonl \
            --model Qwen/Qwen2.5-0.5B \
            --freeze-base-model \
            --lr 1e-3
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
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_callback import TrainerCallback
from scipy.stats import spearmanr, pearsonr
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


TARGET_KEYS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]


def normalize_score(score: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
    """Normalize score from [min_val, max_val] to [0, 1] for better training stability"""
    norm_01 = (score - min_val) / (max_val - min_val)
    return norm_01


def denormalize_score(normalized: float, min_val: float = -1.0, max_val: float = 7.0) -> float:
    """Denormalize score from [0, 1] back to [min_val, max_val]"""
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
            parts = text.split("\n\n")
            augmented_parts = []
            for part in parts:
                sentences = part.strip().split('. ')
                if len(sentences) > 1:
                    np.random.shuffle(sentences)
                    augmented_parts.append('. '.join(sentences))
                else:
                    augmented_parts.append(part)
            return "\n\n".join(augmented_parts)
        
        elif aug_type == 'paraphrase':
            # Simple paraphrase by changing templates
            templates = [
                "Query: {q}\n\nReviewer's view: {c}\n\nSynopsis: {s}",
                "Issue: {q}\n\nAnnotator's perspective: {c}\n\nOverview: {s}",
                "Topic: {q}\n\nOpinion provided: {c}\n\nSummary text: {s}",
            ]
            # Extract parts
            parts = text.split("\n\n")
            if len(parts) == 3:
                q = parts[0].replace("Question:", "").strip()
                c = parts[1].replace("Annotator opinion:", "").strip()
                s = parts[2].replace("Summary:", "").strip()
                template = np.random.choice(templates)
                return template.format(q=q, c=c, s=s)
        
        return text

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        question = rec.get("question", "")
        comment = rec.get("comment", "")
        summary = rec.get("summary", "")

        # Format text for LLM (using natural language format)
        text = f"Question: {question}\n\nAnnotator opinion: {comment}\n\nSummary: {summary}"
        
        # Apply augmentation
        text = self.augment_text(text)

        # Tokenize with padding on the left for causal LM
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
                noise = np.random.normal(0, 0.1, len(labels))
                labels = np.clip(np.array(labels) + noise, -1, 7).tolist()
        
        enc["labels"] = torch.tensor(labels, dtype=torch.float32)
        return enc


class QwenRegressionModel(nn.Module):
    """Qwen model with regression head for multi-output regression"""
    def __init__(self, base_model_name: str, num_dims: int = 4, dropout_rate: float = 0.1, 
                 use_lora: bool = False, lora_r: int = 16, lora_alpha: int = 32, 
                 lora_dropout: float = 0.1, freeze_base_model: bool = False,
                 use_tanh: bool = False, use_sigmoid: bool = False):
        super().__init__()
        
        # Load base Qwen model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            trust_remote_code=True
        )
        
        # Get hidden size from config
        hidden_size = self.base_model.config.hidden_size
        
        # Apply LoRA if requested
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"],
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            print(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}")
            self.base_model.print_trainable_parameters()
        
        # Freeze base model if requested
        if freeze_base_model and not use_lora:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("Froze base model parameters")
        
        # Regression head
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-layer regression head with residual connection
        self.pre_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        self.regression_head = nn.Linear(hidden_size // 4, num_dims)
        
        # Output activation
        self.use_tanh = use_tanh
        self.use_sigmoid = use_sigmoid
        
        if use_tanh:
            self.output_activation = nn.Tanh()  # Output in [-1, 1]
        elif use_sigmoid:
            self.output_activation = nn.Sigmoid()  # Output in [0, 1]
        else:
            self.output_activation = None  # No activation, raw logits
        
        self.num_dims = num_dims
        self.loss_fct = nn.HuberLoss(delta=1.0)  # More robust than MSE
        
        # Add config attribute for compatibility with Trainer
        self.config = self.base_model.config
        
        # Initialize regression head weights
        self._init_regression_weights()
    
    def _init_regression_weights(self):
        """Initialize regression head weights with small values"""
        for module in [self.pre_head, self.regression_head]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Sequential):
                for submodule in module.modules():
                    if isinstance(submodule, nn.Linear):
                        submodule.weight.data.normal_(mean=0.0, std=0.02)
                        if submodule.bias is not None:
                            submodule.bias.data.zero_()
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Remove any unexpected kwargs
        kwargs.pop("num_items_in_batch", None)
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get last hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        
        # Use last token's representation (GPT-style)
        # Find the position of the last non-padding token for each sequence
        if attention_mask is not None:
            # Get the position of last token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            pooled = hidden_states[torch.arange(batch_size), sequence_lengths]
        else:
            # If no attention mask, just use the last token
            pooled = hidden_states[:, -1, :]
        
        # Apply dropout and regression head
        pooled = self.dropout(pooled)
        hidden = self.pre_head(pooled)
        logits = self.regression_head(hidden)  # (batch_size, num_dims)
        
        # Apply output activation if specified
        if self.output_activation is not None:
            predictions = self.output_activation(logits)
        else:
            predictions = logits
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fct(predictions, labels)
        
        return {"loss": loss, "logits": predictions}
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save the model to a directory"""
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the base model with safe_serialization=False to handle shared tensors
        if isinstance(self.base_model, PeftModel):
            self.base_model.save_pretrained(save_directory, safe_serialization=False)
        else:
            self.base_model.save_pretrained(save_directory, safe_serialization=False)
        
        # Save regression head separately
        regression_head_path = os.path.join(save_directory, "regression_head.pt")
        torch.save({
            'pre_head_state_dict': self.pre_head.state_dict(),
            'regression_head_state_dict': self.regression_head.state_dict(),
            'use_tanh': self.use_tanh,
            'use_sigmoid': self.use_sigmoid,
            'num_dims': self.num_dims,
        }, regression_head_path)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a pretrained model from a directory"""
        # Load the base model
        if os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json")):
            # LoRA model
            from peft import AutoPeftModelForCausalLM
            base_model = AutoPeftModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        
        # Load regression head
        regression_head_path = os.path.join(pretrained_model_name_or_path, "regression_head.pt")
        regression_state = torch.load(regression_head_path, map_location='cpu')
        
        # Create model instance
        model = cls(
            base_model_name=pretrained_model_name_or_path,
            num_dims=regression_state['num_dims'],
            use_tanh=regression_state['use_tanh'],
            use_sigmoid=regression_state['use_sigmoid'],
        )
        
        # Load state dicts
        model.pre_head.load_state_dict(regression_state['pre_head_state_dict'])
        model.regression_head.load_state_dict(regression_state['regression_head_state_dict'])
        model.base_model = base_model
        
        return model


class RegressionDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features]
        
        # Pad on the left for causal LM
        batch = self.tokenizer.pad(
            features, 
            padding=True, 
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        batch["labels"] = torch.stack(labels)
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
                if "scores" in data and isinstance(data["scores"], dict):
                    out.append(data)
                else:
                    print(f"Warning: Line {line_num} missing 'scores' field, skipping")
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} invalid JSON: {e}")
                continue
    return out


def split_train_eval(data: List[Dict[str, Any]], eval_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split data ensuring same opinions stay together for ordering accuracy calculation."""
    rng = np.random.default_rng(seed)
    
    # Group data by opinion (comment)
    opinion_groups = {}
    for rec in data:
        opinion = rec.get("comment", "")
        if opinion not in opinion_groups:
            opinion_groups[opinion] = []
        opinion_groups[opinion].append(rec)
    
    # Get list of unique opinions
    unique_opinions = list(opinion_groups.keys())
    print(f"Found {len(unique_opinions)} unique opinions")
    
    # Shuffle opinions (not individual samples)
    rng.shuffle(unique_opinions)
    
    # Split opinions into train/eval
    n_eval_opinions = int(len(unique_opinions) * eval_ratio)
    eval_opinions = set(unique_opinions[:n_eval_opinions])
    train_opinions = set(unique_opinions[n_eval_opinions:])
    
    # Assign samples based on opinion groups
    train, eval_ = [], []
    for opinion in unique_opinions:
        samples = opinion_groups[opinion]
        if opinion in eval_opinions:
            eval_.extend(samples)
        else:
            train.extend(samples)
    
    # Shuffle within train and eval
    rng.shuffle(train)
    rng.shuffle(eval_)
    
    print(f"Split: {len(train_opinions)} train opinions -> {len(train)} samples")
    print(f"       {len(eval_opinions)} eval opinions -> {len(eval_)} samples")
    
    return train, eval_


def compute_metrics(eval_pred, use_normalization=True):
    preds, labels = eval_pred
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    
    if use_normalization:
        # For normalized training, ensure predictions are in [0, 1] range
        preds = np.clip(preds, 0.0, 1.0)
        
        # Denormalize for interpretable MAE
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
    
    # Per-dimension correlations
    for i, key in enumerate(TARGET_KEYS):
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
    
    return metrics


def compute_ordering_accuracy(eval_dataset: Dataset, model, device='cuda') -> Dict[str, float]:
    """Compute ordering accuracy for summaries with the same question and opinion"""
    model.eval()
    
    # Group samples by opinion
    opinion_groups = {}
    all_samples = []
    
    for i in range(len(eval_dataset)):
        sample = eval_dataset[i]
        comment = eval_dataset.records[i].get("comment", "")
        question = eval_dataset.records[i].get("question", "")
        
        key = comment
        
        if key not in opinion_groups:
            opinion_groups[key] = []
        opinion_groups[key].append({
            'idx': i,
            'sample': sample,
            'question': question,
            'comment': comment
        })
        all_samples.append((i, sample))
    
    # Find groups with at least 2 samples
    valid_groups = {k: v for k, v in opinion_groups.items() if len(v) >= 2}
    
    if not valid_groups:
        print("No groups with same opinion found for ordering evaluation")
        return {"ordering_acc_overall": 0.0, "diff_corr_overall": 0.0}
    
    print(f"Found {len(valid_groups)} opinion groups with multiple samples")
    
    # Get predictions for all samples
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for i, sample in all_samples:
            # Prepare input
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"] if "attention_mask" in sample else None
            labels = sample["labels"]
            
            # Convert to tensors if needed
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            if attention_mask is not None and isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask)
            if isinstance(labels, list):
                labels = torch.tensor(labels)
            
            # Add batch dimension and move to device
            input_ids = input_ids.unsqueeze(0).to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            
            # Get prediction
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs["logits"].cpu().numpy()[0]
            label = labels.cpu().numpy()[0]
            
            all_preds.append(pred)
            all_labels.append(label)
            all_indices.append(i)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate ordering accuracy for each dimension
    ordering_accs = {}
    diff_correlations = {}
    
    for dim_idx, key in enumerate(TARGET_KEYS):
        correct_pairs = 0
        total_pairs = 0
        
        pred_diffs = []
        label_diffs = []
        
        for opinion, samples in valid_groups.items():
            if len(samples) < 2:
                continue
                
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    sample1 = samples[i]
                    sample2 = samples[j]
                    idx1 = sample1['idx']
                    idx2 = sample2['idx']
                    
                    pred1_idx = all_indices.index(idx1)
                    pred2_idx = all_indices.index(idx2)
                    
                    pred1_score = all_preds[pred1_idx][dim_idx]
                    pred2_score = all_preds[pred2_idx][dim_idx]
                    label1_score = all_labels[pred1_idx][dim_idx]
                    label2_score = all_labels[pred2_idx][dim_idx]
                    
                    pred_diff = pred1_score - pred2_score
                    label_diff = label1_score - label2_score
                    pred_diffs.append(pred_diff)
                    label_diffs.append(label_diff)
                    
                    if label1_score > label2_score:
                        if pred1_score > pred2_score:
                            correct_pairs += 1
                    elif label1_score < label2_score:
                        if pred1_score < pred2_score:
                            correct_pairs += 1
                    else:
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
        if len(pred_diffs) > 1:
            try:
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
    
    # Overall metrics
    if ordering_accs:
        overall_acc = np.mean(list(ordering_accs.values()))
        ordering_accs["ordering_acc_overall"] = float(overall_acc)
        print(f"Overall ordering accuracy: {overall_acc:.4f}")
    else:
        ordering_accs["ordering_acc_overall"] = 0.0
    
    pearson_diff_corrs = [diff_correlations[f"diff_corr_pearson_{k}"] for k in TARGET_KEYS 
                          if f"diff_corr_pearson_{k}" in diff_correlations and diff_correlations[f"diff_corr_pearson_{k}"] != 0.0]
    spearman_diff_corrs = [diff_correlations[f"diff_corr_spearman_{k}"] for k in TARGET_KEYS 
                           if f"diff_corr_spearman_{k}" in diff_correlations and diff_correlations[f"diff_corr_spearman_{k}"] != 0.0]
    
    if pearson_diff_corrs:
        diff_correlations["diff_corr_pearson_overall"] = float(np.mean(pearson_diff_corrs))
    else:
        diff_correlations["diff_corr_pearson_overall"] = 0.0
    
    if spearman_diff_corrs:
        diff_correlations["diff_corr_spearman_overall"] = float(np.mean(spearman_diff_corrs))
    else:
        diff_correlations["diff_corr_spearman_overall"] = 0.0
    
    results = {**ordering_accs, **diff_correlations}
    return results


class CustomTrainer(Trainer):
    """Custom trainer that computes ordering accuracy during evaluation"""
    
    def __init__(self, eval_dataset_for_ordering=None, **kwargs):
        if 'eval_dataset_for_ordering' in kwargs:
            eval_dataset_for_ordering = kwargs.pop('eval_dataset_for_ordering')
        super().__init__(**kwargs)
        self.eval_dataset_for_ordering = eval_dataset_for_ordering
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if self.eval_dataset_for_ordering is not None:
            try:
                device = next(self.model.parameters()).device
                ordering_results = compute_ordering_accuracy(self.eval_dataset_for_ordering, self.model, device)
                
                for key, value in ordering_results.items():
                    metric_key = f"{metric_key_prefix}_{key}"
                    metrics[metric_key] = value
                    
            except Exception as e:
                print(f"Error computing ordering accuracy: {e}")
                metrics[f"{metric_key_prefix}_ordering_acc_overall"] = 0.0
                metrics[f"{metric_key_prefix}_diff_corr_pearson_overall"] = 0.0
                metrics[f"{metric_key_prefix}_diff_corr_spearman_overall"] = 0.0
        
        return metrics


def evaluate_correlations(trainer: Trainer, eval_dataset: Dataset, wandb_enabled: bool = False):
    """Comprehensive evaluation including correlations and prediction distribution analysis"""
    pred_output = trainer.predict(eval_dataset)
    preds = pred_output.predictions
    labels = pred_output.label_ids
    
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    
    preds = np.clip(preds, 0.0, 1.0)
    
    preds_denorm = denormalize_score(preds)
    labels_denorm = denormalize_score(labels)
    
    results: Dict[str, Any] = {}
    
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
        
        try:
            pr_denorm, _ = pearsonr(label_vals, pred_vals)
        except Exception:
            pr_denorm = np.nan
        
        try:
            sr_denorm = spearmanr(label_vals, pred_vals).correlation
        except Exception:
            sr_denorm = np.nan
        
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
    
    print("\n" + "="*50)
    print(f"Mean Pearson:  {results['pearson_mean']:.4f}")
    print(f"Mean Spearman: {results['spearman_mean']:.4f}")
    print(f"Mean MAE:      {results['mae_mean']:.4f}")
    print("="*50)
    
    # Compute ordering accuracy
    print("\n" + "="*50)
    print("Ordering Accuracy & Difference Correlation Analysis")
    print("="*50)
    
    try:
        device = next(trainer.model.parameters()).device
        ordering_results = compute_ordering_accuracy(eval_dataset, trainer.model, device)
        results.update(ordering_results)
    except Exception as e:
        print(f"Error computing ordering accuracy: {e}")
        results["ordering_acc_overall"] = 0.0
        results["diff_corr_pearson_overall"] = 0.0
        results["diff_corr_spearman_overall"] = 0.0
    
    if wandb_enabled:
        try:
            import wandb
            wandb.log({f"final/{k}": v for k, v in results.items()})
        except Exception:
            pass
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B multi-output regression training")
    parser.add_argument("--data", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/comment_summary_ratings.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--out", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/qwen_regression")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases reporting")
    parser.add_argument("--wandb-project", type=str, default="llm_comment_summary_regression_qwen")
    parser.add_argument("--wandb-run-name", type=str, default="multioutput-regression-qwen")
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-prob", type=float, default=0.3)
    parser.add_argument("--lr-scheduler", type=str, default="linear", 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--no-normalize", action="store_true", 
                        help="Disable normalization and use original label range [-1, 7] directly")
    
    # Qwen-specific arguments
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--freeze-base-model", action="store_true", 
                        help="Freeze base model parameters (only train regression head)")
    parser.add_argument("--use-tanh", action="store_true", help="Use tanh activation for output")
    parser.add_argument("--use-sigmoid", action="store_true", help="Use sigmoid activation for output")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, 
                        help="Gradient accumulation steps (for larger effective batch size)")

    args = parser.parse_args()

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
    
    # Print label distribution
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

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Set padding token for Qwen (it uses eos_token as pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to left for causal LM
    tokenizer.padding_side = "left"
    
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

    # Initialize model
    model = QwenRegressionModel(
        args.model,
        dropout_rate=args.dropout,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_base_model=args.freeze_base_model,
        use_tanh=args.use_tanh,
        use_sigmoid=args.use_sigmoid
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} (trainable)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    collator = RegressionDataCollator(tokenizer=tokenizer)

    # Set WANDB environment variables
    report_backends = ["none"]
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)
        report_backends = ["wandb"]

    # Adjust learning rate based on whether we're freezing the base model
    learning_rate = args.lr
    if args.freeze_base_model and not args.use_lora:
        learning_rate = 1e-3  # Higher LR for just training the head
        print(f"Using higher learning rate {learning_rate} for frozen base model")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=learning_rate,
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if available
        fp16=False,  # Disable fp16 to avoid conflicts
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=args.gradient_clip,
        lr_scheduler_type=args.lr_scheduler,
        logging_dir=f"{args.out}/logs",
        logging_strategy="steps",
        logging_first_step=True,
        eval_accumulation_steps=1,
        save_strategy="steps",
        save_safetensors=False,  # Disable safetensors due to shared tensor issue in Qwen
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Create compute_metrics function
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, use_normalization=use_normalization)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_wrapper,
        eval_dataset_for_ordering=eval_ds,
    )

    print("\nStarting training...")
    trainer.train()
    
    # Save the model
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

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