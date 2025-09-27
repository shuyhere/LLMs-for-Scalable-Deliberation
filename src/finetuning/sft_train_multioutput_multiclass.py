"""
Train a multi-output multiclass model for 4 dimensions with label space {-1, 0, 1, 2, 3, 4, 5, 6, 7}.

Inputs: question + annotator opinion (comment) + summary
Targets: 4 dimensions (perspective_representation, informativeness, neutrality_balance, policy_approval)

Data: JSONL from build_comment_summary_ratings.py
Each line: {"question", "comment", "summary", "scores": {<4 dims>}}
Scores will be rounded to nearest integer and then clipped to [-1, 7] before mapping to class ids (class_id = score + 1 in [0..8]).

Example usage:
    Basic training:
        python improved_training_script.py --data /path/to/data.jsonl --model microsoft/deberta-v3-base

    With custom parameters:
        python improved_training_script.py \
            --data /path/to/data.jsonl \
            --model microsoft/deberta-v3-base \
            --out ./results/my_model \
            --epochs 5 \
            --batch 16 \
            --lr 2e-5 \
            --max-len 1024 \
            --eval-ratio 0.15

    With wandb logging:
        python improved_training_script.py \
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
)


TARGET_KEYS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]

NUM_CLASSES = 9  # mapping: y in {-1..7} -> class_id = y + 1 in {0..8}


def score_to_class_id(value: float) -> int:
    """Convert score to class ID. Scores in [-1, 7] -> class IDs in [0, 8]"""
    y = int(round(value))
    y = max(-1, min(7, y))  # Clip to [-1, 7]
    return y + 1


def class_id_to_score(class_id: int) -> int:
    """Convert class ID back to original score"""
    return class_id - 1


class CommentSummaryRatingsClsDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        question = rec.get("question", "")
        comment = rec.get("comment", "")
        summary = rec.get("summary", "")

        # Use tokenizer's special tokens for better separation
        sep_token = self.tokenizer.sep_token if self.tokenizer.sep_token else "[SEP]"

        text = f"Question: {question} {sep_token} Annotator opinion: {comment} {sep_token} Summary: {summary}"

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        scores = rec.get("scores", {})
        class_ids = [score_to_class_id(scores.get(k, 0.0)) for k in TARGET_KEYS]
        enc["labels"] = torch.tensor(class_ids, dtype=torch.long)  # shape (4,)
        return enc


class MultiOutputClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_dims: int = 4, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, num_dims * num_classes)
        self.num_dims = num_dims
        self.num_classes = num_classes
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Remove any unexpected kwargs
        kwargs.pop("num_items_in_batch", None)

        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = out.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled = self.dropout(pooled)
        logits = self.head(pooled)  # (B, num_dims * num_classes)
        logits = logits.view(logits.size(0), self.num_dims, self.num_classes)  # (B, 4, 9)

        loss = None
        if labels is not None:
            # labels shape: (B, 4) with class ids in [0..8]
            loss = 0.0
            for d in range(self.num_dims):
                loss = loss + self.loss_fct(logits[:, d, :], labels[:, d])
            loss = loss / float(self.num_dims)

        return {"loss": loss, "logits": logits}


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
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_eval = int(len(data) * eval_ratio)
    eval_idx = set(idx[:n_eval].tolist())
    train, eval_ = [], []
    for i, rec in enumerate(data):
        (eval_.append if i in eval_idx else train.append)(rec)
    return train, eval_


class ClsCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = torch.stack(labels)  # (B, 4)
        return batch


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # preds: (B, 4, 9), labels: (B, 4)
    if preds.ndim == 3:
        logits = preds
    else:
        logits = preds.reshape(preds.shape[0], len(TARGET_KEYS), NUM_CLASSES)
    y_hat = logits.argmax(axis=-1)

    metrics: Dict[str, Any] = {}

    # Overall accuracy across all dimensions
    metrics["acc_overall"] = float((y_hat == labels).mean())

    # Per-dimension accuracy
    for i, key in enumerate(TARGET_KEYS):
        dim_acc = float((y_hat[:, i] == labels[:, i]).mean())
        metrics[f"acc_{key}"] = dim_acc
        print(f"Dimension {key}: {dim_acc:.4f}")

    # Additional metrics: mean absolute error in original score space
    y_hat_scores = y_hat - 1  # Convert back to [-1, 7] range
    labels_scores = labels - 1
    mae_overall = float(np.abs(y_hat_scores - labels_scores).mean())
    metrics["mae_overall"] = mae_overall

    for i, key in enumerate(TARGET_KEYS):
        mae_dim = float(np.abs(y_hat_scores[:, i] - labels_scores[:, i]).mean())
        metrics[f"mae_{key}"] = mae_dim

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-output multiclass training (labels -1..7 mapped to 0..8)")
    parser.add_argument("--data", type=str,
                        default="datasets/comment_summary_ratings.jsonl")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--out", type=str,
                        default="results/deberta_multiclass")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llm_comment_summary_multiclass_deberta")
    parser.add_argument("--wandb-run-name", type=str, default="multioutput-multiclass-deberta")
    parser.add_argument("--eval-every-steps", type=int, default=50)

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
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
    print(f"Train: {len(train_recs)}, Eval: {len(eval_recs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = CommentSummaryRatingsClsDataset(train_recs, tokenizer, args.max_len)
    eval_ds = CommentSummaryRatingsClsDataset(eval_recs, tokenizer, args.max_len)

    model = MultiOutputClassifier(args.model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    collator = ClsCollator(tokenizer)

    report_backends = ["none"]
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)
        report_backends = ["wandb"]

    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        logging_steps=10,
        seed=args.seed,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_every_steps,
        save_steps=max(100, args.eval_every_steps),
        report_to=report_backends,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="acc_overall",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.out)

    print("\nRunning final evaluation...")
    metrics = trainer.evaluate()

    print("\nFinal Results:")
    print(f"Overall Accuracy: {metrics['eval_acc_overall']:.4f}")
    print(f"Overall MAE: {metrics['eval_mae_overall']:.4f}")

    print("\nPer-dimension Results:")
    for key in TARGET_KEYS:
        acc = metrics[f'eval_acc_{key}']
        mae = metrics[f'eval_mae_{key}']
        print(f"{key}: Accuracy={acc:.4f}, MAE={mae:.4f}")


if __name__ == "__main__":
    main()