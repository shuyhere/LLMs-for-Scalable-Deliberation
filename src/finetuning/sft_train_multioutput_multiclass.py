#!/usr/bin/env python3
"""
Train a multi-output multiclass model for 4 dimensions with label space {-1, 0, 1, 2, 3, 4, 5, 6, 7}.

Inputs: question + annotator opinion (comment) + summary
Targets: 4 dimensions (perspective_representation, informativeness, neutrality_balance, policy_approval)

Data: JSONL from build_comment_summary_ratings.py
Each line: {"question", "comment", "summary", "scores": {<4 dims>}}
Scores will be rounded to nearest integer and then clipped to [-1, 7] before mapping to class ids (class_id = score + 1 in [0..8]).
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
    y = int(round(value))
    if y < -1:
        y = -1
    if y > 7:
        y = 7
    return y + 1


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

        text = (
            f"Question: {question}\n"
            f"Annotator opinion: {comment}\n"
            f"Summary: {summary}"
        )

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
        kwargs.pop("num_items_in_batch", None)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = out.last_hidden_state[:, 0, :]
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
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
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
    # overall accuracy across all dims
    metrics["acc_overall"] = float((y_hat == labels).mean())
    # per-dimension accuracy
    for i, key in enumerate(TARGET_KEYS):
        metrics[f"acc_{key}"] = float((y_hat[:, i] == labels[:, i]).mean())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-output multiclass training (labels -1..7 mapped to 0..8)")
    parser.add_argument("--data", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/comment_summary_ratings.jsonl")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--out", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/multioutput_multiclass")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="llm_comment_summary_multiclass")
    parser.add_argument("--wandb-run-name", type=str, default="multioutput-multiclass")
    parser.add_argument("--eval-every-steps", type=int, default=50)

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return

    data = read_jsonl(data_path)
    train_recs, eval_recs = split_train_eval(data, args.eval_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = CommentSummaryRatingsClsDataset(train_recs, tokenizer, args.max_len)
    eval_ds = CommentSummaryRatingsClsDataset(eval_recs, tokenizer, args.max_len)

    model = MultiOutputClassifier(args.model)

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.out)

    print("Running final evaluation...")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()


