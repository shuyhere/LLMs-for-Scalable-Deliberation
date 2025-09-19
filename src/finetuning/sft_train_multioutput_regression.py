#!/usr/bin/env python3
"""
Train a multi-output regression model to predict 4 dimensions (1-7 scale) from
inputs constructed as: question + comment + summary.

Targets (in order):
- perspective_representation
- informativeness
- neutrality_balance
- policy_approval

Data format: JSONL produced by build_comment_summary_ratings.py
Each line requires: {"question", "comment", "summary", "scores": { four keys }}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from scipy.stats import spearmanr
from scipy.stats import pearsonr


TARGET_KEYS = [
    "perspective_representation",
    "informativeness",
    "neutrality_balance",
    "policy_approval",
]


class CommentSummaryRatingsDataset(Dataset):
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

        # Build a simple prompt-style input
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
        labels = [float(scores.get(k, 0.0)) for k in TARGET_KEYS]
        enc["labels"] = torch.tensor(labels, dtype=torch.float32)
        return enc


class RegressionDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate labels from features
        labels = [f.pop("labels") for f in features]
        # Pad the remaining inputs using tokenizer
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        # Stack labels to (batch, 4)
        if isinstance(labels[0], torch.Tensor):
            batch["labels"] = torch.stack(labels)
        else:
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


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


def split_train_eval(data: List[Dict[str, Any]], eval_ratio: float, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_eval = int(len(data) * eval_ratio)
    eval_idx = set(idx[:n_eval].tolist())
    train, eval_ = [], []
    for i, rec in enumerate(data):
        if i in eval_idx:
            eval_.append(rec)
        else:
            train.append(rec)
    return train, eval_


def compute_metrics_fn(eval_pred):
    preds, labels = eval_pred
    # preds shape: (batch, 4), labels shape: (batch, 4)
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    mse = ((preds - labels) ** 2).mean(axis=0)
    mae = np.abs(preds - labels).mean(axis=0)
    # overall
    metrics = {
        "mse_overall": float(((preds - labels) ** 2).mean()),
        "mae_overall": float(np.abs(preds - labels).mean()),
    }
    for i, key in enumerate(TARGET_KEYS):
        # Spearman can fail on constant vectors
        try:
            sp = spearmanr(labels[:, i], preds[:, i]).correlation
        except Exception:
            sp = np.nan
        metrics[f"mse_{key}"] = float(mse[i])
        metrics[f"mae_{key}"] = float(mae[i])
        metrics[f"spearman_{key}"] = float(sp) if sp is not None else np.nan
    return metrics


def evaluate_correlations(trainer: Trainer, eval_dataset: Dataset, wandb_enabled: bool = False):
    # Use trainer.predict to get predictions on eval set
    pred_output = trainer.predict(eval_dataset)
    preds = pred_output.predictions
    labels = pred_output.label_ids
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    results: Dict[str, Any] = {}
    pearsons = []
    spearmans = []
    for i, key in enumerate(TARGET_KEYS):
        try:
            pr, _ = pearsonr(labels[:, i], preds[:, i])
        except Exception:
            pr = np.nan
        try:
            sr = spearmanr(labels[:, i], preds[:, i]).correlation
        except Exception:
            sr = np.nan
        results[f"pearson_{key}"] = float(pr) if pr is not None else np.nan
        results[f"spearman_{key}"] = float(sr) if sr is not None else np.nan
        if not np.isnan(results[f"pearson_{key}"]):
            pearsons.append(results[f"pearson_{key}"])
        if not np.isnan(results[f"spearman_{key}"]):
            spearmans.append(results[f"spearman_{key}"])
    results["pearson_mean"] = float(np.nanmean(pearsons)) if pearsons else np.nan
    results["spearman_mean"] = float(np.nanmean(spearmans)) if spearmans else np.nan

    print("Correlation evaluation (eval set):")
    for k in sorted(results.keys()):
        print(f"{k}: {results[k]}")

    # Optional WANDB log
    if wandb_enabled:
        try:
            import wandb  # type: ignore
            wandb.log({f"corr/{k}": v for k, v in results.items()})
        except Exception:
            pass

    return results


class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset: Dataset, every_steps: int, wandb_enabled: bool = False):
        self.eval_dataset = eval_dataset
        self.every_steps = max(1, int(every_steps))
        self.wandb_enabled = wandb_enabled

    def on_step_end(self, args, state, control, **kwargs):
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


def main():
    parser = argparse.ArgumentParser(description="Train 4-dimension regression (1-7) for comment+summary ratings")
    parser.add_argument("--data", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/comment_summary_ratings.jsonl",
                        help="Path to JSONL built by build_comment_summary_ratings.py")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="HF base model")
    parser.add_argument("--out", type=str,
                        default="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/multioutput_regression",
                        help="Output dir for checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases reporting")
    parser.add_argument("--wandb-project", type=str, default="llm_comment_summary_regression", help="WANDB project name")
    parser.add_argument("--wandb-run-name", type=str, default="multioutput-regression", help="WANDB run name")
    parser.add_argument("--eval-every-steps", type=int, default=10, help="Run evaluation every N steps via callback")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return

    data = read_jsonl(data_path)
    train_recs, eval_recs = split_train_eval(data, args.eval_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_ds = CommentSummaryRatingsDataset(train_recs, tokenizer, args.max_len)
    eval_ds = CommentSummaryRatingsDataset(eval_recs, tokenizer, args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=4,
        problem_type="regression",
    )

    # Ensure outputs are raw regression predictions
    if hasattr(model.config, "torch_dtype"):
        pass

    collator = RegressionDataCollator(tokenizer=tokenizer)

    # Set WANDB environment variables if requested
    report_backends = ["none"]
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)
        report_backends = ["wandb"]

    # Use conservative set of arguments for broader Transformers compatibility
    training_args = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        logging_steps=1,
        seed=args.seed,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_every_steps,
        save_steps=max(100, args.eval_every_steps),
        report_to=report_backends,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )

    # Add periodic evaluation callback (compatible with older transformers)
    try:
        trainer.add_callback(PeriodicEvalCallback(eval_ds, args.eval_every_steps, wandb_enabled=(report_backends == ["wandb"])) )
    except Exception:
        pass

    trainer.train()
    trainer.save_model(args.out)

    # Final eval
    metrics = trainer.evaluate()
    print("Final eval metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Additional correlation evaluation on eval set
    evaluate_correlations(trainer, eval_ds, wandb_enabled=(report_backends == ["wandb"]))


if __name__ == "__main__":
    main()


