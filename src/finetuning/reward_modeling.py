#!/usr/bin/env python3
"""
Minimal reward model training (backup-aligned structure) for full_augment RL pairs.

Only differences vs backup:
- Data reader: expects JSONL with fields: prompt, chosen, rejected, weight
- Loss: weighted Bradley–Terry: -w * log(sigmoid(r(chosen) - r(rejected)))
"""

import argparse
import os
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class RLDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        prompt = r.get('prompt', '')
        chosen = r.get('chosen', '')
        rejected = r.get('rejected', '')
        weight = float(r.get('weight', 1.0))

        text_pos = f"{prompt}\n\n{chosen}"
        text_neg = f"{prompt}\n\n{rejected}"

        pos = self.tok(text_pos, truncation=True, max_length=self.max_len, padding=False)
        neg = self.tok(text_neg, truncation=True, max_length=self.max_len, padding=False)

        return {
            'input_ids_pos': pos['input_ids'],
            'attention_mask_pos': pos['attention_mask'],
            'input_ids_neg': neg['input_ids'],
            'attention_mask_neg': neg['attention_mask'],
            'weight': torch.tensor(weight, dtype=torch.float32),
        }


class PairwiseCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tok = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pos = [{'input_ids': f.pop('input_ids_pos'), 'attention_mask': f.pop('attention_mask_pos')} for f in features]
        neg = [{'input_ids': f.pop('input_ids_neg'), 'attention_mask': f.pop('attention_mask_neg')} for f in features]
        bpos = self.tok.pad(pos, padding=True, return_tensors='pt')
        bneg = self.tok.pad(neg, padding=True, return_tensors='pt')
        weights = torch.stack([f['weight'] for f in features])
        return {
            'input_ids_pos': bpos['input_ids'],
            'attention_mask_pos': bpos['attention_mask'],
            'input_ids_neg': bneg['input_ids'],
            'attention_mask_neg': bneg['attention_mask'],
            'weight': weights,
        }


class RewardModel(nn.Module):
    def __init__(self, base: str, peft_config: Any | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        # Optionally apply LoRA to the encoder
        if peft_config is not None:
            try:
                from peft import get_peft_model
                self.encoder = get_peft_model(self.encoder, peft_config)
            except Exception:
                # If PEFT is not available or fails, keep the base encoder
                pass
        hidden = self.encoder.config.hidden_size
        self.value = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.1)
        self.logsigm = nn.LogSigmoid()

    def score(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        pooled = pooled.to(self.value.weight.dtype)
        return self.value(pooled).squeeze(-1)

    def forward(self,
                input_ids_pos=None, attention_mask_pos=None,
                input_ids_neg=None, attention_mask_neg=None,
                weight=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        rpos = self.score(input_ids_pos, attention_mask_pos)
        rneg = self.score(input_ids_neg, attention_mask_neg)
        diff = rpos - rneg
        if weight is None:
            weight = torch.ones_like(diff)
        loss = -(weight * self.logsigm(diff)).mean()
        # Provide logits as 2D tensor for Trainer compatibility
        logits = diff.unsqueeze(-1)
        return {'loss': loss, 'logits': logits}


def split(data: List[Dict[str, Any]], eval_ratio: float) -> tuple[list, list]:
    n = len(data)
    k = int(n * eval_ratio)
    # Ensure both splits are non-empty when possible
    if k <= 0 and n > 1:
        k = 1
    if k >= n and n > 1:
        k = n - 1
    return data[k:], data[:k]


def main():
    p = argparse.ArgumentParser(description='Minimal reward model training')
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--model', type=str, default='microsoft/deberta-v3-base')
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--max-len', type=int, default=1024)
    p.add_argument('--eval-ratio', type=float, default=0.1)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--eval-steps', type=int, default=10)
    p.add_argument('--eval-strategy', type=str, default='steps')
    # Logging / WANDB
    p.add_argument('--report_to', type=str, default='wandb')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--wandb_project', type=str, default='reward-modeling')
    # Best model selection controls
    p.add_argument('--load_best_model_at_end', action='store_true')
    p.add_argument('--metric_for_best_model', type=str, default='runtime')
    p.add_argument('--greater_is_better', action='store_true', default=True)
    # PEFT arguments (names kept identical to backup script)
    p.add_argument("--use_peft", action="store_true", help="Use PEFT for training")
    p.add_argument("--lora_task_type", type=str, default="SEQ_CLS", help="LoRA task type")
    p.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    args = p.parse_args()

    # Configure Weights & Biases project if enabled
    if args.report_to and 'wandb' in args.report_to:
        os.environ['WANDB_PROJECT'] = args.wandb_project

    rows = read_jsonl(Path(args.data))
    train_rows, eval_rows = split(rows, args.eval_ratio)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    train_ds = RLDataset(train_rows, tok, args.max_len)
    eval_ds = RLDataset(eval_rows, tok, args.max_len)

    # Build optional PEFT config using identical parameter names as backup
    peft_config = None
    if getattr(args, "use_peft", False):
        try:
            from peft import LoraConfig
            peft_config = LoraConfig(
                task_type=args.lora_task_type,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        except Exception:
            peft_config = None

    model = RewardModel(args.model, peft_config=peft_config)

    collator = PairwiseCollator(tok)

    targs = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_total_limit=2,
        do_eval=True,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        report_to=[args.report_to] if args.report_to else ['none'],
        run_name=args.run_name,
        gradient_accumulation_steps=args.grad_accum,
        prediction_loss_only=False,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=f"eval_{args.metric_for_best_model}",
        greater_is_better=args.greater_is_better,
        remove_unused_columns=False,
    )

    # Metrics: report accuracy of r(chosen) > r(rejected) and mean margin
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.array(preds).squeeze()
        acc = float((preds > 0).mean())
        margin = float(np.abs(preds).mean())
        # Return both plain and prefixed keys for compatibility across HF versions
        return {
            'accuracy': acc,
            'margin': margin,
            'eval_accuracy': acc,
            'eval_margin': margin,
        }

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.out)

    print('Final eval:')
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()


