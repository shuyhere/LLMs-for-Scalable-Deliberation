#!/usr/bin/env python3
"""
Reward Model Training Script - Based on TRL Official Example
https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py

Usage:
# Regular training
python src/finetuning/reward_modeling.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_path datasets/rl_datasets/trl_format/perspective_trl_dataset.jsonl \
    --output_dir outputs/reward_models/perspective \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_grad_norm 10.0 \
    --num_train_epochs 1 \
    --learning_rate 2.0e-5 \
    --eval_strategy steps \
    --eval_steps 10 \
    --logging_steps 1 \
    --max_length 8192 \
    --seed 42 \
    --report_to wandb \
    --run_name perspective_reward_model

# Cross-validation training
python src/finetuning/reward_modeling.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_path datasets/rl_datasets/trl_format/perspective_trl_dataset.jsonl \
    --train_path datasets/rl_datasets/trl_format/perspective_trl_dataset/train.jsonl \
    --test_path datasets/rl_datasets/trl_format/perspective_trl_dataset/test.jsonl \
    --output_dir outputs/reward_models/perspective_cv \
    --crossvalidation \
    --cv_folds 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_grad_norm 10.0 \
    --num_train_epochs 1 \
    --learning_rate 2.0e-5 \
    --max_length 8192 \
    --seed 42 \
    --report_to wandb \
    --run_name perspective_reward_model_cv 
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sklearn.model_selection import KFold

import torch
from accelerate import logging
from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    HfArgumentParser,
    TrainingArguments
)

from trl import (
    RewardConfig,
    RewardTrainer,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

# Set environment variables to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

logger = logging.get_logger(__name__)

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*Qwen2TokenizerFast.*")
warnings.filterwarnings("ignore", message=".*using the `__call__` method is faster.*")
warnings.filterwarnings("ignore", message=".*huggingface/tokenizers.*")


def load_custom_dataset(dataset_path: str) -> Dataset:
    """Load custom dataset from JSONL format."""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    # Validate required fields
                    if 'chosen' in item and 'rejected' in item:
                        data.append(item)
                    else:
                        logger.warning(f"Skipping invalid item: missing 'chosen' or 'rejected' field")
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue
    
    if len(data) == 0:
        raise ValueError(f"No valid data found in {dataset_path}")
    
    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return Dataset.from_list(data)

def run_crossvalidation(trainer, dataset, cv_folds: int, output_dir: str, args):
    """Run cross-validation training and evaluation."""
    logger.info(f"Starting {cv_folds}-fold cross-validation")
    
    # Convert dataset to list for easier indexing
    data_list = [dataset[i] for i in range(len(dataset))]
    data_indices = np.arange(len(data_list))
    
    # Initialize KFold
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
    
    # Store results for each fold
    fold_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(data_indices)):
        logger.info(f"Training fold {fold_idx + 1}/{cv_folds}")
        
        # Create fold-specific datasets
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Update trainer with fold-specific data
        trainer.train_dataset = train_dataset
        trainer.eval_dataset = val_dataset
        
        # Update training arguments for this fold
        training_args = trainer.args
        training_args.output_dir = fold_output_dir
        training_args.run_name = f"{args.run_name}_fold_{fold_idx + 1}" if args.run_name else f"reward_model_fold_{fold_idx + 1}"
        
        # Train the model for this fold
        logger.info(f"Training fold {fold_idx + 1} with {len(train_dataset)} train, {len(val_dataset)} val examples")
        trainer.train()
        
        # Evaluate on validation set
        logger.info(f"Evaluating fold {fold_idx + 1}")
        val_metrics = trainer.evaluate()
        
        # Save fold results
        fold_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'metrics': val_metrics
        }
        fold_results.append(fold_result)
        
        # Save fold metrics
        trainer.log_metrics(f"eval_fold_{fold_idx + 1}", val_metrics)
        trainer.save_metrics(f"eval_fold_{fold_idx + 1}", val_metrics)
        
        logger.info(f"Fold {fold_idx + 1} metrics: {val_metrics}")
        
        # Save model for this fold
        trainer.save_model()
        logger.info(f"Saved fold {fold_idx + 1} model to {fold_output_dir}")
    
    # Calculate and save cross-validation summary
    cv_summary = calculate_cv_summary(fold_results)
    logger.info(f"Cross-validation summary: {cv_summary}")
    
    # Save CV summary
    summary_path = os.path.join(output_dir, "cv_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    logger.info(f"Cross-validation completed. Summary saved to {summary_path}")
    return cv_summary

def calculate_cv_summary(fold_results: List[Dict]) -> Dict:
    """Calculate cross-validation summary statistics."""
    metrics = {}
    
    # Get all metric names from the first fold
    if fold_results:
        metric_names = list(fold_results[0]['metrics'].keys())
        
        for metric_name in metric_names:
            values = [fold['metrics'][metric_name] for fold in fold_results]
            metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    summary = {
        'num_folds': len(fold_results),
        'metrics': metrics,
        'fold_details': fold_results
    }
    
    return summary


def main():
    # Initialize accelerate state for logging
    _ = PartialState()
    
    parser = argparse.ArgumentParser(description="Train a reward model using TRL official approach")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B",
                       help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--trust_remote_code", action="store_true",
                       help="Trust remote code when loading model")
    parser.add_argument("--dtype", type=str, default="auto",
                       help="Data type for model weights")
    parser.add_argument("--model_revision", type=str, default="main",
                       help="Model revision")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to a dataset JSONL file (used if train/test not provided)")
    parser.add_argument("--train_path", type=str, default=None,
                       help="Optional path to train JSONL (overrides dataset_path train)")
    parser.add_argument("--test_path", type=str, default=None,
                       help="Optional path to test JSONL (used only for final evaluation)")
    parser.add_argument("--dataset_train_split", type=str, default="train",
                       help="Name of the train split")
    parser.add_argument("--dataset_test_split", type=str, default="test", 
                       help="Name of the test split")
    parser.add_argument("--eval_split", type=float, default=0.1,
                       help="Fraction of data to use for evaluation")
    parser.add_argument("--crossvalidation", action="store_true",
                       help="Enable cross-validation mode")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="Number of folds for cross-validation (default: 5)")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for the model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="Batch size per device for evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1.0e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       choices=[
                           "linear",
                           "cosine",
                           "cosine_with_restarts",
                           "polynomial",
                           "constant",
                           "constant_with_warmup",
                       ],
                       help="Learning rate scheduler type")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                       choices=["no", "steps", "epoch"],
                       help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="Number of steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Number of steps between saves")
    parser.add_argument("--save_total_limit", type=int, default=1,
                       help="Maximum number of checkpoints to keep")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Number of steps between logging")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of steps to accumulate gradients")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                       help="Use fp16 mixed precision training")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bf16 mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--remove_unused_columns", action="store_true",
                       help="Remove unused columns from dataset")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push model to hub after training")
    parser.add_argument("--hub_model_id", type=str, default=None,
                       help="Hub model ID for pushing")
    parser.add_argument("--report_to", type=str, default="wandb",
                       help="Reporting tool (wandb, tensorboard, etc.)")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for logging")
    parser.add_argument("--wandb_project", type=str, default="reward-modeling",
                       help="Weights & Biases project name")
    
    # PEFT arguments
    parser.add_argument("--use_peft", action="store_true",
                       help="Use PEFT for training")
    parser.add_argument("--lora_task_type", type=str, default="SEQ_CLS",
                       help="LoRA task type")
    parser.add_argument("--lora_r", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Quantization arguments
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit quantization")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16",
                       help="4-bit quantization compute dtype")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                       help="4-bit quantization type")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true",
                       help="Use double quantization for 4-bit")
    
    # Best model saving arguments
    parser.add_argument("--load_best_model_at_end", action="store_true",
                       help="Load the best model at the end of training")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_accuracy",
                       help="Metric to use for determining the best model")
    parser.add_argument("--greater_is_better", action="store_true", default=True,
                       help="Whether higher metric values are better")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)

    # Ensure Weights & Biases project is set if reporting to wandb
    if args.report_to and "wandb" in args.report_to:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    ################
    # Model & Tokenizer
    ################
    logger.info(f"Loading model: {args.model_name_or_path}")
    
    # Model kwargs
    dtype = args.dtype if args.dtype in ["auto", None] else getattr(torch, args.dtype)
    model_kwargs = dict(
        revision=args.model_revision,
        use_cache=False if args.gradient_checkpointing else True,
        dtype=dtype,
    )
    
    # Quantization config - simplified approach
    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            if args.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                )
            elif args.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            logger.warning("BitsAndBytesConfig not available. Skipping quantization.")
            quantization_config = None
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=args.trust_remote_code, 
        use_fast=True
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        num_labels=1, 
        trust_remote_code=args.trust_remote_code, 
        **model_kwargs
    )
    
    # Align padding tokens between tokenizer and model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Setup chat format if needed
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
    
    # PEFT warning
    if args.use_peft and args.lora_task_type != "SEQ_CLS":
        logger.warning(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. "
            "This will lead to silent bugs. Make sure to pass --lora_task_type SEQ_CLS "
            "when using this script with PEFT."
        )
    
    ##############
    # Load dataset
    ##############
    # Load train/eval datasets
    ##############
    train_dataset = None
    eval_dataset = None
    test_dataset = None
    full_dataset = None  # For cross-validation
    
    if args.train_path is not None:
        # Train from explicit train_path; create training-time eval via eval_split from the same training data (SFT-like)
        logger.info(f"Loading train dataset from {args.train_path}")
        base_train = load_custom_dataset(args.train_path)
        
        if args.crossvalidation:
            # For cross-validation, use the full dataset
            full_dataset = base_train
            train_dataset = base_train  # Also set train_dataset for trainer initialization
            logger.info(f"Using full dataset for cross-validation: {len(full_dataset)} examples")
        else:
            if args.eval_split > 0:
                split = base_train.train_test_split(test_size=args.eval_split, seed=args.seed)
                train_dataset = split['train']
                eval_dataset = split['test']
                logger.info(f"Split train: {len(train_dataset)} train, {len(eval_dataset)} eval (eval_split={args.eval_split})")
            else:
                train_dataset = base_train
                eval_dataset = None
                logger.info(f"Using full train dataset without eval split: {len(train_dataset)} examples")
        
        if args.test_path is not None:
            logger.info(f"Loading held-out test dataset from {args.test_path}")
            test_dataset = load_custom_dataset(args.test_path)
    else:
        # Fallback: single dataset_path; create eval via eval_split from it if requested
        logger.info(f"Loading dataset from {args.dataset_path}")
        dataset = load_custom_dataset(args.dataset_path)
        
        if args.crossvalidation:
            # For cross-validation, use the full dataset
            full_dataset = dataset
            train_dataset = dataset  # Also set train_dataset for trainer initialization
            logger.info(f"Using full dataset for cross-validation: {len(full_dataset)} examples")
        else:
            if args.eval_split > 0:
                dataset = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
                train_dataset = dataset['train']
                eval_dataset = dataset['test']
                logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval (eval_split={args.eval_split})")
            else:
                train_dataset = dataset
                eval_dataset = None
                logger.info(f"Using full dataset for training: {len(train_dataset)} examples")
    
    ##############
    # Training Config
    ##############
    training_args = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=args.remove_unused_columns,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to=args.report_to,
        run_name=args.run_name or f"reward_model_{Path(args.dataset_path).stem}_{Path(args.model_name_or_path).name}",
        # TRL specific settings
        center_rewards_coefficient=0.0,
        disable_dropout=True,
        dataset_num_proc=1,
        # Best model saving
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )
    
    ##########
    # Training
    ##########
    logger.info("Starting training...")
    
    # PEFT config - simplified approach
    peft_config = None
    if args.use_peft:
        try:
            from peft import LoraConfig
            peft_config = LoraConfig(
                task_type=args.lora_task_type,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
        except ImportError:
            logger.warning("PEFT not available. Skipping PEFT configuration.")
            peft_config = None
    
    # Train the model
    if args.crossvalidation:
        # Run cross-validation - create trainer with train_dataset and disable eval during CV
        logger.info("Running cross-validation training...")
        
        # Create a copy of training_args for cross-validation with eval disabled
        cv_training_args = RewardConfig(
            output_dir=training_args.output_dir,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            warmup_ratio=training_args.warmup_ratio,
            lr_scheduler_type=training_args.lr_scheduler_type,
            max_grad_norm=training_args.max_grad_norm,
            max_length=training_args.max_length,
            gradient_checkpointing=training_args.gradient_checkpointing,
            eval_strategy="no",  # Disable eval during cross-validation setup
            eval_steps=training_args.eval_steps,
            save_steps=training_args.save_steps,
            save_total_limit=training_args.save_total_limit,
            logging_steps=training_args.logging_steps,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            seed=training_args.seed,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            dataloader_num_workers=training_args.dataloader_num_workers,
            remove_unused_columns=training_args.remove_unused_columns,
            push_to_hub=training_args.push_to_hub,
            hub_model_id=training_args.hub_model_id,
            report_to=training_args.report_to,
            run_name=training_args.run_name,
            load_best_model_at_end=training_args.load_best_model_at_end,
            metric_for_best_model=training_args.metric_for_best_model,
            greater_is_better=training_args.greater_is_better,
        )
        
        trainer = RewardTrainer(
            model=model,
            processing_class=tokenizer,
            args=cv_training_args,
            train_dataset=train_dataset,  # Use train_dataset instead of full_dataset
            eval_dataset=None,
            peft_config=peft_config,
        )
        cv_summary = run_crossvalidation(trainer, full_dataset, args.cv_folds, args.output_dir, args)
        logger.info("Cross-validation training completed!")
    else:
        # Regular training - create trainer with actual datasets
        trainer = RewardTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
        # Regular training
        trainer.train()
        
        ############################
        # Save model and evaluate
        ############################
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Evaluate if eval dataset exists
        if eval_dataset is not None and args.eval_strategy != "no":
            logger.info("Evaluating model...")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            logger.info(f"Evaluation metrics: {metrics}")
        
        # Evaluate explicitly on provided test set if given (held-out, not used during training-time eval)
        if test_dataset is not None:
            logger.info("Evaluating model on provided test set...")
            # Use the existing trainer's eval_dataset temporarily, then evaluate on test_dataset
            original_eval_dataset = trainer.eval_dataset
            trainer.eval_dataset = test_dataset
            test_metrics = trainer.evaluate()
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
            logger.info(f"Test metrics: {test_metrics}")
            # Restore original eval_dataset
            trainer.eval_dataset = original_eval_dataset
    
    # Push to hub if requested
    if args.push_to_hub:
        logger.info("Pushing model to hub...")
        trainer.push_to_hub(dataset_name=Path(args.dataset_path).stem)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
