#!/usr/bin/env python3
"""
Train Reward Model with Filtered Dataset

This script trains a reward model using the filtered dataset.
Based on the sbatch script configuration.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Configuration from sbatch script
    PROJECT_DIR = "/ibex/project/c2328/LLMs-Scalable-Deliberation"
    MODEL_NAME = "/ibex/project/c2328/LLMs-Scalable-Deliberation/outputs/reward_models/informativeness_reward_model"
    DATASET_DIR = f"{PROJECT_DIR}/datasets/rl_datasets/trl_format"
    OUTPUT_DIR = f"{PROJECT_DIR}/outputs/reward_models_filtered"
    
    # Dataset paths
    dimension = "informativeness"
    dataset_path = f"{DATASET_DIR}/{dimension}_trl_dataset.jsonl"
    train_path = f"{DATASET_DIR}/{dimension}_trl_dataset/filtered_dataset.jsonl"
    test_path = f"{DATASET_DIR}/{dimension}_trl_dataset/test.jsonl"
    output_path = f"{OUTPUT_DIR}/{dimension}_reward_model"
    
    # Training parameters
    params = {
        "--model_name_or_path": MODEL_NAME,
        "--dataset_path": dataset_path,
        "--train_path": train_path,
        "--test_path": test_path,
        "--eval_split": "0.1",
        "--output_dir": output_path,
        "--per_device_train_batch_size": "4",
        "--per_device_eval_batch_size": "4",
        "--max_grad_norm": "10.0",
        "--gradient_accumulation_steps": "4",
        "--num_train_epochs": "3",
        "--learning_rate": "4e-5",
        "--weight_decay": "0.01",
        "--lr_scheduler_type": "cosine",
        "--warmup_ratio": "0.1",
        "--eval_strategy": "steps",
        "--eval_steps": "10",
        "--logging_steps": "1",
        "--save_steps": "100",
        "--save_total_limit": "3",
        "--max_length": "8192",
        "--bf16": "",
        "--seed": "42",
        "--report_to": "wandb",
        "--wandb_project": "reward-modeling-Qwen3-4b-filtered-v0",
        "--run_name": f"rw_{MODEL_NAME.split('/')[-1]}_S42_{dimension}"
    }
    
    # Build command
    cmd = [
        "python3", 
        f"{PROJECT_DIR}/src/finetuning/reward_modeling.py"
    ]
    
    for key, value in params.items():
        cmd.append(key)
        if value:  # Only add value if it's not empty
            cmd.append(value)
    
    # Set environment variables
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
        "CUDA_LAUNCH_BLOCKING": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"
    }
    
    print("Training Reward Model with the following command:")
    print(" ".join(cmd))
    print("\nEnvironment variables:")
    for key, value in env.items():
        print(f"export {key}={value}")
    
    print(f"\nDataset paths:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    print(f"  Output: {output_path}")
    
    # Ask for confirmation
    response = input("\nDo you want to run this command? (y/N): ")
    if response.lower() in ['y', 'yes']:
        try:
            # Run the command
            result = subprocess.run(cmd, env={**os.environ, **env}, check=True)
            print("Training completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error: {e}")
            sys.exit(1)
    else:
        print("Training cancelled.")

if __name__ == "__main__":
    main()
