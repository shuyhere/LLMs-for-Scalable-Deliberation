#!/usr/bin/env python3
"""
Simple inference script for testing a single trained model.
Usage examples:
  python test_single_model.py --model_path outputs/comparison_models/binary_classifier --test_data datasets/sft_dataset/comparison_split/test.jsonl --model_type binary
  python test_single_model.py --model_path outputs/comparison_models/regression --test_data datasets/sft_dataset/comparison_split/test.jsonl --model_type regression
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

try:
    from scipy.stats import spearmanr, pearsonr
except ImportError:
    spearmanr = None
    pearsonr = None

from finetuning.comparison_binary_classifier import ComparisonBinaryClassifier
from finetuning.comparison_regression import ComparisonRegressionModel


def main():
    parser = argparse.ArgumentParser(description="Test a single trained comparison model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model directory")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to the test dataset JSONL file")
    parser.add_argument("--model_type", type=str, choices=["binary", "regression"], required=True,
                       help="Type of model: 'binary' or 'regression'")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results (default: model_type_inference_results.json)")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to test (default: all)")
    
    args = parser.parse_args()
    
    # Set output file
    if args.output_file is None:
        args.output_file = f"{args.model_type}_inference_results.json"
    
    print(f"Testing {args.model_type} model from {args.model_path}")
    print(f"Test data: {args.test_data}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - use the same base model as training
    print(f"Loading {args.model_type} model...")
    if args.model_type == "binary":
        model = ComparisonBinaryClassifier("microsoft/deberta-v3-large")
    else:
        model = ComparisonRegressionModel("microsoft/deberta-v3-large")
    
    # Load model weights - check for safetensors first, then pytorch_model.bin
    safetensors_path = os.path.join(args.model_path, "model.safetensors")
    pytorch_path = os.path.join(args.model_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        print("Loading model from safetensors format...")
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict)
    elif os.path.exists(pytorch_path):
        print("Loading model from pytorch format...")
        model.load_state_dict(torch.load(pytorch_path, map_location=device))
    else:
        print(f"Error: No model file found at {args.model_path}")
        print("Expected either model.safetensors or pytorch_model.bin")
        return
    model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    dataset = load_dataset("json", data_files=args.test_data, split="train")
    
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    
    print(f"Testing on {len(dataset)} examples")
    
    # Prepare data
    test_data = []
    for example in dataset:
        # Extract scores from metadata
        scores = example["metadata"]["comparison_scores"]
        
        # Tokenize the prompt
        tokenized = tokenizer(
            example["prompt"],
            truncation=True,
            padding=False,
            max_length=4096,
            return_tensors="pt"
        )
        
        test_data.append({
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": scores,
            "prompt": example["prompt"][:200] + "..." if len(example["prompt"]) > 200 else example["prompt"],
            "metadata": example["metadata"]
        })
    
    # Run inference
    print("Running inference...")
    results = []
    all_predictions = []
    all_labels = []
    
    dimension_names = ["perspective", "informativeness", "neutrality", "policy"]
    
    with torch.no_grad():
        for i, example in enumerate(test_data):
            # Prepare input
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(device)
            
            # Run model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu().numpy()[0]
            
            # Process predictions
            if args.model_type == "binary":
                probs = torch.sigmoid(torch.tensor(logits)).numpy()
                predictions = (probs > 0.5).astype(int)
                predictions_original = 2 - predictions
            else:
                predictions_original = logits
                probs = logits
            
            # Store results
            all_predictions.append(predictions_original)
            all_labels.append(example["labels"])
            
            result = {
                "example_id": i,
                "logits": logits.tolist(),
                "predictions": predictions_original.tolist(),
                "true_labels": example["labels"],
                "prompt_preview": example["prompt"],
                "triplet_id": example["metadata"]["triplet_id"]
            }
            
            # Add per-dimension details
            for j, dim_name in enumerate(dimension_names):
                result[f"{dim_name}_logit"] = float(logits[j])
                result[f"{dim_name}_prediction"] = float(predictions_original[j])
                result[f"{dim_name}_true_label"] = float(example["labels"][j])
            
            results.append(result)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_data)} examples")
    
    # Compute metrics
    print("Computing metrics...")
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = {}
    
    if args.model_type == "binary":
        # Binary classification metrics
        binary_pred = (all_predictions == 1).astype(int)
        binary_labels = (all_labels == 1).astype(int)
        
        accuracy = accuracy_score(binary_labels.flatten(), binary_pred.flatten())
        f1 = f1_score(binary_labels.flatten(), binary_pred.flatten(), average='weighted')
        metrics["overall_accuracy"] = accuracy
        metrics["overall_f1"] = f1
        
        # Per-dimension metrics
        for i, dim_name in enumerate(dimension_names):
            dim_pred = binary_pred[:, i]
            dim_labels = binary_labels[:, i]
            dim_accuracy = accuracy_score(dim_labels, dim_pred)
            dim_f1 = f1_score(dim_labels, dim_pred, average='weighted', zero_division=0)
            metrics[f"{dim_name}_accuracy"] = dim_accuracy
            metrics[f"{dim_name}_f1"] = dim_f1
    else:
        # Regression metrics
        mse = mean_squared_error(all_labels.flatten(), all_predictions.flatten())
        mae = mean_absolute_error(all_labels.flatten(), all_predictions.flatten())
        rmse = np.sqrt(mse)
        metrics["overall_mse"] = mse
        metrics["overall_mae"] = mae
        metrics["overall_rmse"] = rmse
        
        # Per-dimension metrics
        for i, dim_name in enumerate(dimension_names):
            dim_pred = all_predictions[:, i]
            dim_labels = all_labels[:, i]
            dim_mse = mean_squared_error(dim_labels, dim_pred)
            dim_mae = mean_absolute_error(dim_labels, dim_pred)
            dim_rmse = np.sqrt(dim_mse)
            metrics[f"{dim_name}_mse"] = dim_mse
            metrics[f"{dim_name}_mae"] = dim_mae
            metrics[f"{dim_name}_rmse"] = dim_rmse
    
    # Correlations
    if spearmanr is not None and pearsonr is not None:
        try:
            spearman_corr, _ = spearmanr(all_predictions.flatten(), all_labels.flatten())
            pearson_corr, _ = pearsonr(all_predictions.flatten(), all_labels.flatten())
            metrics["spearman_correlation"] = spearman_corr if not np.isnan(spearman_corr) else 0.0
            metrics["pearson_correlation"] = pearson_corr if not np.isnan(pearson_corr) else 0.0
        except Exception:
            metrics["spearman_correlation"] = 0.0
            metrics["pearson_correlation"] = 0.0
    
    # Classification accuracy (for both model types)
    binary_pred = (all_predictions > 1.5).astype(int) + 1
    binary_labels = all_labels.astype(int)
    accuracy = np.mean(binary_pred.flatten() == binary_labels.flatten())
    metrics["classification_accuracy"] = accuracy
    
    # Save results
    output_data = {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "test_data_path": args.test_data,
        "num_examples": len(test_data),
        "metrics": metrics,
        "results": results
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print(f"INFERENCE RESULTS - {args.model_type.upper()} MODEL")
    print("="*60)
    
    if args.model_type == "binary":
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Overall F1 Score: {metrics['overall_f1']:.4f}")
        print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")
        
        print("\nPer-dimension Accuracy:")
        for dim in dimension_names:
            print(f"  {dim.capitalize()}: {metrics[f'{dim}_accuracy']:.4f}")
    else:
        print(f"Overall MSE: {metrics['overall_mse']:.4f}")
        print(f"Overall MAE: {metrics['overall_mae']:.4f}")
        print(f"Overall RMSE: {metrics['overall_rmse']:.4f}")
        print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")
        print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}")
        print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
        
        print("\nPer-dimension RMSE:")
        for dim in dimension_names:
            print(f"  {dim.capitalize()}: {metrics[f'{dim}_rmse']:.4f}")
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
