#!/usr/bin/env python3
"""
Inference script for testing trained comparison models on test dataset.
Supports both binary classification and regression models.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

try:
    from scipy.stats import spearmanr, pearsonr
except ImportError:
    spearmanr = None
    pearsonr = None


class ComparisonBinaryClassifier(torch.nn.Module):
    """Binary classifier for comparison tasks - predicts which summary is better for each dimension."""

    def __init__(self, base_model_name: str, num_dimensions: int = 4):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.config = self.base_model.config
        self.dropout = torch.nn.Dropout(0.1)
        self.num_dimensions = num_dimensions
        
        # Binary classification head for each dimension
        self.classifier = torch.nn.Linear(self.base_model.config.hidden_size, num_dimensions)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Get logits for each dimension (binary classification)
        logits = self.classifier(pooled_output)
        
        return {
            "logits": logits,
        }


class ComparisonRegressionModel(torch.nn.Module):
    """Regression model for comparison tasks - predicts comparison scores for each dimension."""

    def __init__(self, base_model_name: str, num_dimensions: int = 4):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.config = self.base_model.config
        self.dropout = torch.nn.Dropout(0.1)
        self.num_dimensions = num_dimensions
        
        # Regression head for each dimension
        self.regression_head = torch.nn.Linear(self.base_model.config.hidden_size, num_dimensions)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each dimension
        logits = self.regression_head(pooled_output)
        
        return {
            "logits": logits,
        }


def load_test_dataset(dataset_path: str, tokenizer) -> List[Dict[str, Any]]:
    """Load and preprocess the test dataset."""
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Extract original data for comparison
    original_data = []
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
        
        original_data.append({
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": scores,
            "prompt": example["prompt"],
            "completion": example["completion"],
            "metadata": example["metadata"]
        })
    
    return original_data


def run_inference(model, test_data: List[Dict[str, Any]], device: str, model_type: str) -> List[Dict[str, Any]]:
    """Run inference on test data."""
    model.eval()
    results = []
    
    dimension_names = ["perspective", "informativeness", "neutrality", "policy"]
    
    with torch.no_grad():
        for i, example in enumerate(test_data):
            # Prepare input
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(device)
            
            # Run model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu().numpy()[0]  # Remove batch dimension
            
            # Process predictions based on model type
            if model_type == "binary":
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(torch.tensor(logits)).numpy()
                predictions = (probs > 0.5).astype(int)
                # Convert back to original scale (0->2, 1->1)
                predictions_original = 2 - predictions
            else:  # regression
                predictions_original = logits
                probs = logits
            
            # Create result entry
            result = {
                "example_id": i,
                "logits": logits.tolist(),
                "probabilities": probs.tolist() if model_type == "binary" else None,
                "predictions": predictions_original.tolist(),
                "true_labels": example["labels"],
                "prompt": example["prompt"],
                "completion": example["completion"],
                "metadata": example["metadata"]
            }
            
            # Add per-dimension details
            for j, dim_name in enumerate(dimension_names):
                result[f"{dim_name}_logit"] = float(logits[j])
                result[f"{dim_name}_prediction"] = float(predictions_original[j])
                result[f"{dim_name}_true_label"] = float(example["labels"][j])
                if model_type == "binary":
                    result[f"{dim_name}_probability"] = float(probs[j])
            
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_data)} examples")
    
    return results


def compute_metrics(results: List[Dict[str, Any]], model_type: str) -> Dict[str, float]:
    """Compute evaluation metrics."""
    dimension_names = ["perspective", "informativeness", "neutrality", "policy"]
    
    # Extract predictions and labels
    all_predictions = []
    all_labels = []
    
    for result in results:
        all_predictions.append(result["predictions"])
        all_labels.append(result["true_labels"])
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = {}
    
    if model_type == "binary":
        # Binary classification metrics
        # Convert to binary for accuracy calculation
        binary_pred = (all_predictions == 1).astype(int)
        binary_labels = (all_labels == 1).astype(int)
        
        # Overall metrics
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
    
    else:  # regression
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
    
    # Accuracy for both model types (treating as classification)
    binary_pred = (all_predictions > 1.5).astype(int) + 1
    binary_labels = all_labels.astype(int)
    accuracy = np.mean(binary_pred.flatten() == binary_labels.flatten())
    metrics["classification_accuracy"] = accuracy
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test trained comparison models on test dataset")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model directory")
    parser.add_argument("--test_data_path", type=str, required=True,
                       help="Path to the test dataset JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for inference results")
    parser.add_argument("--model_type", type=str, choices=["binary", "regression"], required=True,
                       help="Type of model: 'binary' or 'regression'")
    parser.add_argument("--base_model_name", type=str, default="microsoft/deberta-v3-base",
                       help="Base model name for loading the model architecture")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for inference (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - use the same base model as training
    print(f"Loading {args.model_type} model from {args.model_path}...")
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
    
    # Load test data
    print(f"Loading test data from {args.test_data_path}...")
    test_data = load_test_dataset(args.test_data_path, tokenizer)
    print(f"Loaded {len(test_data)} test examples")
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, test_data, device, args.model_type)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results, args.model_type)
    
    # Save results
    results_file = os.path.join(args.output_dir, f"{args.model_type}_inference_results.json")
    metrics_file = os.path.join(args.output_dir, f"{args.model_type}_metrics.json")
    
    print(f"Saving results to {results_file}...")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saving metrics to {metrics_file}...")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print(f"INFERENCE RESULTS SUMMARY - {args.model_type.upper()} MODEL")
    print("="*50)
    
    if args.model_type == "binary":
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Overall F1 Score: {metrics['overall_f1']:.4f}")
        print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")
        
        print("\nPer-dimension Accuracy:")
        for dim in ["perspective", "informativeness", "neutrality", "policy"]:
            print(f"  {dim.capitalize()}: {metrics[f'{dim}_accuracy']:.4f}")
    else:
        print(f"Overall MSE: {metrics['overall_mse']:.4f}")
        print(f"Overall MAE: {metrics['overall_mae']:.4f}")
        print(f"Overall RMSE: {metrics['overall_rmse']:.4f}")
        print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")
        print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}")
        print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
        
        print("\nPer-dimension RMSE:")
        for dim in ["perspective", "informativeness", "neutrality", "policy"]:
            print(f"  {dim.capitalize()}: {metrics[f'{dim}_rmse']:.4f}")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
