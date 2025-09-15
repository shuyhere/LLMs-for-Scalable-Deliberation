# Comparison Models Inference Scripts

This directory contains scripts for testing trained comparison models on test datasets.

## Files

- `test_comparison_models.py`: Complete inference script with detailed output
- `test_single_model.py`: Simplified script for testing a single model
- `run_inference.sh`: Bash script to run inference on both model types
- `README.md`: This documentation

## Usage

### 1. Test a Single Model (Recommended)

```bash
# Test binary classifier
python scripts/inference/test_single_model.py \
    --model_path outputs/comparison_models/binary_classifier \
    --test_data datasets/sft_dataset/comparison_split/test.jsonl \
    --model_type binary

# Test regression model
python scripts/inference/test_single_model.py \
    --model_path outputs/comparison_models/regression \
    --test_data datasets/sft_dataset/comparison_split/test.jsonl \
    --model_type regression
```

### 2. Test All Models

```bash
# Make script executable
chmod +x scripts/inference/run_inference.sh

# Run inference on both models
bash scripts/inference/run_inference.sh
```

### 3. Advanced Usage

```bash
# Test with custom output file and limited examples
python scripts/inference/test_single_model.py \
    --model_path outputs/comparison_models/binary_classifier \
    --test_data datasets/sft_dataset/comparison_split/test.jsonl \
    --model_type binary \
    --output_file my_results.json \
    --max_examples 100
```

## Output Format

The inference results include:

### For Each Example:
- `example_id`: Index of the example
- `logits`: Raw model outputs for each dimension
- `predictions`: Final predictions (1 or 2 for each dimension)
- `true_labels`: Ground truth labels
- `prompt_preview`: First 200 characters of the prompt
- `triplet_id`: Original triplet identifier
- Per-dimension details:
  - `{dimension}_logit`: Raw logit for that dimension
  - `{dimension}_prediction`: Final prediction (1 or 2)
  - `{dimension}_true_label`: Ground truth label

### Metrics:
- **Binary Classifier**: Overall accuracy, F1 score, per-dimension accuracy
- **Regression**: MSE, MAE, RMSE, Spearman/Pearson correlations, per-dimension metrics
- **Both**: Classification accuracy (treating predictions as binary choice)

## Example Output

```json
{
  "model_type": "binary",
  "model_path": "outputs/comparison_models/binary_classifier",
  "test_data_path": "datasets/sft_dataset/comparison_split/test.jsonl",
  "num_examples": 450,
  "metrics": {
    "overall_accuracy": 0.7422,
    "overall_f1": 0.7356,
    "classification_accuracy": 0.7422,
    "perspective_accuracy": 0.7511,
    "informativeness_accuracy": 0.7333,
    "neutrality_accuracy": 0.7400,
    "policy_accuracy": 0.7444
  },
  "results": [
    {
      "example_id": 0,
      "logits": [0.234, -0.456, 0.789, -0.123],
      "predictions": [1, 2, 1, 2],
      "true_labels": [1, 2, 1, 2],
      "perspective_logit": 0.234,
      "perspective_prediction": 1.0,
      "perspective_true_label": 1.0,
      ...
    }
  ]
}
```

## Requirements

- PyTorch
- Transformers
- Datasets
- Scikit-learn
- NumPy
- SciPy (optional, for correlations)
- jq (optional, for pretty printing JSON in bash script)
