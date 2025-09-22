# Regression Model Evaluation Scripts

This directory contains scripts for evaluating summary quality using trained DeBERTa regression models.

## Files

- `sbatch_regression_eval.sh` - SLURM batch job script for submitting evaluation jobs
- `../batch_regression_evaluate.py` - Python script for running regression evaluations

## Usage

### 1. Using the SLURM batch script

```bash
# Basic usage
./sbatch_regression_eval.sh [model_name] [comment_number] [eval_model_path]

# Examples
./sbatch_regression_eval.sh gpt-5-nano 10 results/summary_model_for_evaluation/deberta-v3-large
./sbatch_regression_eval.sh gemini-2.5-flash-lite 50 results/summary_model_for_evaluation/deberta-v3-large
```

### 2. Using the Python script directly

```bash
# Basic usage
python scripts/batch_regression_evaluate.py \
    --model-name gpt-5-nano \
    --comment-number 10 \
    --eval-model-path results/summary_model_for_evaluation/deberta-v3-large

# With custom output directory
python scripts/batch_regression_evaluate.py \
    --model-name gpt-5-nano \
    --comment-number 10 \
    --eval-model-path results/summary_model_for_evaluation/deberta-v3-large \
    --output-dir custom_output_dir \
    --max-files 5
```

## Parameters

### Model Names
Available model names: `gpt-5-nano`, `gemini-2.5-flash-lite`, `web-rev-claude-3-7-sonnet-20250219`, `gpt-4o-mini`, `gpt-5`, `gemini-2.5-pro`, `qwen3-32b`, `qwen3-14b`, `qwen3-8b`, `qwen3-4b`, `qwen3-1.7b`, `qwen3-0.6b`

### Comment Numbers
Available comment numbers: `10`, `20`, `30`, `50`, `70`, `90`

### Evaluation Model Path
Path to the trained DeBERTa regression model directory (e.g., `results/summary_model_for_evaluation/deberta-v3-large`)

## Output

The evaluation generates:

1. **JSONL file**: `regression_evaluation_{model_name}_{comment_number}.jsonl`
   - Contains detailed evaluation results for each comment-summary pair
   - Includes regression scores for 4 dimensions: perspective_representation, informativeness, neutrality_balance, policy_approval

2. **Summary JSON file**: `evaluation_summary_{model_name}_{comment_number}.json`
   - Contains statistics and summary information
   - Includes mean, std, min, max for each dimension

## Example Output Structure

### JSONL file entries:
```json
{
  "perspective_representation": 0.75,
  "informativeness": 0.82,
  "neutrality_balance": 0.68,
  "policy_approval": 0.71,
  "file_name": "summary_001.json",
  "comment_index": 0,
  "comment_number": "10",
  "evaluation_timestamp": "2024-01-15T10:30:00"
}
```

### Summary statistics:
```json
{
  "total_files_processed": 10,
  "total_evaluations": 150,
  "successful_evaluations": 148,
  "failed_evaluations": 2,
  "comment_number": "10",
  "model_name": "gpt-5-nano",
  "dimension_statistics": {
    "perspective_representation": {
      "mean": 0.72,
      "std": 0.15,
      "min": 0.45,
      "max": 0.95,
      "count": 148
    }
  }
}
```

## Requirements

- CUDA-capable GPU (recommended)
- Python environment with required packages
- Trained DeBERTa regression model
- Summary files in `results/summary_model_for_evaluation/{model_name}/` directory

## Monitoring

Check job status:
```bash
squeue -u $USER
```

Check logs:
```bash
ls regression_eval_logs/
tail -f regression_eval_logs/regression_eval_{model_name}_{comment_number}_*.out
```
