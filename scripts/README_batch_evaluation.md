# Batch Summary Evaluation Script

This script evaluates the quality of summaries by assessing how well individual comments are represented in the summary text. It processes all summary files in the `results/summary` directory and generates evaluation results.

## Features

- **Automatic Discovery**: Automatically finds all summary files in the results directory
- **Batch Processing**: Processes multiple datasets and models in one run
- **Configurable**: Supports configuration file and command-line arguments
- **Structured Output**: Maintains the same directory structure (`summary/model/dataset`)
- **Comprehensive Evaluation**: Evaluates comment representation using a 1-5 scoring system
- **Detailed Results**: Includes both individual comment evaluations and summary statistics

## Directory Structure

The script maintains the existing directory structure and adds evaluation files:

```
results/summary/
├── model_name_1/
│   ├── dataset_1/
│   │   ├── summary_dataset_1.json          # Original summary
│   │   └── eva_summary_dataset_1.json      # Evaluation results
│   ├── dataset_2/
│   │   ├── summary_dataset_2.json
│   │   └── eva_summary_dataset_2.json
│   └── batch_summary_report.json
├── model_name_2/
│   └── ...
```

## Evaluation Scoring System

The evaluation uses a 1-5 scale to assess comment representation:

1. **Not represented at all**: The summary ignored the comment entirely
2. **Some relevant material**: Contains some material but missing most content
3. **Substantially represented**: Covers main points but missing important details
4. **Mostly covered**: Covers most information but missing nuance/detail
5. **Fully covered**: Covers all information in the comment

## Usage

### Quick Start

1. **Run with default configuration**:
```bash
cd LLMs-Scalable-Deliberation
python scripts/batch_evaluate_summaries.py
```

2. **Run with custom evaluation model**:
```bash
python scripts/batch_evaluate_summaries.py --evaluation-model gpt-4o
```

3. **Run with custom configuration file**:
```bash
python scripts/batch_evaluate_summaries.py --config config/my_evaluation_config.yaml
```

### Command Line Arguments

- `--config`: Path to configuration file (default: `config/batch_evaluation_config.yaml`)
- `--results-dir`: Path to results directory (overrides config file)
- `--evaluation-model`: Model to use for evaluation (overrides config file)
- `--output-dir`: Output directory for evaluation results (overrides config file)

### Configuration File

Edit `config/batch_evaluation_config.yaml` to customize:

```yaml
# Model configuration for evaluation
evaluation_model:
  name: "gpt-4o-mini"  # Model to use for evaluation
  system_prompt: "You are a helpful assistant"  # System prompt for evaluation
  temperature: 1.0  # Temperature for model generation (0.0 to 2.0)

# Input/Output settings
paths:
  results_directory: "results/summary"  # Directory containing summary files
  output_directory: "results/summary"   # Directory to save evaluation results
  datasets_directory: "datasets"        # Directory containing original datasets

# Evaluation settings
evaluation:
  # Summary types to evaluate (in order of preference)
  summary_types: ["main_points", "topic_modeling", "custom_analysis"]
  
  # Comment sampling settings
  comment_sampling:
    # Maximum comments to evaluate per dataset (null = all comments used in summary)
    max_comments_per_dataset: null
    
    # Sampling strategy: "random", "first_n", "last_n", "stratified"
    sampling_strategy: "first_n"
    
    # Whether to sample from comments used in summary (comment_indices) or all original comments
    sample_from_summary_comments: true
    
    # Random seed for reproducible sampling (only used with "random" strategy)
    random_seed: 42
    
    # For stratified sampling: number of comments per score level (1-5)
    stratified_samples_per_level: 10
  
  # Whether to skip datasets with disabled summary types
  skip_disabled_summaries: true
```

## Output Format

Each evaluation file (`eva_summary_*.json`) contains:

1. **Optimized Metadata**: Consolidated metadata without duplicates
   - `dataset_name`: Name of the dataset
   - `question`: Original question from the dataset
   - `summary_model`: Model used for summary generation
   - `evaluation_model`: Model used for evaluation
   - `num_samples`: Number of samples used for summary
   - `summary_timestamp`: When summary was generated
   - `evaluation_timestamp`: When evaluation was performed
   - `version`: File version

2. **Summary Parameters**: Parameters used for summary generation
   - `summary_types`: Types of summaries generated
   - `custom_system_prompt`: Custom system prompt for summarization
   - `custom_user_prompt`: Custom user prompt for summarization
   - `script_version`: Summarization script version

3. **Evaluation Parameters**: Parameters used for evaluation
   - `evaluation_model`: Model used for evaluation
   - `temperature`: Temperature setting for evaluation
   - `system_prompt`: System prompt for evaluation
   - `sampling_strategy`: Comment sampling strategy used
   - `comments_evaluated`: Number of comments evaluated

4. **Content Sections**:
   - `summaries`: Generated summary texts
   - `comment_indices`: Indices of comments used in summary
   - `evaluation`: Complete evaluation results and statistics

### Detailed Evaluation Information

Each comment evaluation now includes comprehensive information:

- **`evaluation_response`**: Complete model response with full reasoning (not just the final score)

**Important**: The `evaluation_response` field now contains the complete model output, including the full reasoning process that led to the final score. This allows you to understand how the model arrived at its evaluation, not just what the final score was.
- **`extracted_score`**: Numerically extracted score (1-5)
- **`score`**: Backward compatibility field (same as extracted_score)
- **`evaluation_model`**: Model used for this specific comment evaluation

**Note**: This field is particularly useful when you want to track which model evaluated each individual comment, especially if you're comparing results from different evaluation models or running multiple evaluations with different models.
- **`evaluation_details`**: Simplified evaluation metadata
  - `evaluation_timestamp`: When this specific comment was evaluated
3. **Evaluation Info**: Root-level metadata for easy identification
   - `file_type`: Always "evaluation_result"
   - `evaluation_date`: Date when evaluation was performed (YYYY-MM-DD format)

### Example Evaluation Output

```json
{
  "metadata": {
    "dataset_name": "bowling-green",
    "question": "What do you believe should change in Bowling Green/Warren County?",
    "summary_model": "gpt-4o-mini",
    "evaluation_model": "gpt-5-nano",
    "num_samples": 100,
    "summary_timestamp": "2025-01-27T10:00:00.000000",
    "evaluation_timestamp": "2025-01-27T10:30:00.000000",
    "version": "1.0"
  },
  "summary_parameters": {
    "summary_types": { "main_points": true, "topic_modeling": false },
    "custom_system_prompt": "You are an expert data analyst...",
    "custom_user_prompt": "Analyze the sentiment and key themes...",
    "script_version": "1.0"
  },
  "evaluation_parameters": {
    "evaluation_model": "gpt-5-nano",
    "temperature": 1.0,
    "system_prompt": "You are a helpful assistant",
    "sampling_strategy": "first_n",
    "comments_evaluated": 10
  },
  "summaries": { ... },
  "comment_indices": [ ... ],
  "evaluations": {
    "gpt-4o-mini": {
      "evaluation_data": {
        "evaluation_results": [
          {
            "comment_index": 0,
            "comment": "Comment text...",
            "evaluation_response": "Based on my analysis... \\boxed{4}",
            "extracted_score": 4,
            "score": 4,
            "status": "success",
            "evaluation_model": "gpt-4o-mini",
            "evaluation_details": {
              "evaluation_timestamp": "2025-01-27T10:00:00.000000"
            }
          }
        ],
        "statistics": { "average_score": 3.8, ... },
        "evaluation_model": "gpt-4o-mini",
        "total_comments_evaluated": 10
      },
      "evaluation_timestamp": "2025-01-27T10:00:00.000000",
      "model_parameters": { "temperature": 0.7, ... },
      "sampling_info": { "sampling_strategy": "first_n", ... }
    },
    "gpt-5-nano": {
      "evaluation_data": {
        "evaluation_results": [
          {
            "comment_index": 0,
            "comment": "Comment text...",
            "evaluation_response": "Based on my analysis... \\boxed{4}",
            "extracted_score": 4,
            "score": 4,
            "status": "success",
            "evaluation_model": "gpt-5-nano",
            "evaluation_details": {
              "evaluation_timestamp": "2025-01-27T11:00:00.000000"
            }
          }
        ],
        "statistics": { "average_score": 3.9, ... },
        "evaluation_model": "gpt-5-nano",
        "total_comments_evaluated": 10
      },
      "evaluation_timestamp": "2025-01-27T11:00:00.000000",
      "model_parameters": { "temperature": 1.0, ... },
      "sampling_info": { "sampling_strategy": "first_n", ... }
    }
  },
  "evaluation_info": {
    "file_type": "evaluation_result",
    "evaluation_date": "2025-01-27"
  }
}
```

## Requirements

- Python 3.7+
- Required packages: `yaml`, `json`, `pathlib`
- Access to LLM API (OpenAI, etc.)
- Original dataset files in `datasets/` directory
- Summary files in `results/summary/` directory

## Notes

- The script automatically finds summary files by walking through the directory structure
- It loads original datasets to get comments for evaluation
- Evaluation results are saved in the same directory as the original summary
- The script prefers `main_points` summaries, then falls back to `topic_modeling` or `custom_analysis`
- Failed evaluations are logged but don't stop the batch process
- API rate limits should be considered when processing large numbers of comments

## Multiple Evaluation Models

The script now supports running evaluations with different models and storing all results in the same file:

### Multi-Model Evaluation Structure

When you run evaluations with different models, the results are automatically merged:

```json
{
  "evaluations": {
    "gpt-4o-mini": {
      "evaluation_data": { ... },
      "evaluation_timestamp": "2025-01-27T10:00:00.000000",
      "model_parameters": { "temperature": 0.7, ... },
      "sampling_info": { ... }
    },
    "gpt-5-nano": {
      "evaluation_data": { ... },
      "evaluation_timestamp": "2025-01-27T11:00:00.000000",
      "model_parameters": { "temperature": 1.0, ... },
      "sampling_info": { ... }
    }
  },
  "evaluation_info": {
    "file_type": "multi_model_evaluation_result",
    "models_evaluated": ["gpt-4o-mini", "gpt-5-nano"]
  }
}
```

### Benefits of Multi-Model Evaluation

- **Easy Comparison**: Compare results from different models side by side
- **Single File**: All evaluations stored in one place
- **Model Tracking**: Clear identification of which model produced which results
- **Timestamp Tracking**: Know when each model evaluation was performed
- **Parameter Comparison**: Compare different temperature and prompt settings

### How to Use

1. **First run** with one model (e.g., gpt-4o-mini):
   ```bash
   python scripts/batch_evaluate_summaries.py --evaluation-model gpt-4o-mini
   ```

2. **Second run** with another model (e.g., gpt-5-nano):
   ```bash
   python scripts/batch_evaluate_summaries.py --evaluation-model gpt-5-nano
   ```

3. **Results are automatically merged** into the same evaluation file

### Individual Comment Evaluation

Each comment evaluation still includes the model identifier:

- **`evaluation_model`**: Model used for this specific comment evaluation
- **`evaluation_response`**: Complete model reasoning for this comment
- **`extracted_score`**: Score extracted from this model's evaluation

## Data Structure Optimization

The evaluation files are now optimized to eliminate duplicate information and provide a cleaner, more organized structure:

### Before (with duplicates):
- `metadata.model` and `parameters.model` both contained the same model name
- `metadata.num_samples` and `parameters.num_samples` were duplicated
- Information was scattered across multiple sections

### After (optimized):
- **Consolidated metadata**: Single source of truth for dataset and model information
- **Separated parameters**: Clear distinction between summary and evaluation parameters
- **Eliminated redundancy**: No more duplicate fields
- **Better organization**: Logical grouping of related information

### Summary File Structure (Updated):
```json
{
  "metadata": {
    "dataset_name": "bowling-green",
    "question": "What do you believe should change...",
    "summary_model": "gpt-4o-mini",  // 明确标识summary模型
    "num_samples": 100,
    "timestamp": "2025-01-27T10:00:00",
    "version": "1.0"
  },
  "summary_parameters": {              // 专门用于summary的参数
    "summary_types": { "main_points": true },
    "custom_system_prompt": "...",
    "custom_user_prompt": "...",
    "script_version": "1.0"
  }
}
```

## Comment Sampling Control

The script provides flexible control over which comments to evaluate:

### Sampling Strategies

1. **`first_n`** (default): Evaluate the first N comments
2. **`last_n`**: Evaluate the last N comments  
3. **`random`**: Randomly sample N comments (with configurable seed)

### Sampling Source

- **`sample_from_summary_comments: true`** (default): Sample from comments actually used in summary generation (`comment_indices`)
- **`sample_from_summary_comments: false`**: Sample from all available comments in the original dataset

### Configuration Examples

```yaml
# Evaluate only first 50 comments used in summary
comment_sampling:
  max_comments_per_dataset: 50
  sampling_strategy: "first_n"
  sample_from_summary_comments: true

# Randomly sample 100 comments from all available
comment_sampling:
  max_comments_per_dataset: 100
  sampling_strategy: "random"
  sample_from_summary_comments: false
  random_seed: 42

# Evaluate all comments used in summary (default)
comment_sampling:
  max_comments_per_dataset: null  # No limit
  sampling_strategy: "first_n"
  sample_from_summary_comments: true
```

### Benefits

- **Efficiency**: Avoid evaluating all comments when not necessary
- **Consistency**: Sample from the same comments used to generate the summary
- **Flexibility**: Choose sampling strategy based on your needs
- **Reproducibility**: Random sampling with fixed seed for consistent results
- **Cost Control**: Limit API calls by controlling comment count

## Troubleshooting

1. **No summary files found**: Check that the results directory path is correct
2. **Dataset loading errors**: Ensure original dataset files exist in the datasets directory
3. **API errors**: Check API key and rate limits
4. **Memory issues**: Consider limiting the number of comments per dataset in the config
