# Batch Summarization Script

Batch summarization script that supports specifying datasets, models, and sample sizes, and saves results to JSON files.

### 1. Basic usage

```bash
# Process a single dataset with default parameters
python scripts/batch_summarization.py

# Process a specific dataset
python scripts/batch_summarization.py --datasets datasets/protest.json

# Process multiple datasets
python scripts/batch_summarization.py --datasets datasets/protest.json datasets/gun_use.json

# Specify model and sample size
python scripts/batch_summarization.py --model gpt-4o --num-samples 200

# Specify output directory
python scripts/batch_summarization.py --output-dir results/my_summaries
```

### 2. Advanced usage

```bash
# Use custom prompts
python scripts/batch_summarization.py \
  --custom-system-prompt "You are an expert social scientist." \
  --custom-user-prompt "Analyze the political views in these comments: {comments}"

# Process all datasets, using all comments
python scripts/batch_summarization.py \
  --datasets datasets/*.json \
  --num-samples 0
```

## Output file structure

### Single dataset result file

```json
{
  "metadata": {
    "dataset_name": "protest",
    "question": "Do you support the use of military force...",
    "model": "gpt-4o-mini",
    "num_samples": 100,
    "timestamp": "2024-01-15T10:30:00",
    "version": "1.0"
  },
  "parameters": {
    "model": "gpt-4o-mini",
    "num_samples": 100,
    "custom_system_prompt": "...",
    "custom_user_prompt": "...",
    "script_version": "1.0"
  },
  "summaries": {
    "topic_modeling": "Based on the comments, the main topics are...",
    "main_points": "The key points from the discussion include...",
    "custom_analysis": "Analysis of sentiment and themes reveals..."
  },
  "comments_used": [
    "Peaceful protest is a freedom of expression...",
    "The use of force against civilians...",
    "..."
  ],
  "statistics": {
    "total_comments_in_dataset": 100,
    "comments_used_for_summary": 100,
    "topic_summary_length": 245,
    "main_summary_length": 312,
    "custom_summary_length": 298
  }
}
```

### Batch processing report file

```json
{
  "metadata": {
    "total_datasets_processed": 4,
    "model_used": "gpt-4o-mini",
    "timestamp": "2024-01-15T10:30:00",
    "parameters": {...}
  },
  "datasets_processed": ["protest", "gun_use", "operation", "bowling-green"],
  "results": [...]
}
```

## Configuration file

You can use `config/batch_summarization_config.yaml` to configure default parameters:

```yaml
# Model configuration
model:
  name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 2000

# Dataset configuration
datasets:
  - "datasets/protest.json"
  - "datasets/gun_use.json"

# Processing configuration
processing:
  num_samples: 100
  parallel: false
```

## Example scenarios

### 1. Quick test

```bash
# Use a small number of samples for quick testing
python scripts/batch_summarization.py --num-samples 10
```

### 2. Complete analysis

```bash
# Process all datasets, using all comments
python scripts/batch_summarization.py \
  --datasets datasets/*.json \
  --num-samples 0 \
  --model gpt-4o
```

### 3. Custom analysis

```bash
# Use specific analysis prompts
python scripts/batch_summarization.py \
  --custom-system-prompt "You are a political analyst." \
  --custom-user-prompt "Analyze the political implications: {comments}"
```

## Output directory structure

```
results/
└── summary/
    ├── summary_protest_20240115_103000.json
    ├── summary_gun_use_20240115_103000.json
    ├── summary_operation_20240115_103000.json
    ├── summary_bowling-green_20240115_103000.json
    └── batch_summary_report_20240115_103000.json
```
