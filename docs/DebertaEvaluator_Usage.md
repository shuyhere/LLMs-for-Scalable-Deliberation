# DebertaEvaluator Usage Guide

The `DebertaEvaluator` class allows you to evaluate comment-summary pairs using a trained DeBERTa regression model that predicts 4 dimensions: perspective_representation, informativeness, neutrality_balance, and policy_approval.

## Features

- Load trained DeBERTa models from checkpoint directories
- Evaluate single comment-summary pairs or batches
- Process data from JSONL files in the same format as training data
- Calculate comprehensive evaluation statistics
- Support for both pytorch_model.bin and model.safetensors formats

## Installation Requirements

```bash
pip install torch transformers numpy
# Optional: for safetensors format support
pip install safetensors
```

## Basic Usage

### 1. Initialize the Evaluator

```python
from src.llm_evaluation.evaluator import DebertaEvaluator

# Load a trained model
evaluator = DebertaEvaluator(
    model_path="/path/to/trained/deberta/model",
    device="cuda"  # or "cpu"
)
```

### 2. Single Evaluation

```python
# Evaluate a single comment-summary pair
result = evaluator.evaluate_single(
    question="What are your thoughts on AI development?",
    comment="I believe AI will revolutionize healthcare and education.",
    summary="The discussion covered various perspectives on AI development...",
    max_length=2048
)

print("Predictions:")
for key, value in result['predictions'].items():
    print(f"  {key}: {value:.4f}")
```

### 3. Batch Evaluation

```python
# Evaluate multiple pairs
data = [
    {
        "question": "What are your thoughts on AI development?",
        "comment": "I believe AI will revolutionize healthcare and education.",
        "summary": "The discussion covered various perspectives on AI development..."
    },
    {
        "question": "What is your view on climate change policies?",
        "comment": "We need immediate action on carbon emissions reduction.",
        "summary": "Participants discussed various climate policy approaches..."
    }
]

results = evaluator.evaluate_batch(data)
```

### 4. JSONL File Evaluation

```python
# Evaluate from a JSONL file (same format as training data)
results = evaluator.evaluate_from_jsonl("/path/to/data.jsonl")
```

### 5. Get Statistics

```python
# Calculate evaluation statistics
stats = evaluator.get_evaluation_statistics(results)

print(f"Total items: {stats['total_items']}")
print(f"Successful evaluations: {stats['successful_evaluations']}")
print("Average scores:")
for key, value in stats['average_scores'].items():
    print(f"  {key}: {value:.4f}")
```

## Command Line Usage

Use the provided script for command-line evaluation:

```bash
# Evaluate from JSONL file
python scripts/evaluate_with_deberta.py \
    --model /path/to/trained/model \
    --data /path/to/data.jsonl \
    --output results.json

# Single evaluation
python scripts/evaluate_with_deberta.py \
    --model /path/to/trained/model \
    --question "What are your thoughts on AI?" \
    --comment "I believe AI will help humanity." \
    --summary "The discussion covered AI perspectives..."

# Use sample data
python scripts/evaluate_with_deberta.py --model /path/to/trained/model
```

## Data Format

The evaluator expects data in the same format as the training data:

```json
{
    "question": "What are your thoughts on artificial intelligence development?",
    "comment": "I believe AI will revolutionize healthcare and education, but we need strong regulations to prevent misuse.",
    "summary": "The discussion on AI development revealed mixed perspectives. Some participants emphasized the potential benefits in healthcare and education, while others expressed concerns about the need for proper regulation and oversight to ensure responsible development and deployment of AI technologies."
}
```

## Model Architecture

The evaluator loads models with the following architecture:
- **Base Model**: DeBERTa (microsoft/deberta-v3-base)
- **Input Format**: `"Question: {q} [SEP] Annotator opinion: {c} [SEP] Summary: {s}"`
- **Output**: 4-dimensional regression predictions
- **Score Range**: No specific range restriction (model output as-is)

## Output Format

Each evaluation result contains:

```python
{
    "question": "Original question",
    "comment": "Annotator's comment", 
    "summary": "Summary text",
    "predictions": {
        "perspective_representation": 4.2,
        "informativeness": 3.8,
        "neutrality_balance": 4.0,
        "policy_approval": 3.5
    },
    "normalized_predictions": {
        "perspective_representation": 4.2,
        "informativeness": 3.8,
        "neutrality_balance": 4.0,
        "policy_approval": 3.5
    },
    "status": "success"  # or "error"
}
```

## Error Handling

The evaluator includes robust error handling:
- Missing model files fall back to randomly initialized weights
- Invalid JSON lines are skipped with warnings
- Individual evaluation errors don't stop batch processing
- Error details are included in results

## Performance Tips

1. **GPU Usage**: Use `device="cuda"` for faster evaluation
2. **Batch Size**: Process multiple items at once for efficiency
3. **Sequence Length**: Adjust `max_length` based on your data (default: 2048)
4. **Memory**: Large batches may require more GPU memory

## Example: Complete Evaluation Pipeline

```python
from src.llm_evaluation.evaluator import DebertaEvaluator
import json

# 1. Load evaluator
evaluator = DebertaEvaluator("/path/to/model", device="cuda")

# 2. Load data
with open("data.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# 3. Evaluate
results = evaluator.evaluate_batch(data)

# 4. Get statistics
stats = evaluator.get_evaluation_statistics(results)

# 5. Save results
with open("evaluation_results.json", "w") as f:
    json.dump({
        "results": results,
        "statistics": stats
    }, f, indent=2)

print(f"Evaluated {len(results)} items")
print(f"Average perspective_representation: {stats['average_scores']['perspective_representation']:.4f}")
```

## Troubleshooting

**Model Loading Issues:**
- Ensure the model directory contains `pytorch_model.bin` or `model.safetensors`
- Check that the model was trained with the same architecture

**CUDA Issues:**
- Use `device="cpu"` if CUDA is not available
- Check GPU memory if getting OOM errors

**Data Format Issues:**
- Ensure JSONL files have `question`, `comment`, `summary` fields
- Check for valid JSON formatting

**Import Issues:**
- Install required packages: `pip install torch transformers numpy`
- Add the src directory to your Python path
