# Summary Evaluation with LLMs

This system evaluates how well comments are represented in summaries using LLM models. It's designed to work with JSON datasets containing questions and comments.

## Features

- **JSON Dataset Support**: Automatically processes JSON files with `question` and `comments` fields
- **Comment Representation Evaluation**: Uses a 1-5 scoring system to evaluate comment representation
- **Batch Processing**: Process multiple datasets automatically
- **Automatic Summary Generation**: Always generates summaries from comments using LLM
- **Detailed Results**: Save individual evaluation results and summary reports
- **Configurable**: Easy configuration through YAML files

## Dataset Format

Your JSON files should follow this format:

```json
{
  "question": "What do you believe should change in Bowling Green/Warren County?",
  "comments": [
    {
      "index": 0,
      "comment": "Smallhouse Rd needs to be widened to have room for three lanes."
    },
    {
      "index": 1,
      "comment": "More free meeting space for nonprofits."
    }
  ]
}
```

## Scoring System

The evaluation uses a 1-5 scale:

1. **Not represented at all**: The summary ignored the comment entirely
2. **Some relevant material**: Contains some material but missing most content
3. **Substantially represented**: Covers main points but missing important details
4. **Mostly covered**: Covers most information but missing nuance/detail
5. **Fully covered**: Covers all information in the comment

## Usage

### Quick Start

1. **Process all datasets** (generates summaries from comments):
```bash
cd LLMs-Scalable-Deliberation
python scripts/evaluate_json_datasets.py
```

2. **Process single dataset**:
```bash
python scripts/test_evaluate_summaries.py datasets/protest.json
```

3. **Customize summary generation** (edit config file):
```bash
# Edit config/evaluation_config.yaml to customize model and system prompt
python scripts/evaluate_json_datasets.py
```

### Configuration

Edit `config/evaluation_config.yaml` to customize:

- **Model selection**: Choose different LLM models
- **System prompt**: Customize the system prompt for summary generation
- **Output format**: Choose output directory and format
- **Processing options**: Set batch sizes and retry settings

### Output Files

Results are saved to the `evaluation_results/` directory:

- `evaluation_[dataset_name].txt`: Individual dataset results
- `summary_report.txt`: Overall summary across all datasets

## Scripts

### `evaluate_json_datasets.py`

Main script for batch processing all JSON datasets:

- Automatically finds all `.json` files in `datasets/` directory
- Processes each dataset sequentially
- Generates comprehensive reports
- Handles errors gracefully

### `test_evaluate_summaries.py`

Single dataset evaluation script:

- Process individual JSON or CSV files
- Good for testing and development
- Supports both JSON and CSV formats

## Example Output

```
Processing: protest.json
============================================================
Question: Do you support the use of military force against civilians if there are peaceful protests when the next president takes office?
Number of comments: 1901
Summary source: generated
Summary: Based on the comments, the discussion reveals strong opposition to using military force against peaceful protesters. Most participants emphasize that peaceful protest is a fundamental right and that military intervention should only occur if there is actual violence. Many commenters stress the importance of maintaining democratic principles and protecting civil liberties...

Evaluating comment representation...

Evaluation completed!
Successful evaluations: 1901
Average score: 3.45
Score distribution: {1: 45, 2: 123, 3: 456, 4: 789, 5: 488}
```

## Customization

### Adding New Models

Edit `src/models/LanguageModel.py` to add support for new LLM providers.

### Modifying Prompts

Edit `src/utils/prompts/evaluation.py` to customize the evaluation prompt format.

### Changing Scoring

Modify the `_extract_score_from_response` method in `SummaryEvaluator` to handle different response formats.
