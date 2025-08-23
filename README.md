# LLMs-Scalable-Deliberation
Scaling evaluation of LLM based opinion summarization.

# Enviroment setup
We use `uv` to manage the environment.

if you did not have `uv` installed, you can install it by:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, you can install the environment by:

```bash
uv python install
uv sync --frozen
```

Also you need to create .env file at the root of this project folder
```bash
OPENAI_API_KEY=YOUR_OPENAI_KEY

ZHI_API_KEY=YOUR_ZHIZENGZENG_KEY
```


# Data
We use the following datasets:
- protest.json
- gun_use.json
- operation.json
- bowling-green.json


# Demo usage

You can check the usage of the scripts in batch submission and batch evaluation documentation: [README_batch_evaluation.md](./scripts/README_batch_evaluation.md) and [README_batch_summarization.md](./scripts/README_batch_summarization.md)

default setting can be found in [config/batch_summarization_config.yaml](./config/batch_summarization_config.yaml) and [config/batch_evaluation_config.yaml](./config/batch_evaluation_config.yaml)

## Summary with LLM
`uv run scripts/batch_gen_summarization.py`

## Evaluation with LLMs
`uv run scripts/batch_evaluate_summaries.py --evaluation-model gpt-4o-mini`

`uv run scripts/batch_evaluate_summaries.py --evaluation-model gpt-5-nano`









