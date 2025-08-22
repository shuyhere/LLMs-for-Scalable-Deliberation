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


# Demo usage
## Summary with LLM
`uv run scripts/test_sum_comments.py`









