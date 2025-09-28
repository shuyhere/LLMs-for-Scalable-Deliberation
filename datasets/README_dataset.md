# Datasets for LLM-based Opinion Summarization Evaluation

This directory contains datasets used for evaluating large language models (LLMs) in the context of scalable deliberation and opinion summarization tasks.

## Dataset Structure

### Opinion Dataset (`opinion_dataset/`)

The opinion dataset contains 10 different topics with public opinion data collected through online surveys. Each topic includes a question and a collection of user comments expressing diverse viewpoints.

#### Topics Covered

**Binary Policy Questions (5 topics):**
- **Binary-Health-Care-Policy.json**: "Do you support the government provide basic health insurance for everyone?"
- **Binary-Online-Identity-Policies.json**: "Do you support requiring real-name registration on social media platforms?"
- **Binary-Refugee-Policies.json**: "Do you support accepting more refugees into the country?"
- **Binary-Tariff-Policy.json**: "Do you support imposing tariffs on imported goods?"
- **Binary-Vaccination-Policy.json**: "Do you support mandatory vaccination policies?"

**Open-ended Questions (5 topics):**
- **Openqa-AI-changes-human-life.json**: "How has AI changed your life?"
- **Openqa-Influencers-as-a-job.json**: "What is your opinion on internet influencers becoming a recognized profession?"
- **Openqa-Tipping-System.json**: "What are your thoughts on the current tipping system in restaurants?"
- **Openqa-Trump-cutting-funding.json**: "What do you think about Trump's policies on cutting funding?"
- **Openqa-Updates-of-electronic-products.json**: "How do you feel about frequent updates of electronic products?"

#### Data Format

Each dataset file follows this JSON structure:

```json
{
  "question": "The survey question asked to participants",
  "comments": [
    {
      "index": 0,
      "comment": "User's response to the question"
    },
    {
      "index": 1,
      "comment": "Another user's response"
    }
  ]
}
```

#### Dataset Characteristics

- **Total Topics**: 10 (5 binary policy questions + 5 open-ended questions)
- **Response Length**: 2-3 sentences per comment (as requested in surveys)

## Usage for Evaluation

These datasets are specifically designed for evaluating LLM performance in:

1. **Opinion Summarization**: Testing how well LLMs can synthesize diverse viewpoints into coherent summaries
2. **Deliberation Quality**: Assessing the ability to capture nuanced arguments and counter-arguments
3. **Bias Detection**: Evaluating whether LLMs fairly represent different perspectives
4. **Scalability**: Testing performance across different topic domains and comment volumes

## Data Collection

The datasets were collected through online surveys where participants were asked to provide brief (2-3 sentence) responses to various policy and opinion questions. The data represents diverse viewpoints from the general public on contemporary issues.

## File Organization

```
datasets/
├── README_dataset.md          # This file
└── opinion_dataset/           # Main evaluation datasets
    ├── Binary-Health-Care-Policy.json
    ├── Binary-Online-Identity-Policies.json
    ├── Binary-Refugee-Policies.json
    ├── Binary-Tariff-Policy.json
    ├── Binary-Vaccination-Policy.json
    ├── Openqa-AI-changes-human-life.json
    ├── Openqa-Influencers-as-a-job.json
    ├── Openqa-Tipping-System.json
    ├── Openqa-Trump-cutting-funding.json
    └── Openqa-Updates-of-electronic-products.json
```

## Citation

If you use these datasets in your research, please cite the original paper and acknowledge the data collection methodology used in this study.