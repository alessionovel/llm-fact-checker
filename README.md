# LLM Fact Checker

How reliable are large language models as fact-checkers, and do they stay **logically consistent**? This university course project asks three models to judge the same claims phrased in three ways and measures how accurate they are, whether their answers contradict each other, and whether they change their mind when asked to reconsider.

## The experiment

The [dataset](data/dataset.xlsx) contains 142 **triplets** of statements, each labelled with its ground truth:

| Type | Example | Ground truth |
| --- | --- | --- |
| Affirmation | Affirmative action is necessary in schools. | YES |
| Negation | Affirmative action is not necessary in schools. | NO |
| Antonym | Affirmative action is dispensable in schools. | NO |

A logically consistent model must give opposite verdicts to the affirmation and its negation, and the same verdict to the negation and the antonym.

Each statement is sent to the model with **two prompts**: a neutral one, and one that also reminds the model of classical logic (*if A is true, not-A is false*). For each prompt the model answers `TRUE`, `FALSE` or `INSUFFICIENT INFO` with a confidence from 0 to 100. Then it is asked to **reconsider** its answer, so every statement gets four verdicts.

## Results

| Model | Accuracy | Affirmation/negation consistency | Full-triplet consistency | Verdicts changed on reconsideration |
| --- | --- | --- | --- | --- |
| GPT-4o | 94.2% | 84.7% | 65.9% | ~2% |
| Ministral 3 (14B) | 85.5% | 63.2% | 51.8% | ~19% |
| Llama 3.2 | 57.1% | 50.2% | 27.1% | ~63% |

Accuracy excludes `INSUFFICIENT INFO` answers. The full reports, with McNemar significance tests, confidence analysis and breakdowns by statement type, are in [`analysis/`](analysis).

## Repository structure

```
queryLLM.py           Query a local Ollama model (default: llama3.2)
queryLLM-cloud.py     Query GPT-4o on Azure OpenAI
queryLLM-ollama.py    Query ministral-3:14b-cloud through the Ollama client
data/
├── dataset.xlsx      The 426 statements with type and ground truth
└── sample.xlsx       12-statement sample for a quick test
analysis/
├── run_analysis.R    Statistical analysis
├── results-*.csv     Model answers for each model
└── report-*/         Generated reports (accuracy, consistency, confidence, flip rate)
```

## Running it

Requires Python 3.9+ and, for the analysis, R.

```bash
pip install -r requirements.txt
```

**1. Query a model.** The input Excel file needs a `statement` column; the other columns are kept in the output.

```bash
# Local model with Ollama (run `ollama pull llama3.2` and `ollama serve` first)
python queryLLM.py --file data/sample.xlsx --output results.xlsx --model llama3.2

# GPT-4o on Azure OpenAI (set AZURE_OPENAI_API_KEY in a .env file)
python queryLLM-cloud.py --file data/sample.xlsx --output results.xlsx

# Ministral through the Ollama client
python queryLLM-ollama.py --file data/sample.xlsx --output results.xlsx
```

Add `--verbose` to print every model response.

**2. Analyse the results.** Export the output as CSV into `analysis/` and run:

```bash
Rscript analysis/run_analysis.R analysis/results-llama3.2.csv
```

The reports are written to `analysis/report-<model>/`:

| Report | Content |
| --- | --- |
| `task1-accuracy.csv` | Accuracy, false positives/negatives and abstentions per prompt and step |
| `task2-consistency.csv` | Consistency between affirmation, negation and antonym, with error patterns |
| `task3-confidence.csv` | Average confidence, and confidence for correct vs. wrong answers |
| `task4-fliprate.csv` | How often the verdict changes after reconsideration |
