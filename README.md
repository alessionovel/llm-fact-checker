# LLM Fact Checker

Automated fact-checking tool that processes statements from Excel files using locally hosted Large Language Models (LLMs) (via Ollama and datapizza-ai) to determine their veracity and provide confidence scores. The main script is `queryLLM.py` for local use. For cloud/remote APIs (e.g., Azure OpenAI), see `queryLLM-cloud.py` (easily adaptable to other APIs).

## Overview

This tool reads statements from an Excel file, sends each statement to an LLM for analysis, and generates a results file containing the verification status (`TRUE` / `FALSE` / `INSUFFICIENT INFO`) along with confidence scores. The main workflow runs locally using Ollama, leveraging the `datapizza-ai` libraries for structured interaction. For cloud or remote API providers, use the `queryLLM-cloud.py` script (see below).

## Prerequisites

Before you begin, make sure you have the following installed for local use:

1. **Python 3.7+**: Check your version with `python --version` or `python3 --version`
2. **Ollama**: Download and install from [https://ollama.com](https://ollama.com)
3. **Llama 3.2 model (default)**: After installing Ollama, pull the model with `ollama pull llama3.2`

For cloud/remote API use (e.g., Azure OpenAI), see the `queryLLM-cloud.py` script and ensure you have the required API keys and Python packages (see script for details).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alessionovel/llm-fact-checker.git
cd llm-fact-checker
```

2. **Create a virtual environment** (recommended):

   Using conda:
   ```bash
   conda create -n llm-fact-checker python=3.10
   conda activate llm-fact-checker
   ```

   Alternatively, using venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies (includes `datapizza-ai` and the OpenAI-like client wrapper):
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is running:
```bash
ollama serve
```

## Usage


### Local (Ollama/datapizza-ai)

Basic usage:
```bash
python queryLLM.py --file input.xlsx
```

Specify a different local Ollama model (e.g. `llama3.1`) with:
```bash
python queryLLM.py --file input.xlsx --model llama3.1
```

With custom output and verbose mode:
```bash
python queryLLM.py --file input.xlsx --output my_results.xlsx --verbose
```

### Cloud/Remote API (Azure OpenAI)

To use Azure OpenAI (or adapt to other APIs), use the `queryLLM-cloud.py` script. This version is designed for Azure but can be easily modified for other providers. 

**Setup for Azure OpenAI:**
1. Create a `.env` file in the project root directory
2. Add your Azure OpenAI API key: `AZURE_OPENAI_API_KEY=your_api_key_here`

Example usage:
```bash
python queryLLM-cloud.py --file input.xlsx --output my_results.xlsx --verbose
```

### Input Format

Your Excel file should contain a column named `Statement` with the statements to verify:

| Statement |
|-----------|
| The Earth is flat |
| Water boils at 100°C at sea level |
| ... |

### Output Format

The tool generates an Excel file with the following columns (plus additional columns for each prompt and reconsideration step):

| statement | verdict-prompt1-initial | confidence-prompt1-initial | verdict-prompt1-reconsidered | confidence-prompt1-reconsidered | verdict-prompt2-initial | confidence-prompt2-initial | verdict-prompt2-reconsidered | confidence-prompt2-reconsidered |
|-----------|------------------------|----------------------------|------------------------------|----------------------------------|------------------------|----------------------------|------------------------------|----------------------------------|
| The Earth is flat | FALSE | 95 | FALSE | 95 | FALSE | 95 | FALSE | 95 |
| Water boils at 100°C at sea level | TRUE | 98 | TRUE | 98 | TRUE | 98 | TRUE | 98 |
| ... | INSUFFICIENT INFO | (blank) | INSUFFICIENT INFO | (blank) | INSUFFICIENT INFO | (blank) | INSUFFICIENT INFO | (blank) |

Rules:
- `verdict` is one of `TRUE`, `FALSE`, `INSUFFICIENT INFO`.
- `confidence` is an integer in `[0,100]` only when verdict is `TRUE` or `FALSE`.
- When verdict is `INSUFFICIENT INFO`, confidence is left empty (null).

## Command Line Arguments

### queryLLM.py (local)
- `--file` (required): Path to input Excel file (`.xlsx`) containing a `Statement` column
- `--output` (optional): Path for output file (default: `output.xlsx`)
- `--verbose` (optional): Enable detailed logging
- `--model` (optional): Name of the Ollama model to use (default: `llama3.2`)

### queryLLM-cloud.py (Azure/cloud)
- `--file` (required): Path to input Excel file (`.xlsx`) containing a `Statement` column
- `--output` (optional): Path for output file (default: `output.xlsx`)
- `--verbose` (optional): Enable detailed logging

## Requirements

- Python 3.7+
- Ollama (running locally, for local use)
- pandas
- openpyxl
- tqdm
- datapizza-ai
- datapizza-ai-clients-openai-like
- pydantic

All Python requirements are listed in `requirements.txt` and can be installed with:

```bash
pip install -r requirements.txt
```

For cloud/remote API use, see the top of `queryLLM-cloud.py` for any additional requirements (e.g., `openai`, `python-dotenv`).

## Notes


- The model you want to use must be downloaded first using `ollama pull <model-name>` (e.g., `ollama pull llama3.2`).
- Ensure Ollama is serving (`ollama serve`) before running the script.
- For cloud/remote API use, set your API keys and endpoints as required (see `queryLLM-cloud.py`).

## Analysis Folder (R Reports)

After you have generated results with the Python pipeline above, move into the `analysis/` folder to post-process them.

### Prerequisites
- Install R (e.g., on macOS: `brew install r` or from https://cloud.r-project.org/`).
- Place your output CSVs inside `analysis/` (e.g., `results-llama3.2.csv`, `results-gpt4o.csv`). `.rds` is also accepted.

### How to run
From the repo root (or after `cd analysis`):
```bash
Rscript analysis/run_analysis.R analysis/results-llama3.2.csv
# or, inside the folder
Rscript run_analysis.R results-llama3.2.csv
```
If you omit the argument, the script looks for `results.csv`, `results.rds`, or `example.csv` located in `analysis/`.

### What the script does

The R script performs comprehensive statistical analysis and generates four detailed CSV reports in a `report-<model-name>/` directory:

#### 1. Accuracy Analysis (`task1-accuracy.csv`)
Evaluates binary correctness of model verdicts against ground truth:
- **Accuracy percentages** for each scenario (Prompt 1/2, Initial/Reconsidered)
- **Correct counts**: Number of correct predictions (excluding abstentions)
- **Abstained counts**: Cases where model returned `INSUFFICIENT INFO`
- **Total attempted**: Valid predictions (excluding abstentions)
- **Overall pooled statistics**: Aggregate accuracy across all scenarios
- **McNemar tests**: Statistical significance comparisons between:
  - Prompt 1 vs Prompt 2 (Initial and Reconsidered)
  - Initial vs Reconsidered (within each prompt)
  - Includes chi-square values, p-values, and significance indicators

#### 2. Consistency Analysis (`task2-consistency.csv`)
Checks logical consistency across triplets (affirmation, negation, antonym):
- **Affirmation vs Negation consistency**: Checks if these have opposite verdicts (or both abstained)
- **Affirmation vs Antonym consistency**: Checks if these have opposite verdicts (or both abstained)
- **Negation vs Antonym consistency**: Checks if these have the same verdict (or both abstained)
- **Full Triplet consistency**: Checks if all three statements maintain logical coherence
- For each consistency type:
  - Consistency percentages by scenario
  - Consistent triplet counts vs total triplets
  - McNemar tests comparing prompts and reconsideration effects
  - Statistical significance indicators (chi-square, p-values)

#### 3. Confidence Analysis (`task3-confidence.csv`)
Analyzes confidence scores and their relationship to accuracy:
- **Overall average confidence**: Mean confidence scores for each scenario
- **Confidence by correctness**: Separate averages for:
  - Correct predictions
  - Incorrect predictions
  - Total statistics across all scenarios
- **Confidence by statement type**: Average confidence for affirmation, negation, and antonym statements
- **Correlation analysis**: Pearson correlation coefficients between confidence and accuracy
  - Includes valid data point counts (excludes `INSUFFICIENT INFO` cases)
  - Measures if higher confidence correlates with better accuracy

#### 4. Flip Rate Analysis (`task4-fliprate.csv`)
Measures verdict stability between initial and reconsidered responses:
- **Overall flip rate**: Percentage of cases where verdicts changed (Initial → Reconsidered)
- **Flip statistics**: For each prompt:
  - Flip rate percentage
  - Flipped cases count
  - Stable cases count (no change)
  - Abstained cases (where either verdict was `INSUFFICIENT INFO`)
  - Total cases analyzed
- **Flip rate by statement type**: Breakdown for affirmation, negation, and antonym
- **Flip rate by initial correctness**: Separate analysis for:
  - Initially correct answers (how often correct answers flip)
  - Initially incorrect answers (how often incorrect answers flip)
  - Helps identify if reconsideration improves or degrades accuracy

### Output Structure
All reports are saved in `analysis/report-<model-name>/` where `<model-name>` is extracted from the input filename (e.g., `results-llama3.2.csv` → `report-llama3.2/`).