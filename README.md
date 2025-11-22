# LLM Fact Checker

Automated fact-checking tool that processes statements from Excel files using locally hosted Large Language Models (LLMs) to determine their veracity and provide confidence scores.

## Overview

This tool reads statements from an Excel file, sends each statement to an LLM for analysis, and generates a results file containing the verification status (`TRUE` / `FALSE` / `INSUFFICIENT INFO`) along with confidence scores. The tool runs locally using Ollama, leveraging the `datapizza-ai` libraries for structured interaction. It can be adapted to remote/API providers with minimal changes.

## Prerequisites

Before you begin, make sure you have the following installed:

1. **Python 3.7+**: Check your version with `python --version` or `python3 --version`
2. **Ollama**: Download and install from [https://ollama.com](https://ollama.com)
3. **Llama 3.2 model (default)**: After installing Ollama, pull the model with `ollama pull llama3.2`

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

### Input Format

Your Excel file should contain a column named `Statement` with the statements to verify:

| Statement |
|-----------|
| The Earth is flat |
| Water boils at 100°C at sea level |
| ... |

### Output Format

The tool generates an Excel file with the following columns:

| statement | verdict | confidence |
|-----------|---------|------------|
| The Earth is flat | FALSE | 0.95 |
| Water boils at 100°C at sea level | TRUE | 0.98 |
| ... | INSUFFICIENT INFO | (blank) |

Rules:
- `verdict` is one of `TRUE`, `FALSE`, `INSUFFICIENT INFO`.
- `confidence` is a float in `[0,1]` only when verdict is `TRUE` or `FALSE`.
- When verdict is `INSUFFICIENT INFO`, confidence is left empty (null).

## Command Line Arguments

- `--file` (required): Path to input Excel file (`.xlsx`) containing a `Statement` column
- `--output` (optional): Path for output file (default: `output.xlsx`)
- `--verbose` (optional): Enable detailed logging
- `--model` (optional): Name of the Ollama model to use (default: `llama3.2`)

## Requirements

- Python 3.7+
- Ollama (running locally)
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

## Notes

- Ensure Ollama is serving (`ollama serve`) before running the script.