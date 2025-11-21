# LLM Fact Checker

An automated fact-checking tool that processes statements from Excel files using Large Language Models (LLMs) to determine their veracity and provide confidence scores.

## Overview

This tool reads statements from an Excel file, sends each statement to an LLM for analysis, and generates a results file containing the verification status (TRUE/FALSE/INSUFFICIENT INFO) along with confidence scores. The tool runs locally using Ollama, though it can be adapted to work with API-based providers if desired.

## Prerequisites

Before you begin, make sure you have the following installed:

1. **Python 3.7+**: Check your version with `python --version` or `python3 --version`
2. **Ollama**: Download and install from [https://ollama.com](https://ollama.com)
3. **A compatible LLM model**: After installing Ollama, pull a model (e.g., `ollama pull llama2`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alessionovel/llm-fact-checker.git
cd llm-fact-checker
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running:
```bash
ollama serve
```

## Usage

Basic usage:
```bash
python queryLLM.py --file input.xlsx
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

| statement | response | confidence |
|-----------|----------|------------|
| The Earth is flat | FALSE | 0.95 |
| Water boils at 100°C at sea level | TRUE | 0.98 |
| ... | INSUFFICIENT INFO | N/A |

## Command Line Arguments

- `--file` (required): Path to input Excel file
- `--output` (optional): Path for output file (default: `output.xlsx`)
- `--verbose` (optional): Enable detailed logging

## Requirements

- Python 3.7+
- Ollama (running locally)
- pandas
- openpyxl
- tqdm

All Python requirements are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
