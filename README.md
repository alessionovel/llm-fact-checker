# LLM Fact Checker

An automated fact-checking tool that processes statements from Excel files using Large Language Models (LLMs) to determine their veracity and provide confidence scores.

## Overview

This tool reads statements from an Excel file, sends each statement to an LLM for analysis, and generates a results file containing the verification status (TRUE/FALSE/INSUFFICIENT INFO) along with confidence scores.

## Installation

```bash
# Clone the repository
git clone https://github.com/alessionovel/llm-fact-checker.git
cd llm-fact-checker

# Install dependencies
pip install -r requirements.txt

# Set up your environment variables
cp .env.example .env
# Add your HUGGINGFACE_API_TOKEN to .env
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

## Configuration

The tool uses environment variables for API configuration:
- `HUGGINGFACE_API_TOKEN`: Your Hugging Face API token

## Command Line Arguments

- `--file` (required): Path to input Excel file
- `--output` (optional): Path for output file (default: `output.xlsx`)
- `--verbose` (optional): Enable detailed logging

## Requirements

- Python 3.7+
- pandas
- huggingface_hub
- python-dotenv
- openpyxl

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
