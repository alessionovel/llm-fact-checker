# LLM Fact Checker

An automated fact-checking tool that processes statements from Excel files using Large Language Models (LLMs) to determine their veracity and provide confidence scores.

## Overview

This tool reads statements from an Excel file, sends each statement to an LLM for analysis, and generates a results file containing the verification status (TRUE/FALSE/INSUFFICIENT INFO) along with confidence scores. The tool supports multiple LLM providers including Hugging Face, Google Gemini, and Azure OpenAI.

## Installation

```bash
# Clone the repository
git clone https://github.com/alessionovel/llm-fact-checker.git
cd llm-fact-checker

# Install dependencies
pip install -r requirements.txt

# Set up your environment variables
cp .env.example .env
# Add your API token to .env:
# - HUGGINGFACE_API_TOKEN (if using Hugging Face)
# - GEMINI_API_KEY (if using Google Gemini)
# - AZURE_OPENAI_API_KEY (if using Azure OpenAI)
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
- `HUGGINGFACE_API_TOKEN`: Your Hugging Face API token (required if using Hugging Face model)
- `GEMINI_API_KEY`: Your Google Gemini API key (required if using Gemini model)
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key (required if using Azure OpenAI model)

**Note**: The current version uses Google Gemini (gemini-2.0-flash-lite) by default. To use a different provider, modify the `process_statements()` function in `queryLLM.py` to call the appropriate function:
- `query_llm_huggingface()` for Hugging Face
- `query_llm_gemini()` for Google Gemini (default)
- `query_llm_azure_openai()` for Azure OpenAI

## Command Line Arguments

- `--file` (required): Path to input Excel file
- `--output` (optional): Path for output file (default: `output.xlsx`)
- `--verbose` (optional): Enable detailed logging

## Requirements

- Python 3.7+
- pandas
- huggingface_hub
- google-genai
- openai
- python-dotenv
- openpyxl

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
