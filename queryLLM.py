import argparse
import pandas as pd
import os
import sys
from tqdm import tqdm
from datapizza.clients.openai_like import OpenAILikeClient
from pydantic import BaseModel, field_validator
from typing import Literal

class Truth(BaseModel):
    """Structured response model for LLM fact-checking output.

    Attributes:
        verdict: One of TRUE, FALSE, INSUFFICIENT INFO
        confidence: Integer in [0,100] if verdict is TRUE or FALSE, else None
    """
    verdict: Literal["TRUE", "FALSE", "INSUFFICIENT INFO"]
    confidence: int | None
    
    @field_validator("verdict", mode="before")
    def normalize_verdict(cls, v):  # noqa: D401
        # Normalize case and common variants
        if not isinstance(v, str):
            raise ValueError("Verdict must be a string")
        vv = v.strip().upper()
        if vv in {"TRUE", "FALSE", "INSUFFICIENT INFO"}:
            return vv
        # Allow some loose variants
        if vv in {"INSUFFICIENT", "UNKNOWN", "NOT ENOUGH INFO"}:
            return "INSUFFICIENT INFO"
        raise ValueError(f"Invalid verdict value: {v}")

    @field_validator("confidence")
    def validate_confidence(cls, v, info):  # noqa: D401
        verdict = info.data.get("verdict")
        if verdict == "INSUFFICIENT INFO":
            return None
        if v is None:
            raise ValueError("Confidence required for TRUE/FALSE verdicts")
        if not isinstance(v, (int, float)):
            raise ValueError("Confidence must be a number")
        # Convert float to int if needed
        v_int = int(v) if isinstance(v, float) else v
        if not (0 <= v_int <= 100):
            raise ValueError("Confidence must be between 0 and 100")
        return v_int
    

def create_prompts(statement: str) -> list[str]:
    """Return four differently worded prompts for fact-checking a statement.

    Args:
        statement: The statement to embed into each prompt.

    Returns:
        List of four prompt strings.
    """
    return [
        (
            "You are evaluating short statements one at a time.\n"
            "Respond with exactly one of: True, False, or Insufficient info.\n"
            "Rely on standard definitions and widely accepted facts only.\n"
            "After your label, include Confidence: <0–100> on a new line.\n"
            "Do not provide explanations.\n"
            f"Statement: {statement} "
        ),
        (
            "You are evaluating short statements one at a time.\n"
            "Respond with exactly one of: True, False, or Insufficient info.\n"
            "Rely on standard definitions and widely accepted facts only. Also use classical logical: If A is true, not-A must be false and vice versa.\n"
            "After your label, include Confidence: <0–100> on a new line.\n"
            "Do not provide explanations.\n"
            f"Statement: {statement} "
        ),
        (
            "You are evaluating short statements one at a time.\n"
            "Respond with exactly one of: True, False, or Insufficient info.\n"
            "Rely on standard definitions and widely accepted facts only.\n"
            "After your label, include Confidence: <0–100> on a new line.\n"
            "Do not provide explanations.\n"
            "First give an output for this. After that, reconsider and make a new best judgment, accompanied by confidence <0-100>.\n"
            f"Statement: {statement}"
        ),
        (
            "You are evaluating short statements one at a time.\n"
            "Respond with exactly one of: True, False, or Insufficient info.\n"
            "Rely on standard definitions and widely accepted facts only. Also use classical logical: If A is true, not-A must be false and vice versa.\n"
            "After your label, include Confidence: <0–100> on a new line.\n"
            "Do not provide explanations.\n"
            "First give an output for this. After that, reconsider and make a new best judgment, accompanied by confidence <0-100>.\n"
            f"Statement: {statement}"
        ),
    ]

def query_llm_ollama(statement, client, verbose=False):
    """
    Sends a statement to the LLM using four prompt variants and collects results.

    Args:
        statement (str): The statement to analyze
        client (OpenAILikeClient): The LLM client instance
        verbose (bool): If True, print the LLM responses

    Returns:
        dict: Keys for each prompt variant: 'verdict-prompt1'..'verdict-prompt4' and
              'confidence-prompt1'..'confidence-prompt4'.
    """

    prompts = create_prompts(statement)

    results = {}
    for idx, prompt in enumerate(prompts, start=1):
        key_v = f"verdict-prompt{idx}"
        key_c = f"confidence-prompt{idx}"
        try:
            response = client.structured_response(input=prompt, output_cls=Truth)
            if verbose:
                print(f"Raw structured response (prompt {idx}): {response}")
            data_list = getattr(response, "structured_data", [])
            if not data_list:
                raise ValueError("No structured data returned by LLM")
            truth: Truth = data_list[0]
            results[key_v] = truth.verdict
            results[key_c] = truth.confidence
        except Exception as e:
            if verbose:
                print(f"Error during LLM query for statement (prompt {idx}): {statement}\n{e}")
            results[key_v] = "INSUFFICIENT INFO"
            results[key_c] = None

    return results

def process_statements(statements, client, verbose=False):
    """
    Processes each statement by querying the LLM.
    
    Args:
        statements (list): List of statements to process
        client (OpenAILikeClient): The LLM client instance
        verbose (bool): If True, print detailed information
    
    Returns:
        list: List of dictionaries containing the results for each statement
    """
    results = []
    
    for statement in tqdm(statements, desc="Processing statements", unit="statement"):
        llm_results = query_llm_ollama(statement, client, verbose)
        row = {'statement': statement}
        row.update(llm_results)
        results.append(row)
    
    return results

def save_results_to_excel(results, output_path):
    """
    Saves the results to an Excel file.
    
    Args:
        results (list): List of result dictionaries
        output_path (str): Path where to save the output Excel file
    """
    df_results = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    df_results.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")

def read_excel_file(file_path, verbose=False):
    """
    Reads the Excel file and extracts statements from the 'Statement' column.
    
    Args:
        file_path (str): Path to the Excel file
        verbose (bool): If True, print detailed information
    
    Returns:
        pandas.DataFrame: The full DataFrame read from the Excel file
    """
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Normalize to lowercase 'statement' column
    if 'statement' not in df.columns and 'Statement' in df.columns:
        df = df.rename(columns={'Statement': 'statement'})
    
    if 'statement' not in df.columns:
        raise KeyError("Input file must contain a 'statement' column (or 'Statement' which will be normalized)")
    
    if verbose:
        print(f"--- Total rows found: {len(df)} ---")
        print(f"Successfully validated presence of 'statement' column.")
    
    return df


def process_dataframe(df: pd.DataFrame, client, verbose=False) -> pd.DataFrame:
    """Process the entire DataFrame, appending per-prompt results after existing columns.

    Args:
        df: Input DataFrame containing at least a 'Statement' column.
        client: LLM client
        verbose: If True, print detailed information

    Returns:
        pandas.DataFrame: Original columns plus result columns appended at the end.
    """
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing statements", unit="statement"):
        statement = str(row['statement']) if pd.notna(row['statement']) else ''
        llm_results = query_llm_ollama(statement, client, verbose)
        base = row.to_dict()
        base.update(llm_results)
        rows.append(base)

    df_aug = pd.DataFrame(rows)
    result_cols = [
        'verdict-prompt1', 'confidence-prompt1',
        'verdict-prompt2', 'confidence-prompt2',
        'verdict-prompt3', 'confidence-prompt3',
        'verdict-prompt4', 'confidence-prompt4',
    ]
    ordered_cols = list(df.columns) + [c for c in result_cols if c in df_aug.columns]
    return df_aug[ordered_cols]

def create_client(model_name):
    """
    Creates an OpenAI-like client configured for Ollama.
    
    Args:
        model_name (str): Name of the Ollama model to use (e.g., "llama3.2")
    
    Returns:
        OpenAILikeClient: Configured client instance
    """
    client = OpenAILikeClient(
        api_key="",  # Ollama doesn't require an API key
        model=model_name,
        system_prompt=(
            """You are a rigorous Fact-Checking Analyst. You function deterministically: identical inputs must always yield identical reasoning paths and conclusions."""
        ),
        base_url="http://localhost:11434/v1",  # Default Ollama API endpoint
        temperature=0.0,
    )
    return client

def test_client_connection(client, verbose=False):
    """
    Tests if the Ollama client is working by sending a simple test query.
    
    Args:
        client (OpenAILikeClient): The LLM client instance to test
        verbose (bool): If True, print detailed information
    
    Raises:
        ConnectionError: If Ollama is not running or the connection fails
        ValueError: If the model is not available
    """
    if verbose:
        print("Testing connection to Ollama...")
    
    try:
        # Send a simple test query
        test_prompt = "Respond with the word 'ok' only."
        response = client.structured_response(input=test_prompt, output_cls=Truth)
        
        if verbose:
            print("✓ Connection to Ollama successful!")
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for common connection errors
        if "connection" in error_msg or "refused" in error_msg or "unreachable" in error_msg:
            raise ConnectionError(
                f"Cannot connect to Ollama. Please ensure Ollama is running with 'ollama serve'.\n"
                f"Error details: {e}"
            )
        elif "model" in error_msg or "not found" in error_msg:
            raise ValueError(
                f"Model '{client.model}' not found. Please download it first with 'ollama pull {client.model}'.\n"
                f"Error details: {e}"
            )
        else:
            raise ConnectionError(
                f"Failed to communicate with Ollama: {e}"
            )

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Process Excel file statements through LLM and save results with confidence scores."
    )

    # Add command-line arguments
    parser.add_argument(
        '--file', 
        dest='file_input', 
        required=True, 
        type=str, 
        help='Path to the input Excel file containing statements (.xlsx)'
    )

    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Print more details during execution'
    )

    parser.add_argument(
        '--output',
        dest='output_file',
        type=str,
        default='output.xlsx',
        help='Path for the output Excel file (default: output.xlsx)'
    )

    parser.add_argument(
        '--model',
        dest='model_name',
        type=str,
        default='llama3.2',
        help='Name of the Ollama model to use (default: llama3.2)'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Create LLM client with specified model
    client = create_client(args.model_name)
    
    # Test client connection before processing
    try:
        test_client_connection(client, args.verbose)
    except (ConnectionError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get file paths from arguments
    file_path = args.file_input
    output_path = args.output_file

    # Validate input file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    if args.verbose:
        print(f"--- Starting processing of file: {file_path} ---")

    try:
        df_input = read_excel_file(file_path, args.verbose)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        sys.exit(1)

    try:
        df_results = process_dataframe(df_input, client, args.verbose)
        save_results_to_excel(df_results, output_path)
    except Exception as e:
        print(f"Processing error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()