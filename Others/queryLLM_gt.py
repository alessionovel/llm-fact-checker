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
    

def query_llm_ollama(statement, client, verbose=False):
    """
    Sends a statement to the LLM and receives a response using Ollama.
    
    Args:
        statement (str): The statement to analyze
        client (OpenAILikeClient): The LLM client instance
        verbose (bool): If True, print the LLM response
    
    Returns:
        Truth: Pydantic model with 'verdict' ("TRUE", "FALSE", or "INSUFFICIENT INFO")
               and 'confidence' (integer between 0-100, or None if verdict is "INSUFFICIENT INFO")
    """

    prompt = (
        "Given the statement below, "
        "respond ONLY with a JSON object matching the schema: {\n"
        "  'verdict': 'TRUE' | 'FALSE' | 'INSUFFICIENT INFO',\n"
        "  'confidence': <integer between 0 and 100 or null>\n"
        "}. If you don't have enough information or you are not enough confident, your verdict should be INSUFFICIENT INFO, and in that case you set confidence to null. Statement: "
        f"{statement}"
    )

    try:
        response = client.structured_response(input=prompt, output_cls=Truth)
        if verbose:
            print(f"Raw structured response: {response}")
        data_list = getattr(response, "structured_data", [])
        if not data_list:
            raise ValueError("No structured data returned by LLM")
        return data_list[0]
    except Exception as e:
        if verbose:
            print(f"Error during LLM query for statement: {statement}\n{e}")
        # Fallback neutral response
        return Truth(verdict="INSUFFICIENT INFO", confidence=None)

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
        # Query the LLM for this statement
        llm_result = query_llm_ollama(statement, client, verbose)
        
        results.append({
            'statement': statement,
            'verdict': llm_result.verdict,
            'confidence': llm_result.confidence
        })
    
    return results

def save_results_to_excel(results, ground_truths, output_path):
    """
    Saves the results to an Excel file.
    
    Args:
        results (list): List of result dictionaries
        output_path (str): Path where to save the output Excel file
    """
    df_results = pd.DataFrame(results)
    df_results["GroundTruth"] = ground_truths
    df_results.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")

def read_excel_file(file_path, verbose=False):
    """
    Reads the Excel file and extracts statements from the 'Statement' column.
    
    Args:
        file_path (str): Path to the Excel file
        verbose (bool): If True, print detailed information
    
    Returns:
        list: List of statements from the 'Statement' column
    """
    # Read the Excel file
    df = pd.read_excel(file_path)

    if 'Statement' not in df.columns:
        raise KeyError("Input file must contain a 'Statement' column")

    statements = df['Statement'].dropna().astype(str).tolist()
    ground_truths = df['GroundTruth'].dropna().astype(str).tolist() if 'GroundTruth' in df.columns else []
    
    if verbose:
        print(f"--- Total rows found: {len(df)} ---")
        print(f"--- Total statements extracted: {len(statements)} ---")
        print(f"Successfully extracted {len(statements)} statements from the 'Statement' column.")
    
    return statements, ground_truths

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
        statements, _ = read_excel_file(file_path, args.verbose)
        _, ground_truths = read_excel_file(file_path, args.verbose)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        sys.exit(1)

    try:
        for i in range(5):
            results = process_statements(statements, client, args.verbose)
            save_results_to_excel(results, ground_truths, output_path)
    except Exception as e:
        print(f"Processing error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()