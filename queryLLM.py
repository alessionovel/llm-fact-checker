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
        confidence: Float in [0,1] if verdict is TRUE or FALSE, else None
    """
    verdict: Literal["TRUE", "FALSE", "INSUFFICIENT INFO"]
    confidence: float | None

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
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")
        return v

def query_llm_ollama(statement, client, verbose=False):
    """
    Sends a statement to the LLM and receives a response using Ollama.
    
    Args:
        statement (str): The statement to analyze
        client (OpenAILikeClient): The LLM client instance
        verbose (bool): If True, print the LLM response
    
    Returns:
        Truth: Pydantic model with 'verdict' ("TRUE", "FALSE", or "INSUFFICIENT INFO")
               and 'confidence' (float between 0-1, or None if verdict is "INSUFFICIENT INFO")
    """

    prompt = (
        "Given the statement below, "
        "respond ONLY with a JSON object matching the schema: {\n"
        "  'verdict': 'TRUE' | 'FALSE' | 'INSUFFICIENT INFO',\n"
        "  'confidence': <float between 0 and 1 or null>\n"
        "}. If you don't have enough information, your verdict should be INSUFFICIENT INFO, and in that case you set confidence to null. Statement: "
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

def save_results_to_excel(results, output_path):
    """
    Saves the results to an Excel file.
    
    Args:
        results (list): List of result dictionaries
        output_path (str): Path where to save the output Excel file
    """
    df_results = pd.DataFrame(results)
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
    
    if verbose:
        print(f"--- Total rows found: {len(df)} ---")
        print(f"--- Total statements extracted: {len(statements)} ---")
        print(f"Successfully extracted {len(statements)} statements from the 'Statement' column.")
    
    return statements

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
            """You are a rigorous Fact-Checking Analyst. You function deterministically: identical inputs must always yield identical reasoning paths and conclusions.

            Your process is as follows:

            STEP 1: DECONSTRUCTION
            Break the user's input into specific, atomic claims that can be verified independently.

            STEP 2: EVIDENCE RETRIEVAL (Internal Knowledge)
            Retrieve only well-established facts, scientific consensus, or historical records. explicitely exclude speculation, conspiracy theories, or highly partisan rhetoric.

            STEP 3: LOGICAL COMPARISON
            Compare the atomic claims against the evidence. Look for logical fallacies, omitted context, or statistical manipulation.

            STEP 4: VERDICT
            Based *only* on the steps above, provide a final verdict."""
        ),
        base_url="http://localhost:11434/v1",  # Default Ollama API endpoint
        temperature=0.0,
    )
    return client

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
        statements = read_excel_file(file_path, args.verbose)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        sys.exit(1)

    try:
        results = process_statements(statements, client, args.verbose)
        save_results_to_excel(results, output_path)
    except Exception as e:
        print(f"Processing error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()