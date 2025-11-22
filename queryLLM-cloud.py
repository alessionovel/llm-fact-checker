import argparse
import pandas as pd
import os
import sys
import json
from tqdm import tqdm
from ollama import Client
from pydantic import BaseModel, field_validator
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

def query_llm_ollama(statement, client, model, verbose=False):
    """
    Sends a statement to the LLM and receives a response using Ollama.
    
    Args:
        statement (str): The statement to analyze
        client (Client): The Ollama client instance
        model (str): The model name to use
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

    messages = [
        {
            'role': 'system',
            'content': 'You are a rigorous Fact-Checking Analyst. You function deterministically: identical inputs must always yield identical reasoning paths and conclusions.'
        },
        {
            'role': 'user',
            'content': prompt
        }
    ]

    try:
        # Collect the full response from streaming
        full_response = ''
        for part in client.chat(model, messages=messages, stream=True):
            full_response += part['message']['content']
        
        if verbose:
            print(f"Raw LLM response: {full_response}")
        
        # Parse JSON response
        # Try to extract JSON if it's embedded in text
        response_text = full_response.strip()
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.find('```', start)
            response_text = response_text[start:end].strip()
        
        # Parse JSON and validate with Pydantic
        json_data = json.loads(response_text)
        return Truth(**json_data)
    except Exception as e:
        if verbose:
            print(f"Error during LLM query for statement: {statement}\n{e}")
        # Fallback neutral response
        return Truth(verdict="INSUFFICIENT INFO", confidence=None)

def process_statements(statements, client, model, verbose=False):
    """
    Processes each statement by querying the LLM.
    
    Args:
        statements (list): List of statements to process
        client (Client): The Ollama client instance
        model (str): The model name to use
        verbose (bool): If True, print detailed information
    
    Returns:
        list: List of dictionaries containing the results for each statement
    """
    results = []
    
    for statement in tqdm(statements, desc="Processing statements", unit="statement"):
        # Query the LLM for this statement
        llm_result = query_llm_ollama(statement, client, model, verbose)
        
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

def create_client():
    """
    Creates an Ollama client configured with API key from environment.
    
    Returns:
        Client: Configured Ollama client instance
    """
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        raise ValueError(
            "OLLAMA_API_KEY environment variable is not set. "
            "Please set it with: export OLLAMA_API_KEY='your-api-key'"
        )
    
    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + api_key}
    )
    return client

def test_client_connection(client, model, verbose=False):
    """
    Tests if the Ollama client is working by sending a simple test query.
    
    Args:
        client (Client): The Ollama client instance to test
        model (str): The model name to test
        verbose (bool): If True, print detailed information
    
    Raises:
        ConnectionError: If Ollama is not reachable or the connection fails
        ValueError: If the model is not available
    """
    if verbose:
        print("Testing connection to Ollama...")
    
    try:
        # Send a simple test query
        messages = [
            {
                'role': 'user',
                'content': 'Respond with the word "ok" only.'
            }
        ]
        
        # Try to get a response
        response_received = False
        for part in client.chat(model, messages=messages, stream=True):
            response_received = True
            break  # Just check if we get at least one part
        
        if not response_received:
            raise ValueError("No response received from model")
        
        if verbose:
            print("✓ Connection to Ollama successful!")
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for common connection errors
        if "connection" in error_msg or "refused" in error_msg or "unreachable" in error_msg:
            raise ConnectionError(
                f"Cannot connect to Ollama at https://ollama.com. Please check your internet connection and API key.\n"
                f"Error details: {e}"
            )
        elif "model" in error_msg or "not found" in error_msg:
            raise ValueError(
                f"Model '{model}' not found or not accessible.\n"
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

    # Create LLM client
    try:
        client = create_client()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Test client connection before processing
    try:
        test_client_connection(client, args.model_name, args.verbose)
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
        statements = read_excel_file(file_path, args.verbose)
    except Exception as e:
        print(f"Failed to read input file: {e}")
        sys.exit(1)

    try:
        results = process_statements(statements, client, args.model_name, args.verbose)
        save_results_to_excel(results, output_path)
    except Exception as e:
        print(f"Processing error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()