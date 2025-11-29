import argparse
import pandas as pd
import os
import sys
from tqdm import tqdm
from datapizza.clients.openai_like import OpenAILikeClient

def create_prompts(statement: str) -> list[str]:
    """Return two base prompts for fact-checking a statement.

    Prompt 1: Neutral factual assessment.
    Prompt 2: Adds classical logic consistency requirement.

    Each will be used twice: initial and reconsideration steps.
    """
    return [
        (
            "You are evaluating short statements one at a time.\n"
            "Respond in exactly this format:\n"
            "- First line: 'TRUE', 'FALSE', or 'INSUFFICIENT INFO' (nothing else)\n"
            "- Second line: confidence value 0-100 (ONLY if first line is TRUE or FALSE, omit for INSUFFICIENT INFO)\n"
            "Rely on standard definitions and widely accepted facts only.\n"
            "Do not provide explanations or additional text.\n"
            f"Statement: {statement} "
        ),
        (
            "You are evaluating short statements one at a time.\n"
            "Respond in exactly this format:\n"
            "- First line: 'TRUE', 'FALSE', or 'INSUFFICIENT INFO' (nothing else)\n"
            "- Second line: confidence value 0-100 (ONLY if first line is TRUE or FALSE, omit for INSUFFICIENT INFO)\n"
            "Rely on standard definitions and widely accepted facts only. Also use classical logic: If A is true, not-A must be false and vice versa.\n"
            "Do not provide explanations or additional text.\n"
            f"Statement: {statement} "
        ),
    ]

RECONSIDER_PROMPT = (
    "Reconsider your previous answer and provide your final judgment in the same format:\n"
    "- First line: 'TRUE', 'FALSE', or 'INSUFFICIENT INFO' (nothing else)\n"
    "- Second line: confidence value 0-100 (ONLY if first line is TRUE or FALSE, omit for INSUFFICIENT INFO)\n"
    "Do not provide explanations or additional text."
)

def parse_response(response_text: str, verbose: bool = False) -> tuple:
    """Parse LLM response to extract verdict and confidence.
    
    Args:
        response_text: The raw response from the LLM
        verbose: If True, print parsing details
        
    Returns:
        Tuple of (verdict, confidence)
    """
    import re
    
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    verdict = None
    confidence = None
    
    if len(lines) >= 1:
        first_line = lines[0].upper()
        
        # Check first line for verdict
        if first_line == 'TRUE':
            verdict = "TRUE"
        elif first_line == 'FALSE':
            verdict = "FALSE"
        elif 'INSUFFICIENT' in first_line:
            verdict = "INSUFFICIENT INFO"
        # Fallback: check if verdict is part of the line
        elif 'TRUE' in first_line and 'FALSE' not in first_line:
            verdict = "TRUE"
        elif 'FALSE' in first_line:
            verdict = "FALSE"
        
        # Extract confidence from second line if present and verdict is TRUE or FALSE
        if len(lines) >= 2 and verdict in ["TRUE", "FALSE"]:
            # Second line should be just the number, but handle "Confidence: X" format too
            confidence_line = lines[1]
            numbers = re.findall(r'\d+', confidence_line)
            if numbers:
                confidence = int(numbers[0])
    
    return verdict, confidence

def query_llm(statement, client, verbose=False):
    """Query LLM with two base prompts, each with initial + reconsideration steps.

    Returns a dict containing initial and reconsidered verdict/confidence for prompts 1 and 2.
    """
    prompts = create_prompts(statement)
    results = {}

    for idx, prompt in enumerate(prompts, start=1):
        try:
            # Initial call
            initial_response = client.invoke(prompt)
            # Extract text content from ClientResponse
            if hasattr(initial_response, 'content') and initial_response.content:
                initial_text = initial_response.content[0].content if hasattr(initial_response.content[0], 'content') else str(initial_response.content[0])
            else:
                initial_text = str(initial_response)
            if verbose:
                print(f"Initial response (prompt {idx}): {initial_text}")
            verdict_initial, confidence_initial = parse_response(initial_text, verbose)
            results[f"verdict-prompt{idx}-initial"] = verdict_initial
            results[f"confidence-prompt{idx}-initial"] = confidence_initial

            # Reconsideration call
            reconsider_prompt = f"{prompt}\n\nPrevious response: {initial_text}\n\n{RECONSIDER_PROMPT}"
            reconsider_response = client.invoke(reconsider_prompt)
            # Extract text content from ClientResponse
            if hasattr(reconsider_response, 'content') and reconsider_response.content:
                reconsider_text = reconsider_response.content[0].content if hasattr(reconsider_response.content[0], 'content') else str(reconsider_response.content[0])
            else:
                reconsider_text = str(reconsider_response)
            if verbose:
                print(f"Reconsidered response (prompt {idx}): {reconsider_text}")
            verdict_final, confidence_final = parse_response(reconsider_text, verbose)
            results[f"verdict-prompt{idx}-reconsidered"] = verdict_final
            results[f"confidence-prompt{idx}-reconsidered"] = confidence_final
        except Exception as e:
            if verbose:
                print(f"Error during LLM query for statement (prompt {idx}): {statement}\n{e}")
            # Detect Azure content filtering to mark policy violation in output
            err_str = str(e).lower()
            policy_violation = (
                "content_filter" in err_str or
                "responsibleaipolicyviolation" in err_str or
                ("error code: 400" in err_str and "policy" in err_str)
            )

            if policy_violation:
                results[f"verdict-prompt{idx}-initial"] = "POLICY VIOLATION"
                results[f"confidence-prompt{idx}-initial"] = None
                # For reconsidered fields, also mark violation to keep columns consistent
                results[f"verdict-prompt{idx}-reconsidered"] = "POLICY VIOLATION"
                results[f"confidence-prompt{idx}-reconsidered"] = None
            else:
                results[f"verdict-prompt{idx}-initial"] = "INSUFFICIENT INFO"
                results[f"confidence-prompt{idx}-initial"] = None
                results[f"verdict-prompt{idx}-reconsidered"] = "INSUFFICIENT INFO"
                results[f"confidence-prompt{idx}-reconsidered"] = None

    return results

def process_statements(statements, client, verbose=False):
    """
    Processes each statement by querying the LLM.
    
    Args:
        statements (list): List of statements to process
        client (OpenAI): The LLM client instance
        verbose (bool): If True, print detailed information
    
    Returns:
        list: List of dictionaries containing the results for each statement
    """
    results = []
    
    for statement in tqdm(statements, desc="Processing statements", unit="statement"):
        llm_results = query_llm(statement, client, verbose)
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
        llm_results = query_llm(statement, client, verbose)
        base = row.to_dict()
        base.update(llm_results)
        rows.append(base)

    df_aug = pd.DataFrame(rows)
    result_cols = [
        'verdict-prompt1-initial', 'confidence-prompt1-initial',
        'verdict-prompt1-reconsidered', 'confidence-prompt1-reconsidered',
        'verdict-prompt2-initial', 'confidence-prompt2-initial',
        'verdict-prompt2-reconsidered', 'confidence-prompt2-reconsidered',
    ]
    ordered_cols = list(df.columns) + [c for c in result_cols if c in df_aug.columns]
    return df_aug[ordered_cols]

def create_client(model_name):
    """
    Creates an OpenAILikeClient configured for local Ollama.
    
    Args:
        model_name (str): Name of the Ollama model to use
    
    Returns:
        OpenAILikeClient: Configured client instance
    """
    client = OpenAILikeClient(
        api_key="",  # Ollama doesn't require an API key
        model=model_name,
        system_prompt="You are a rigorous Fact-Checking Analyst. You function deterministically: identical inputs must always yield identical reasoning paths and conclusions.",
        base_url="http://localhost:11434/v1",  # Default Ollama API endpoint
        temperature=0.0,
    )
    return client

def test_client_connection(client, verbose=False):
    """
    Tests if the Ollama client is working by sending a simple test query.
    
    Args:
        client (OpenAILikeClient): The Ollama client instance to test
        verbose (bool): If True, print detailed information
    
    Raises:
        ConnectionError: If Ollama is not reachable or the connection fails
        ValueError: If the model is not available
    """
    if verbose:
        print("Testing connection to Ollama...")
    
    try:
        # Send a simple test query
        response = client.invoke("Respond with the word 'ok' only.")
        
        if verbose:
            print("✓ Connection to Ollama successful!")
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for common connection errors
        if "connection" in error_msg or "refused" in error_msg or "unreachable" in error_msg:
            raise ConnectionError(
                f"Cannot connect to Ollama. Please check that Ollama is running on localhost:11434.\n"
                f"Error details: {e}"
            )
        elif "model" in error_msg or "not found" in error_msg:
            raise ValueError(
                f"Model not found. Please verify the model is pulled in Ollama.\n"
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

    # Create Azure OpenAI client
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