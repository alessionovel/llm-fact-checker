import argparse
import pandas as pd
import os
import sys
import ollama
from tqdm import tqdm



def query_llm_api(statement):
    """
    Template function showing how to integrate any LLM API service.
    This function is not called but serves as a guide for implementation.
    
    To use this template:
    1. Choose your preferred API service (OpenAI, Anthropic, Hugging Face, Google Gemini, etc.)
    2. Install the required SDK/library for your chosen service
    3. Set up authentication (API keys, endpoints) via environment variables
    4. Initialize the client with your service-specific configuration
    5. Make the API call using the service's SDK
    6. Parse the response according to the expected format below
    
    Args:
        statement (str): The statement to analyze
    
    Returns:
        dict: Dictionary with keys 'response' ("TRUE", "FALSE", or "INSUFFICIENT INFO")
              and 'confidence' (float, only if response is "TRUE" or "FALSE")
    
    Example implementations:
    
    # OpenAI
    # from openai import OpenAI
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # completion = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    # response_text = completion.choices[0].message.content
    
    # Anthropic
    # from anthropic import Anthropic
    # client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # message = client.messages.create(model="claude-3-opus-20240229", messages=[{"role": "user", "content": prompt}])
    # response_text = message.content[0].text
    
    # Hugging Face
    # from huggingface_hub import InferenceClient
    # client = InferenceClient(api_key=os.environ["HUGGINGFACE_API_TOKEN"])
    # completion = client.chat.completions.create(model="model-name", messages=[{"role": "user", "content": prompt}])
    # response_text = completion.choices[0].message.content
    
    # Google Gemini
    # from google import genai
    # client = genai.Client()
    # response = client.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
    # response_text = response.text
    """
    
    # Construct the prompt
    prompt = f"""Analyze the following statement and determine if it is TRUE, FALSE, or if there is INSUFFICIENT INFO to make a determination.

    Statement: {statement}

    Please respond in the following format:
    Response: [TRUE/FALSE/INSUFFICIENT INFO]
    Confidence: [0.0-1.0 or N/A if INSUFFICIENT INFO]

    Provide only these two lines in your response."""

    # TODO: Initialize your API client here
    # client = YourAPIClient(api_key=os.environ.get("YOUR_API_KEY"))
    
    # TODO: Make the API call
    # response = client.your_api_method(prompt)
    # response_text = response.your_text_field
    
    # For demonstration purposes:
    response_text = "Response: INSUFFICIENT INFO\nConfidence: N/A"
    
    # Parse the response (this logic remains the same regardless of API)
    result = {
        'response': 'INSUFFICIENT INFO',
        'confidence': None
    }
    
    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('Response:'):
                response_value = line.split(':', 1)[1].strip().upper()
                if response_value in ['TRUE', 'FALSE', 'INSUFFICIENT INFO']:
                    result['response'] = response_value
            elif line.startswith('Confidence:'):
                confidence_value = line.split(':', 1)[1].strip()
                if confidence_value.upper() != 'N/A':
                    try:
                        result['confidence'] = float(confidence_value)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
    
    return result

def query_llm_ollama(statement, verbose=False):
    """
    Sends a statement to the LLM and receives a response using Ollama.
    
    Args:
        statement (str): The statement to analyze
        verbose (bool): If True, print the LLM response
    
    Returns:
        dict: Dictionary with keys 'response' ("TRUE", "FALSE", or "INSUFFICIENT INFO")
              and 'confidence' (float, only if response is "TRUE" or "FALSE")
    """

    prompt = f"""Analyze the following statement and determine if it is TRUE, FALSE, or if there is INSUFFICIENT INFO to make a determination.

    Statement: {statement}

    Please respond in the following format:
    Response: [TRUE/FALSE/INSUFFICIENT INFO]
    Confidence: [0.0-1.0 or N/A if INSUFFICIENT INFO]

    Provide only these two lines in your response."""

    response = ollama.chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    response_text = response['message']['content']
    if verbose:
        print(f"LLM Response: {response_text}")

    # Parse the response
    result = {
        'response': 'INSUFFICIENT INFO',
        'confidence': None
    }
    
    try:
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('Response:'):
                response_value = line.split(':', 1)[1].strip().upper()
                if response_value in ['TRUE', 'FALSE', 'INSUFFICIENT INFO']:
                    result['response'] = response_value
            elif line.startswith('Confidence:'):
                confidence_value = line.split(':', 1)[1].strip()
                if confidence_value.upper() != 'N/A':
                    try:
                        result['confidence'] = float(confidence_value)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
    
    return result

def process_statements(statements, verbose=False):
    """
    Processes each statement by querying the LLM.
    
    Args:
        statements (list): List of statements to process
        verbose (bool): If True, print detailed information
    
    Returns:
        list: List of dictionaries containing the results for each statement
    """
    results = []
    
    for idx, statement in enumerate(tqdm(statements, desc="Processing statements", unit="statement")):
        if verbose:
            print(f"Processing statement {idx + 1}/{len(statements)}")
        
        # Query the LLM for this statement
        llm_result = query_llm_ollama(statement, verbose)
        
        results.append({
            'statement': statement,
            'response': llm_result['response'],
            'confidence': llm_result['confidence']
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
    
    # Store sentences from the "Statement" column in a vector (list)
    statements = df['Statement'].tolist()
    
    if verbose:
        print(f"--- Total rows found: {len(df)} ---")
        print(f"--- Total statements extracted: {len(statements)} ---")
        print(f"Successfully extracted {len(statements)} statements from the 'Statement' column.")
    
    return statements

def main():

    # 1. Create the parser
    parser = argparse.ArgumentParser(
        description="Process Excel file statements through LLM and save results with confidence scores."
    )

    # 2. Add arguments (parameters)
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

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Script Logic
    file_path = args.file_input
    output_path = args.output_file

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    if args.verbose:
        print(f"--- Starting processing of file: {file_path} ---")

    try:
        # Read statements from Excel
        statements = read_excel_file(file_path, args.verbose)
        
        # Process each statement with LLM
        results = process_statements(statements, args.verbose)
        
        # Save results to output Excel file
        save_results_to_excel(results, output_path)
        
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    main()