import argparse
from xmlrpc import client
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from google import genai
from openai import OpenAI



def query_llm_huggingface(statement):
    """
    Sends a statement to the LLM and receives a response.
    
    Args:
        statement (str): The statement to analyze
    
    Returns:
        dict: Dictionary with keys 'response' ("TRUE", "FALSE", or "INSUFFICIENT INFO")
              and 'confidence' (float, only if response is "TRUE" or "FALSE")
    """

    client = InferenceClient(
        api_key=os.environ["HUGGINGFACE_API_TOKEN"],
    )

    prompt = f"""Analyze the following statement and determine if it is TRUE, FALSE, or if there is INSUFFICIENT INFO to make a determination.

    Statement: {statement}

    Please respond in the following format:
    Response: [TRUE/FALSE/INSUFFICIENT INFO]
    Confidence: [0.0-1.0 or N/A if INSUFFICIENT INFO]

    Provide only these two lines in your response."""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    response_text = completion.choices[0].message.content
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

def query_llm_gemini(statement):
    """
    Sends a statement to the LLM and receives a response using Google Gemini.
    
    Args:
        statement (str): The statement to analyze
    
    Returns:
        dict: Dictionary with keys 'response' ("TRUE", "FALSE", or "INSUFFICIENT INFO")
              and 'confidence' (float, only if response is "TRUE" or "FALSE")
    """

    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    prompt = f"""Analyze the following statement and determine if it is TRUE, FALSE, or if there is INSUFFICIENT INFO to make a determination.

    Statement: {statement}

    Please respond in the following format:
    Response: [TRUE/FALSE/INSUFFICIENT INFO]
    Confidence: [0.0-1.0 or N/A if INSUFFICIENT INFO]

    Provide only these two lines in your response."""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    response_text = response.text
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

def query_llm_azure_openai(statement):
    """
    Sends a statement to the LLM and receives a response using Azure OpenAI.
    
    Args:
        statement (str): The statement to analyze
    
    Returns:
        dict: Dictionary with keys 'response' ("TRUE", "FALSE", or "INSUFFICIENT INFO")
              and 'confidence' (float, only if response is "TRUE" or "FALSE")
    """

    endpoint = "https://course-gpt4o-resource.openai.azure.com/openai/v1"
    deployment_name = "gpt-4o"
    
    client = OpenAI(
        base_url=endpoint,
        api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    )

    prompt = f"""Analyze the following statement and determine if it is TRUE, FALSE, or if there is INSUFFICIENT INFO to make a determination.

    Statement: {statement}

    Please respond in the following format:
    Response: [TRUE/FALSE/INSUFFICIENT INFO]
    Confidence: [0.0-1.0 or N/A if INSUFFICIENT INFO]

    Provide only these two lines in your response."""

    completion = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response_text = completion.choices[0].message.content
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
    
    for idx, statement in enumerate(statements):
        if verbose:
            print(f"Processing statement {idx + 1}/{len(statements)}")
        
        # Query the LLM for this statement
        llm_result = query_llm_gemini(statement)
        
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
    load_dotenv()

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