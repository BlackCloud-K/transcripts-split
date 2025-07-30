import os
import json
import time
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

def setup_gemini_model():
    """Setup Gemini 2.5 Flash model"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        print(f"Error setting up Gemini model: {e}")
        return None

def read_prompt() -> str:
    """Read the prompt from prompt.txt"""
    try:
        with open('prompt.txt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("prompt.txt not found")
        return ""

def read_transcript_file(file_path: str) -> str:
    """Read transcript file and return content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_sentence_numbers(content: str) -> List[int]:
    """Extract sentence numbers from transcript content"""
    sentences = []
    for line in content.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Extract the sentence number
            parts = line.split('.', 1)
            if len(parts) > 1:
                try:
                    sentence_num = int(parts[0])
                    sentences.append(sentence_num)
                except ValueError:
                    continue
    return sentences

def call_gemini_api(model, prompt: str, transcript_content: str) -> str:
    """Call Gemini API with prompt and transcript"""
    try:
        # Combine prompt and transcript
        full_prompt = f"{prompt}\n\n{transcript_content}"
        
        # Call Gemini API
        response = model.generate_content(full_prompt)
        
        # Extract the response text
        result = response.text.strip()
        
        # Try to extract the list from the response
        # Look for Python-style list in the response
        import re
        list_pattern = r'\[[\d,\s]+\]'
        match = re.search(list_pattern, result)
        
        if match:
            list_str = match.group()
            # Convert from 1-based to 0-based indexing
            try:
                # Parse the list and convert to 0-based
                import ast
                numbers = ast.literal_eval(list_str)
                zero_based = [n - 1 for n in numbers if n > 0]  # Convert to 0-based and filter out 0
                return str(zero_based)
            except:
                # If parsing fails, return the original string
                return list_str
        else:
            print(f"Warning: Could not extract list from response: {result}")
            return result
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return ""

def save_chunk_boundaries(filename: str, chunk_boundaries: str):
    """Save chunk boundaries to file"""
    output_dir = Path("chunked_transcripts")
    output_dir.mkdir(exist_ok=True)
    
    # Create output filename
    base_name = filename.replace('_sentences.txt', '')
    output_file = output_dir / f"{base_name}_chunks.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chunk_boundaries)
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

def process_all_transcripts():
    """Process all transcript files"""
    # Setup model
    model = setup_gemini_model()
    if not model:
        return
    
    # Read prompt
    prompt = read_prompt()
    if not prompt:
        return
    
    # Get all transcript files
    transcripts_dir = Path("processed_transcripts")
    if not transcripts_dir.exists():
        print("processed_transcripts directory not found")
        return
    
    transcript_files = list(transcripts_dir.glob("*_sentences.txt"))
    print(f"Found {len(transcript_files)} transcript files to process")
    
    # Process each file
    for i, file_path in enumerate(transcript_files, 1):
        print(f"\nProcessing {i}/{len(transcript_files)}: {file_path.name}")
        
        # Read transcript content
        transcript_content = read_transcript_file(str(file_path))
        if not transcript_content:
            print(f"Skipping {file_path.name} - empty content")
            continue
        
        # Call Gemini API
        print("Calling Gemini API...")
        chunk_boundaries = call_gemini_api(model, prompt, transcript_content)
        
        if chunk_boundaries:
            # Save results
            save_chunk_boundaries(file_path.name, chunk_boundaries)
        else:
            print(f"Failed to get chunk boundaries for {file_path.name}")
        
        # Add delay to avoid rate limiting
        time.sleep(6)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    process_all_transcripts() 