import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

def read_chunk_boundaries(chunk_file_path: str) -> List[int]:
    """Read chunk boundary file, return 0-based index list"""
    try:
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Parse Python list format
            import ast
            boundaries = ast.literal_eval(content)
            return boundaries
    except Exception as e:
        print(f"Error reading chunk file {chunk_file_path}: {e}")
        return []

def read_sentences(sentences_file_path: str) -> List[str]:
    """Read sentences file, return sentence list"""
    try:
        with open(sentences_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            sentences = []
            for line in content.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    # Extract sentence content (remove numbering)
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        sentences.append(parts[1].strip())
            return sentences
    except Exception as e:
        print(f"Error reading sentences file {sentences_file_path}: {e}")
        return []

def create_labels(sentences: List[str], chunk_boundaries: List[int]) -> List[int]:
    """Create labels for each sentence position: chunk start=1, others=0"""
    labels = [0] * len(sentences)
    
    # Set chunk boundaries to 1
    for boundary in chunk_boundaries:
        if 0 <= boundary < len(sentences):
            labels[boundary] = 1
    
    return labels

def create_training_sample(sentences: List[str], labels: List[int], filename: str) -> Dict:
    """Create single training sample"""
    return {
        "filename": filename,
        "sentences": sentences,
        "labels": labels,
        "num_sentences": len(sentences),
        "num_chunks": sum(labels)
    }

def process_all_files():
    """Process all files and generate training data"""
    chunked_dir = Path("chunked_transcripts")
    processed_dir = Path("processed_transcripts")
    
    if not chunked_dir.exists():
        print("chunked_transcripts directory not found")
        return
    
    if not processed_dir.exists():
        print("processed_transcripts directory not found")
        return
    
    # Get all chunk files
    chunk_files = list(chunked_dir.glob("*_chunks.txt"))
    print(f"Found {len(chunk_files)} chunk files")
    
    training_data = []
    successful_files = 0
    
    for chunk_file in chunk_files:
        # Construct corresponding sentences filename
        base_name = chunk_file.name.replace('_chunks.txt', '')
        sentences_file = processed_dir / f"{base_name}_sentences.txt"
        
        if not sentences_file.exists():
            print(f"Warning: {sentences_file} not found, skipping {chunk_file.name}")
            continue
        
        print(f"Processing: {chunk_file.name}")
        
        # Read data
        chunk_boundaries = read_chunk_boundaries(str(chunk_file))
        sentences = read_sentences(str(sentences_file))
        
        if not chunk_boundaries or not sentences:
            print(f"Warning: Empty data for {chunk_file.name}, skipping")
            continue
        
        # Create labels
        labels = create_labels(sentences, chunk_boundaries)
        
        # Create training sample
        sample = create_training_sample(sentences, labels, base_name)
        training_data.append(sample)
        
        successful_files += 1
        print(f"  - Sentences: {len(sentences)}, Chunks: {sum(labels)}")
    
    # Save training data
    output_file = "training_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful_files} files")
    print(f"Total samples: {len(training_data)}")
    print(f"Training data saved to: {output_file}")
    
    # Statistics
    if training_data:
        total_sentences = sum(sample['num_sentences'] for sample in training_data)
        total_chunks = sum(sample['num_chunks'] for sample in training_data)
        avg_sentences = total_sentences / len(training_data)
        avg_chunks = total_chunks / len(training_data)
        
        print(f"\nStatistics:")
        print(f"  - Total sentences: {total_sentences}")
        print(f"  - Total chunks: {total_chunks}")
        print(f"  - Average sentences per transcript: {avg_sentences:.1f}")
        print(f"  - Average chunks per transcript: {avg_chunks:.1f}")
        print(f"  - Chunk ratio: {total_chunks/total_sentences*100:.2f}%")


if __name__ == "__main__":
    process_all_files() 