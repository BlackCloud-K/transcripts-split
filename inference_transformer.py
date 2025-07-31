import os
import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from train_transformer_encoder import TransformerEncoderModel, TransformerEncoderConfig

class TransformerChunkPredictor:
    """TransformerEncoder Chunk predictor - Best solution"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """Initialize predictor"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            checkpoint = torch.load(config_path, map_location='cpu')
            self.config = checkpoint['config']
        else:
            # Use default configuration
            self.config = TransformerEncoderConfig()
        
        # Load sentence encoder
        self.sentence_encoder = SentenceTransformer(self.config.sentence_encoder)
        
        # Create model
        self.model = TransformerEncoderModel(self.config)
        
        # Load model weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model: {model_path}")
            print(f"Best F1 score: {checkpoint.get('best_f1', 'N/A')}")
        else:
            print(f"Warning: Model file does not exist: {model_path}")
            return
        
        self.model.to(self.device)
        self.model.eval()
        
    def predict_sentence_level(self, sentences: List[str]) -> List[int]:
        """Sentence-level chunk boundary prediction"""
        if not sentences:
            return []
        
        # Limit sentence count
        if len(sentences) > self.config.max_sentences:
            sentences = sentences[:self.config.max_sentences]
            print(f"Warning: Sentence count exceeds limit, truncating to first {self.config.max_sentences} sentences")
        
        # Extract sentence embeddings
        sentence_embeddings = self.sentence_encoder.encode(sentences, convert_to_tensor=True)
        sentence_embeddings = sentence_embeddings.unsqueeze(0)  # Add batch dimension
        
        # Create attention mask
        batch_size, seq_len = sentence_embeddings.shape[:2]
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        # Move to device
        sentence_embeddings = sentence_embeddings.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sentence_embeddings, attention_mask)
            probs = torch.sigmoid(outputs)  # Convert to probabilities
            predictions = (probs > 0.5).int()  # Binarize
        
        # Extract predictions
        predictions = predictions.squeeze(0).cpu().numpy()
        
        # Find chunk boundaries (predictions of 1)
        chunk_boundaries = [i for i, pred in enumerate(predictions) if pred == 1]
        
        return chunk_boundaries
    
    def predict_with_confidence(self, sentences: List[str], threshold: float = 0.5) -> Dict:
        """Predict with confidence"""
        if not sentences:
            return {'chunk_boundaries': [], 'confidences': [], 'sentences': []}
        
        # Limit sentence count
        if len(sentences) > self.config.max_sentences:
            sentences = sentences[:self.config.max_sentences]
        
        # Extract sentence embeddings
        sentence_embeddings = self.sentence_encoder.encode(sentences, convert_to_tensor=True)
        sentence_embeddings = sentence_embeddings.unsqueeze(0)
        
        # Create attention mask
        batch_size, seq_len = sentence_embeddings.shape[:2]
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        # Move to device
        sentence_embeddings = sentence_embeddings.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sentence_embeddings, attention_mask)
            probs = torch.sigmoid(outputs)
        
        # Extract results
        probs = probs.squeeze(0).cpu().numpy()
        predictions = (probs > threshold).astype(int)
        
        # Find chunk boundaries and confidences
        chunk_boundaries = []
        confidences = []
        
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            if pred == 1:
                chunk_boundaries.append(i)
                confidences.append(prob)
        
        return {
            'chunk_boundaries': chunk_boundaries,
            'confidences': confidences,
            'all_probs': probs.tolist(),
            'sentences': sentences
        }
    
    def predict_single_text(self, text: str, threshold: float = 0.5) -> Dict:
        """Predict single text"""
        # Simple sentence splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return {
                'text': text,
                'sentences': [],
                'chunk_boundaries': [],
                'num_chunks': 0,
                'confidences': []
            }
        
        # Predict
        result = self.predict_with_confidence(sentences, threshold)
        
        return {
            'text': text,
            'sentences': result['sentences'],
            'chunk_boundaries': result['chunk_boundaries'],
            'num_chunks': len(result['chunk_boundaries']) + 1,
            'confidences': result['confidences'],
            'all_probs': result['all_probs']
        }
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """Batch prediction"""
        results = []
        for text in texts:
            result = self.predict_single_text(text, threshold)
            results.append(result)
        return results

def test_inference():
    """Test inference"""
    # Check if model exists
    model_path = "./transformer_encoder_final/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        print("Please run training script: python train_transformer_encoder.py")
        return
    
    # Create predictor
    predictor = TransformerChunkPredictor(model_path)
    
    # Test text
    test_text = """
    Hello everyone, how are you doing tonight? I'm really excited to be here. 
    Let me tell you about my day. It was absolutely crazy. So I woke up this morning. 
    And the first thing I noticed. My cat was missing. I looked everywhere for her. 
    But then I found her in the kitchen. She was eating my breakfast.
    """
    
    # Predict
    result = predictor.predict_single_text(test_text)
    
    print("Inference results:")
    print(f"Text: {result['text'][:100]}...")
    print(f"Sentence count: {len(result['sentences'])}")
    print(f"Chunk boundaries: {result['chunk_boundaries']}")
    print(f"Chunk count: {result['num_chunks']}")
    print(f"Confidence: {result['confidences']}")
    
    # Display probability for each sentence
    print("\nProbability for each sentence:")
    for i, (sentence, prob) in enumerate(zip(result['sentences'], result['all_probs'])):
        marker = " [CHUNK]" if i in result['chunk_boundaries'] else ""
        print(f"{i+1:2d}. [{prob:.3f}] {sentence[:50]}{'...' if len(sentence) > 50 else ''}{marker}")

def test_with_real_data():
    """Test with real data"""
    # Check if model exists
    model_path = "./transformer_encoder_final/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return
    
    # Create predictor
    predictor = TransformerChunkPredictor(model_path)
    
    # Read a real transcript file
    transcript_file = "processed_transcripts/12 Comics You Need to See - Comedy Central Stand-Up Presents_sentences.txt"
    if not os.path.exists(transcript_file):
        print(f"File does not exist: {transcript_file}")
        return
    
    # Read sentences
    with open(transcript_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    sentences = []
    for line in content.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            parts = line.split('.', 1)
            if len(parts) > 1:
                sentences.append(parts[1].strip())
    
    # Only take first 50 sentences for testing
    test_sentences = sentences[:50]
    
    print(f"Testing with real data: {len(test_sentences)} sentences")
    
    # Predict
    result = predictor.predict_with_confidence(test_sentences)
    
    print(f"Predicted chunk boundaries: {result['chunk_boundaries']}")
    print(f"Confidence: {result['confidences']}")
    
    # Display results
    print("\nPrediction results:")
    for i, sentence in enumerate(test_sentences):
        prob = result['all_probs'][i]
        marker = " [CHUNK]" if i in result['chunk_boundaries'] else ""
        print(f"{i+1:2d}. [{prob:.3f}] {sentence[:60]}{'...' if len(sentence) > 60 else ''}{marker}")

if __name__ == "__main__":
    print("Testing TransformerEncoder inference...")
    test_inference()
    print("\n" + "="*50 + "\n")
    test_with_real_data() 