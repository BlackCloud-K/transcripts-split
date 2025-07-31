#!/usr/bin/env python3
"""
Test trained model for English comedy transcript chunk boundary prediction
"""

import torch
import torch.nn as nn
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from train_transformer_encoder import TransformerEncoderModel, TransformerEncoderConfig

def load_model(model_path):
    """Load trained model"""
    print(f"Loading model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = TransformerEncoderModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully, using device: {device}")
    print(f"Best F1 score: {checkpoint.get('best_f1', 'N/A')}")
    
    return model, config, device

def preprocess_transcript(sentences, sentence_encoder, max_sentences=150):
    """Preprocess transcript"""
    # Limit sentence count
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    
    # Encode sentences
    embeddings = sentence_encoder.encode(sentences, convert_to_tensor=True)
    
    return embeddings, sentences

def predict_chunks(model, embeddings, sentences, device, threshold=0.5):
    """Predict chunk boundaries"""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        embeddings = embeddings.unsqueeze(0).to(device)
        
        # Create attention mask
        batch_size, seq_len = embeddings.shape[:2]
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Forward pass
        outputs = model(embeddings, attention_mask)
        
        # Apply sigmoid
        probabilities = torch.sigmoid(outputs[0])
        
        # Convert to numpy
        probs = probabilities.cpu().numpy()
        
        # Predict chunk boundaries
        predictions = (probs > threshold).astype(int)
        
        return probs, predictions

def format_output(sentences, probabilities, predictions, threshold=0.5):
    """Format output results"""
    print(f"\n{'='*80}")
    print(f"CHUNK BOUNDARY PREDICTION RESULTS (threshold: {threshold})")
    print(f"{'='*80}")
    
    chunk_boundaries = []
    current_chunk = []
    
    for i, (sentence, prob, pred) in enumerate(zip(sentences, probabilities, predictions)):
        current_chunk.append(sentence)
        
        if pred == 1:
            # This is a chunk boundary
            chunk_boundaries.append({
                'start_sentence': len(current_chunk) - len(sentence.split()),
                'end_sentence': i,
                'sentences': current_chunk.copy(),
                'confidence': prob
            })
            current_chunk = []
    
    # Add last chunk
    if current_chunk:
        chunk_boundaries.append({
            'start_sentence': len(sentences) - len(current_chunk),
            'end_sentence': len(sentences) - 1,
            'sentences': current_chunk,
            'confidence': 0.0
        })
    
    print(f"Detected {len(chunk_boundaries)} chunks:")
    print()
    
    for i, chunk in enumerate(chunk_boundaries):
        print(f"Chunk {i+1} (confidence: {chunk['confidence']:.3f}):")
        print(f"Sentences {chunk['start_sentence']+1}-{chunk['end_sentence']+1}:")
        for j, sent in enumerate(chunk['sentences']):
            print(f"  {chunk['start_sentence']+j+1}. {sent}")
        print()
    
    # Show prediction probabilities for each sentence
    print(f"\n{'='*80}")
    print("DETAILED PREDICTION PROBABILITIES:")
    print(f"{'='*80}")
    for i, (sentence, prob, pred) in enumerate(zip(sentences, probabilities, predictions)):
        marker = "🔴 CHUNK BOUNDARY" if pred == 1 else "  "
        print(f"{i+1:2d}. [{prob:.3f}] {marker} {sentence}")
    
    return chunk_boundaries

def test_sample_transcripts():
    """Test sample transcripts"""
    
    # Load model
    model, config, device = load_model("./model/transformer_encoder_final/best_model.pth")
    
    # Load sentence encoder
    sentence_encoder = SentenceTransformer(config.sentence_encoder)
    
    # Sample 1: Typical stand-up comedy segment
    sample1 = [
        "So I was at the grocery store yesterday, and this woman was arguing with the cashier about expired coupons.",
        "I mean, who does that?",
        "You're literally fighting over 50 cents.",
        "Just pay the damn money and move on with your life.",
        "But no, she's standing there for 20 minutes, holding up the entire line.",
        "I'm thinking, lady, your time is worth more than 50 cents.",
        "Unless you're making less than $1.50 an hour, which would explain a lot about your life choices.",
        "Anyway, this got me thinking about how we value our time.",
        "We'll spend hours on social media but won't wait 5 minutes for a table at a restaurant.",
        "We'll drive 20 minutes to save $2 on gas, but we'll pay $5 for a coffee we could make at home.",
        "It's like we're all terrible at math when it comes to our own time.",
        "Speaking of time, I've been trying to learn guitar lately.",
        "I thought it would be easy, you know?",
        "Just strum a few chords, write some songs, become famous.",
        "Turns out, it's actually really hard.",
        "My fingers hurt, my neighbors hate me, and I still can't play 'Wonderwall'.",
        "I mean, if you can't play 'Wonderwall', can you really call yourself a guitarist?",
        "That's like saying you can cook but you can't make toast.",
        "But I'm determined to keep practicing.",
        "Someday, I'll be able to play at open mic nights and make people regret coming to the bar."
    ]
    
    # Sample 2: Another comedy segment
    sample2 = [
        "You know what's weird about dating apps?",
        "Everyone's photos look like they were taken by a professional photographer.",
        "But then you meet them in person and they look like they were drawn by a child.",
        "It's like they hired a magician to take their pictures.",
        "I'm not saying you should catfish people, but maybe don't use photos from 10 years ago.",
        "Or photos where you're clearly using filters that make you look like a cartoon character.",
        "I matched with this girl once, and her profile said she was 'adventure-seeking'.",
        "I thought, great, she likes hiking and travel.",
        "Turns out, her idea of adventure is trying a new flavor of ice cream.",
        "I mean, that's fine, but maybe be more specific.",
        "Like, 'I'm adventurous with dessert choices'.",
        "Or 'I like to try new restaurants'.",
        "Don't make it sound like you're going to climb Mount Everest when you're really just going to try the mint chocolate chip.",
        "But you know what's worse than dating apps?",
        "Dating in real life.",
        "At least with apps, you know they're single and looking.",
        "In real life, you have to figure out if they're interested, if they're single, if they're not a serial killer.",
        "It's like playing detective, but with higher stakes.",
        "And worse consequences if you get it wrong."
    ]
    
    # Test sample 1
    print("\n" + "="*100)
    print("TEST SAMPLE 1: Grocery Store Experience + Guitar Learning")
    print("="*100)
    
    embeddings1, sentences1 = preprocess_transcript(sample1, sentence_encoder, config.max_sentences)
    probs1, preds1 = predict_chunks(model, embeddings1, sentences1, device, threshold=0.5)
    chunks1 = format_output(sentences1, probs1, preds1, threshold=0.5)
    
    # Test sample 2
    print("\n" + "="*100)
    print("TEST SAMPLE 2: Dating App Rant")
    print("="*100)
    
    embeddings2, sentences2 = preprocess_transcript(sample2, sentence_encoder, config.max_sentences)
    probs2, preds2 = predict_chunks(model, embeddings2, sentences2, device, threshold=0.5)
    chunks2 = format_output(sentences2, probs2, preds2, threshold=0.5)
    
    # Test different thresholds
    print("\n" + "="*100)
    print("TESTING DIFFERENT THRESHOLDS")
    print("="*100)
    
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        print(f"\nThreshold {threshold}:")
        probs, preds = predict_chunks(model, embeddings1, sentences1, device, threshold=threshold)
        num_chunks = sum(preds) + 1  # +1 for the last chunk
        print(f"Detected {num_chunks} chunks")
        print(f"Average confidence: {np.mean(probs):.3f}")

if __name__ == "__main__":
    test_sample_transcripts() 