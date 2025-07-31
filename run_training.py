#!/usr/bin/env python3
"""
Improved TransformerEncoder training script
With enhanced imbalanced data handling strategies
"""

import argparse
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_transformer_encoder import main as train_main

def main():
    parser = argparse.ArgumentParser(description="Improved TransformerEncoder training")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, default="training_data.json",
                       help="Training data path")
    parser.add_argument("--output_dir", type=str, default="./transformer_encoder_output",
                       help="Output directory")
    parser.add_argument("--model_save_dir", type=str, default="./model/transformer_encoder_final",
                       help="Model save directory")
    
    # Model configuration
    parser.add_argument("--sentence_encoder", type=str, default="BAAI/bge-small-en-v1.5",
                       help="Sentence encoder model")
    parser.add_argument("--hidden_size", type=int, default=256,
                       help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Attention heads")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20,
                       help="Number of epochs")
    parser.add_argument("--max_sentences", type=int, default=150,
                       help="Maximum sentences")
    
    # Imbalance handling configuration
    parser.add_argument("--use_combined_loss", action="store_true", default=True,
                       help="Use combined loss function")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                       help="Focal Loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal Loss gamma parameter")
    parser.add_argument("--positive_weight", type=float, default=8.0,
                       help="Positive sample weight")
    parser.add_argument("--focal_weight", type=float, default=0.7,
                       help="Focal Loss weight")
    parser.add_argument("--oversample_ratio", type=float, default=3.0,
                       help="Oversampling ratio")
    
    # Other configuration
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use WandB logging")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Import and modify configuration
    from train_transformer_encoder import TransformerEncoderConfig
    
    config = TransformerEncoderConfig()
    
    # Update configuration
    config.data_path = args.data_path
    config.output_dir = args.output_dir
    config.model_save_dir = args.model_save_dir
    config.sentence_encoder = args.sentence_encoder
    config.hidden_size = args.hidden_size
    config.num_layers = args.num_layers
    config.num_heads = args.num_heads
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_epochs = args.num_epochs
    config.max_sentences = args.max_sentences
    config.use_combined_loss = args.use_combined_loss
    config.focal_alpha = args.focal_alpha
    config.focal_gamma = args.focal_gamma
    config.positive_weight = args.positive_weight
    config.focal_weight = args.focal_weight
    config.oversample_ratio = args.oversample_ratio
    config.use_wandb = args.use_wandb
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    print("============================================================")
    print("Improved TransformerEncoder Training Configuration")
    print("============================================================")
    print(f"Data path: {config.data_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Model save directory: {config.model_save_dir}")
    print(f"Sentence encoder: {config.sentence_encoder}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Max sentences: {config.max_sentences}")
    print(f"Use combined loss: {config.use_combined_loss}")
    print(f"Focal Loss parameters: alpha={config.focal_alpha}, gamma={config.focal_gamma}")
    print(f"Positive weight: {config.positive_weight}")
    print(f"Oversampling ratio: {config.oversample_ratio}")
    print(f"Use wandb: {config.use_wandb}")
    print("============================================================")
    
    # Run training
    train_main()

if __name__ == "__main__":
    main() 