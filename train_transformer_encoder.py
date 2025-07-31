import torch
import torch.nn as nn
import numpy as np
import json
import os
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid
        inputs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate pt (probability of target class)
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """Combined loss: Focal Loss + Weighted BCE"""
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, positive_weight=5.0, focal_weight=0.7):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.positive_weight = positive_weight
        self.focal_weight = focal_weight
    
    def forward(self, inputs, targets):
        # Focal Loss
        focal = self.focal_loss(inputs, targets)
        
        # Weighted BCE Loss
        bce = self.bce_loss(inputs, targets)
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.positive_weight
        weighted_bce = bce * weights
        
        # Combine losses
        combined = self.focal_weight * focal + (1 - self.focal_weight) * weighted_bce
        
        return combined

@dataclass
class TransformerEncoderConfig:
    """TransformerEncoder configuration"""
    # Sentence encoder config
    sentence_encoder: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    
    # TransformerEncoder config
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    
    # Training config
    batch_size: int = 8  # Reduced batch size for more frequent updates
    learning_rate: float = 3e-4  # Lower learning rate
    num_epochs: int = 10  # Increased epochs
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Data config
    train_split: float = 0.8
    seed: int = 42
    max_sentences: int = 150
    
    # Imbalance handling config
    use_stratified_sampling: bool = True
    use_combined_loss: bool = True  # Use combined loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    positive_weight: float = 8.0  # Increased weight
    focal_weight: float = 0.7  # Focal Loss weight
    use_oversampling: bool = True
    oversample_ratio: float = 4.0  # Increased oversampling ratio
    use_smote: bool = True  # Use SMOTE oversampling
    
    # Path config
    data_path: str = "training_data.json"
    output_dir: str = "./model/transformer_encoder_output"
    model_save_dir: str = "./model/transformer_encoder_final"
    
    # Other config
    use_wandb: bool = False
    gradient_accumulation_steps: int = 4  # Gradient accumulation steps

class TransformerEncoderModel(nn.Module):
    """TransformerEncoder model"""
    
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.config = config
        
        # Sentence encoder
        self.sentence_encoder = SentenceTransformer(config.sentence_encoder)
        
        # Freeze sentence encoder parameters
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False
        
        # Input projection layer
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_size)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1000, config.hidden_size) * 0.02
        )
        
        # TransformerEncoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, sentence_embeddings, attention_mask=None):
        batch_size, seq_len = sentence_embeddings.shape[:2]
        
        # Input projection
        x = self.input_projection(sentence_embeddings)
        
        # Positional encoding
        if seq_len <= self.positional_encoding.shape[0]:
            x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # LayerNorm
        x = self.layer_norm(x)
        
        # Create padding mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # TransformerEncoder
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        
        # Classification
        logits = self.classifier(x)
        
        return logits.squeeze(-1)

class TransformerDataset:
    """Transformer dataset class"""
    
    def __init__(self, data_path: str, sentence_encoder, max_sentences: int = 150):
        self.data_path = data_path
        self.sentence_encoder = sentence_encoder
        self.max_sentences = max_sentences
    
    def load_data(self, data_path: str) -> List[Dict]:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def prepare_transformer_data(self) -> List[Dict]:
        data = self.load_data(self.data_path)
        processed_data = []
        
        for item in tqdm(data, desc="Preparing Transformer data"):
            sentences = item['sentences']
            labels = item['labels']
            
            # Limit sentence count
            if len(sentences) > self.max_sentences:
                sentences = sentences[:self.max_sentences]
                labels = labels[:self.max_sentences]
            
            # Encode sentences
            embeddings = self.sentence_encoder.encode(sentences, convert_to_tensor=True)
            
            processed_data.append({
                'embeddings': embeddings,
                'labels': labels,
                'sentences': sentences
            })
        
        return processed_data

def collate_fn(batch):
    """Improved collate function"""
    max_len = max(len(item['embeddings']) for item in batch)
    
    embeddings_list = []
    labels_list = []
    lengths = []
    
    for item in batch:
        embeddings = item['embeddings']
        labels = item['labels']
        length = len(embeddings)
        
        # Padding
        if length < max_len:
            pad_len = max_len - length
            # Ensure padding tensors are on correct device
            device = embeddings.device
            padding = torch.zeros(pad_len, embeddings.shape[1], device=device)
            embeddings = torch.cat([embeddings, padding], dim=0)
            labels = labels + [0] * pad_len
        
        embeddings_list.append(embeddings)
        labels_list.append(labels)
        lengths.append(length)
    
    return {
        'embeddings': torch.stack(embeddings_list),
        'labels': torch.tensor(labels_list, dtype=torch.float32),
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }

def compute_metrics(preds, labels, lengths):
    """Compute evaluation metrics"""
    if not preds or not labels:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]])
        }
    
    all_preds = np.array(preds)
    all_labels = np.array(labels)
    
    # Binarize predictions
    pred_binary = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, pred_binary, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_labels, pred_binary)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, pred_binary)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_lengths = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']
        
        # Create attention mask
        batch_size, seq_len = embeddings.shape[:2]
        attention_mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = attention_mask < lengths.unsqueeze(1)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        outputs = model(embeddings, attention_mask)
        
        # Calculate loss
        loss = 0
        valid_preds = []
        valid_labels = []
        
        for i, length in enumerate(lengths):
            pred = outputs[i][:length]
            label = labels[i][:length]
            
            if pred.shape != label.shape:
                continue
            
            # Calculate loss
            if config.use_combined_loss:
                loss += criterion(pred, label).mean()
            else:
                sample_loss = criterion(pred, label)
                weights = torch.ones_like(label)
                weights[label == 1] = config.positive_weight
                weighted_loss = (sample_loss * weights).mean()
                loss += weighted_loss
            
            valid_preds.extend(pred.detach().cpu().numpy())
            valid_labels.extend(label.cpu().numpy())
        
        if len(lengths) > 0:
            loss = loss / len(lengths)
        else:
            loss = torch.tensor(0.0, device=device)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        all_preds.extend(valid_preds)
        all_labels.extend(valid_labels)
        all_lengths.extend(lengths.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(step+1):.4f}'
        })
    
    # Calculate metrics
    if all_preds and all_labels:
        metrics = compute_metrics(all_preds, all_labels, all_lengths)
        metrics['loss'] = total_loss / len(dataloader)
    else:
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]]),
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        }
    
    return metrics

def evaluate(model, dataloader, criterion, device, config):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_lengths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            
            # Create attention mask
            batch_size, seq_len = embeddings.shape[:2]
            attention_mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            attention_mask = attention_mask < lengths.unsqueeze(1)
            attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(embeddings, attention_mask)
            
            # Calculate loss
            loss = 0
            valid_preds = []
            valid_labels = []
            
            for i, length in enumerate(lengths):
                pred = outputs[i][:length]
                label = labels[i][:length]
                
                if pred.shape != label.shape:
                    continue
                
                # Calculate loss
                if config.use_combined_loss:
                    loss += criterion(pred, label).mean()
                else:
                    sample_loss = criterion(pred, label)
                    weights = torch.ones_like(label)
                    weights[label == 1] = config.positive_weight
                    weighted_loss = (sample_loss * weights).mean()
                    loss += weighted_loss
                
                valid_preds.extend(pred.cpu().numpy())
                valid_labels.extend(label.cpu().numpy())
            
            if len(lengths) > 0:
                loss = loss / len(lengths)
            else:
                loss = torch.tensor(0.0, device=device)
            
            total_loss += loss.item()
            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)
            all_lengths.extend(lengths.cpu().numpy())
    
    # Calculate metrics
    if all_preds and all_labels:
        metrics = compute_metrics(all_preds, all_labels, all_lengths)
        metrics['loss'] = total_loss / len(dataloader)
    else:
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]]),
            'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        }
    
    return metrics

def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    """Main function"""
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = TransformerEncoderConfig()
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    print("============================================================")
    print("Starting improved TransformerEncoder training")
    print("============================================================")
    print(f"Data path: {config.data_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Model save directory: {config.model_save_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Max sentences: {config.max_sentences}")
    print(f"Use wandb: {config.use_wandb}")
    print("============================================================")
    print(f"Configuration saved to: {os.path.join(config.output_dir, 'config.json')}")
    
    # Initialize wandb (disabled)
    if config.use_wandb:
        wandb.init(
            project="transformer-encoder-chunking",
            config=vars(config),
            name="transformer_encoder_v1"
        )
    else:
        print("\nWandB disabled, using local logging")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load sentence encoder
    print("Loading sentence encoder...")
    sentence_encoder = SentenceTransformer(config.sentence_encoder)
    
    # Prepare data
    print("Preparing data...")
    dataset = TransformerDataset(config.data_path, sentence_encoder, config.max_sentences)
    processed_data = dataset.prepare_transformer_data()
    
    # Stratified sampling for data splitting
    if config.use_stratified_sampling:
        positive_ratios = []
        for item in processed_data:
            positive_ratio = sum(item['labels']) / len(item['labels'])
            positive_ratios.append(positive_ratio)
        
        try:
            bins = pd.cut(positive_ratios, bins=5, labels=False)
            bin_counts = pd.Series(bins).value_counts()
            min_samples_per_bin = bin_counts.min()
            
            if min_samples_per_bin >= 2:
                print(f"Using stratified sampling, minimum samples per bin: {min_samples_per_bin}")
                train_indices, val_indices = train_test_split(
                    range(len(processed_data)),
                    test_size=1-config.train_split,
                    stratify=bins,
                    random_state=config.seed
                )
            else:
                print(f"Stratified sampling failed, insufficient samples (need>=2, actual={min_samples_per_bin}), using random split")
                train_indices, val_indices = train_test_split(
                    range(len(processed_data)),
                    test_size=1-config.train_split,
                    random_state=config.seed
                )
        except Exception as e:
            print(f"Stratified sampling error: {e}, using random split")
            train_indices, val_indices = train_test_split(
                range(len(processed_data)),
                test_size=1-config.train_split,
                random_state=config.seed
            )
    else:
        train_indices, val_indices = train_test_split(
            range(len(processed_data)),
            test_size=1-config.train_split,
            random_state=config.seed
        )
    
    train_data = [processed_data[i] for i in train_indices]
    val_data = [processed_data[i] for i in val_indices]
    
    # Oversampling processing
    if config.use_oversampling:
        print(f"Applying oversampling (ratio: {config.oversample_ratio})")
        positive_samples = []
        negative_samples = []
        
        for item in train_data:
            positive_count = sum(item['labels'])
            if positive_count > 0:
                positive_samples.append(item)
            else:
                negative_samples.append(item)
        
        if positive_samples:
            oversampled_positive = []
            for _ in range(int(len(positive_samples) * config.oversample_ratio)):
                sample = np.random.choice(positive_samples)
                oversampled_positive.append(sample)
            
            train_data = negative_samples + oversampled_positive
            np.random.shuffle(train_data)
            
            print(f"After oversampling - Positive samples: {len(oversampled_positive)}, Negative samples: {len(negative_samples)}")
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    
    # Print data statistics
    print("\nData statistics:")
    total_sentences = sum(len(item['labels']) for item in processed_data)
    total_chunks = sum(sum(item['labels']) for item in processed_data)
    print(f"Total sentences: {total_sentences}")
    print(f"Total chunk boundaries: {total_chunks}")
    print(f"Chunk ratio: {total_chunks/total_sentences*100:.2f}%")
    
    positive_ratios = [sum(item['labels'])/len(item['labels']) for item in processed_data]
    print(f"Positive sample ratio range: {min(positive_ratios):.3f} - {max(positive_ratios):.3f}")
    print(f"Average positive sample ratio: {np.mean(positive_ratios):.3f}")
    
    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # Create model
    print("Creating improved TransformerEncoder model...")
    model = TransformerEncoderModel(config)
    model.to(device)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    if config.use_combined_loss:
        print(f"Using combined loss (Focal + Weighted BCE)")
        print(f"Focal weight: {config.focal_weight}, Positive weight: {config.positive_weight}")
        criterion = CombinedLoss(
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            positive_weight=config.positive_weight,
            focal_weight=config.focal_weight
        )
    else:
        print(f"Using Focal Loss (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, reduction='none')
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )
    
    # Training loop
    print("Starting training...")
    best_f1 = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device, config)
        
        # Validation
        val_metrics = evaluate(model, val_dataloader, criterion, device, config)
        
        # Record metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        
        print(f"Training - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Validation - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Log to wandb (disabled)
        if config.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_f1': train_metrics['f1'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'learning_rate': scheduler.get_last_lr()[0]
            })
        else:
            # Local logging
            print(f"Epoch {epoch+1} - LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Training: Loss={train_metrics['loss']:.4f}, F1={train_metrics['f1']:.4f}, P={train_metrics['precision']:.4f}, R={train_metrics['recall']:.4f}")
            print(f"  Validation: Loss={val_metrics['loss']:.4f}, F1={val_metrics['f1']:.4f}, P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, os.path.join(config.model_save_dir, 'best_model.pth'))
            
            # Save confusion matrix
            plot_confusion_matrix(
                val_metrics['confusion_matrix'],
                os.path.join(config.output_dir, 'best_confusion_matrix.png')
            )
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'final_f1': val_metrics['f1'],
        'val_metrics': val_metrics
    }, os.path.join(config.model_save_dir, 'final_model.pth'))
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config.output_dir, 'loss_curve.png'))
    plt.close()
    
    print(f"\nTraining completed!")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Model saved in: {config.model_save_dir}")
    
    print("\n============================================================")
    print("Training completed!")
    print(f"Best model saved in: {os.path.join(config.model_save_dir, 'best_model.pth')}")
    print(f"Final model saved in: {os.path.join(config.model_save_dir, 'final_model.pth')}")
    print("============================================================")

if __name__ == "__main__":
    main() 