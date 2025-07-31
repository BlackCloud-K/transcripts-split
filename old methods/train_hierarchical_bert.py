import torch
import torch.nn as nn
import numpy as np
import json
import os
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

@dataclass
class HierarchicalBertConfig:
    """Hierarchical BERT configuration"""
    # Sentence encoder configuration
    sentence_encoder: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # BGE-small的维度
    
    # 序列模型配置
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    
    # 训练配置
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 10
    weight_decay: float = 0.01
    
    # 数据配置
    train_split: float = 0.8
    seed: int = 42
    max_sentences: int = 100  # 每个transcript最大句子数
    
    # 不平衡处理配置
    use_stratified_sampling: bool = True
    positive_weight: float = 2.0
    
    # 路径配置
    data_path: str = "training_data.json"
    output_dir: str = "./hierarchical_bert_output"
    model_save_dir: str = "./hierarchical_bert_final"
    
    # 其他配置
    use_wandb: bool = True

class HierarchicalBertModel(nn.Module):
    """Hierarchical BERT model"""
    
    def __init__(self, config: HierarchicalBertConfig):
        super().__init__()
        self.config = config
        
        # Sentence encoder
        self.sentence_encoder = SentenceTransformer(config.sentence_encoder)
        
        # Freeze sentence encoder parameters
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sentence_embeddings, lengths=None):
        # sentence_embeddings: (batch_size, max_len, embedding_dim)
        batch_size, max_len, _ = sentence_embeddings.shape
        
        # Create packed sequence
        if lengths is not None:
            packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                sentence_embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed_embeddings)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=max_len)
        else:
            lstm_out, _ = self.lstm(sentence_embeddings)
        
        # Classification
        logits = self.classifier(lstm_out)  # (batch_size, max_len, 1)
        return logits.squeeze(-1)  # (batch_size, max_len)

class HierarchicalDataset:
    """Hierarchical dataset"""
    
    def __init__(self, data_path: str, sentence_encoder, max_sentences: int = 100):
        self.sentence_encoder = sentence_encoder
        self.max_sentences = max_sentences
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict]:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} transcript files")
        return data
    
    def prepare_hierarchical_data(self) -> List[Dict]:
        """Prepare hierarchical data - sentence-level balancing"""
        processed_data = []
        
        for sample in tqdm(self.data, desc="Preparing hierarchical data"):
            sentences = sample['sentences']
            labels = sample['labels']
            
            # Limit sentence count
            if len(sentences) > self.max_sentences:
                sentences = sentences[:self.max_sentences]
                labels = labels[:self.max_sentences]
            
            # Extract sentence embeddings
            sentence_embeddings = self.sentence_encoder.encode(sentences, convert_to_tensor=True)
            
            processed_data.append({
                'sentence_embeddings': sentence_embeddings.cpu().numpy(),
                'labels': labels,
                'length': len(sentences),
                'filename': sample['filename']
            })
        
        print(f"Original data statistics:")
        print(f"    Total files: {len(processed_data)}")
        
        # Count positive and negative samples at sentence level
        total_sentences = 0
        total_positive = 0
        for d in processed_data:
            total_sentences += d['length']
            total_positive += sum(d['labels'])
        
        print(f"    Total sentences: {total_sentences}")
        print(f"    Positive samples: {total_positive}")
        print(f"    Negative samples: {total_sentences - total_positive}")
        print(f"    Positive sample ratio: {total_positive/total_sentences*100:.1f}%")
        
        # Sentence-level data balancing
        print(f"\nStarting sentence-level data balancing...")
        
        # Collect all sentence-level samples
        all_sentence_samples = []
        for d in processed_data:
            for i in range(d['length']):
                all_sentence_samples.append({
                    'embedding': d['sentence_embeddings'][i],
                    'label': d['labels'][i],
                    'filename': d['filename'],
                    'sentence_index': i
                })
        
        # Separate positive and negative samples
        positive_samples = [s for s in all_sentence_samples if s['label'] == 1]
        negative_samples = [s for s in all_sentence_samples if s['label'] == 0]
        
        print(f"    Sentence level - Positive samples: {len(positive_samples)}")
        print(f"    Sentence level - Negative samples: {len(negative_samples)}")
        
        # Target: at least 1 positive sample in every 3 samples
        target_positive_ratio = 0.333
        target_negative_count = int(len(positive_samples) * (1 - target_positive_ratio) / target_positive_ratio)
        
        # Randomly sample negative samples
        if len(negative_samples) > target_negative_count:
            negative_samples = np.random.choice(negative_samples, target_negative_count, replace=False)
            if isinstance(negative_samples, np.ndarray):
                negative_samples = negative_samples.tolist()
        
        # Merge balanced sentence samples
        balanced_sentence_samples = positive_samples + negative_samples
        np.random.shuffle(balanced_sentence_samples)
        
        print(f"    Balanced - Total samples: {len(balanced_sentence_samples)}")
        print(f"    Balanced - Positive samples: {len(positive_samples)}")
        print(f"    Balanced - Negative samples: {len(negative_samples)}")
        print(f"    Balanced - Positive sample ratio: {len(positive_samples)/len(balanced_sentence_samples)*100:.1f}%")
        
        # Reorganize as file format
        balanced_data = []
        current_file = None
        current_embeddings = []
        current_labels = []
        
        for sample in balanced_sentence_samples:
            if current_file != sample['filename']:
                # Save previous file
                if current_file is not None and len(current_embeddings) > 0:
                    balanced_data.append({
                        'sentence_embeddings': np.array(current_embeddings),
                        'labels': current_labels,
                        'length': len(current_embeddings),
                        'filename': current_file
                    })
                
                # Start new file
                current_file = sample['filename']
                current_embeddings = [sample['embedding']]
                current_labels = [sample['label']]
            else:
                # Continue current file
                current_embeddings.append(sample['embedding'])
                current_labels.append(sample['label'])
        
        # Save last file
        if current_file is not None and len(current_embeddings) > 0:
            balanced_data.append({
                'sentence_embeddings': np.array(current_embeddings),
                'labels': current_labels,
                'length': len(current_embeddings),
                'filename': current_file
            })
        
        print(f"\n📊 Balanced file statistics:")
        print(f"    Total files: {len(balanced_data)}")
        
        total_sentences = 0
        total_positive = 0
        for d in balanced_data:
            total_sentences += d['length']
            total_positive += sum(d['labels'])
        
        print(f"    Total sentences: {total_sentences}")
        print(f"    Positive samples: {total_positive}")
        print(f"    Negative samples: {total_sentences - total_positive}")
        print(f"    Positive sample ratio: {total_positive/total_sentences*100:.1f}%")
        
        return balanced_data

def collate_fn(batch):
    """Custom collate function"""
    # Get max length
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)
    
    # Initialize tensors
    embeddings = torch.zeros(batch_size, max_len, 384)  # BGE-small维度
    labels = torch.zeros(batch_size, max_len)
    lengths = []
    
    for i, item in enumerate(batch):
        length = item['length']
        lengths.append(length)
        
        # Fill embeddings
        embeddings[i, :length] = torch.tensor(item['sentence_embeddings'])
        
        # Fill labels
        labels[i, :length] = torch.tensor(item['labels'])
    
    return {
        'embeddings': embeddings,
        'labels': labels,
        'lengths': lengths
    }

def compute_metrics(preds, labels, lengths):
    """Compute evaluation metrics"""
    all_preds = []
    all_labels = []
    
    for i, length in enumerate(lengths):
        pred = preds[i][:length]
        label = labels[i][:length]
        
        all_preds.extend(pred.detach().cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Convert to binary predictions
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, binary_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_labels, binary_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = np.sum((all_labels == 1) & (binary_preds == 1))
        tn = np.sum((all_labels == 0) & (binary_preds == 0))
        fp = np.sum((all_labels == 0) & (binary_preds == 1))
        fn = np.sum((all_labels == 1) & (binary_preds == 0))
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_lengths = []
    
    for batch in tqdm(dataloader, desc="Training"):
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths']
        
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(embeddings, lengths)
        
        # Compute loss (only consider valid positions)
        loss = 0
        for i, length in enumerate(lengths):
            pred = preds[i, :length]
            label = labels[i, :length]
            loss += criterion(pred, label.float())
        
        loss = loss / len(lengths)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        for i, length in enumerate(lengths):
            all_preds.append(preds[i, :length])
            all_labels.append(labels[i, :length])
            all_lengths.append(length)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels, all_lengths)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def evaluate(model, dataloader, criterion, device):
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
            
            # Forward pass
            preds = model(embeddings, lengths)
            
            # Compute loss
            loss = 0
            for i, length in enumerate(lengths):
                pred = preds[i, :length]
                label = labels[i, :length]
                loss += criterion(pred, label.float())
            
            loss = loss / len(lengths)
            total_loss += loss.item()
            
            # 收集预测结果
            for i, length in enumerate(lengths):
                all_preds.append(preds[i, :length])
                all_labels.append(labels[i, :length])
                all_lengths.append(length)
    
    # 计算指标
    metrics = compute_metrics(all_preds, all_labels, all_lengths)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def main():
    """Main training function"""
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = HierarchicalBertConfig()
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    if config.use_wandb:
        try:
            wandb.init(project="comedy-chunk-detection", name="hierarchical-bert-training")
        except:
            print("Wandb initialization failed, continuing training...")
    
    # Load sentence encoder
    print("Loading sentence encoder...")
    sentence_encoder = SentenceTransformer(config.sentence_encoder)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = HierarchicalDataset(
        config.data_path, 
        sentence_encoder, 
        config.max_sentences
    )
    
    # Prepare hierarchical data
    processed_data = dataset.prepare_hierarchical_data()
    
    # Hierarchical split data
    if config.use_stratified_sampling:
        print("Using stratified sampling to split data...")
        # Stratify by whether there is a chunk
        has_chunk = [sum(d['labels']) > 0 for d in processed_data]
        
        train_data, val_data = train_test_split(
            processed_data, 
            test_size=1-config.train_split, 
            stratify=has_chunk,
            random_state=config.seed
        )
    else:
        print("Using random split...")
        train_data, val_data = train_test_split(
            processed_data, 
            test_size=1-config.train_split, 
            random_state=config.seed
        )
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Create model
    print("Creating model...")
    model = HierarchicalBertModel(config).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Training loop
    print("Starting training...")
    best_f1 = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_dataloader, criterion, device)
        
        # Print results
        print(f"Training - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Validation - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"Validation - True positives: {val_metrics['true_positives']}, False positives: {val_metrics['false_positives']}, False negatives: {val_metrics['false_negatives']}")
        
        # Record to wandb
        if wandb.run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_true_positives': val_metrics['true_positives'],
                'val_false_positives': val_metrics['false_positives'],
                'val_false_negatives': val_metrics['false_negatives'],
            })
        
        # Save best model
    if val_metrics['f1'] > best_f1:
        best_f1 = val_metrics['f1']
        os.makedirs(config.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_model.pth'))
        print(f"Saved best model, F1: {best_f1:.4f}")
    
    # Save final model
    print("Saving final model...")
    os.makedirs(config.model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.model_save_dir, 'final_model.pth'))
    
    # Save configuration
    with open(os.path.join(config.model_save_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    print("Training completed!")
    print(f"Best F1 score: {best_f1:.4f}")

if __name__ == "__main__":
    main() 