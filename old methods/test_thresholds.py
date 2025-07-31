import torch
import numpy as np
import json
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from train_hierarchical_bert import HierarchicalBertModel, HierarchicalBertConfig, HierarchicalDataset, collate_fn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

def test_thresholds():
    """测试不同阈值对F1分数的影响"""
    
    # 配置
    config = HierarchicalBertConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print("加载模型...")
    model = HierarchicalBertModel(config).to(device)
    
    # 加载最佳模型权重
    model_path = os.path.join(config.output_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"加载模型权重: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return
    
    # 加载句子编码器
    sentence_encoder = SentenceTransformer(config.sentence_encoder)
    
    # 准备数据集
    print("准备数据集...")
    dataset = HierarchicalDataset(
        config.data_path, 
        sentence_encoder, 
        config.max_sentences
    )
    
    # 准备数据
    processed_data = dataset.prepare_hierarchical_data()
    
    # 分割数据（使用相同的随机种子）
    from sklearn.model_selection import train_test_split
    has_chunk = [sum(d['labels']) > 0 for d in processed_data]
    train_data, val_data = train_test_split(
        processed_data, 
        test_size=1-config.train_split, 
        stratify=has_chunk,
        random_state=config.seed
    )
    
    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_data, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 测试不同阈值
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    print(f"\n🔍 测试不同阈值...")
    print(f"验证集大小: {len(val_data)} 文件")
    
    model.eval()
    with torch.no_grad():
        # 收集所有预测和标签
        all_preds = []
        all_labels = []
        all_lengths = []
        
        for batch in val_dataloader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            
            # 前向传播
            preds = model(embeddings, lengths)
            
            # 收集结果
            for i, length in enumerate(lengths):
                all_preds.append(preds[i][:length])
                all_labels.append(labels[i][:length])
                all_lengths.append(length)
        
        # 转换为numpy数组
        all_preds_np = []
        all_labels_np = []
        for i, length in enumerate(all_lengths):
            all_preds_np.extend(all_preds[i].cpu().numpy())
            all_labels_np.extend(all_labels[i].cpu().numpy())
        
        all_preds_np = np.array(all_preds_np)
        all_labels_np = np.array(all_labels_np)
        
        print(f"总句子数: {len(all_preds_np)}")
        print(f"正样本数: {np.sum(all_labels_np)}")
        print(f"负样本数: {len(all_labels_np) - np.sum(all_labels_np)}")
        
        # 测试每个阈值
        for threshold in thresholds:
            # 应用阈值
            binary_preds = (all_preds_np > threshold).astype(int)
            
            # 计算指标
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels_np, binary_preds, average='binary', zero_division=0
            )
            acc = accuracy_score(all_labels_np, binary_preds)
            
            # 混淆矩阵
            cm = confusion_matrix(all_labels_np, binary_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tp = np.sum((all_labels_np == 1) & (binary_preds == 1))
                tn = np.sum((all_labels_np == 0) & (binary_preds == 0))
                fp = np.sum((all_labels_np == 0) & (binary_preds == 1))
                fn = np.sum((all_labels_np == 1) & (binary_preds == 0))
            
            results.append({
                'threshold': float(threshold),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(acc),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            })
            
            print(f"阈值 {threshold:.1f}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, TP={tp}, FP={fp}, FN={fn}")
    
    # 找到最佳阈值
    best_result = max(results, key=lambda x: x['f1'])
    print(f"\n🏆 最佳结果:")
    print(f"   阈值: {best_result['threshold']:.1f}")
    print(f"   F1: {best_result['f1']:.4f}")
    print(f"   Precision: {best_result['precision']:.4f}")
    print(f"   Recall: {best_result['recall']:.4f}")
    print(f"   真正例: {best_result['true_positives']}")
    print(f"   假正例: {best_result['false_positives']}")
    print(f"   假负例: {best_result['false_negatives']}")
    
    # 保存结果
    with open('threshold_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: threshold_test_results.json")
    
    return results

def analyze_predictions():
    """分析预测结果分布"""
    print(f"\n📊 分析预测概率分布...")
    
    # 配置
    config = HierarchicalBertConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = HierarchicalBertModel(config).to(device)
    model_path = os.path.join(config.output_dir, 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 加载句子编码器
    sentence_encoder = SentenceTransformer(config.sentence_encoder)
    
    # 准备数据集
    dataset = HierarchicalDataset(
        config.data_path, 
        sentence_encoder, 
        config.max_sentences
    )
    processed_data = dataset.prepare_hierarchical_data()
    
    # 分割数据
    from sklearn.model_selection import train_test_split
    has_chunk = [sum(d['labels']) > 0 for d in processed_data]
    train_data, val_data = train_test_split(
        processed_data, 
        test_size=1-config.train_split, 
        stratify=has_chunk,
        random_state=config.seed
    )
    
    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_data, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 收集预测概率
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            
            preds = model(embeddings, lengths)
            
            for i, length in enumerate(lengths):
                all_preds.extend(preds[i][:length].cpu().numpy())
                all_labels.extend(labels[i][:length].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 分析分布
    positive_preds = all_preds[all_labels == 1]
    negative_preds = all_preds[all_labels == 0]
    
    print(f"正样本预测分布:")
    print(f"   数量: {len(positive_preds)}")
    print(f"   均值: {np.mean(positive_preds):.4f}")
    print(f"   标准差: {np.std(positive_preds):.4f}")
    print(f"   最小值: {np.min(positive_preds):.4f}")
    print(f"   最大值: {np.max(positive_preds):.4f}")
    print(f"   中位数: {np.median(positive_preds):.4f}")
    
    print(f"\n负样本预测分布:")
    print(f"   数量: {len(negative_preds)}")
    print(f"   均值: {np.mean(negative_preds):.4f}")
    print(f"   标准差: {np.std(negative_preds):.4f}")
    print(f"   最小值: {np.min(negative_preds):.4f}")
    print(f"   最大值: {np.max(negative_preds):.4f}")
    print(f"   中位数: {np.median(negative_preds):.4f}")
    
    # 计算重叠程度
    overlap_threshold = 0.5
    positive_above_threshold = np.sum(positive_preds > overlap_threshold)
    negative_above_threshold = np.sum(negative_preds > overlap_threshold)
    
    print(f"\n重叠分析 (阈值={overlap_threshold}):")
    print(f"   正样本 > {overlap_threshold}: {positive_above_threshold}/{len(positive_preds)} ({positive_above_threshold/len(positive_preds)*100:.1f}%)")
    print(f"   负样本 > {overlap_threshold}: {negative_above_threshold}/{len(negative_preds)} ({negative_above_threshold/len(negative_preds)*100:.1f}%)")

if __name__ == "__main__":
    print("🔍 开始阈值测试...")
    results = test_thresholds()
    
    print("\n" + "="*50)
    analyze_predictions() 