import torch
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
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    TrainerCallback,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class DebugTrainingConfig:
    """调试训练配置"""
    # 模型配置
    model_name: str = "allenai/longformer-base-4096"
    max_length: int = 512  # 减少长度，避免内存问题
    
    # 训练配置
    batch_size: int = 2  # 减少batch size
    learning_rate: float = 1e-4  # 增加学习率
    num_epochs: int = 5  # 增加epoch数
    warmup_steps: int = 50
    weight_decay: float = 0.01
    
    # LoRA配置
    lora_r: int = 16  # 增加rank
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 数据配置
    train_split: float = 0.8
    seed: int = 42
    context_window: int = 5  # 减少上下文窗口
    
    # 不平衡处理配置
    use_stratified_sampling: bool = True
    use_weighted_loss: bool = False
    
    # 路径配置
    data_path: str = "training_data.json"
    output_dir: str = "./chunk_model_debug_output"
    model_save_dir: str = "./chunk_model_debug_final"
    
    # 其他配置
    use_wandb: bool = True
    gradient_accumulation_steps: int = 2
    fp16: bool = False  # 关闭fp16避免精度问题

class DebugCallback(TrainerCallback):
    """调试回调函数"""
    def __init__(self):
        self.step = 0
        
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        if self.step % 10 == 0 and logs:
            print(f"步骤 {self.step}: 损失 = {logs.get('loss', 'N/A'):.4f}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"🔍 评估结果:")
            print(f"  损失: {metrics.get('eval_loss', 'N/A'):.4f}")
            print(f"  F1: {metrics.get('eval_f1', 'N/A'):.4f}")
            print(f"  准确率: {metrics.get('eval_accuracy', 'N/A'):.4f}")
            print(f"  真正例: {metrics.get('eval_true_positives', 'N/A')}")
            print(f"  假正例: {metrics.get('eval_false_positives', 'N/A')}")
            print(f"  假负例: {metrics.get('eval_false_negatives', 'N/A')}")

class DebugDataset:
    """调试数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, context_window: int = 5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict]:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载了 {len(data)} 个transcript文件")
        return data
    
    def prepare_sentence_level_data(self) -> List[Dict]:
        """准备句子级别的数据 - 简化版本"""
        processed_data = []
        
        for sample in tqdm(self.data, desc="准备句子级数据"):
            sentences = sample['sentences']
            labels = sample['labels']
            
            # 为每个句子创建一个样本
            for i, (sentence, label) in enumerate(zip(sentences, labels)):
                # 创建简化的上下文窗口
                context_start = max(0, i - self.context_window)
                context_end = min(len(sentences), i + self.context_window + 1)
                context_sentences = sentences[context_start:context_end]
                
                # 合并上下文
                context_text = " ".join(context_sentences)
                
                # 创建训练样本
                processed_data.append({
                    'text': context_text,
                    'label': label,
                    'sentence_index': i,
                    'total_sentences': len(sentences),
                    'context_size': len(context_sentences),
                    'is_chunk_start': label == 1
                })
        
        # 数据平衡处理
        positive_samples = [d for d in processed_data if d['label'] == 1]
        negative_samples = [d for d in processed_data if d['label'] == 0]
        
        print(f"📊 原始数据分布:")
        print(f"   正样本数: {len(positive_samples)}")
        print(f"   负样本数: {len(negative_samples)}")
        print(f"   正负比例: 1:{len(negative_samples)/len(positive_samples):.1f}")
        
        # 目标：每3个样本中至少有1个正样本
        target_positive_ratio = 0.333
        target_negative_ratio = 1 - target_positive_ratio
        target_negative_count = int(len(positive_samples) * target_negative_ratio / target_positive_ratio)
        
        # 随机采样负样本
        if len(negative_samples) > target_negative_count:
            negative_samples = np.random.choice(negative_samples, target_negative_count, replace=False)
            if isinstance(negative_samples, np.ndarray):
                negative_samples = negative_samples.tolist()
        
        # 合并平衡后的数据
        balanced_data = positive_samples + negative_samples
        np.random.shuffle(balanced_data)
        
        # 详细的数据分布分析
        total_samples = len(balanced_data)
        final_positive_samples = sum(d['label'] for d in balanced_data)
        final_negative_samples = total_samples - final_positive_samples
        
        print(f"📊 平衡后数据分布:")
        print(f"   总样本数: {total_samples}")
        print(f"   正样本数: {final_positive_samples} ({final_positive_samples/total_samples*100:.1f}%)")
        print(f"   负样本数: {final_negative_samples} ({final_negative_samples/total_samples*100:.1f}%)")
        print(f"   正负比例: 1:{final_negative_samples/final_positive_samples:.1f}")
        
        # 验证标签分布
        labels = [d['label'] for d in balanced_data]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"标签分布: {dict(zip(unique_labels, counts))}")
        
        return balanced_data
    
    def tokenize_function(self, examples):
        """tokenize函数"""
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # 确保labels字段被保留
        tokenized['labels'] = examples['label']
        
        return tokenized

def compute_debug_metrics(pred):
    """计算调试评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # 确保数据类型正确
    labels = labels.astype(int)
    preds = preds.astype(int)
    
    # 调试信息
    print(f"🔍 预测调试:")
    print(f"  标签分布: {np.bincount(labels)}")
    print(f"  预测分布: {np.bincount(preds)}")
    print(f"  前10个标签: {labels[:10]}")
    print(f"  前10个预测: {preds[:10]}")
    
    # 基础指标
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    print(f"混淆矩阵:\n{cm}")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = np.sum((labels == 1) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'eval_true_positives': int(tp),
        'eval_false_positives': int(fp),
        'eval_false_negatives': int(fn),
    }

def setup_debug_model_and_tokenizer(config: DebugTrainingConfig):
    """设置调试模型和tokenizer"""
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print("加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # 配置LoRA
    print("配置LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
    
    return model, tokenizer

def main():
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 配置
    config = DebugTrainingConfig()
    
    # 检查GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化wandb
    if config.use_wandb:
        try:
            wandb.init(project="comedy-chunk-detection", name="longformer-debug-training")
        except:
            print("Wandb初始化失败，继续训练...")
    
    # 加载模型和tokenizer
    model, tokenizer = setup_debug_model_and_tokenizer(config)
    
    # 准备数据集
    print("准备数据集...")
    dataset = DebugDataset(
        config.data_path, 
        tokenizer, 
        config.max_length, 
        config.context_window
    )
    
    # 准备句子级数据
    processed_data = dataset.prepare_sentence_level_data()
    
    # 分层分割数据
    if config.use_stratified_sampling:
        print("使用分层采样分割数据...")
        labels = [d['label'] for d in processed_data]
        
        # 检查每个类别的样本数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"类别分布: {dict(zip(unique_labels, counts))}")
        
        # 如果某个类别样本太少，使用随机分割
        if min(counts) < 2:
            print(f"警告：某个类别样本数量不足（最少{min(counts)}个），切换到随机分割")
            train_data, val_data = train_test_split(
                processed_data, 
                test_size=1-config.train_split, 
                random_state=config.seed
            )
        else:
            train_data, val_data = train_test_split(
                processed_data, 
                test_size=1-config.train_split, 
                stratify=labels,
                random_state=config.seed
            )
    else:
        print("使用随机分割数据...")
        train_data, val_data = train_test_split(
            processed_data, 
            test_size=1-config.train_split, 
            random_state=config.seed
        )
    
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    # 显示分割后的数据分布
    train_pos = sum(d['label'] for d in train_data)
    train_neg = len(train_data) - train_pos
    val_pos = sum(d['label'] for d in val_data)
    val_neg = len(val_data) - val_pos
    
    print(f"训练集分布: 正样本 {train_pos} ({train_pos/len(train_data)*100:.1f}%), 负样本 {train_neg} ({train_neg/len(train_data)*100:.1f}%)")
    print(f"验证集分布: 正样本 {val_pos} ({val_pos/len(val_data)*100:.1f}%), 负样本 {val_neg} ({val_neg/len(val_data)*100:.1f}%)")
    
    # 创建HuggingFace数据集
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Tokenize数据集
    train_dataset = train_dataset.map(
        dataset.tokenize_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        dataset.tokenize_function, 
        batched=True, 
        remove_columns=val_dataset.column_names
    )
    
    # 数据收集器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy="steps",
        eval_steps=10,  # 每10步评估一次
        save_strategy="steps",
        save_steps=10,  # 每10步保存一次，与评估策略匹配
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=config.warmup_steps,
        logging_steps=5,
        save_total_limit=2,
        report_to="wandb" if wandb.run is not None else None,
        remove_unused_columns=False,
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_dir="./logs",
        log_level="info",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )
    
    # 创建调试回调
    debug_callback = DebugCallback()
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_debug_metrics,
        callbacks=[debug_callback],
    )
    
    # 开始训练
    print("开始调试训练...")
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model(config.model_save_dir)
    tokenizer.save_pretrained(config.model_save_dir)
    
    # 最终评估
    print("最终评估...")
    results = trainer.evaluate()
    print(f"最终结果: {results}")
    
    # 保存训练结果
    with open("debug_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("调试训练完成！")

if __name__ == "__main__":
    main() 