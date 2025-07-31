import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from typing import List, Dict

class ChunkPredictor:
    """Chunk预测器 - 句子级别分类（增强版本）"""
    
    def __init__(self, model_path: str, context_window: int = 15):
        """初始化预测器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_window = context_window
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # 如果是LoRA模型，需要加载适配器
        if hasattr(self.model, 'peft_config'):
            self.model = PeftModel.from_pretrained(self.model, model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
    def predict_sentence_level(self, sentences: List[str]) -> List[int]:
        """句子级别的chunk边界预测"""
        chunk_boundaries = []
        
        # 为每个句子预测是否为chunk开始
        for i, sentence in enumerate(sentences):
            # 创建上下文窗口
            context_start = max(0, i - self.context_window)
            context_end = min(len(sentences), i + self.context_window + 1)
            context_sentences = sentences[context_start:context_end]
            
            # 合并上下文
            context_text = " ".join(context_sentences)
            
            # Tokenize
            inputs = self.tokenizer(
                context_text,
                truncation=True,
                padding=True,
                max_length=1024,  # 匹配训练时的max_length
                return_tensors="pt"
            )
            
            # 移动到GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
            
            # 如果预测为chunk边界，记录位置
            if prediction == 1:
                chunk_boundaries.append(i)
        
        return chunk_boundaries
    
    def predict_single_text(self, text: str) -> Dict:
        """预测单个文本"""
        # 简单分割成句子
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunk_boundaries = self.predict_sentence_level(sentences)
        
        return {
            'text': text,
            'sentences': sentences,
            'chunk_boundaries': chunk_boundaries,
            'num_chunks': len(chunk_boundaries) + 1
        }

def test_inference():
    """测试推理"""
    # 检查模型是否存在
    model_path = "./chunk_model_final"
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请先运行训练脚本")
        return
    
    # 创建预测器
    predictor = ChunkPredictor(model_path)
    
    # 测试文本
    test_text = """
    Hello everyone, how are you doing tonight? I'm really excited to be here. 
    Let me tell you about my day. It was absolutely crazy. So I woke up this morning. 
    And the first thing I noticed. My cat was missing. I looked everywhere for her. 
    But then I found her in the kitchen. She was eating my breakfast.
    """
    
    # 预测
    result = predictor.predict_single_text(test_text)
    
    print("推理结果:")
    print(f"文本: {result['text'][:100]}...")
    print(f"句子数: {len(result['sentences'])}")
    print(f"Chunk边界: {result['chunk_boundaries']}")
    print(f"Chunk数量: {result['num_chunks']}")

if __name__ == "__main__":
    test_inference() 