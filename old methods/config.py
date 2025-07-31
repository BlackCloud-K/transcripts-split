from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model configuration
    model_name: str = "allenai/longformer-base-4096"
    max_length: int = 1024  # Increase max length to support more context
    
    # Training configuration
    batch_size: int = 4     # Decrease batch_size to adapt to longer sequences
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Data configuration
    train_split: float = 0.8
    seed: int = 42
    context_window: int = 15  # Significantly increase context window
    
    # Path configuration
    data_path: str = "training_data.json"
    output_dir: str = "./chunk_model_output"
    model_save_dir: str = "./chunk_model_final"
    
    # Other configuration
    use_wandb: bool = False  # Whether to use wandb
    gradient_accumulation_steps: int = 4  # Increase gradient accumulation
    fp16: bool = True  # Whether to use mixed precision training 