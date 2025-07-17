from dataclasses import dataclass

@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Model configs
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training configs
    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    batch_size: int = 32
    max_new_tokens: int = 10
    gamma: float = 0.99
    lam: float = 0.95
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2
    
    # Generation configs
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95 