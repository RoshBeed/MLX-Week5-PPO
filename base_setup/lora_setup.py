from peft import LoraConfig

class LoRASetup:
    """Handles LoRA configuration setup"""
    
    def __init__(self, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM"
        ) 