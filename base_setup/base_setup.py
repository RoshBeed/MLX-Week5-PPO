from .tokenizer_setup import TokenizerSetup
from .lora_setup import LoRASetup
from .model_setup import ModelSetup
from config import PPOConfig

class BaseModelSetup:
    """Handles base model and tokenizer setup"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.tokenizer_setup = TokenizerSetup(config.model_name)
        self.lora_setup = LoRASetup(
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )
        self.model_setup = ModelSetup(config.model_name, self.lora_setup)
    
    @property
    def tokenizer(self):
        return self.tokenizer_setup.tokenizer
    
    def create_base_model(self):
        return self.model_setup.create_base_model() 