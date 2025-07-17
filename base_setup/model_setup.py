from transformers import AutoModelForCausalLM
from peft import get_peft_model
from .lora_setup import LoRASetup

class ModelSetup:
    """Handles base model setup with LoRA"""
    
    def __init__(self, model_name: str, lora_setup: LoRASetup):
        self.model_name = model_name
        self.lora_setup = lora_setup
    
    def create_base_model(self):
        """Create base model with LoRA"""
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        return get_peft_model(base_model, self.lora_setup.lora_config) 