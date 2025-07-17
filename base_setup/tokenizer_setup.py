from transformers import AutoTokenizer

class TokenizerSetup:
    """Handles tokenizer setup"""
    
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token 