from typing import List, Dict

class TokenExtractor:
    """Extracts tokens from generated sequences"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def extract_tokens(self, generated_text: str) -> tuple:
        """Extract tokens and token IDs from generated text"""
        gen_tokens = self.tokenizer.tokenize(generated_text)
        gen_token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
        return gen_tokens, gen_token_ids 