import torch
from typing import List, Dict

class ValueAdder:
    """Adds value predictions to (s, a) pairs"""
    
    def __init__(self, value_model, tokenizer, batch_size: int = 32, device=None):
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device or next(value_model.parameters()).device
    
    def add_values_to_sa_pairs(self, sa_pairs: List[Dict]) -> List[Dict]:
        """Add value predictions to (s, a) pairs"""
        state_texts = [pair['state_text'] for pair in sa_pairs]
        values = []
        
        for i in range(0, len(state_texts), self.batch_size):
            batch_texts = state_texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                batch_values = self.value_model(inputs["input_ids"], inputs["attention_mask"])
            
            batch_values = batch_values.squeeze().cpu().tolist()
            if isinstance(batch_values, float):
                batch_values = [batch_values]
            values.extend(batch_values)
        
        for pair, value in zip(sa_pairs, values):
            pair['value'] = value
        
        return sa_pairs 