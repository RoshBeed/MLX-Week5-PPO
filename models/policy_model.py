import torch
import torch.nn.functional as F
from .base_model import BaseModel

class PolicyModel(BaseModel):
    """Policy Model - The model we're training with PPO"""
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    
    def get_logprobs(self, input_ids, attention_mask=None):
        """Get log probabilities for the last token"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Last token logits
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs) 