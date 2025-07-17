import torch
import torch.nn.functional as F
from .base_model import BaseModel

class SFTModel(BaseModel):
    """Supervised Fine-Tuned Model - Reference model for KL divergence"""
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
    
    def get_logprobs(self, input_ids, attention_mask=None):
        """Get log probabilities for the last token"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Last token logits
            logprobs = F.log_softmax(logits, dim=-1)
        return logprobs 