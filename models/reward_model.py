import torch.nn as nn
from .base_model import BaseModel

class RewardModel(BaseModel):
    """Reward Model - Predicts scalar reward for sequences"""
    
    def __init__(self, base_model_setup):
        super().__init__(base_model_setup)
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]
        # Get the last token's hidden state
        last_token_idx = attention_mask.sum(dim=1) - 1
        last_token_idx = last_token_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, last_hidden.size(-1))
        last_hidden_state = last_hidden.gather(1, last_token_idx).squeeze(1)
        reward = self.value_head(last_hidden_state).squeeze(-1)
        return reward 