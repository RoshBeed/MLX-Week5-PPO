import torch
import torch.nn.functional as F
from typing import List
from models import PolicyModel

class PolicyRatio:
    """Computes policy ratios for PPO"""
    
    @staticmethod
    def get_new_logprobs(policy_model: PolicyModel, tokenizer, 
                        state_texts: List[str], action_token_ids: List[int], 
                        device, batch_size: int = 32) -> torch.Tensor:
        """Get new logprobs from current policy"""
        new_logprobs = []
        
        for i in range(0, len(state_texts), batch_size):
            batch_states = state_texts[i:i+batch_size]
            batch_action_ids = action_token_ids[i:i+batch_size]
            
            inputs = tokenizer(batch_states, return_tensors="pt", 
                             padding=True, truncation=True).to(device)
            outputs = policy_model(inputs["input_ids"], 
                                 attention_mask=inputs["attention_mask"])
            logits = outputs.logits[:, -1, :]  # Last token logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            batch_action_ids_tensor = torch.tensor(batch_action_ids, 
                                                 dtype=torch.long, device=device)
            batch_logprobs = log_probs[torch.arange(len(batch_states)), 
                                     batch_action_ids_tensor]
            new_logprobs.append(batch_logprobs)
        
        return torch.cat(new_logprobs, dim=0) 