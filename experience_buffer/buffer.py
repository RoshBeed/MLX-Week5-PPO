import torch
from typing import List, Dict

class ExperienceBuffer:
    """Stores experience data for training"""
    
    def __init__(self):
        self.buffer = []
    
    def add_experience(self, sa_pairs: List[Dict]):
        """Add experience data to buffer"""
        for pair in sa_pairs:
            experience = {
                'state_text': pair['state_text'],
                'action_token_id': pair['action_token_id'],
                'old_logprob': pair['logprob'],
                'advantage': pair['advantage'],
                'reward_to_go': pair['reward_to_go']
            }
            self.buffer.append(experience)
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get a batch of experiences"""
        if len(self.buffer) < batch_size:
            return self.buffer
        indices = torch.randperm(len(self.buffer))[:batch_size]
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = [] 