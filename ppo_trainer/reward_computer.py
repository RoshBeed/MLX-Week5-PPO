import torch
from typing import List

class RewardComputer:
    """Computes rewards for generated sequences"""
    
    def __init__(self, reward_model, tokenizer, device):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
    
    def compute_rewards(self, prompts: List[str], 
                       generated_texts: List[str]) -> List[float]:
        """Compute rewards for generated sequences"""
        full_texts = [p + gt for p, gt in zip(prompts, generated_texts)]
        inputs = self.tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            rewards = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
        
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.squeeze()
            if rewards.dim() == 0:
                rewards = [rewards.item()]
            else:
                rewards = rewards.cpu().tolist()
        elif isinstance(rewards, float):
            rewards = [rewards]
        
        return rewards 