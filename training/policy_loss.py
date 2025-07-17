import torch

class PolicyLoss:
    """Computes PPO policy loss"""
    
    @staticmethod
    def ppo_clip_loss(new_logprobs: torch.Tensor, old_logprobs: torch.Tensor, 
                     advantages: torch.Tensor, clip_epsilon: float = 0.2) -> torch.Tensor:
        """Compute PPO-clip loss"""
        ratio = torch.exp(new_logprobs - old_logprobs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss = -torch.mean(torch.min(unclipped, clipped))
        return loss 