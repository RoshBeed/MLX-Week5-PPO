import torch
import torch.nn.functional as F

class ValueLoss:
    """Computes value function loss"""
 
    @staticmethod
    def mse_loss(predicted_values: torch.Tensor, 
                target_values: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss for value function"""
        return F.mse_loss(predicted_values, target_values) 