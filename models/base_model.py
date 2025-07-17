import torch.nn as nn
from base_setup import BaseModelSetup

class BaseModel(nn.Module):
    """Base model class with common functionality"""
    
    def __init__(self, base_model_setup: BaseModelSetup):
        super().__init__()
        self.model = base_model_setup.create_base_model()
        self.tokenizer = base_model_setup.tokenizer 