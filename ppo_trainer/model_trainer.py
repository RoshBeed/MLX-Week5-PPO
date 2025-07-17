import torch
from typing import List, Dict
from training import PolicyRatio, PolicyLoss, ValueLoss

class ModelTrainer:
    """Handles training of policy and value models"""
    
    def __init__(self, policy_model, value_model, tokenizer, config, device):
        self.policy_model = policy_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Initialize optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(), lr=config.value_learning_rate
        )
    
    def train_value_model(self, sa_pairs: List[Dict]):
        """Train value model on reward-to-go targets"""
        self.value_model.train()
        state_texts = [pair['state_text'] for pair in sa_pairs]
        reward_to_go_targets = [pair['reward_to_go'] for pair in sa_pairs]
        
        # Shuffle data
        indices = torch.randperm(len(state_texts))
        state_texts = [state_texts[i] for i in indices]
        reward_to_go_targets = [reward_to_go_targets[i] for i in indices]
        
        total_loss = 0.0
        
        for i in range(0, len(state_texts), self.config.batch_size):
            batch_states = state_texts[i:i+self.config.batch_size]
            batch_targets = torch.tensor(
                reward_to_go_targets[i:i+self.config.batch_size], 
                dtype=torch.float32
            ).to(self.device)
            
            inputs = self.tokenizer(
                batch_states, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            
            preds = self.value_model(inputs["input_ids"], inputs["attention_mask"]).squeeze(-1)
            loss = ValueLoss.mse_loss(preds, batch_targets)
            
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()
            
            total_loss += loss.item() * len(batch_states)
        
        avg_loss = total_loss / len(state_texts)
        self.value_model.eval()
        return avg_loss
    
    def train_policy_model(self, sa_pairs: List[Dict]):
        """Train policy model using PPO"""
        self.policy_model.train()
        
        state_texts = [pair['state_text'] for pair in sa_pairs]
        action_token_ids = [pair['action_token_id'] for pair in sa_pairs]
        old_logprobs = torch.tensor(
            [pair['logprob'] for pair in sa_pairs], dtype=torch.float32
        ).to(self.device)
        advantages = torch.tensor(
            [pair['advantage'] for pair in sa_pairs], dtype=torch.float32
        ).to(self.device)
        
        # Get new logprobs from current policy
        new_logprobs = PolicyRatio.get_new_logprobs(
            self.policy_model, self.tokenizer, state_texts, 
            action_token_ids, self.device, self.config.batch_size
        )
        
        # Compute PPO-clip loss
        loss = PolicyLoss.ppo_clip_loss(
            new_logprobs.to(self.device), old_logprobs, advantages, 
            self.config.clip_epsilon
        )
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        self.policy_model.eval()
        return loss.item() 