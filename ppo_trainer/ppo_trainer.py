import torch
from typing import List, Dict
from config import PPOConfig
from base_setup import BaseModelSetup
from models import SFTModel, PolicyModel, RewardModel, ValueModel
from token_attribution import TokenAttribution
from reward_kl import KLDivergence, RewardAdjuster
from advantage import TDError, GAEAdvantage, ReturnCalculator
from experience_buffer import ExperienceBuffer
from .generator import SequenceGenerator
from .reward_computer import RewardComputer
from .value_adder import ValueAdder
from .model_trainer import ModelTrainer

class PPOTrainer:
    """Main PPO trainer that orchestrates all components"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup base components
        self.base_setup = BaseModelSetup(config)
        
        # Initialize models
        self.sft_model = SFTModel(self.base_setup).to(self.device)
        self.policy_model = PolicyModel(self.base_setup).to(self.device)
        self.reward_model = RewardModel(self.base_setup).to(self.device)
        self.value_model = ValueModel(self.base_setup).to(self.device)
        
        # Initialize components
        self.token_attribution = TokenAttribution(self.base_setup.tokenizer)
        self.kl_divergence = KLDivergence()
        self.reward_adjuster = RewardAdjuster()
        self.td_error = TDError()
        self.gae_advantage = GAEAdvantage()
        self.return_calculator = ReturnCalculator()
        self.experience_buffer = ExperienceBuffer()
        
        # Initialize sub-components
        self.generator = SequenceGenerator(self.policy_model, self.sft_model, self.base_setup.tokenizer, config, self.device)
        self.reward_computer = RewardComputer(self.reward_model, self.base_setup.tokenizer, self.device)
        self.value_adder = ValueAdder(self.value_model, self.base_setup.tokenizer, config.batch_size, self.device)
        self.model_trainer = ModelTrainer(self.policy_model, self.value_model, self.base_setup.tokenizer, config, self.device)
        
        # Set models to eval mode initially
        self.sft_model.eval()
        self.policy_model.eval()
        self.reward_model.eval()
        self.value_model.eval()
    
    def ppo_step(self, prompts: List[str]):
        """Execute one PPO step"""
        # Step 1: Generate sequences and extract (s, a) pairs
        sa_pairs, generated_texts = self.generator.generate_sequences(prompts)
        
        # Step 2: Add KL divergence
        sa_pairs = self.kl_divergence.add_kl_to_sa_pairs(sa_pairs)
        
        # Step 3: Compute rewards
        rewards = self.reward_computer.compute_rewards(prompts, generated_texts)
        
        # Step 4: Adjust rewards with KL penalty
        sa_pairs = self.reward_adjuster.adjust_rewards_with_kl(
            sa_pairs, rewards, self.config.kl_coef
        )
        
        # Step 5: Add value predictions
        sa_pairs = self.value_adder.add_values_to_sa_pairs(sa_pairs)
        
        # Step 6: Compute TD errors
        sa_pairs = self.td_error.compute_td_errors(sa_pairs, self.config.gamma)
        
        # Step 7: Compute GAE advantages
        sa_pairs = self.gae_advantage.compute_gae_advantages(
            sa_pairs, self.config.gamma, self.config.lam
        )
        
        # Step 8: Compute reward-to-go
        sa_pairs = self.return_calculator.add_reward_to_go(sa_pairs)
        
        # Step 9: Train value model
        value_loss = self.model_trainer.train_value_model(sa_pairs)
        
        # Step 10: Train policy model
        policy_loss = self.model_trainer.train_policy_model(sa_pairs)
        
        return {
            'sa_pairs': sa_pairs,
            'generated_texts': generated_texts,
            'rewards': rewards,
            'value_loss': value_loss,
            'policy_loss': policy_loss
        } 