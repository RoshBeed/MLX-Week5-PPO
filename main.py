import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from config import PPOConfig
from base_setup import BaseModelSetup
from models import SFTModel, PolicyModel, RewardModel, ValueModel
from token_attribution import TokenAttribution
from reward_kl import KLDivergence, RewardAdjuster
from advantage import TDError, GAEAdvantage, ReturnCalculator
from experience_buffer import ExperienceBuffer
from training import PolicyRatio, PolicyLoss, ValueLoss

from ppo_trainer import PPOTrainer
from ppo_trainer import SequenceGenerator
from ppo_trainer import RewardComputer
from ppo_trainer import ValueAdder
from ppo_trainer import ModelTrainer

def main():
    # Configuration
    config = PPOConfig()
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Example prompts
    prompts = [
        "SUBREDDIT: r/relationships\nTITLE: Should I admit to snooping?\nPOST: ...\nTL;DR:",
        "SUBREDDIT: r/advice\nTITLE: Need help with decision\nPOST: ...\nTL;DR:"
    ]
    
    # Run PPO step
    results = trainer.ppo_step(prompts)
    
    print("PPO Step Results:")
    print(f"Generated texts: {results['generated_texts']}")
    print(f"Rewards: {results['rewards']}")
    print(f"Value loss: {results['value_loss']:.4f}")
    print(f"Policy loss: {results['policy_loss']:.4f}")
    
    # Print first (s, a) pair details
    first_pair = results['sa_pairs'][0]
    print(f"\nFirst (s, a) pair:")
    print(f"State: {first_pair['state_text'][:100]}...")
    print(f"Action: {first_pair['action_token']}")
    print(f"KL divergence: {first_pair['kl_div']:.4f}")
    print(f"Advantage: {first_pair['advantage']:.4f}")
    print(f"Reward-to-go: {first_pair['reward_to_go']:.4f}")

if __name__ == "__main__":
    main()
