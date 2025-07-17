#!/usr/bin/env python3
"""
Test PPO Components

This file contains tests for all PPO components to ensure
they work correctly together.
"""

import sys
import os
import unittest
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PPOConfig
from base_setup import BaseModelSetup
from models import SFTModel, PolicyModel, RewardModel, ValueModel
from token_attribution import TokenAttribution
from reward_kl import KLDivergence, RewardAdjuster
from advantage import TDError, GAEAdvantage, ReturnCalculator
from experience_buffer import ExperienceBuffer
from training import PolicyRatio, PolicyLoss, ValueLoss
from ppo_trainer import PPOTrainer

class TestPPOComponents(unittest.TestCase):
    """Test cases for PPO components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = PPOConfig()
        self.device = torch.device("cpu")  # Use CPU for testing
        
        # Create base setup
        self.base_setup = BaseModelSetup(self.config)
        
        # Create models
        self.sft_model = SFTModel(self.base_setup).to(self.device)
        self.policy_model = PolicyModel(self.base_setup).to(self.device)
        self.reward_model = RewardModel(self.base_setup).to(self.device)
        self.value_model = ValueModel(self.base_setup).to(self.device)
        
        # Test data
        self.test_prompts = [
            "SUBREDDIT: r/test\nTITLE: Test prompt\nPOST: This is a test.\nTL;DR:",
            "SUBREDDIT: r/test2\nTITLE: Another test\nPOST: Another test post.\nTL;DR:"
        ]
        
        self.test_sa_pairs = [
            {
                'state_text': 'SUBREDDIT: r/test\nTITLE: Test prompt\nPOST: This is a test.\nTL;DR:',
                'action_token': ' Test',
                'action_token_id': 1234,
                'logprob': -2.5,
                'ref_logprob': -2.8,
                'prompt_idx': 0,
                'step': 0
            },
            {
                'state_text': 'SUBREDDIT: r/test\nTITLE: Test prompt\nPOST: This is a test.\nTL;DR: Test',
                'action_token': ' response',
                'action_token_id': 5678,
                'logprob': -1.8,
                'ref_logprob': -2.1,
                'prompt_idx': 0,
                'step': 1
            }
        ]
    
    def test_config(self):
        """Test configuration"""
        self.assertIsNotNone(self.config.model_name)
        self.assertGreater(self.config.batch_size, 0)
        self.assertGreater(self.config.learning_rate, 0)
        self.assertGreater(self.config.value_learning_rate, 0)
    
    def test_base_setup(self):
        """Test base setup"""
        self.assertIsNotNone(self.base_setup.tokenizer)
        self.assertIsNotNone(self.base_setup.lora_config)
    
    def test_models(self):
        """Test model creation"""
        # Test that models can be created
        self.assertIsNotNone(self.sft_model)
        self.assertIsNotNone(self.policy_model)
        self.assertIsNotNone(self.reward_model)
        self.assertIsNotNone(self.value_model)
        
        # Test that models have the expected methods
        self.assertTrue(hasattr(self.sft_model, 'forward'))
        self.assertTrue(hasattr(self.policy_model, 'forward'))
        self.assertTrue(hasattr(self.reward_model, 'forward'))
        self.assertTrue(hasattr(self.value_model, 'forward'))
    
    def test_kl_divergence(self):
        """Test KL divergence computation"""
        kl_div = KLDivergence()
        
        # Test adding KL to SA pairs
        sa_pairs = self.test_sa_pairs.copy()
        sa_pairs = kl_div.add_kl_to_sa_pairs(sa_pairs)
        
        for pair in sa_pairs:
            self.assertIn('kl_div', pair)
            expected_kl = pair['logprob'] - pair['ref_logprob']
            self.assertAlmostEqual(pair['kl_div'], expected_kl, places=5)
    
    def test_reward_adjuster(self):
        """Test reward adjustment"""
        reward_adjuster = RewardAdjuster()
        
        # Test reward adjustment
        sa_pairs = self.test_sa_pairs.copy()
        sa_pairs = KLDivergence().add_kl_to_sa_pairs(sa_pairs)
        
        rewards = [0.5, -0.3]  # Test rewards
        kl_coef = 0.1
        
        sa_pairs = reward_adjuster.adjust_rewards_with_kl(sa_pairs, rewards, kl_coef)
        
        for pair in sa_pairs:
            self.assertIn('adjusted_reward', pair)
            expected_reward = rewards[pair['prompt_idx']] - kl_coef * pair['kl_div']
            self.assertAlmostEqual(pair['adjusted_reward'], expected_reward, places=5)
    
    def test_td_error(self):
        """Test TD error computation"""
        td_error = TDError()
        
        # Add values to SA pairs
        sa_pairs = self.test_sa_pairs.copy()
        for pair in sa_pairs:
            pair['value'] = 0.1
            pair['adjusted_reward'] = 0.5
        
        # Add a third pair to test TD computation
        sa_pairs.append({
            'state_text': 'SUBREDDIT: r/test\nTITLE: Test prompt\nPOST: This is a test.\nTL;DR: Test response',
            'action_token': ' end',
            'action_token_id': 9999,
            'logprob': -1.2,
            'ref_logprob': -1.5,
            'prompt_idx': 0,
            'step': 2,
            'value': 0.2,
            'adjusted_reward': 0.3
        })
        
        sa_pairs = td_error.compute_td_errors(sa_pairs, gamma=0.99)
        
        for pair in sa_pairs:
            self.assertIn('td_error', pair)
    
    def test_gae_advantage(self):
        """Test GAE advantage computation"""
        gae = GAEAdvantage()
        
        # Add TD errors to SA pairs
        sa_pairs = self.test_sa_pairs.copy()
        for pair in sa_pairs:
            pair['value'] = 0.1
            pair['adjusted_reward'] = 0.5
            pair['td_error'] = 0.2
        
        sa_pairs = gae.compute_gae_advantages(sa_pairs, gamma=0.99, lam=0.95)
        
        for pair in sa_pairs:
            self.assertIn('advantage', pair)
    
    def test_return_calculator(self):
        """Test return calculation"""
        return_calc = ReturnCalculator()
        
        # Add advantages to SA pairs
        sa_pairs = self.test_sa_pairs.copy()
        for pair in sa_pairs:
            pair['advantage'] = 0.3
            pair['value'] = 0.1
        
        sa_pairs = return_calc.add_reward_to_go(sa_pairs)
        
        for pair in sa_pairs:
            self.assertIn('reward_to_go', pair)
            expected_return = pair['advantage'] + pair['value']
            self.assertAlmostEqual(pair['reward_to_go'], expected_return, places=5)
    
    def test_experience_buffer(self):
        """Test experience buffer"""
        buffer = ExperienceBuffer()
        
        # Add experience
        sa_pairs = self.test_sa_pairs.copy()
        for pair in sa_pairs:
            pair['advantage'] = 0.3
            pair['reward_to_go'] = 0.4
        
        buffer.add_experience(sa_pairs)
        
        # Test getting batch
        batch = buffer.get_batch(1)
        self.assertEqual(len(batch), 1)
        
        # Test clearing
        buffer.clear()
        self.assertEqual(len(buffer.buffer), 0)
    
    def test_policy_loss(self):
        """Test policy loss computation"""
        new_logprobs = torch.tensor([-1.0, -2.0])
        old_logprobs = torch.tensor([-1.1, -2.1])
        advantages = torch.tensor([0.5, -0.3])
        
        loss = PolicyLoss.ppo_clip_loss(new_logprobs, old_logprobs, advantages, clip_epsilon=0.2)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_value_loss(self):
        """Test value loss computation"""
        predicted_values = torch.tensor([0.5, 0.3])
        target_values = torch.tensor([0.6, 0.4])
        
        loss = ValueLoss.mse_loss(predicted_values, target_values)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_ppo_trainer_creation(self):
        """Test PPO trainer creation"""
        trainer = PPOTrainer(self.config)
        
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.generator)
        self.assertIsNotNone(trainer.reward_computer)
        self.assertIsNotNone(trainer.value_adder)
        self.assertIsNotNone(trainer.model_trainer)
    
    def test_token_attribution(self):
        """Test token attribution"""
        token_attribution = TokenAttribution(self.base_setup.tokenizer)
        
        # This is a basic test - in practice, you'd need actual generated sequences
        self.assertIsNotNone(token_attribution)

def run_tests():
    """Run all tests"""
    print("🧪 Running PPO Component Tests")
    print("=" * 40)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPPOComponents)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("   ✅ All tests passed!")
    else:
        print("   ❌ Some tests failed!")
        for failure in result.failures:
            print(f"     - {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"     - {error[0]}: {error[1]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 