#!/usr/bin/env python3
"""
Basic PPO Usage Example

This example demonstrates how to use the PPO implementation
for language model fine-tuning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PPOConfig
from ppo_trainer import PPOTrainer

def main():
    """Basic PPO training example"""
    
    print("🚀 Starting PPO Training Example")
    print("=" * 50)
    
    # 1. Initialize configuration
    print("📋 Initializing configuration...")
    config = PPOConfig()
    print(f"   Model: {config.model_name}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Max new tokens: {config.max_new_tokens}")
    
    # 2. Create trainer
    print("\n🏗️  Creating PPO trainer...")
    trainer = PPOTrainer(config)
    print("   ✓ Trainer created successfully")
    
    # 3. Example prompts
    print("\n📝 Preparing example prompts...")
    prompts = [
        "SUBREDDIT: r/relationships\nTITLE: Should I admit to snooping?\nPOST: I found some suspicious messages on my partner's phone. Should I confront them?\nTL;DR:",
        "SUBREDDIT: r/advice\nTITLE: Career decision help\nPOST: I have two job offers. One pays more but the other has better work-life balance.\nTL;DR:",
        "SUBREDDIT: r/personalfinance\nTITLE: Should I buy a house?\nPOST: I have enough savings for a down payment but I'm not sure if it's the right time.\nTL;DR:"
    ]
    
    print(f"   ✓ Prepared {len(prompts)} prompts")
    
    # 4. Run PPO training steps
    print("\n🔄 Running PPO training steps...")
    
    for step in range(3):
        print(f"\n   Step {step + 1}/3:")
        print("   " + "-" * 30)
        
        # Run PPO step
        results = trainer.ppo_step(prompts)
        
        # Print results
        print(f"   Generated texts:")
        for i, text in enumerate(results['generated_texts']):
            print(f"     {i+1}. {text[:50]}...")
        
        print(f"   Rewards: {[f'{r:.3f}' for r in results['rewards']]}")
        print(f"   Value loss: {results['value_loss']:.4f}")
        print(f"   Policy loss: {results['policy_loss']:.4f}")
        
        # Print first SA pair details
        if results['sa_pairs']:
            first_pair = results['sa_pairs'][0]
            print(f"   First (s,a) pair:")
            print(f"     State: {first_pair['state_text'][:60]}...")
            print(f"     Action: '{first_pair['action_token']}'")
            print(f"     KL divergence: {first_pair['kl_div']:.4f}")
            print(f"     Advantage: {first_pair['advantage']:.4f}")
    
    print("\n✅ PPO training example completed!")
    print("\n📊 Summary:")
    print("   • Successfully ran 3 PPO training steps")
    print("   • Generated text sequences for each prompt")
    print("   • Computed rewards, advantages, and losses")
    print("   • Updated both policy and value models")
    
    print("\n🎯 Next steps:")
    print("   • Try different prompts and configurations")
    print("   • Experiment with hyperparameters")
    print("   • Add custom reward models")
    print("   • Implement evaluation metrics")

if __name__ == "__main__":
    main() 