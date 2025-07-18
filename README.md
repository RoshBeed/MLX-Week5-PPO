# PPO Language Model Fine-tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PPO](https://img.shields.io/badge/PPO-Implemented-orange.svg)](https://arxiv.org/abs/1707.06347)

A production-ready implementation of Proximal Policy Optimization (PPO) for language model fine-tuning. This project provides a modular, scalable architecture for training language models with human feedback through reinforcement learning.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MLX-Week5-PPO

# Install dependencies using uv
uv sync
```

### Basic Usage

```python
from config import PPOConfig
from ppo_trainer import PPOTrainer

# Initialize configuration
config = PPOConfig()

# Create trainer
trainer = PPOTrainer(config)

# Example prompts
prompts = [
    "SUBREDDIT: r/relationships\nTITLE: Should I admit to snooping?\nPOST: ...\nTL;DR:",
    "SUBREDDIT: r/advice\nTITLE: Need help with decision\nPOST: ...\nTL;DR:"
]

# Run PPO step
results = trainer.ppo_step(prompts)

print(f"Generated texts: {results['generated_texts']}")
print(f"Rewards: {results['rewards']}")
print(f"Value loss: {results['value_loss']:.4f}")
print(f"Policy loss: {results['policy_loss']:.4f}")
```

## 📁 Project Structure

```
MLX-Week5-PPO/
├── config/
│   └── ppo_config.py          # PPO configuration
├── base_setup/
│   ├── base_setup.py          # Base model setup
│   ├── lora_setup.py          # LoRA configuration
│   ├── model_setup.py         # Model initialization
│   └── tokenizer_setup.py     # Tokenizer setup
├── models/
│   ├── base_model.py          # Base model class
│   ├── policy_model.py        # Policy model
│   ├── reward_model.py        # Reward model
│   ├── sft_model.py           # SFT model
│   └── value_model.py         # Value model
├── ppo_trainer/
│   ├── ppo_trainer.py         # Main PPO trainer
│   ├── generator.py           # Sequence generator
│   ├── reward_computer.py     # Reward computation
│   ├── value_adder.py         # Value addition
│   └── model_trainer.py       # Model training
├── token_attribution/
│   ├── token_attribution.py   # Token attribution
│   ├── extractor.py           # Feature extraction
│   └── sa_pair_builder.py     # State-action pair builder
├── reward_kl/
│   ├── kl_divergence.py       # KL divergence computation
│   └── reward_adjuster.py     # Reward adjustment
├── advantage/
│   ├── gae_advantage.py       # GAE advantage computation
│   ├── return_calculator.py   # Return calculation
│   └── td_error.py            # TD error computation
├── experience_buffer/
│   └── buffer.py              # Experience buffer
├── training/
│   ├── policy_loss.py         # Policy loss computation
│   ├── policy_ratio.py        # Policy ratio computation
│   └── value_loss.py          # Value loss computation
├── main.py                    # Main execution script
└── pyproject.toml            # Project configuration
```

## 🔧 Configuration

The `PPOConfig` class in `config/ppo_config.py` contains all configurable parameters:

```python
@dataclass
class PPOConfig:
    # Model configs
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training configs
    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    batch_size: int = 32
    max_new_tokens: int = 10
    gamma: float = 0.99
    lam: float = 0.95
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2
    
    # Generation configs
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
```

## 🏗️ Architecture

### Core Components

#### PPOTrainer
The main orchestrator that coordinates all PPO components and executes the training loop.

**Key Methods:**
- `__init__(config)`: Initializes all models and components
- `ppo_step(prompts)`: Executes one complete PPO training step

**Responsibilities:**
- Orchestrates the entire PPO training process
- Manages model initialization and device placement
- Coordinates data flow between all components

#### Models

**PolicyModel**: The main model being trained with PPO
- Generates text sequences token by token
- Provides log probabilities for actions
- Updated using PPO policy loss

**SFTModel**: Supervised Fine-Tuned reference model
- Used to compute KL divergence with policy model
- Provides reference log probabilities for each action
- Remains frozen during training

**ValueModel**: Estimates state values
- Predicts expected future rewards for each state
- Trained to minimize MSE with reward-to-go targets
- Used for advantage computation

**RewardModel**: Computes rewards for generated sequences
- Evaluates the quality of generated text
- Returns scalar reward values
- Can be human feedback or automated scoring

#### SequenceGenerator
Handles text generation and state-action pair extraction.

**Key Methods:**
- `generate_sequences(prompts)`: Generates text and extracts (s,a) pairs

**Process:**
1. Generates text sequences using policy model
2. For each token, extracts:
   - State: prompt + previously generated tokens
   - Action: next token
   - Policy log probability
   - Reference log probability (from SFT model)

#### KLDivergence
Computes KL divergence between policy and reference models.

**Key Methods:**
- `add_kl_to_sa_pairs(sa_pairs)`: Adds KL divergence to each pair

**Process:**
- Calculates KL divergence per token: `logprob_policy - logprob_ref`
- Used for reward adjustment to prevent policy drift

#### RewardComputer
Computes rewards for generated sequences.

**Key Methods:**
- `compute_rewards(prompts, generated_texts)`: Returns reward values

**Process:**
- Takes full text (prompt + generated) as input
- Returns scalar reward values for each sequence

#### RewardAdjuster
Applies KL penalty to rewards to prevent policy drift.

**Process:**
- Adjusts rewards: `reward = reward - kl_coef * kl_divergence`
- Balances reward maximization with staying close to reference model

#### ValueAdder
Adds value predictions to state-action pairs.

**Key Methods:**
- `add_values_to_sa_pairs(sa_pairs)`: Adds value predictions

**Process:**
- Computes value predictions for each state
- Adds values to state-action pairs for advantage computation

#### TDError
Computes temporal difference errors.

**Process:**
- Calculates TD error: `reward + gamma * next_value - current_value`
- Used for advantage computation

#### GAEAdvantage
Computes Generalized Advantage Estimation.

**Key Methods:**
- `compute_gae_advantages(sa_pairs, gamma, lam)`: Returns GAE advantages

**Process:**
- Groups pairs by prompt sequence
- Computes GAE recursively: `gae = td_error + gamma * lam * next_gae`
- Provides stable advantage estimates

#### ReturnCalculator
Computes reward-to-go values.

**Process:**
- Calculates cumulative future rewards for each state
- Used as targets for value model training

#### ModelTrainer
Handles training of policy and value models.

**Key Methods:**
- `train_value_model(sa_pairs)`: Trains value model on reward-to-go targets
- `train_policy_model(sa_pairs)`: Trains policy model using PPO loss

**Process:**
- Value model: Minimizes MSE with reward-to-go targets
- Policy model: Maximizes PPO-clipped objective with advantages

### PPO Training Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Prompts      │───▶│ generate_sequences│───▶│ add_kl_to_sa_pairs│───▶│ compute_rewards │
└─────────────────┘    │                 │    └─────────────────┘    └─────────────────┘
                       │ • PolicyModel   │              │                       │
                       │ • SFTModel      │              │                       │
                       │ • Tokenizer     │              │                       │
                       │ • Both models   │              │                       │
                       │   used during   │              │                       │
                       │   generation    │              │                       │
                       └─────────────────┘              │                       │
                                │                       │                       │
                                ▼                       ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                       │ State-Action    │    │ sa_pairs with   │    │ Rewards         │
                       │ Pairs with:     │    │ KL divergence   │    │ Sequence-level  │
                       │ • Policy logprobs│    │ added           │    │                 │
                       │ • SFT logprobs  │    │                 │    │                 │
                       └─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │                       │
                                └───────────────────────┼───────────────────────┘
                                                        ▼
                                               ┌─────────────────┐
                                               │ adjust_rewards_with_kl│
                                               │ • Combines KL   │
                                               │   and rewards   │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ add_values_to_sa_pairs│
                                               │ • ValueModel    │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ compute_td_errors│
                                               │ • TD errors     │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ compute_gae_advantages│
                                               │ • GAE advantages│
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ add_reward_to_go│
                                               │ • Returns       │
                                               └─────────────────┘
                                                        │
                                                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ train_value_model│    │ train_policy_model│
                       │ • ValueModel    │    │ • PolicyModel   │
                       │ • MSE Loss      │    │ • PPO Loss      │
                       └─────────────────┘    └─────────────────┘
```

**Step-by-Step Process (from ppo_step method):**

1. **generate_sequences(prompts)**: `SequenceGenerator` uses both `PolicyModel` and `SFTModel` to:
   - Generate text sequences using policy model
   - Extract state-action pairs for each token
   - Get policy log probabilities from policy model
   - Get reference log probabilities from SFT model (during generation)
   - Returns: `sa_pairs` (with both policy and SFT logprobs) and `generated_texts`

2. **add_kl_to_sa_pairs(sa_pairs)**: `KLDivergence` computes KL divergence between policy and SFT log probabilities per token
   - Input: `sa_pairs` (already containing both policy and SFT logprobs)
   - Output: `sa_pairs` with KL divergence added

3. **compute_rewards(prompts, generated_texts)**: `RewardComputer` uses `RewardModel` to compute sequence-level rewards
   - Input: `prompts` and `generated_texts`
   - Output: `rewards` (sequence-level)

4. **adjust_rewards_with_kl(sa_pairs, rewards, kl_coef)**: `RewardAdjuster` combines KL divergence and rewards
   - Input: `sa_pairs` (with KL), `rewards`, `kl_coef`
   - Output: `sa_pairs` with adjusted rewards

5. **add_values_to_sa_pairs(sa_pairs)**: `ValueAdder` uses `ValueModel` to predict values for each state
   - Input: `sa_pairs`
   - Output: `sa_pairs` with values added

6. **compute_td_errors(sa_pairs, gamma)**: `TDError` computes temporal difference errors
   - Input: `sa_pairs` (with values and rewards)
   - Output: `sa_pairs` with TD errors

7. **compute_gae_advantages(sa_pairs, gamma, lam)**: `GAEAdvantage` calculates Generalized Advantage Estimation
   - Input: `sa_pairs` (with TD errors)
   - Output: `sa_pairs` with advantages

8. **add_reward_to_go(sa_pairs)**: `ReturnCalculator` computes reward-to-go for value model targets
   - Input: `sa_pairs` (with advantages)
   - Output: `sa_pairs` with returns

9. **train_value_model(sa_pairs)**: `ModelTrainer` updates value model using MSE loss with reward-to-go targets

10. **train_policy_model(sa_pairs)**: `ModelTrainer` updates policy model using PPO-clipped objective with advantages

## 🚀 Running the Project

```bash
# Run the main script
python main.py
```

## 📚 Dependencies

The project uses `uv` for dependency management. Key dependencies include:

- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers library
- `peft` - Parameter-efficient fine-tuning
- `datasets` - Dataset handling
- `numpy` - Numerical computing
- `tqdm` - Progress bars

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.