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
┌───────────────────────────────────────────────────────────────────────────────┐
│                          PPO Training Pipeline                                │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────┐                                                       │
│  │    Input Data      │                                                       │
│  │────────────────────│                                                       │
│  │ • Prompts          │                                                       │
│  │ • Datasets         │                                                       │
│  │ • Human Feedback   │                                                       │
│  └────────────────────┘                                                       │
│         │                                                                     │
│         ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────-────────┐ │
│  │                   Generation & Token Attribution                         │ │
│  │                                                                          │ │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                   │ │
│  │  │ Policy Model  │ │   SFT Model   │ │   Tokenizer   │                   │ │
│  │  │───────────────│ │───────────────│ │───────────────│                   │ │
│  │  │ • generate()  │ │ • Ref Logprobs│ │ • Tokenize    │                   │ │
│  │  │ • Logprobs    │ │ • KL Baseline │ │ • Decode      │                   │ │
│  │  │               │ │               │ │ • SA Extract  │                   │ │
│  │  └───────────────┘ └───────────────┘ └───────────────┘                   │ │
│  │         │                  │                    │                        │ │
│  │         └──────────────────┴────────────────────┘                        │ │
│  │                            ▼                                             │ │
│  │             ┌────────────────────────────┐                               │ │
│  │             │    generate_sequences()    │                               │ │
│  │             │────────────────────────────│                               │ │
│  │             │ • Input: List[str]         │                               │ │
│  │             │ • Output: SA pairs, text   │                               │ │
│  │             └--------────────────────────┘                               │ │
│  └───────────────────────────────────────────────────────────────-──────────┘ │
│         │                                     │                               │
│         ▼                                     ▼                               │
│  ┌────────────────────┐       ┌─────────────────────────────┐                 │
│  │   Reward Model     │       │               KL Divergence │                 │
│  │────────────────────│       │─────────────────────────────│                 │
│  │ • compute_rewards()│       │ • add_kl_to_sa_pairs()      │                 │
│  │ • Output: floats   │       │ • Output: SA pairs w/ KL    │                 │
│  └────────────────────┘       └─────────────────────────────┘                 │
│         │                                     │                               │
│         └───────────────────────┬─────────────┘                               │
│                                 ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────-──┐ │
│  │                      Reward Adjustment Pipeline                          │ │
│  │                                                                          │ │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                   │ │
│  │  │  Adjuster     │ │ Enhanced SA   │ │ Value Model   │                   │ │
│  │  │───────────────│ │───────────────│ │───────────────│                   │ │
│  │  │ • adjust_kl() │ │ • Adjusted    │ │ • add_values()│                   │ │
│  │  │               │ │   Rewards     │ │               │                   │ │
│  │  └───────────────┘ └───────────────┘ └───────────────┘                   │ │
│  └─────────┼────────────────┼────────────────────┼──────────────────────────┘ │
│            │                │                    │                            │
│            └────────────────┴────────────────────┘                            │
│                             ▼                                                 │
│               ┌──────────────────────────────┐                                │
│               │  Advantage Computation       │                                │
│               │──────────────────────────────│                                │
│               │ • TD Errors                  │                                │
│               │ • GAE Advantages             │                                │
│               │ • Reward-to-go               │                                │
│               └──────────────────────────────┘                                │
│                            │                                                  │
│                            ▼                                                  │
│  ┌───────────────────────────────────────────────────────────────-──────────┐ │
│  │                          Model Training                                  │ │
│  │                                                                          │ │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                   │ │
│  │  │ Value Trainer │ │ Policy Trainer│ │ Experience Buf│                   │ │
│  │  │───────────────│ │───────────────│ │───────────────│                   │ │
│  │  │ • MSE Loss    │ │ • PPO Loss    │ │ • SA Batching │                   │ │
│  │  │ • Updates     │ │ • Grad Step   │ │               │                   │ │
│  │  └───────────────┘ └───────────────┘ └───────────────┘                   │ │
│  │         │                  │                 │                           │ │
│  │         └──────────────────┼──----───────────┘                           │ │
│  │                            ▼                                             │ │
│  │                   ┌─────────────────┐                                    │ │
│  │                   │ Updated Models  │                                    │ │
│  │                   │─--──────────────│                                    │ │
│  │                   │ • Policy Model  │                                    │ │
│  │                   │ • Value Model   │                                    │ │
│  │                   │ • Ready for     │                                    │ │
│  │                   │   Next Step     │                                    │ │
│  │                   └─────────────────┘                                    │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │ 
└───────────────────────────────────────────────────────────────────────────────┘
```

Component Details:
==================

1. INPUT DATA LAYER
   - Prompts: Text inputs for generation
   - Datasets: Training and evaluation data
   - Human Feedback: Reward signals and preferences

2. GENERATION & TOKEN ATTRIBUTION LAYER
   - Policy Model: Current policy for text generation
   - SFT Model: Reference model for KL divergence
   - Tokenizer: Text tokenization and state-action extraction
   - generate_sequences(): Creates SA pairs and generated text

3. KL DIVERGENCE & REWARDS LAYER 
   - KL Divergence: Distance from reference model (add_kl_to_sa_pairs)
   - Reward Model: Learned reward function (compute_rewards)
   - Reward Adjuster: KL penalty application (adjust_rewards_with_kl)

4. ADVANTAGE & VALUE COMPUTATION LAYER
   - Value Model: State value predictions (add_values_to_sa_pairs)
   - TD Error: Temporal difference error computation (compute_td_errors)
   - GAE Advantage: Generalized advantage estimation (compute_gae_advantages)
   - Return Calculator: Reward-to-go computation (add_reward_to_go)

5. MODEL TRAINING LAYER
   - Value Model Trainer: MSE loss and value updates (train_value_model)
   - Policy Model Trainer: PPO-clip loss and policy updates (train_policy_model)
   - Experience Buffer: Training data storage


Data Flow:
==========

1. Prompts → generate_sequences() → State-Action Pairs + Generated Text
2. Generated Text → compute_rewards() → Rewards
3. State-Action Pairs → add_kl_to_sa_pairs() → Enhanced SA Pairs
4. Enhanced SA Pairs + Rewards → adjust_rewards_with_kl() → Adjusted SA Pairs
5. Adjusted SA Pairs → add_values_to_sa_pairs() → SA Pairs with Values
6. SA Pairs with Values → compute_td_errors() + compute_gae_advantages() + add_reward_to_go() → Training Targets
7. Training Targets → train_value_model() + train_policy_model() → Updated Models
8. Updated Models → Next Generation Cycle (Feedback Loop)

This architecture enables efficient, stable, and scalable PPO training for language models
with human feedback, supporting both research and production deployments. 


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