# MLX PPO Language Model Fine-tuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PPO](https://img.shields.io/badge/PPO-Implemented-orange.svg)](https://arxiv.org/abs/1707.06347)

A production-ready implementation of Proximal Policy Optimization (PPO) for language model fine-tuning. This project provides a modular, scalable architecture for training language models with human feedback through reinforcement learning.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/RoshBeed/MLX-Week5-PPO.git
cd MLX-Week5-PPO

# Install dependencies
uv sync

# Run a quick demo
python main.py
```

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete PPO training pipeline for language models, featuring:

- **Modular Architecture**: Clean separation of concerns with dedicated components
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Extensible Design**: Easy to add new models, reward functions, and training strategies
- **Research Friendly**: Supports experimentation with different PPO variants

### Key Features

- ✅ **Token-level PPO**: Fine-grained control over language generation
- ✅ **KL Divergence Penalty**: Prevents policy collapse
- ✅ **GAE Advantage Estimation**: Stable advantage computation
- ✅ **Value Function Learning**: Separate value network for better estimates
- ✅ **Experience Buffering**: Efficient memory management
- ✅ **Modular Components**: Pluggable architecture for easy customization

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PPO Training Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Prompt    │───▶│   Policy    │───▶│  Generated  │          │
│  │  Dataset    │    │   Model     │    │    Text     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                           │                       │             │
│                           ▼                       ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Reward    │◀───│   Token     │◀───│   Value     │          │
│  │   Model     │    │ Attribution │    │   Model     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Reward    │    │   KL &      │    │   Advantage │          │
│  │ Computation │    │   Penalty   │    │ Computation │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                    ┌─────────────┐                              │
│                    │ Experience  │                              │
│                    │   Buffer    │                              │
│                    └─────────────┘                              │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          |
│  │   Policy    │◀───│   Model     │───▶│   Value     │          |
│  │   Update    │    │  Training   │    │   Update    │          |
│  └─────────────┘    └─────────────┘    └─────────────┘          |
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
config/
├── ppo_config.py          # PPO hyperparameters and settings
├── model_config.py        # Model architecture configurations
└── training_config.py     # Training loop configurations

base_setup/
├── tokenizer_setup.py     # Tokenizer initialization and management
└── lora_setup.py          # LoRA adapter configuration

models/
├── sft_model.py           # Supervised Fine-Tuned base model
├── policy_model.py        # PPO policy network
├── value_model.py         # Value function network
└── reward_model.py        # Reward model for human feedback

token_attribution/
├── extractor.py           # Token-level state-action extraction
└── logprob_computer.py    # Log probability computation

reward_kl/
├── reward_computer.py     # Reward signal computation
└── kl_divergence.py       # KL divergence calculation

advantage/
├── td_error.py            # Temporal Difference error computation
├── gae_advantage.py       # Generalized Advantage Estimation
└── reward_to_go.py        # Reward-to-go calculation

experience_buffer/
├── buffer_manager.py      # Experience buffer management
└── data_structures.py     # Buffer data structures

training/
├── policy_loss.py         # PPO policy loss computation
├── value_loss.py          # Value function loss
└── model_trainer.py       # Training loop orchestration

ppo_trainer/
├── ppo_trainer.py         # Main PPO training orchestrator
├── generator.py           # Text generation utilities
└── evaluator.py           # Model evaluation utilities
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- 16GB+ RAM recommended
- 50GB+ free disk space for model storage

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/RoshBeed/MLX-Week5-PPO.git
   cd MLX-Week5-PPO
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

## 🚀 Usage

### Basic Usage

```python
from ppo_trainer.ppo_trainer import PPOTrainer
from config.ppo_config import PPOConfig

# Initialize configuration
config = PPOConfig(
    model_name="Qwen/Qwen3-0.6B-Base",
    max_new_tokens=50,
    batch_size=4,
    learning_rate=1e-5,
    kl_coef=0.1
)

# Create trainer
trainer = PPOTrainer(config)

# Train the model
trainer.train(
    prompts=["Your prompt here"],
    num_epochs=10
)
```

### Advanced Usage

```python
from ppo_trainer.ppo_trainer import PPOTrainer
from config.ppo_config import PPOConfig
from models.custom_reward_model import CustomRewardModel

# Custom configuration
config = PPOConfig(
    model_name="Qwen/Qwen3-0.6B-Base",
    max_new_tokens=100,
    batch_size=8,
    learning_rate=5e-6,
    kl_coef=0.2,
    gamma=0.99,
    lam=0.95,
    clip_epsilon=0.2
)

# Custom reward model
reward_model = CustomRewardModel()

# Initialize trainer with custom components
trainer = PPOTrainer(
    config=config,
    reward_model=reward_model
)

# Training with custom callbacks
def on_epoch_end(epoch, metrics):
    print(f"Epoch {epoch}: Reward = {metrics['avg_reward']:.4f}")

trainer.train(
    prompts=prompts,
    num_epochs=20,
    callbacks=[on_epoch_end]
)
```

## ⚙️ Configuration

### PPO Configuration

```python
@dataclass
class PPOConfig:
    # Model settings
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    max_new_tokens: int = 50
    
    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 1e-5
    kl_coef: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
```

### Model Configuration

```python
@dataclass
class ModelConfig:
    # Architecture settings
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12
    
    # Training settings
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
```

## 📚 API Reference

### PPOTrainer

Main training orchestrator for PPO fine-tuning.

```python
class PPOTrainer:
    def __init__(self, config: PPOConfig, **kwargs):
        """Initialize PPO trainer with configuration."""
        
    def train(self, prompts: List[str], num_epochs: int = 10) -> Dict:
        """Train the model using PPO."""
        
    def evaluate(self, prompts: List[str]) -> Dict:
        """Evaluate the trained model."""
        
    def save_model(self, path: str):
        """Save the trained model."""
        
    def load_model(self, path: str):
        """Load a trained model."""
```

### Key Components

#### Policy Model
```python
class PolicyModel:
    def forward(self, input_ids, attention_mask=None) -> ModelOutput:
        """Forward pass through policy network."""
        
    def generate(self, input_ids, **kwargs) -> torch.Tensor:
        """Generate text using current policy."""
```

#### Value Model
```python
class ValueModel:
    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        """Predict value for given state."""
```

#### Reward Model
```python
class RewardModel:
    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        """Compute reward for given text."""
```



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the original PPO algorithm
- **Hugging Face** for the transformers library
- **Qwen Team** for the base language models

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/RoshBeed/MLX-Week5-PPO/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RoshBeed/MLX-Week5-PPO/discussions)
- **Email**: rosh.beed@roshbeed.com

---

**Made with ❤️ by Rosh**
