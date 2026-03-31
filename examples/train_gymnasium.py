"""
Example: training on standard Gymnasium environments with different
algorithm and backbone combinations.

Run from the repo root:
    python examples/train_gymnasium.py
"""

import gymnasium as gym
from config import NetworkConfig, NetworkType
from algorithms.sac.model import SAC
from algorithms.ppo.model import PPO
from algorithms.ddpg.model import DDPG
from algorithms.d4pg.model import D4PG
from algorithms.mpo.model import MPO

# ── Environment ───────────────────────────────────────────────────────────────

env = gym.make("Pendulum-v1")

# ── Algorithm examples ────────────────────────────────────────────────────────

# SAC with default MLP backbone
model = SAC(env, device="cpu")

# PPO with larger hidden layers
model = PPO(
    env,
    config={
        "model": {
            "actor_config": NetworkConfig(
                network_type=NetworkType.MLP,
                hidden_sizes=[512, 512],
            ).model_dump(),
        }
    },
    device="cpu",
)

# SAC with separate actor / critic architectures
model = SAC(
    env,
    config={
        "model": {
            "actor_config": NetworkConfig(
                network_type=NetworkType.MLP,
                hidden_sizes=[256, 256],
            ).model_dump(),
            "critic_config": NetworkConfig(
                network_type=NetworkType.MLP,
                hidden_sizes=[512, 512, 256],
            ).model_dump(),
        }
    },
    device="cpu",
)

# ── Backbone examples ─────────────────────────────────────────────────────────

# CNN over an observation history window
cnn_config = NetworkConfig(
    network_type=NetworkType.CNN,
    hidden_sizes=[256, 256],
    cnn_sizes=[[3, 32, 2], [3, 32, 2]],  # [kernel, out_channels, stride] per layer
)

# LSTM for sequential observations
lstm_config = NetworkConfig(
    network_type=NetworkType.LSTM,
    hidden_size=256,
    num_layers=2,
)

# Causal Transformer for long history with positional encoding
transformer_config = NetworkConfig(
    network_type=NetworkType.TRANSFORMER,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
)

# Pass any of these as actor_config / critic_config (see SAC example above)

# ── Train ─────────────────────────────────────────────────────────────────────

model = SAC(env, device="cpu")
model.train(steps=50_000)
