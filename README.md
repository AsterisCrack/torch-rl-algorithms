# torch-rl-algorithms

**Compact PyTorch implementations of SAC, PPO, DDPG, D4PG and MPO, with swappable network backbones, built for training on thousands of vectorised GPU environments.**
Written for the [BipedRobot](https://github.com/AsterisCrack/BipedRobot) project (a small biped that learns
to walk in Isaac Lab), but nothing in it is specific to robots: any environment that returns batched
tensors can be trained on.

---

## Features

- **Five algorithms** behind one interface: SAC, PPO, DDPG, D4PG and MPO.
- **Four backbones**, MLP, CNN, LSTM and causal Transformer, chosen with one config field. The
  last three read a history of stacked observation frames.
- **Built for vectorised environments.** Rollouts, replay buffer, n-step returns and logging all stay on
  the environment's device, with no per-step host copies.
- **Asymmetric actor-critic.** The critic can see privileged observations the actor never gets.
- **Smooth, symmetric policies (SAC):**
  - CAPS action-smoothness regulariser;
  - mirror augmentation of the replay buffer and an explicit mirror loss;
  - an actor that is mirror-symmetric exactly, by construction.
- **Online observation normalisation** built into every network and saved with its checkpoints.
- **Trainer included:** TensorBoard logging, periodic checkpoints and Ctrl+C-safe state saving.

---

## Install

```bash
git clone https://github.com/AsterisCrack/torch-rl-algorithms
cd torch-rl-algorithms
pip install -r requirements.txt     # or: pip install -e .
```

Python 3.9+, PyTorch 2+, Gymnasium, NumPy, Pydantic, TensorBoardX and tqdm. For GPU training, install the
CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first.

The code uses top-level imports (`algorithms`, `models` and `config`), so the repository root has to
be on `sys.path`. `pip install -e .` registers `algorithms` and `models` but not the single-file
`config` module, so either run from the repository root or set `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD              # PowerShell: $env:PYTHONPATH = $PWD
```

To use it from another project, add it as a submodule and put its folder on `sys.path` (an install
alone is not enough, for the same reason):

```bash
git submodule add https://github.com/AsterisCrack/torch-rl-algorithms
pip install -e torch-rl-algorithms/
export PYTHONPATH=$PWD/torch-rl-algorithms
```

If your project has its own top-level `config` package, append the submodule folder to `sys.path`
instead of prepending it, so the two do not shadow each other.

---

## Algorithms

```python
from algorithms.sac.model import SAC
from algorithms.ppo.model import PPO
from algorithms.ddpg.model import DDPG
from algorithms.d4pg.model import D4PG
from algorithms.mpo.model import MPO
```

| Algorithm | Kind | Notes | Paper |
|---|---|---|---|
| **SAC** | Off-policy, stochastic | Twin critics, automatic temperature, tanh-squashed Gaussian policy | [Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290) |
| **PPO** | On-policy, stochastic | Clipped surrogate, GAE, clipped value loss, minibatch epochs | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) |
| **DDPG** | Off-policy, deterministic | Target networks, Gaussian exploration noise | [Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971) |
| **D4PG** | Off-policy, deterministic | DDPG with a categorical distributional critic (51 atoms on [-150, 150]) | [Barth-Maron et al. 2018](https://arxiv.org/abs/1804.08617) |
| **MPO** | Off-policy, stochastic | E-step / M-step with dual variables and per-dimension KL constraints | [Abdolmaleki et al. 2018](https://arxiv.org/abs/1806.06920) |

Every class takes the same arguments (`SAC` and the others share `env`, `model_path`, `use_history`,
`history_size`, `device` and `config`) and trains with `model.train(...)`. Actions are assumed to lie in
[-1, 1]: the actors squash with `tanh` and nothing rescales them, so scale actions inside your
environment.

### What each algorithm supports

| | SAC | PPO | DDPG | D4PG | MPO |
|---|:-:|:-:|:-:|:-:|:-:|
| MLP / CNN / LSTM / Transformer backbones | yes | yes | yes | yes | yes |
| Dict observations (actor + privileged critic) | yes | yes | yes | yes | yes |
| n-step returns, replay buffer | yes | n/a (GAE) | yes | yes | yes |
| Observation normalisation | yes | yes | yes | yes | yes |
| Learning-rate schedulers | yes | yes | yes | yes | yes |
| CAPS smoothness loss | yes | | | | |
| Mirror augmentation, mirror loss, equivariant actor | yes | | | | |

### CAPS (SAC)

[CAPS](https://arxiv.org/abs/2012.06644) adds two penalties to the actor loss so the policy output
stays smooth. Both act on the deterministic action `mu(s)`:

| Term | Penalises | Config |
|---|---|---|
| Temporal | `‖mu(s_t) − mu(s_t+1)‖²`, skipping episode boundaries | `train.caps_temporal_weight` |
| Spatial | `‖mu(s) − mu(s + ε)‖²` with `ε ~ N(0, σ²)` in normalised observation space | `train.caps_spatial_weight`, `train.caps_spatial_sigma` |

Both default to 0 (off). The three terms share one actor forward pass.

### Mirror symmetry (SAC)

For robots with a left/right symmetry there are three independent switches, from soft to exact:

| Option | What it does |
|---|---|
| `train.symmetry_augmentation` | Stores a mirrored copy of every transition in the replay buffer |
| `train.symmetry_loss_weight` | Adds `‖mu(M s) − M mu(s)‖²` to the actor loss ([Yu et al. 2018](https://arxiv.org/abs/1801.08093)); needs augmentation on |
| `train.equivariant_actor` | Wraps the actor so that `mu(M s) = M mu(s)` holds exactly, not just approximately |

The equivariant actor evaluates the network on `s` and on the mirrored `M s` and averages
`loc = ½ (f(s) + M f(M s))` in pre-tanh space. The mirror tables are stored inside the checkpoint, so a
trained policy can be reloaded and exported without the environment. Plain checkpoints load into an
equivariant model, but not the other way round (that would silently drop the symmetry).

These options need the environment to describe its mirror. It provides (on itself, or on its `.env`
attribute if it is a wrapper):

| Method | Needed by | Returns |
|---|---|---|
| `mirror_obs(obs)` | augmentation, mirror loss | The mirrored observation, in the same structure the environment returns (tensor, or `{"actor": ..., "critic": ...}`) |
| `mirror_action(actions)` | augmentation, mirror loss | The mirrored actions |
| `actor_mirror_tables()` | equivariant actor | `(obs_perm, obs_sign, act_perm, act_sign)` for the actor input, where `M x = sign * x[perm]` and both maps are involutions |

---

## Quick start

The library trains on **vectorised environments that return torch tensors**, not on a plain
`gym.make(...)` environment (see [Environment interface](#environment-interface)). Here is a small
adapter around Gymnasium's vector env, and a SAC run on Pendulum:

```python
import gymnasium as gym
import torch

class VecEnv:
    """Gymnasium vector env -> the start()/step() interface the library trains on."""

    def __init__(self, env_id, num_envs=8, device="cpu"):
        self.env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(num_envs)])
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.observation_space = self.env.single_observation_space
        self.action_space = self.env.single_action_space

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def start(self):
        obs, _ = self.env.reset()
        return self._t(obs)

    def step(self, actions):
        obs, rew, terminated, truncated, _ = self.env.step(actions.cpu().numpy())
        obs = self._t(obs)
        return obs, {
            "observations": obs,                         # observation after the step
            "next_observations": obs,                    # same tensor, read by PPO
            "rewards": self._t(rew),
            "terminations": self._t(terminated),         # true terminal state: no bootstrapping
            "resets": self._t(terminated | truncated),   # episode ended for any reason
        }
```

```python
from algorithms.sac.model import SAC

env = VecEnv("Pendulum-v1", num_envs=8)
model = SAC(env, device="cpu", config=config)     # `config` is defined in the next section
model.train(steps=5000, log_dir="runs", checkpoint_path="checkpoints/pendulum")
```

`steps` counts iterations of the vectorised environment (each one is `num_envs` transitions).
Progress is printed per epoch, scalars go to TensorBoard (`tensorboard --logdir runs`) and
checkpoints are written to `checkpoints/pendulum/step_<N>.pt`. Swap `SAC` for `PPO`, `DDPG`, `D4PG` or
`MPO` to change algorithm.

Run the trained policy greedily:

```python
obs = env.start()
with torch.no_grad():
    for _ in range(200):
        obs, _ = env.step(model.step(obs))        # model.step returns the deterministic action
```

---

## Configuration

Hyperparameters live in one config object with four sections: `train`, `model`, `buffer` and `ppo`.
The library reads it in three ways (`config["buffer"]["batch_size"]`, `config.get("ppo")` and
`config.train.caps_temporal_weight`), so a plain dict does not work. BipedRobot uses a Pydantic schema loaded
from YAML ([`config/schema.py`](https://github.com/AsterisCrack/BipedRobot/blob/main/config/schema.py)).
For standalone use, this small adapter is enough:

```python
from config import NetworkConfig, NetworkType

class Section:
    """One config section: item access, .get() and attribute access."""
    def __init__(self, **values):
        self.__dict__.update(values)
    def __getitem__(self, key):
        return self.__dict__.get(key)             # missing -> None, the library falls back to its default
    def get(self, key, default=None):
        return self.__dict__.get(key, default)

class Config:
    def __init__(self, train=None, model=None, buffer=None, ppo=None):
        model = {"actor_config": None, "critic_config": None, **(model or {})}
        self.train, self.model = Section(**(train or {})), Section(**model)
        self.buffer, self.ppo = Section(**(buffer or {})), Section(**(ppo or {}))
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return vars(getattr(self, key, None) or Section()) or default
    def copy_in_file(self, path):                  # called at every checkpoint; save the config here if you like
        pass

config = Config(
    train=dict(caps_temporal_weight=1.0, caps_spatial_weight=1.0, caps_spatial_sigma=0.1),
    model=dict(
        actor_config=NetworkConfig(network_type=NetworkType.MLP, hidden_sizes=[256, 256]),
        critic_config=NetworkConfig(network_type=NetworkType.MLP, hidden_sizes=[256, 256]),
        actor_lr=3e-4,
        critic_lr=3e-4,
    ),
    buffer=dict(batch_size=256, batch_iterations=4, steps_before_batches=2000),
)
```

Anything left out falls back to the default in the tables below. The `train(...)` keyword arguments
(`steps`, `epoch_steps`, `save_steps`, ...) are used for values the config does not set.

### `train`

| Key | Default | |
|---|---|---|
| `steps` | `train()` argument | Iterations of the vectorised env to run |
| `epoch_steps` | 5000 | Iterations between progress reports (and test episodes, if a test env is given) |
| `save_steps` | 5000 | Iterations between checkpoints (one is always written at the end) |
| `checkpoint_path` | `checkpoints/<timestamp>` | Output directory |
| `log`, `log_interval` | on, 1 | TensorBoard logging; scalars are averaged over `log_interval` iterations |
| `replace_checkpoint` | off | Delete the previous checkpoint of the same name when saving |
| `caps_temporal_weight`, `caps_spatial_weight`, `caps_spatial_sigma` | 0, 0, 0.05 | [CAPS](#caps-sac) |
| `symmetry_augmentation`, `symmetry_loss_weight`, `equivariant_actor` | off, 0, off | [Mirror symmetry](#mirror-symmetry-sac) |

### `model`

| Key | Default | |
|---|---|---|
| `actor_config`, `critic_config` | MLP, `[256, 256]` | A [`NetworkConfig`](#network-backbones) per network |
| `actor_lr`, `critic_lr` | 1e-3 | Adam learning rates |
| `lr_scheduler` | cosine | `scheduler_type`: `cosine` (`T_max`, `eta_min`), `plateau` (`factor`, `patience`, ...) or `none` |
| `exploration` | scale 0.3, min 0.03, decay 1e-6, start 10000 | Gaussian action noise for DDPG and D4PG (`scale`, `min_scale`, `decay_rate`, `start_steps`) |

SAC explores through its stochastic policy, after taking uniform random actions for its first 10000
transitions. DDPG and D4PG do the same for `exploration.start_steps` transitions, then add the decaying noise.

### `buffer`

| Key | Default | |
|---|---|---|
| `size` | 1e6 | Replay capacity, in transitions across all environments |
| `return_steps` | 5 | n-step returns (rewards are accumulated and the bootstrap moved n steps ahead, cut at episode ends) |
| `batch_size` | 1024 | Samples per gradient step |
| `batch_iterations` | 50 | Gradient steps after each environment iteration |
| `steps_before_batches` | 1e4 | Transitions to collect before learning starts |
| `discount_factor` | 0.99 | |

### `ppo`

`clip_param` (0.2), `ppo_epoch` (4), `num_mini_batches` (4), `value_loss_coef` (0.5), `entropy_coef`
(0.01), `gamma` (0.99), `gae_lambda` (0.95), `max_grad_norm` (0.5) and `num_steps` (2048, the rollout length
per environment between updates).

---

## Network backbones

The backbone is chosen per network with a [`NetworkConfig`](config.py), so the actor and the critic can
differ. `NetworkFactory` builds the matching actor or critic, and the algorithm code does not change.

| `network_type` | Class | Reads | Fields used |
|---|---|---|---|
| `mlp` | `MLPActor`, `MLPCritic` | The flat observation | `hidden_sizes` |
| `cnn` | `CNNActor`, `CNNCritic` | A history window, as a 1-D signal over time, then an MLP | `cnn_sizes`, `hidden_sizes` |
| `lstm` | `LSTMActor`, `LSTMCritic` | A history window, last output | `hidden_size`, `num_layers` |
| `transformer` | `TransformerActor`, `TransformerCritic` | A history window, one token per frame | `d_model`, `nhead`, `num_layers`, `dim_feedforward` |

```python
NetworkConfig(network_type=NetworkType.MLP, hidden_sizes=[1024, 512, 256])
NetworkConfig(network_type=NetworkType.CNN, hidden_sizes=[256, 256],
              cnn_sizes=[[3, 32, 2], [3, 32, 2]])                  # [kernel, channels, stride] x 2
NetworkConfig(network_type=NetworkType.LSTM, hidden_size=128, num_layers=2)
NetworkConfig(network_type=NetworkType.TRANSFORMER, d_model=128, nhead=4, num_layers=2, dim_feedforward=256)
```

In YAML:

```yaml
model:
  actor_config:
    network_type: transformer
    d_model: 128
    nhead: 4
    num_layers: 2
    dim_feedforward: 256
  critic_config:
    network_type: mlp
    hidden_sizes: [512, 512, 256]
```

**Observation history.** The sequence backbones expect the environment to return the last `K` frames
concatenated into one flat vector, oldest first (`K * frame_size` values). Tell the networks how to
split it:

```python
model = SAC(env, use_history=True, history_size=K, config=config)
```

With `use_history=True` and no `actor_config`, the default backbone is the CNN. An MLP ignores the
history and sees the flat vector. The LSTM is not stateful: it runs over the `K` stacked frames on every call.

**Without network configs**, PPO, DDPG, D4PG and MPO also take `model_sizes=[[256, 256], [256, 256]]`
(actor, critic) and `network_type=`. SAC does not; it takes its networks from the config.

**Always pass a config object when you call `train()`**, even an empty one such as `Config()` above.
With `config=None` the trainer fails at the first checkpoint save, after the weights are written
(`'NoConfig' object is not callable`), because it calls `config.copy_in_file(...)`. A custom config
class must therefore implement `copy_in_file(path)`.

**Normalisation.** Every network normalises its input with a running mean and standard deviation that
updates from the data it trains on and is stored in the checkpoint.

---

## Environment interface

The trainer talks to the environment through a small batched interface. All tensors are float32, with
the number of parallel environments `N` as the first dimension, on the device you train on.

| Member | |
|---|---|
| `observation_space`, `action_space` | Gymnasium `Box` spaces of **one** environment (not batched) |
| `num_envs` | `N` (PPO sizes its rollout buffer from it) |
| `device` | Where tensors live; the replay buffer and trainer use it |
| `start()` | Resets all environments and returns the first observation, shape `(N, obs_dim)` |
| `step(actions)` | Takes `(N, act_dim)` actions in [-1, 1] and returns `(obs, infos)`; see below |

`infos` must contain:

| Key | Shape | |
|---|---|---|
| `observations` | `(N, obs_dim)` | The observation after the step |
| `rewards` | `(N,)` | |
| `terminations` | `(N,)` | 1 where the episode ended in a true terminal state (no bootstrapping) |
| `resets` | `(N,)` | 1 where the episode ended for any reason, including time limits. The environment must auto-reset these rows. |
| `next_observations` | `(N, obs_dim)` | Same as `observations`; PPO reads it for its GAE bootstrap |
| `log` *(optional)* | dict of scalars | Written to TensorBoard as-is (e.g. per-term rewards) |

Optional methods: `save(path)` and `load(path)` are called next to every checkpoint and when resuming,
for example to store the environment's own observation statistics.

**Privileged critic.** Return a `Dict` space with `actor` and `critic` entries and the same dict from
`start()` and `step()`. The actor only sees `obs["actor"]`; the critic sees `obs["critic"]`. In `infos`, PPO
reads `next_observations_actor` / `next_observations_critic` instead of `next_observations`.

```python
from gymnasium import spaces
self.observation_space = spaces.Dict({
    "actor":  spaces.Box(-float("inf"), float("inf"), (n_actor,),  "float32"),
    "critic": spaces.Box(-float("inf"), float("inf"), (n_critic,), "float32"),
})
```

### Plugging in a GPU simulator

BipedRobot trains on thousands of Isaac Lab environments with this same interface. Its
[`BaseIsaacLabWrapper`](https://github.com/AsterisCrack/BipedRobot/blob/main/src/isaaclab/common.py) is
one short class (about 200 lines) and shows what a wrapper for any batched simulator needs:

- **Spaces.** Isaac Lab reports batched spaces; the wrapper exposes the single-environment spaces, and
  builds the `actor` / `critic` dict when the task has privileged observations.
- **`step`.** It maps `(obs, reward, terminated, truncated, info)` to `(obs, infos)` with the keys above:
  `resets = terminated | truncated`, plus `log` for the per-term rewards. Isaac Lab resets finished
  environments itself, so nothing else is needed.
- **Everything stays on the GPU.** Observations, rewards and flags are never copied to the host, and
  the replay buffer is allocated on the same device.
- **Observation scaling.** The wrapper can also normalise observations before they reach the
  library (the library exports `RunningMeanStd` for this) and implements `save` / `load` so that those
  statistics travel with each checkpoint.

The training script then only builds the wrapper and hands it to the algorithm:

```python
wrapped_env = BaseIsaacLabWrapper(env, config)
model = SAC(env=wrapped_env, device=device, config=config, use_history=True, history_size=5)
model.train(steps=config.train.steps, save_steps=config.train.save_steps)
```

---

## Checkpoints and resuming

Training writes to `checkpoint_path`:

| File | Contents |
|---|---|
| `step_<N>.pt` | Network weights (actor, critics, targets, normalisers, mirror tables), `N` = transitions so far |
| Whatever `env.save(path)` writes | The environment's own state, if it implements `save(path)` (the BipedRobot wrapper writes `step_<N>_obs_scalers.pt`) |
| `config.yaml` | The config, if it implements `copy_in_file(path)` |
| `trainer_state/` | Replay buffer, optimisers and temperature, written when you stop with Ctrl+C |

To resume or evaluate, build the model with the same network config and pass the checkpoint:

```python
model = SAC(env, config=config, model_path="checkpoints/pendulum/step_40000.pt")
```

This restores the weights and calls `env.load(...)` if it exists. Optimiser state and the replay
buffer start fresh. Checkpoints trained with `equivariant_actor` need it set again when loading.

---

## Project structure

```
torch-rl-algorithms/
├── algorithms/
│   ├── sac/        SAC, with CAPS, mirror loss and replay augmentation
│   ├── ppo/        PPO and its rollout buffer with GAE
│   ├── ddpg/       DDPG: the base of the off-policy step / update loop
│   ├── d4pg/       D4PG (extends DDPG)
│   ├── mpo/        MPO
│   └── utils.py    Replay buffer, Trainer, entropy and noise schedules, MeanStd, Model base class
├── models/
│   ├── mlp.py, cnn.py, lstm.py, transformer.py   Backbones with their actor and critic heads
│   ├── networks.py     ActorCritic, ActorCriticWithTargets, ActorTwinCriticWithTargets
│   ├── factory.py      NetworkFactory: NetworkConfig -> actor / critic
│   └── equivariant.py  Mirror-equivariant actor wrapper
├── config.py           NetworkType and NetworkConfig
├── examples/           Network configuration snippets (not a runnable script; see Quick start)
├── tests_caps/         pytest suite
├── pyproject.toml
└── requirements.txt
```

SAC and D4PG extend `DDPG`: the step, replay and update loop lives once in `algorithms/ddpg/ddpg.py`, and each
algorithm only supplies its own actor and critic updaters.

---

## Tests

```bash
pip install pytest
python -m pytest tests_caps -q       # from the repository root
```

The suite covers the CAPS terms, the mirror loss, replay augmentation, symmetric observation
statistics and the equivariant actor (exact symmetry, table validation, checkpoint loading). Tests that
need `onnxruntime` or the BipedRobot export script are skipped when those are missing.

---

## Known limitations

- `buffer.steps_between_batches` is accepted but does not set an update cadence: updates run every
  iteration once `steps_before_batches` is reached (a larger value only delays the first update). Use
  `batch_iterations` to control the update ratio.
- The soft target update coefficient is fixed at 0.005 for SAC (and the other target-network
  algorithms); there is no config key for it.
- `examples/train_gymnasium.py` constructs models from a plain `gym.make` environment and dict configs,
  which the library does not accept. Follow the Quick start instead.

---

## License

[MIT](LICENSE)
