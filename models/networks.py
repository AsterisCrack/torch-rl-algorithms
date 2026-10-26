import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from collections import OrderedDict
try:
    from algorithms.utils import MeanStd
    from models.factory import NetworkFactory
    from models.equivariant import EquivariantActor, TABLE_KEYS
except ImportError:
    from torch_rl_algorithms.algorithms.utils import MeanStd
    from torch_rl_algorithms.models.factory import NetworkFactory
    from torch_rl_algorithms.models.equivariant import EquivariantActor, TABLE_KEYS
from config import NetworkType, NetworkConfig
from gymnasium import spaces
    
class BaseActorCritic(nn.Module):
    def __init__(self, obs_space, target_coeff=0.005):
        super().__init__()
        self.obs_space = obs_space
        self.target_coeff = target_coeff
        
        # Initialize normalizers
        if isinstance(obs_space, spaces.Dict):
            self.actor_obs_space = obs_space["actor"]
            self.critic_obs_space = obs_space["critic"]
            self.actor_observation_normalizer = MeanStd(shape=self.actor_obs_space.shape)
            self.critic_observation_normalizer = MeanStd(shape=self.critic_obs_space.shape)
        else:
            self.actor_obs_space = obs_space
            self.critic_obs_space = obs_space
            self.actor_observation_normalizer = MeanStd(shape=obs_space.shape)
            self.critic_observation_normalizer = self.actor_observation_normalizer

        self.observation_normalizer = self.actor_observation_normalizer if not isinstance(obs_space, spaces.Dict) else None
        self.return_normalizer = None

    def _resolve_configs(self, config, network_type, use_history, actor_sizes, critic_sizes):
        # Resolve Actor Network Config
        if config and hasattr(config, "model") and config.model.actor_config:
            actor_config = config.model.actor_config
        else:
            nt = network_type or (NetworkType.CNN if use_history else NetworkType.MLP)
            actor_config = NetworkConfig(network_type=nt, hidden_sizes=actor_sizes or [256, 256])
        
        # Resolve Critic Network Config
        if config and hasattr(config, "model") and config.model.critic_config:
            critic_config = config.model.critic_config
        else:
            nt = network_type or (NetworkType.CNN if use_history else NetworkType.MLP)
            critic_config = NetworkConfig(network_type=nt, hidden_sizes=critic_sizes or [256, 256])
            
        return actor_config, critic_config

    def to(self, device):
        super().to(device)
        if self.observation_normalizer:
            self.observation_normalizer.to(device)
        if self.actor_observation_normalizer:
            self.actor_observation_normalizer.to(device)
        if self.critic_observation_normalizer and self.critic_observation_normalizer is not self.actor_observation_normalizer:
            self.critic_observation_normalizer.to(device)
        return self

    def _get_sync_pairs(self):
        """ Returns a list of (target_module, source_module) pairs for synchronization. """
        raise NotImplementedError

    def assign_targets(self):
        """ Hard copy parameters to target networks. """
        for target, source in self._get_sync_pairs():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def update_targets(self):
        """ Soft update of target networks: target <- (1 - c) target + c source, as one fused op. """
        with torch.no_grad():
            pairs = getattr(self, "_target_lists", None)
            if pairs is None:
                # The Parameter objects, not .data: they survive .to(device) / load_state_dict.
                tgt, src = [], []
                for target, source in self._get_sync_pairs():
                    for target_param, param in zip(target.parameters(), source.parameters()):
                        tgt.append(target_param)
                        src.append(param)
                # A target tensor listed twice would be lerped twice by the old loop; keep that case on it.
                pairs = (tgt, src) if len({t.data_ptr() for t in tgt}) == len(tgt) else False
                self._target_lists = pairs
            if pairs:
                torch._foreach_lerp_(pairs[0], pairs[1], self.target_coeff)
                return
            for target, source in self._get_sync_pairs():
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.data.mul_(1 - self.target_coeff)
                    target_param.data.add_(self.target_coeff * param.data)

class ActorCriticWithTargets(BaseActorCritic):
    def __init__(self, obs_space, action_space, actor_sizes=None, critic_sizes=None, actor_type="gaussian", critic_type="deterministic", use_history=False, history_size=0, target_coeff=0.005, device=torch.device("cpu"), network_type=None, config=None):
        super().__init__(obs_space, target_coeff)
        
        actor_config, critic_config = self._resolve_configs(config, network_type, use_history, actor_sizes, critic_sizes)

        # Build networks using factory
        self.actor = NetworkFactory.build_actor(actor_config, self.actor_obs_space, action_space, self.actor_observation_normalizer, actor_type, history_size, device)
        self.critic = NetworkFactory.build_critic(critic_config, self.critic_obs_space, action_space, self.critic_observation_normalizer, critic_type, history_size, device)
            
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.assign_targets()

    def _get_sync_pairs(self):
        return [(self.target_actor, self.actor), (self.target_critic, self.critic)]

class ActorCritic(BaseActorCritic):
    def __init__(self, obs_space, action_space, actor_sizes=None, critic_sizes=None, actor_type="gaussian", critic_type="deterministic", use_history=False, history_size=0, device=torch.device("cpu"), network_type=None, config=None):
        super().__init__(obs_space, target_coeff=0.0) # No targets
        
        actor_config, critic_config = self._resolve_configs(config, network_type, use_history, actor_sizes, critic_sizes)

        # Build networks using factory
        self.actor = NetworkFactory.build_actor(actor_config, self.actor_obs_space, action_space, self.actor_observation_normalizer, actor_type, history_size, device)
        self.critic = NetworkFactory.build_critic(critic_config, self.critic_obs_space, action_space, self.critic_observation_normalizer, critic_type, history_size, device)

    def _get_sync_pairs(self):
        return []

class ActorTwinCriticWithTargets(BaseActorCritic):
    def __init__(self, obs_space, action_space, actor_sizes=None, critic_sizes=None, actor_type="gaussian", use_history=False, history_size=0, target_coeff=0.005, device=torch.device("cpu"), network_type=None, config=None, equivariant_actor=False, mirror_tables=None):
        super().__init__(obs_space, target_coeff)
        
        actor_config, critic_config = self._resolve_configs(config, network_type, use_history, actor_sizes, critic_sizes)

        # Build networks using factory
        self.actor = NetworkFactory.build_actor(actor_config, self.actor_obs_space, action_space, self.actor_observation_normalizer, actor_type, history_size, device)
        self.equivariant_actor = bool(equivariant_actor)
        if self.equivariant_actor:
            # Wrap BEFORE the target copy below, so every user of `.actor` (and `.target_actor`) sees the
            # symmetrised policy. mirror_tables = (obs_perm, obs_sign, act_perm, act_sign) over the actor
            # input as it reaches the actor; None builds a placeholder whose tables must come from a checkpoint.
            if actor_type != "gaussian_multivariate":
                raise ValueError(f"equivariant_actor needs the gaussian_multivariate actor head, got {actor_type!r}")
            obs_dim, act_dim = self.actor_obs_space.shape[0], action_space.shape[0]
            if mirror_tables is not None:
                if len(mirror_tables[0]) != obs_dim or len(mirror_tables[2]) != act_dim:
                    raise ValueError(f"mirror tables have obs/action length {len(mirror_tables[0])}/{len(mirror_tables[2])}, "
                                     f"but the actor sees obs {obs_dim} / action {act_dim}")
                self.actor = EquivariantActor(self.actor, *mirror_tables)
            else:
                self.actor = EquivariantActor(self.actor, obs_dim=obs_dim, act_dim=act_dim)
        self.critic_1 = NetworkFactory.build_critic(critic_config, self.critic_obs_space, action_space, self.critic_observation_normalizer, "deterministic", history_size, device)
        # critic_2 must use the SAME stage-2 normaliser object as critic_1. A plain deepcopy gave
        # critic_2 a private MeanStd copy that is never record()ed / update()d (the DDPG loop only
        # updates self.critic_observation_normalizer, which critic_1 and the actor-side attribute
        # reference), so critic_2 kept mean 0 / std 1 forever while critic_1 was normalised.
        # Pre-seeding the deepcopy memo keeps the normaliser shared (also through the target copy).
        _norm = self.critic_observation_normalizer
        self.critic_2 = copy.deepcopy(self.critic_1, memo={id(_norm): _norm} if _norm is not None else None)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.assign_targets()

    _CRITIC_NORM_KEY = ".torso.observation_normalizer."

    def _adapt_actor_keys(self, state_dict):
        """Reconcile the checkpoint's actor layout with this model's (plain `actor.*` vs equivariant
        `actor.net.*` + `actor.mirror_*`). Plain checkpoint -> equivariant model: the inner network weights
        are loaded and the mirror buffers keep the values built from the env. Equivariant checkpoint ->
        plain model: refused with a clear message (it would silently drop the symmetrisation)."""
        ckpt_equiv = any(k.startswith("actor.mirror_") for k in state_dict)
        if ckpt_equiv and not self.equivariant_actor:
            raise ValueError("this checkpoint was trained with an equivariant actor (keys 'actor.mirror_*'); "
                             "set train.equivariant_actor: true in the config, otherwise the symmetrisation "
                             "would be silently dropped")
        if not self.equivariant_actor or ckpt_equiv:
            return state_dict
        if not self.actor._ready:
            raise ValueError("this checkpoint has no mirror tables (trained with a plain actor) and the model was built "
                             "without an env that provides actor_mirror_tables(), so they cannot be reconstructed. "
                             "Load it with equivariant_actor false, or construct the model from the training env.")
        sd = OrderedDict()
        md = getattr(state_dict, "_metadata", None)
        if md is not None:
            sd._metadata = md
        for k, v in state_dict.items():
            for name in ("actor", "target_actor"):
                if k.startswith(name + "."):
                    k = name + ".net." + k[len(name) + 1:]
                    break
            sd[k] = v
        for name in ("actor", "target_actor"):
            for key in TABLE_KEYS:
                sd[f"{name}.{key}"] = getattr(getattr(self, name), key).detach().cpu()
        print("[equivariant_actor] plain-actor checkpoint: loaded its weights into the equivariant actor, "
              "mirror tables taken from the env")
        return sd

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict = self._adapt_actor_keys(state_dict)
        # Legacy checkpoints stored critic_2's private (never updated, identity) normaliser copy.
        # critic_2 now shares critic_1's module, and load_state_dict writes keys in module order
        # (critic_2 last), so the stale identity values would overwrite critic_1's good ones.
        # Overwrite the critic_2 entries with critic_1's before loading (same for the targets).
        pairs = [(k, "critic_2" + k[len("critic_1"):]) for k in state_dict.keys()
                 if k.startswith("critic_1" + self._CRITIC_NORM_KEY)]
        pairs += [(k, "target_critic_2" + k[len("target_critic_1"):]) for k in state_dict.keys()
                  if k.startswith("target_critic_1" + self._CRITIC_NORM_KEY)]
        if pairs:
            sd = OrderedDict(state_dict)
            md = getattr(state_dict, "_metadata", None)
            if md is not None:
                sd._metadata = md
            for src, dst in pairs:
                sd[dst] = state_dict[src]
            state_dict = sd
        return super().load_state_dict(state_dict, *args, **kwargs)

    def _get_sync_pairs(self):
        return [
            (self.target_actor, self.actor),
            (self.target_critic_1, self.critic_1),
            (self.target_critic_2, self.critic_2)
        ]
                