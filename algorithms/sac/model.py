try:
    from models.networks import ActorTwinCriticWithTargets
    from models.equivariant import find_actor_mirror_tables
    from algorithms.sac.sac import SAC as SACAlgorithm
    from algorithms.utils import Model
except ImportError:
    from torch_rl_algorithms.models.networks import ActorTwinCriticWithTargets
    from torch_rl_algorithms.models.equivariant import find_actor_mirror_tables
    from torch_rl_algorithms.algorithms.sac.sac import SAC as SACAlgorithm
    from torch_rl_algorithms.algorithms.utils import Model
import torch

class SAC(Model):
    def __init__(self, env, model_path=None, use_history=False, history_size=0, device=torch.device("cpu"), config=None):
        # Exactly mirror-equivariant actor (train.equivariant_actor): the mirror tables come from the env
        # when it has them; env-less callers (sim2real, policy_gate) get them from the checkpoint's buffers.
        equivariant = bool(config) and bool(getattr(config.train, "equivariant_actor", False))
        mirror_tables = None
        if equivariant:
            mirror_tables = find_actor_mirror_tables(env)
            if mirror_tables is None and not model_path:
                raise ValueError(
                    "train.equivariant_actor is true but the environment provides no mirror tables "
                    "(no actor_mirror_tables() on the env or its wrappers) and there is no checkpoint to read "
                    "them from. Use an env with a mirror transform (BipedEnv) or set train.equivariant_actor: false.")

        # Initialize networks
        self.model = ActorTwinCriticWithTargets(env.observation_space, env.action_space, actor_type="gaussian_multivariate", device=device, use_history=use_history, history_size=history_size, config=config, equivariant_actor=equivariant, mirror_tables=mirror_tables)

        super().__init__(env, model_path, device, config)

        # Build symmetry function if the underlying env supports it and config requests it
        symmetry_fn = None
        use_sym = False
        if config:
            use_sym = bool(getattr(config.train, "symmetry_augmentation", False))
        if use_sym:
            raw_env = env.env if hasattr(env, "env") else env
            if hasattr(raw_env, "mirror_obs") and hasattr(raw_env, "mirror_action"):
                def _symmetry_fn(obs_dict, actions):
                    return raw_env.mirror_obs(obs_dict), raw_env.mirror_action(actions)
                symmetry_fn = _symmetry_fn

        self.agent = SACAlgorithm(
            action_space=env.action_space,
            model=self.model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            device=device,
            config=config,
            symmetry_fn=symmetry_fn,
        )
        if hasattr(self.agent, "set_env"):
            self.agent.set_env(env)

        