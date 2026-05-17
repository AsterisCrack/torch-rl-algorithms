try:
    from models.networks import ActorTwinCriticWithTargets
    from algorithms.sac.sac import SAC as SACAlgorithm
    from algorithms.utils import Model
except ImportError:
    from torch_rl_algorithms.models.networks import ActorTwinCriticWithTargets
    from torch_rl_algorithms.algorithms.sac.sac import SAC as SACAlgorithm
    from torch_rl_algorithms.algorithms.utils import Model
import torch

class SAC(Model):
    def __init__(self, env, model_path=None, use_history=False, history_size=0, device=torch.device("cpu"), config=None):
        # Initialize networks
        self.model = ActorTwinCriticWithTargets(env.observation_space, env.action_space, actor_type="gaussian_multivariate", device=device, use_history=use_history, history_size=history_size, config=config)

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

        