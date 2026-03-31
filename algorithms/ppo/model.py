from models.networks import ActorCritic
from algorithms.ppo.ppo import PPO as PPOAlgorithm
from algorithms.utils import Model
import torch

class PPO(Model):
    def __init__(self, env, model_path=None, use_history=False, history_size=0, model_sizes=[[256, 256], [256, 256]], device=torch.device("cpu"), config=None, network_type=None):
         # Initialize networks
        self.model = ActorCritic(
            env.observation_space, 
            env.action_space, 
            model_sizes[0], 
            model_sizes[1], 
            actor_type="gaussian", 
            critic_type="value", 
            device=device, 
            use_history=use_history, 
            history_size=history_size, 
            network_type=network_type, 
            config=config
        )
        
        super().__init__(env, model_path, device, config)
        
        # PPO Config
        ppo_cfg = config.get("ppo", {}) if config else {}
        
        # Initialize algorithm
        self.agent = PPOAlgorithm(
            model=self.model,
            action_space=env.action_space,
            device=device,
            config=config,
            optimizer=self.actor_optimizer,
            clip_param=ppo_cfg.get("clip_param", 0.2),
            ppo_epoch=ppo_cfg.get("ppo_epoch", 4),
            num_mini_batches=ppo_cfg.get("num_mini_batches", 4),
            value_loss_coef=ppo_cfg.get("value_loss_coef", 0.5),
            entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            num_steps=ppo_cfg.get("num_steps", 2048), # Horizon
            num_envs=env.num_envs,
        )
        
        if hasattr(self, "actor_optimizer"):
            # self.actor_optimizer is a lambda that takes params
            self.agent.optimizer = self.actor_optimizer(self.model.parameters())
