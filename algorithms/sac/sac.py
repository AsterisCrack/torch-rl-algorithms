import os
import torch
import numpy as np
try:
    import algorithms.ddpg.ddpg as ddpg
    from algorithms.utils import NoActionNoise, DecayingEntropyCoeff, AutoEntropyCoeff, SquashedMultivariateNormalDiag, to_tensor
except ImportError:
    import torch_rl_algorithms.algorithms.ddpg.ddpg as ddpg
    from torch_rl_algorithms.algorithms.utils import NoActionNoise, DecayingEntropyCoeff, AutoEntropyCoeff, SquashedMultivariateNormalDiag, to_tensor


def _batched_actor_forward(actor, parts):
    """actor over torch.cat(parts) in ONE forward -> (distribution of parts[0], [squashed mean of each part]).

    Row-wise networks (MLP, the equivariant wrapper) give every row the same value and gradient as a
    separate forward; this only saves kernel launches. Distributions other than
    SquashedMultivariateNormalDiag cannot be split here, so those fall back to separate forwards."""
    sizes = [p.shape[0] for p in parts]
    out = actor(torch.cat(parts, dim=0))
    if isinstance(out, SquashedMultivariateNormalDiag):
        locs = out._distribution.loc.split(sizes)
        scales = out._distribution.scale.split(sizes)
        return SquashedMultivariateNormalDiag(locs[0], scales[0]), [torch.tanh(loc) for loc in locs]
    first = actor(parts[0])
    return first, [first.mean] + [actor(p).mean for p in parts[1:]]

class TwinCriticSoftDeterministicPolicyGradient:
    def __init__(self, model, action_space, device=torch.device("cpu"), seq_length=1, optimizer=None, entropy_coeff=DecayingEntropyCoeff(), gradient_clip=0, recurrent_model = False, caps_temporal_weight=0.0, caps_spatial_weight=0.0, caps_spatial_sigma=0.05, symmetry_loss_weight=0.0, mirror_fn=None):
        self.device = device
        self.recurrent_model = recurrent_model
        self.seq_length = seq_length
        self.optimizer = (lambda params: torch.optim.Adam(params, lr=3e-4)) if optimizer is None else optimizer
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip
        # CAPS smoothness regularization (Mysore et al. 2021), off when weights are 0.
        self.caps_temporal_weight = caps_temporal_weight
        self.caps_spatial_weight = caps_spatial_weight
        self.caps_spatial_sigma = caps_spatial_sigma
        # Explicit mirror-symmetry loss (Yu et al. 2018), off when the weight is 0.
        # mirror_fn(actor_obs, actions) -> (mirrored_actor_obs, mirrored_actions), same transform as
        # the replay symmetry augmentation, acting on the observations as they reach the actor.
        self.symmetry_loss_weight = symmetry_loss_weight
        self.mirror_fn = mirror_fn
        if self.symmetry_loss_weight > 0 and self.mirror_fn is None:
            raise ValueError(
                "symmetry_loss_weight > 0 requires a mirror function (symmetry_fn), but none is available. "
                "Set train.symmetry_augmentation: True and use an env that provides mirror_obs/mirror_action, "
                "or set train.symmetry_loss_weight: 0.")

        self.model = model
        self.variables = [param for param in self.model.actor.parameters() if param.requires_grad]
        self.optimizer = self.optimizer(self.variables)

    def save_train_state(self, path):
        # Save the model and the dual variables
        # Also save optimizers
        path = path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'variables': [v.detach() for v in self.variables],
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"Saved mpo model to {path}")
    
    def load_train_state(self, path):
        # Load the model and the dual variables
        # Also load optimizers
        path = path + '.pt'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        for v, state in zip(self.variables, checkpoint['variables']):
            v.data.copy_(state)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded mpo model from {path}")
        
    def __call__(self, observations, critic_observations=None, steps=0, next_observations=None, resets=None):
        critic_1_variables = [param for param in self.model.critic_1.parameters() if param.requires_grad]
        critic_2_variables = [param for param in self.model.critic_2.parameters() if param.requires_grad]
        critic_variables = critic_1_variables + critic_2_variables

        for var in critic_variables:
            var.requires_grad = False

        if self.recurrent_model:
            observations = observations.reshape(observations.shape[0], self.seq_length, -1)
            observations = observations.transpose(0, 1)
            
        self.optimizer.zero_grad()
        use_caps = self.caps_temporal_weight > 0 or self.caps_spatial_weight > 0
        caps_temporal_on = use_caps and not self.recurrent_model and self.caps_temporal_weight > 0 and next_observations is not None
        caps_spatial_on = use_caps and not self.recurrent_model and self.caps_spatial_weight > 0
        next_mu = noisy_mu = None
        if caps_spatial_on:
            # The actor normalizes its own input, so scale the noise by the normalizer std
            # to get sigma in the normalized space the network consumes.
            noise = torch.randn_like(observations) * self.caps_spatial_sigma
            normalizer = getattr(self.model.actor.torso, 'observation_normalizer', None)
            if normalizer is not None:
                noise = noise * normalizer._std
        if caps_temporal_on or caps_spatial_on:
            # One actor forward over [s; s'; s + eps] instead of three. The MLP acts row-wise, so
            # every row gets exactly the values (and gradients) the separate forwards gave.
            parts = [observations]
            if caps_temporal_on:
                parts.append(next_observations)
            if caps_spatial_on:
                parts.append(observations + noise)
            distributions, means = _batched_actor_forward(self.model.actor, parts)
            k = 1
            if caps_temporal_on:
                next_mu, k = means[k], k + 1
            if caps_spatial_on:
                noisy_mu = means[k]
        else:
            distributions = self.model.actor(observations)
        if hasattr(distributions, 'rsample_with_log_prob'):
            actions, log_probs = distributions.rsample_with_log_prob()
        else:
            actions = distributions.rsample()
            log_probs = distributions.log_prob(actions)
        log_probs = log_probs.sum(dim=-1)
        
        # Use critic_observations if provided (for privileged critic)
        critic_obs = critic_observations if critic_observations is not None else observations
        values_1 = self.model.critic_1(critic_obs, actions)
        values_2 = self.model.critic_2(critic_obs, actions)
        values = torch.min(values_1, values_2)
        loss = (self.entropy_coeff.value * log_probs - values).mean()

        caps_temporal = torch.zeros((), device=observations.device)
        caps_spatial = torch.zeros((), device=observations.device)
        if use_caps and not self.recurrent_model:
            # Deterministic (squashed) action mean; gradients flow through every term.
            mu = distributions.mean
            if caps_temporal_on:
                sq = torch.square(mu - next_mu).sum(dim=-1)
                if resets is not None:
                    # DirectRLEnv auto-resets, so the next obs of a reset row is an unrelated state
                    keep = 1.0 - resets
                    caps_temporal = (keep * sq).sum() / keep.sum().clamp(min=1.0)
                else:
                    caps_temporal = sq.mean()
                loss = loss + self.caps_temporal_weight * caps_temporal
            if caps_spatial_on:
                caps_spatial = torch.square(mu - noisy_mu).sum(dim=-1).mean()
                loss = loss + self.caps_spatial_weight * caps_spatial
        # Recurrent actors take sequence-shaped inputs, so CAPS is skipped for them.

        symmetry_loss = torch.zeros((), device=observations.device)
        if self.symmetry_loss_weight > 0 and not self.recurrent_model:
            # L_sym = mean_batch(sum_a (mu(M_s s) - M_a mu(s))^2) on the deterministic tanh mean;
            # gradients flow through both branches.
            mu = distributions.mean
            mirrored_obs, mirrored_mu = self.mirror_fn(observations, mu)
            mu_of_mirror = self.model.actor(mirrored_obs).mean
            symmetry_loss = torch.square(mu_of_mirror - mirrored_mu).sum(dim=-1).mean()
            loss = loss + self.symmetry_loss_weight * symmetry_loss

        loss.backward()
        if self.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.variables, float('inf'))
        self.optimizer.step(loss.detach(), steps=steps)   # no .item(): only a plateau scheduler reads it

        for var in critic_variables:
            var.requires_grad = True

        return dict(
            loss=loss.detach(),
            grad_norm=grad_norm.detach(),
            log_prob=log_probs.mean().detach(),
            q=values.mean().detach(),
            caps_temporal=caps_temporal.detach(),
            caps_spatial=caps_spatial.detach(),
            symmetry_loss=symmetry_loss.detach(),
        )

class TwinCriticSoftQLearning:
    def __init__(
        self, model, loss=None, optimizer=None, entropy_coeff=DecayingEntropyCoeff(), gradient_clip=0, device=torch.device("cpu"), recurrent_model = False, seq_length=1
    ):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4))
        self.entropy_coeff = entropy_coeff
        self.gradient_clip = gradient_clip
        self.device = device
        self.recurrent_model = recurrent_model
        self.seq_length = seq_length
        self.model = model
        variables_1 = [param for param in self.model.critic_1.parameters() if param.requires_grad]
        variables_2 = [param for param in self.model.critic_2.parameters() if param.requires_grad]
        self.variables = variables_1 + variables_2
        self.optimizer = self.optimizer(self.variables)

    def save_train_state(self, path):
        # Save the model and the optimizer
        path = path + '.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'variables': [v.detach() for v in self.variables],
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"Saved mpo model to {path}")
        
    def load_train_state(self, path):
        # Load the model and the optimizer
        path = path + '.pt'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if 'variables' in checkpoint:
            for v, state in zip(self.variables, checkpoint['variables']):
                v.data.copy_(state)
        else:
            print("Warning: No variables found in the checkpoint. Loading model only.")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded mpo model from {path}")
        
    def __call__(
        self, observations, actions, next_observations, rewards, discounts,
        next_actor_observations=None, steps=0
    ):
        with torch.no_grad():
            if self.recurrent_model:
                observations = observations.reshape(observations.shape[0], self.seq_length, -1)
                observations = observations.transpose(0, 1)
                next_observations = next_observations.reshape(next_observations.shape[0], self.seq_length, -1)
                next_observations = next_observations.transpose(0, 1)
            
            # If actor and critic have different observation spaces, 
            # use next_actor_observations for the target actor.
            actor_obs = next_actor_observations if next_actor_observations is not None else next_observations
            next_distributions = self.model.actor(actor_obs)
            if hasattr(next_distributions, 'rsample_with_log_prob'):
                outs = next_distributions.rsample_with_log_prob()
                next_actions, next_log_probs = outs
            else:
                next_actions = next_distributions.rsample()
                next_log_probs = next_distributions.log_prob(next_actions)
            next_log_probs = next_log_probs.sum(dim=-1)
            next_values_1 = self.model.target_critic_1(
                next_observations, next_actions)
            next_values_2 = self.model.target_critic_2(
                next_observations, next_actions)
            next_values = torch.min(next_values_1, next_values_2)
            rewards_copy = rewards.clone()
            returns = rewards_copy + discounts * (
                next_values - self.entropy_coeff.value * next_log_probs)

        # Update temperature outside no_grad so alpha_loss retains its grad_fn
        self.entropy_coeff.update(next_log_probs)

        self.optimizer.zero_grad()
        values_1 = self.model.critic_1(observations, actions)
        values_2 = self.model.critic_2(observations, actions)
        loss_1 = self.loss(values_1, returns)
        loss_2 = self.loss(values_2, returns)
        loss = loss_1 + loss_2

        loss.backward()
        if self.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.variables, float('inf'))
        self.optimizer.step(loss.detach(), steps=steps)   # no .item(): only a plateau scheduler reads it

        return dict(
            loss=loss.detach(),
            grad_norm=grad_norm.detach(),
            alpha=torch.as_tensor(self.entropy_coeff.value).detach(),
            mean_q=values_1.mean().detach(),
            max_abs_q=values_1.abs().max().detach(),
            mean_target=returns.mean().detach(),
            max_abs_target=returns.abs().max().detach(),
            mean_log_prob=next_log_probs.mean().detach(),
        )
       
class SAC(ddpg.DDPG):
    '''Soft Actor-Critic.
    SAC: https://arxiv.org/pdf/1801.01290.pdf
    '''

    def __init__(
        self, action_space, model, max_seq_length=1, num_workers=1, seed=None, replay=None, exploration=None, actor_updater=None,
        critic_updater=None, recurrent_model=False, actor_optimizer=None, critic_optimizer=None, device=torch.device("cpu"), config=None,
        symmetry_fn=None,
    ):
        model = model
        action_dim = action_space.shape[0]
        self.entropy_coeff = AutoEntropyCoeff(action_dim=action_dim, device=device)
        exploration = NoActionNoise(policy=self._policy, action_space=action_space, seed=seed, device=device) if exploration is None else exploration
        train_cfg = getattr(config, 'train', None)
        caps_kwargs = dict(
            caps_temporal_weight=getattr(train_cfg, 'caps_temporal_weight', 0.0),
            caps_spatial_weight=getattr(train_cfg, 'caps_spatial_weight', 0.0),
            caps_spatial_sigma=getattr(train_cfg, 'caps_spatial_sigma', 0.05),
        )
        # Explicit mirror loss: reuse the augmentation's symmetry_fn on the actor observations only.
        symmetry_loss_weight = getattr(train_cfg, 'symmetry_loss_weight', 0.0)
        mirror_fn = None
        if symmetry_fn is not None:
            from gymnasium import spaces
            _dict_obs = isinstance(model.obs_space, spaces.Dict)
            def mirror_fn(actor_obs, actions):
                # symmetry_fn mirrors obs dicts; for dict-obs models pass/take only the actor part
                mirrored, mirrored_actions = symmetry_fn({'actor': actor_obs} if _dict_obs else actor_obs, actions)
                return (mirrored['actor'] if _dict_obs else mirrored), mirrored_actions
        caps_kwargs.update(symmetry_loss_weight=symmetry_loss_weight, mirror_fn=mirror_fn)
        actor_updater = TwinCriticSoftDeterministicPolicyGradient(model=model, action_space=action_space, device=device, optimizer=actor_optimizer, entropy_coeff=self.entropy_coeff, gradient_clip=0, recurrent_model=recurrent_model, seq_length=max_seq_length, **caps_kwargs) if actor_updater is None else actor_updater
        critic_updater = TwinCriticSoftQLearning(model=model, device=device, optimizer=critic_optimizer, entropy_coeff=self.entropy_coeff, gradient_clip=1.0, recurrent_model=recurrent_model, seq_length=max_seq_length) if critic_updater is None else critic_updater
        
        from gymnasium import spaces
        self.is_dict_obs = isinstance(model.obs_space, spaces.Dict)
        
        super().__init__(action_space=action_space, model=model, recurrent_model=recurrent_model, max_seq_length=max_seq_length, num_workers=num_workers, seed=seed, replay=replay, exploration=exploration,
            actor_updater=actor_updater, critic_updater=critic_updater, device=device, config=config, symmetry_fn=symmetry_fn)

    def _ensure_actor_tensor(self, observations):
        if self.is_dict_obs and isinstance(observations, dict):
            observations = observations["actor"]
        return to_tensor(observations, self.device)

    def _stochastic_actions(self, observations):
        observations = self._ensure_actor_tensor(observations)
        if self.recurrent_model:
            observations = observations.reshape(observations.shape[0], self.seq_length, -1)
            observations = observations.transpose(0, 1)
        with torch.no_grad():
            return self.model.actor(observations).sample()

    def _policy(self, observations):
        # Return tensors (stay on device)
        return self._stochastic_actions(observations)

    def _greedy_actions(self, observations):
        observations = self._ensure_actor_tensor(observations)
        if self.recurrent_model:
            observations = observations.reshape(observations.shape[0], self.seq_length, -1)
            observations = observations.transpose(0, 1)
        with torch.no_grad():
            return self.model.actor(observations).loc