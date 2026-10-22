"""Exactly mirror-equivariant actor wrapper (E16).

Learning a symmetric policy through replay augmentation and a mirror loss only makes the actor
*approximately* equivariant (measured |pi(Ms) - M pi(s)| ~ 22% of |a|), and the residual breaks the
symmetry visibly (the robot limps on the same leg from mirrored starts). This wrapper makes it exact
by projection onto the equivariant subspace, in the pre-squash (pre-tanh) space:

    (loc1, scale1) = f(s)            (loc2, scale2) = f(M_s s)
    loc   = 1/2 (loc1   + M_a loc2)         scale = 1/2 (scale1 + P_a scale2)

M_s / M_a are the observation / action mirrors (index permutation * sign, both involutions) and P_a is
the action permutation WITHOUT signs, because a scale is >= 0 and must not be negated. Then
loc(M_s s) = M_a loc(s) exactly, and since tanh is odd and elementwise tanh(loc) is exactly
equivariant too; the stochastic policy is equivariant in distribution (scale(M_s s) = P_a scale(s)).
The result is again a SquashedMultivariateNormalDiag, so rsample / log_prob / entropy bookkeeping in the
SAC updaters is unchanged and uses the symmetrised loc/scale.

The mirror tables are registered BUFFERS (persistent), so they travel with the checkpoint and are
available at export / sim2real time without the environment. The wrapped network is `self.net`; checkpoint
keys are therefore `actor.net.*` + `actor.mirror_*` (the plain actor stores `actor.*`).

Input convention: the actor input exactly as the actor receives it (stage-1 normalised actor obs when
the env wrapper normalises). The inner actor's own stage-2 MeanStd is part of `net`; the construction is
exact whatever its statistics are (the symmetry follows from f(M s) and f(M M s) = f(s)); they only need
to be symmetric for the learned f to be well-conditioned on mirrored inputs.
"""
import torch
import torch.nn as nn

try:
    from algorithms.utils import SquashedMultivariateNormalDiag
except ImportError:
    from torch_rl_algorithms.algorithms.utils import SquashedMultivariateNormalDiag

TABLE_KEYS = ("mirror_obs_perm", "mirror_obs_sign", "mirror_act_perm", "mirror_act_sign")


def check_involution(perm, sign, name="mirror"):
    """Raise ValueError unless (perm, sign) describes an involution M x = sign * x[perm], M M = I."""
    perm = torch.as_tensor(perm).long().flatten()
    sign = torch.as_tensor(sign).float().flatten()
    n = perm.numel()
    if sign.numel() != n:
        raise ValueError(f"{name}: perm has {n} entries but sign has {sign.numel()}")
    if not torch.equal(perm.sort().values, torch.arange(n)):
        raise ValueError(f"{name}: perm is not a permutation of 0..{n - 1}")
    if not torch.equal(perm[perm], torch.arange(n)):
        raise ValueError(f"{name}: perm is not an involution (perm[perm[i]] != i)")
    if not torch.all(sign.abs() == 1):
        raise ValueError(f"{name}: sign entries must be +-1")
    if not torch.equal(sign * sign[perm], torch.ones(n)):
        raise ValueError(f"{name}: sign[i] * sign[perm[i]] != 1, so M is not an involution")


def find_actor_mirror_tables(env):
    """(obs_perm, obs_sign, act_perm, act_sign) of the actor layout from the training env, or None.

    Looks for BipedEnv.actor_mirror_tables() on the env, its `.env` chain (RL wrapper -> gym wrappers)
    and their `.unwrapped`, since gymnasium wrappers such as RecordVideo do not forward custom methods.
    """
    seen, stack = set(), [env]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        fn = getattr(e, "actor_mirror_tables", None)
        if callable(fn):
            return fn()
        stack.append(getattr(e, "env", None))
        stack.append(getattr(e, "unwrapped", None))
    return None


class EquivariantActor(nn.Module):
    """Projects a Gaussian-squashed actor onto its mirror-equivariant subspace (see module docstring)."""

    def __init__(self, net, obs_perm=None, obs_sign=None, act_perm=None, act_sign=None, obs_dim=None, act_dim=None):
        """Give all four tables, or none (then obs_dim/act_dim are required): a placeholder whose real
        tables must come from a checkpoint (`load_state_dict`) before the first forward."""
        super().__init__()
        self.net = net
        given = [t is not None for t in (obs_perm, obs_sign, act_perm, act_sign)]
        if any(given) and not all(given):
            raise ValueError("EquivariantActor needs all four mirror tables or none")
        self._ready = all(given)
        if self._ready:
            obs_perm, obs_sign = torch.as_tensor(obs_perm).long().flatten().cpu(), torch.as_tensor(obs_sign).float().flatten().cpu()
            act_perm, act_sign = torch.as_tensor(act_perm).long().flatten().cpu(), torch.as_tensor(act_sign).float().flatten().cpu()
            check_involution(obs_perm, obs_sign, "observation mirror")
            check_involution(act_perm, act_sign, "action mirror")
        else:
            if obs_dim is None or act_dim is None:
                raise ValueError("placeholder EquivariantActor needs obs_dim and act_dim")
            obs_perm, obs_sign = torch.arange(obs_dim), torch.ones(obs_dim)
            act_perm, act_sign = torch.arange(act_dim), torch.ones(act_dim)
        self.register_buffer("mirror_obs_perm", obs_perm.clone())
        self.register_buffer("mirror_obs_sign", obs_sign.clone())
        self.register_buffer("mirror_act_perm", act_perm.clone())
        self.register_buffer("mirror_act_sign", act_sign.clone())

    # -- plain-actor attribute passthrough (torso, head_type, action_size, observation_size, ...) --------
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            net = self.__dict__.get("_modules", {}).get("net")
            if net is None or name == "net":
                raise
            return getattr(net, name)

    # -- mirror transforms (also the ONNX-exported path) ----------------------------------------------
    def mirror_obs(self, obs):
        return obs.index_select(-1, self.mirror_obs_perm) * self.mirror_obs_sign

    def mirror_action(self, act):
        return act.index_select(-1, self.mirror_act_perm) * self.mirror_act_sign

    def _check_ready(self):
        if not self._ready:
            raise RuntimeError(
                "EquivariantActor has placeholder (identity) mirror tables: load a checkpoint that contains "
                "them (state_dict keys 'actor.mirror_*') or build the actor with an env that provides "
                "actor_mirror_tables().")

    def loc_scale(self, obs):
        """Symmetrised pre-squash (loc, scale)."""
        self._check_ready()
        d1 = self.net(obs)._distribution
        d2 = self.net(self.mirror_obs(obs))._distribution
        loc = 0.5 * (d1.loc + self.mirror_action(d2.loc))
        scale = 0.5 * (d1.scale + d2.scale.index_select(-1, self.mirror_act_perm))
        return loc, scale

    def forward(self, observations):
        loc, scale = self.loc_scale(observations)
        return SquashedMultivariateNormalDiag(loc, scale)

    def mean_action(self, observations):
        """tanh(loc), the deterministic action, without building a distribution (ONNX-friendly)."""
        return torch.tanh(self.loc_scale(observations)[0])

    def get_action(self, observations):
        # Must NOT fall through to net.get_action (that would bypass the symmetrisation).
        return self.forward(observations).mean

    # -- checkpointing ---------------------------------------------------------------------------------
    def validate_tables(self):
        check_involution(self.mirror_obs_perm, self.mirror_obs_sign, "observation mirror")
        check_involution(self.mirror_act_perm, self.mirror_act_sign, "action mirror")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if all(prefix + k in state_dict for k in TABLE_KEYS):
            try:
                self.validate_tables()
            except ValueError as e:
                error_msgs.append(f"{prefix}mirror tables in the checkpoint are invalid: {e}")
            else:
                self._ready = True
