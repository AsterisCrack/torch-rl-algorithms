"""The 2026-10-22 speed changes must not change the maths: batched actor forwards, fused target update,
GPU-resident alpha and the buffered TensorBoard logger are checked against the old formulas."""
import numpy as np
import torch
from gymnasium import spaces

from algorithms.sac.sac import _batched_actor_forward
from algorithms.utils import AutoEntropyCoeff, MeanStd, SquashedMultivariateNormalDiag, _ScalarLogBuffer
from models.equivariant import EquivariantActor
from models.mlp import MLPActor
from models.networks import ActorTwinCriticWithTargets

OBS, ACT = 8, 4
# mirror: swap the two halves of obs and actions, negate the first entry of each half
OBS_PERM = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])
OBS_SIGN = torch.tensor([-1.0, 1, 1, 1, -1, 1, 1, 1])
ACT_PERM = torch.tensor([2, 3, 0, 1])
ACT_SIGN = torch.tensor([-1.0, 1, -1, 1])


def _plain_actor(seed=0):
    torch.manual_seed(seed)
    norm = MeanStd(shape=(OBS,))
    norm._update(torch.randn(OBS) * 0.3, torch.rand(OBS) + 0.5)
    return MLPActor(spaces.Box(-np.inf, np.inf, (OBS,)), spaces.Box(-1, 1, (ACT,)), [32, 32], norm,
                    "gaussian_multivariate")


def _equiv_actor(seed=0):
    return EquivariantActor(_plain_actor(seed), OBS_PERM, OBS_SIGN, ACT_PERM, ACT_SIGN)


def _obs(n, seed=1):
    return torch.randn(n, OBS, generator=torch.Generator().manual_seed(seed))


def test_equivariant_batched_forward_equals_two_forwards():
    actor, s = _equiv_actor(), _obs(64)
    loc, scale = actor.loc_scale(s)
    d1 = actor.net(s)._distribution
    d2 = actor.net(actor.mirror_obs(s))._distribution
    loc_ref = 0.5 * (d1.loc + actor.mirror_action(d2.loc))
    scale_ref = 0.5 * (d1.scale + d2.scale.index_select(-1, actor.mirror_act_perm))
    assert torch.allclose(loc, loc_ref, atol=1e-6)
    assert torch.allclose(scale, scale_ref, atol=1e-6)


def _grads(actor):
    return [p.grad.clone() for p in actor.parameters() if p.grad is not None]


def test_batched_actor_forward_matches_separate_forwards_values_and_grads():
    for make in (_plain_actor, _equiv_actor):
        a_sep, a_bat = make(), make()
        parts = [_obs(16, 1), _obs(16, 2), _obs(16, 3)]
        # separate (old path)
        d0 = a_sep(parts[0])
        means_sep = [d0.mean] + [a_sep(p).mean for p in parts[1:]]
        torch.manual_seed(5)
        act_sep, lp_sep = d0.rsample_with_log_prob()
        loss_sep = lp_sep.sum() + act_sep.sum() + sum((m ** 2).sum() for m in means_sep)
        loss_sep.backward()
        # batched (new path)
        d0b, means_bat = _batched_actor_forward(a_bat, parts)
        assert isinstance(d0b, SquashedMultivariateNormalDiag)
        torch.manual_seed(5)
        act_bat, lp_bat = d0b.rsample_with_log_prob()
        loss_bat = lp_bat.sum() + act_bat.sum() + sum((m ** 2).sum() for m in means_bat)
        loss_bat.backward()
        for ms, mb in zip(means_sep, means_bat):
            assert torch.allclose(ms, mb, atol=1e-6)
        assert torch.allclose(act_sep, act_bat, atol=1e-6) and torch.allclose(lp_sep, lp_bat, atol=1e-5)
        for gs, gb in zip(_grads(a_sep), _grads(a_bat)):
            assert torch.allclose(gs, gb, atol=1e-5, rtol=1e-4)


def _model(seed=0):
    torch.manual_seed(seed)
    obs_space = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (OBS,)), "critic": spaces.Box(-np.inf, np.inf, (OBS + 2,))})
    return ActorTwinCriticWithTargets(obs_space, spaces.Box(-1, 1, (ACT,)), actor_sizes=[16, 16], critic_sizes=[16, 16],
                                      actor_type="gaussian_multivariate", target_coeff=0.01)


def test_fused_target_update_equals_the_loop():
    fused, ref = _model(), _model()
    for m in (fused, ref):     # make sources differ from the targets (same draws for both models)
        torch.manual_seed(1)
        with torch.no_grad():
            for p in list(m.actor.parameters()) + list(m.critic_1.parameters()) + list(m.critic_2.parameters()):
                p.add_(torch.randn_like(p))
    for _ in range(3):
        fused.update_targets()
        with torch.no_grad():
            for target, source in ref._get_sync_pairs():
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.mul_(1 - ref.target_coeff)
                    tp.data.add_(ref.target_coeff * sp.data)
    for a, b in zip(fused.state_dict().values(), ref.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_auto_entropy_value_is_a_device_tensor_with_the_old_value():
    ec = AutoEntropyCoeff(action_dim=ACT, initial_log_alpha=0.3)
    assert torch.is_tensor(ec.value) and abs(float(ec.value) - float(np.exp(0.3))) < 1e-6
    ec.update(torch.randn(32))
    assert abs(float(ec.value) - float(ec.log_alpha.exp())) < 1e-7
    assert not ec.value.requires_grad


class _Writer:
    def __init__(self):
        self.rows = []

    def add_scalar(self, key, value, step):
        self.rows.append((key, value, step))


def test_scalar_log_buffer_writes_window_means_and_episode_means():
    buf, w = _ScalarLogBuffer(), _Writer()
    buf.add("a", torch.tensor(1.0))
    buf.add("a", torch.tensor(3.0))
    buf.add("b", 2.0)
    buf.add("c", torch.tensor([1.0, 2.0, 3.0]))     # vectors are averaged first
    buf.add_episodes(torch.tensor([10.0, 40.0, 2.0]))   # 2 episodes: score sum 10, length sum 40
    buf.flush(w, 7)
    got = {k: v for k, v, _ in w.rows}
    assert got == {"a": 2.0, "b": 2.0, "c": 2.0, "train/episode_score": 5.0, "train/episode_length": 20.0}
    assert all(s == 7 for _, _, s in w.rows)
    w.rows.clear()
    buf.add_episodes(torch.zeros(3))                  # no finished episode: no episode rows
    buf.flush(w, 8)
    assert w.rows == []
