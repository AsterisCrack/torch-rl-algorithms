import types
import pytest
import torch
import numpy as np
from gymnasium import spaces

from algorithms.sac.sac import TwinCriticSoftDeterministicPolicyGradient, SAC
from algorithms.utils import DecayingEntropyCoeff, SquashedMultivariateNormalDiag

from test_caps import FakeModel, StepsAdam, OBS, ACT

# Toy left/right mirror: permutation + sign flips, both involutions (M(M(x)) = x).
OBS_PERM = torch.tensor([1, 0, 2, 4, 3, 5])
OBS_SIGNS = torch.tensor([1., 1., -1., 1., 1., -1.])
ACT_PERM = torch.tensor([1, 0, 2])
ACT_SIGNS = torch.tensor([1., 1., -1.])


def mirror_obs(o):
    return o[:, OBS_PERM] * OBS_SIGNS


def mirror_action(a):
    return a[:, ACT_PERM] * ACT_SIGNS


class SymmetricActor(torch.nn.Module):
    """mu_sym(s) = 0.5 (f(s) + M_a f(M_s s)) in pre-tanh space, so mu_sym(M_s s) = M_a mu_sym(s) exactly."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        d, dm = self.base(x), self.base(mirror_obs(x))
        loc = 0.5 * (d._distribution.mean + mirror_action(dm._distribution.mean))
        scale = 0.5 * (d._distribution.stddev + mirror_action(dm._distribution.stddev).abs())
        return SquashedMultivariateNormalDiag(loc, scale)


class CountingMirror:
    def __init__(self):
        self.calls = 0

    def __call__(self, obs, actions):
        self.calls += 1
        return mirror_obs(obs), mirror_action(actions)


def make_updater(symmetric_actor=False, mirror_fn="default", **kw):
    torch.manual_seed(0)
    model = FakeModel()
    if symmetric_actor:
        model.actor = SymmetricActor(model.actor)
    if mirror_fn == "default":
        mirror_fn = CountingMirror()
    return TwinCriticSoftDeterministicPolicyGradient(
        model=model, action_space=None, entropy_coeff=DecayingEntropyCoeff(),
        optimizer=lambda params: StepsAdam(params, lr=1e-2), mirror_fn=mirror_fn, **kw)


def run(updater, obs, seed=1):
    torch.manual_seed(seed)
    return updater(obs)


def sym_error(updater, obs):
    with torch.no_grad():
        mu = updater.model.actor(obs).mean
        return (updater.model.actor(mirror_obs(obs)).mean - mirror_action(mu)).square().sum(-1).mean()


@pytest.fixture
def obs():
    return torch.randn(64, OBS, generator=torch.Generator().manual_seed(3))


def test_weight_zero_matches_plain_and_never_calls_mirror(obs):
    plain = run(make_updater(mirror_fn=None), obs)
    upd = make_updater(symmetry_loss_weight=0.0)
    info = run(upd, obs)
    assert upd.mirror_fn.calls == 0
    assert torch.equal(info['loss'], plain['loss'])
    assert info['symmetry_loss'] == 0 and plain['symmetry_loss'] == 0


def test_symmetric_actor_has_zero_loss(obs):
    upd = make_updater(symmetric_actor=True, symmetry_loss_weight=2.0)
    base = run(make_updater(symmetric_actor=True, mirror_fn=None), obs)
    info = run(upd, obs)
    assert upd.mirror_fn.calls == 1
    assert info['symmetry_loss'].abs() < 1e-10
    assert torch.allclose(info['loss'], base['loss'], atol=1e-6)


def test_asymmetric_actor_positive_and_step_reduces_it(obs):
    upd = make_updater(symmetry_loss_weight=100.0)
    before = sym_error(upd, obs)
    info = run(upd, obs)
    assert info['symmetry_loss'] > 0
    assert torch.allclose(info['symmetry_loss'], before, rtol=1e-4)  # logged value is the pre-step loss
    plain = run(make_updater(mirror_fn=None), obs)
    assert torch.allclose(info['loss'], plain['loss'] + 100.0 * before, rtol=1e-4)
    assert sym_error(upd, obs) < before
    for _ in range(30):
        run(upd, obs)
    assert sym_error(upd, obs) < 0.1 * before


def test_gradient_flows_through_both_branches(obs):
    # Gradient of the sym loss w.r.t. actor params must differ when either branch is detached.
    def grads(detach):
        upd = make_updater()
        upd.model.actor.zero_grad()
        mu = upd.model.actor(obs).mean
        mu_m = upd.model.actor(mirror_obs(obs)).mean
        target = mirror_action(mu)
        if detach == 'target':
            target = target.detach()
        if detach == 'mirror':
            mu_m = mu_m.detach()
        (mu_m - target).square().sum(-1).mean().backward()
        return torch.cat([p.grad.flatten() for p in upd.model.actor.parameters() if p.grad is not None])
    full, no_t, no_m = grads(None), grads('target'), grads('mirror')
    assert not torch.allclose(full, no_t) and not torch.allclose(full, no_m)
    assert torch.allclose(full, no_t + no_m, atol=1e-6)


def test_recurrent_skips_symmetry(obs):
    # seq_length=1: the (1, B, OBS) sequence layout still runs through the toy MLP actor
    upd = make_updater(symmetry_loss_weight=2.0, recurrent_model=True, seq_length=1)
    info = run(upd, obs)
    assert upd.mirror_fn.calls == 0 and info['symmetry_loss'] == 0


def test_error_when_weight_positive_without_mirror_fn():
    with pytest.raises(ValueError, match="symmetry_loss_weight"):
        make_updater(mirror_fn=None, symmetry_loss_weight=1.0)


def _sac(weight, symmetry_fn, dict_obs=False):
    model = FakeModel()
    obs_space = spaces.Box(-np.inf, np.inf, (OBS,))
    model.obs_space = spaces.Dict({'actor': obs_space, 'critic': obs_space}) if dict_obs else obs_space
    model.observation_normalizer = None
    model.return_normalizer = None
    cfg = types.SimpleNamespace(train=types.SimpleNamespace(symmetry_loss_weight=weight))
    return SAC(action_space=spaces.Box(-1, 1, (ACT,)), model=model, replay=object(), exploration=types.SimpleNamespace(),
               config=cfg, symmetry_fn=symmetry_fn,
               actor_optimizer=lambda params: StepsAdam(params, lr=1e-2))


def test_sac_construction_error_and_dict_wrapper(obs):
    with pytest.raises(ValueError, match="symmetry_loss_weight"):
        _sac(1.0, None)
    # dict-obs model: the wrapper passes {'actor': obs} to symmetry_fn and returns only the actor part
    seen = []

    def symmetry_fn(obs_dict, actions):
        seen.append(set(obs_dict))
        return {k: mirror_obs(v) for k, v in obs_dict.items()}, mirror_action(actions)

    agent = _sac(2.0, symmetry_fn, dict_obs=True)
    info = agent.actor_updater(obs)
    assert seen == [{'actor'}] and info['symmetry_loss'] > 0
    # plain-obs model: tensors passed straight through
    agent = _sac(2.0, lambda o, a: (mirror_obs(o), mirror_action(a)))
    assert agent.actor_updater(obs)['symmetry_loss'] > 0
    # weight 0 -> no symmetry term even with symmetry_fn available
    assert _sac(0.0, symmetry_fn, dict_obs=True).actor_updater(obs)['symmetry_loss'] == 0
