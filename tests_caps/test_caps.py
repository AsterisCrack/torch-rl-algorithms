import copy
import pytest
import torch
from gymnasium import spaces
import numpy as np

from algorithms.sac.sac import TwinCriticSoftDeterministicPolicyGradient
from algorithms.utils import DecayingEntropyCoeff, MeanStd
from models.mlp import MLPActor

OBS, ACT = 6, 3


class FakeCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(OBS + ACT, 1)

    def forward(self, obs, actions):
        return self.net(torch.cat([obs, actions], dim=-1)).squeeze(-1)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        obs_space = spaces.Box(-np.inf, np.inf, (OBS,))
        act_space = spaces.Box(-1, 1, (ACT,))
        normalizer = MeanStd(shape=obs_space.shape)
        self.actor = MLPActor(obs_space, act_space, [16, 16], normalizer, "gaussian_multivariate")
        self.critic_1 = FakeCritic()
        self.critic_2 = FakeCritic()


class StepsAdam(torch.optim.Adam):
    # The library's optimizers accept (loss, steps=...) in step()
    def step(self, loss=None, steps=0):
        return super().step()


def make_updater(**caps):
    torch.manual_seed(0)
    model = FakeModel()
    return TwinCriticSoftDeterministicPolicyGradient(
        model=model, action_space=None, entropy_coeff=DecayingEntropyCoeff(),
        optimizer=lambda params: StepsAdam(params, lr=1e-2), **caps)


def run(updater, obs, next_obs=None, seed=1):
    torch.manual_seed(seed)
    return updater(obs, next_observations=next_obs)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(3)
    obs = torch.randn(32, OBS, generator=g)
    next_obs = obs + torch.randn(32, OBS, generator=g)
    return obs, next_obs


def test_weights_zero_matches_plain(batch):
    obs, next_obs = batch
    a = run(make_updater(), obs)
    b = run(make_updater(), obs, next_obs)
    assert torch.allclose(a['loss'], b['loss'])
    assert a['caps_temporal'] == 0 and b['caps_spatial'] == 0


def test_temporal_term(batch):
    obs, next_obs = batch
    base = run(make_updater(), obs, next_obs)
    upd = make_updater(caps_temporal_weight=1.0)
    before = [p.detach().clone() for p in upd.variables]
    info = run(upd, obs, next_obs)
    assert info['caps_temporal'] > 0
    assert info['loss'] > base['loss']
    assert info['caps_spatial'] == 0
    assert any(not torch.equal(b, p) for b, p in zip(before, upd.variables))
    # No next observations: temporal term is skipped
    assert run(make_updater(caps_temporal_weight=1.0), obs)['caps_temporal'] == 0


def test_temporal_gradient_contribution(batch):
    obs, next_obs = batch
    # Parameters after one step differ between the two settings
    u0, u1 = make_updater(), make_updater(caps_temporal_weight=5.0)
    run(u0, obs, next_obs)
    run(u1, obs, next_obs)
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(u0.variables, u1.variables))


def test_spatial_term(batch):
    obs, _ = batch
    info = run(make_updater(caps_spatial_weight=1.0, caps_spatial_sigma=0.5), obs)
    assert info['caps_spatial'] > 0
    zero = run(make_updater(caps_spatial_weight=1.0, caps_spatial_sigma=0.0), obs)
    assert zero['caps_spatial'] == 0


def test_replay_caps_keys_are_one_step_and_reset_masked():
    from algorithms.utils import Buffer
    T, W = 12, 2
    buf = Buffer(size=100, return_steps=5, batch_size=8, steps_before_batches=0, seed=0)
    # obs_t = [t, t, ...]; a reset happens after step 6 on worker 0
    resets_at = {6: [1.0, 0.0]}
    for t in range(T):
        obs = torch.full((W, OBS), float(t))
        next_obs = torch.full((W, OBS), float(t + 1))
        resets = torch.tensor(resets_at.get(t, [0.0, 0.0]))
        buf.store(observations=obs, actions=torch.zeros(W, ACT), next_observations=next_obs,
                  rewards=torch.ones(W), resets=resets, terminations=torch.zeros(W),
                  caps_next_actor=next_obs, caps_reset=resets)
    n = buf.size
    rows = torch.arange(n)
    for w in range(W):
        cur = buf.buffers['observations'][rows, w][:, 0]
        caps_next = buf.buffers['caps_next_actor'][rows, w][:, 0]
        n_step_next = buf.buffers['next_observations'][rows, w][:, 0]
        assert torch.equal(caps_next, cur + 1)  # true 1-step next obs
        assert (n_step_next[:-1] > cur[:-1] + 1).any()  # n-step key was overwritten as expected
    assert buf.buffers['caps_reset'][6, 0] == 1 and buf.buffers['caps_reset'][6, 1] == 0
    batch = next(buf.get('caps_next_actor', 'caps_reset', steps=0))
    assert set(batch) == {'caps_next_actor', 'caps_reset'}


def test_temporal_term_masks_resets(batch):
    obs, next_obs = batch
    resets = torch.zeros(obs.shape[0])
    resets[:16] = 1.0
    upd = make_updater(caps_temporal_weight=1.0)
    torch.manual_seed(1)
    info = upd(obs, next_observations=next_obs, resets=resets)
    # Reference: mean of squared diff over the 16 non-reset rows only
    ref = make_updater(caps_temporal_weight=1.0)
    with torch.no_grad():
        mu = ref.model.actor(obs).mean
        nmu = ref.model.actor(next_obs).mean
        sq = (mu - nmu).square().sum(-1)
    assert torch.allclose(info['caps_temporal'], sq[16:].mean(), atol=1e-5)
    # All rows reset: term is 0 without NaN
    allr = run_with_resets(make_updater(caps_temporal_weight=1.0), obs, next_obs, torch.ones(obs.shape[0]))
    assert allr['caps_temporal'] == 0


def run_with_resets(updater, obs, next_obs, resets):
    torch.manual_seed(1)
    return updater(obs, next_observations=next_obs, resets=resets)
