"""Symmetric observation statistics (E15). CPU only.

Replay symmetry augmentation mirrors the NORMALISED observations with a plain perm*sign. That is
exact only if the stage-1 scaler satisfies  mean_i = s_i * mean_p(i)  and  var_i = var_p(i).
The Isaac wrapper (src/isaaclab/common.py, `symmetric_obs_stats=True`) therefore updates each
RunningMeanStd with the real raw batch AND its raw mirror. `_wrapper_update` below is that exact
operation (the wrapper itself needs Isaac to import; BipedRobot/tests/test_symmetric_obs_stats.py
exercises the real class with stubs).

Also covers the stage-2 MeanStd (DDPG.update records the mirrored obs) and the critic_2
normaliser bug (critic_2 was a deepcopy with a private, never-updated normaliser).

Run from torch_rl_algorithms/:  python -m pytest tests_caps/test_symmetric_stats.py -q
"""
import torch
import numpy as np
from collections import OrderedDict
from gymnasium import spaces

from algorithms.utils import RunningMeanStd, MeanStd
from algorithms.ddpg.ddpg import DDPG
from models.networks import ActorTwinCriticWithTargets

TOL = 1e-6
DIM_A, DIM_C, ACT = 6, 8, 3

# Toy involutive mirrors (s_i == s_p(i)): a self-paired antisymmetric slot (2 -> -1) and swapped pairs.
A_PERM = torch.tensor([1, 0, 2, 4, 3, 5])
A_SIGN = torch.tensor([1., 1., -1., 1., 1., -1.])
C_PERM = torch.tensor([1, 0, 2, 4, 3, 5, 7, 6])
C_SIGN = torch.tensor([1., 1., -1., 1., 1., -1., -1., -1.])
ACT_PERM = torch.tensor([1, 0, 2])
ACT_SIGN = torch.tensor([1., 1., -1.])


def mirror_actor(x):
    return x[:, A_PERM] * A_SIGN


def mirror_critic(x):
    return x[:, C_PERM] * C_SIGN


def mirror_dict(d):
    out = {}
    if "actor" in d:
        out["actor"] = mirror_actor(d["actor"])
    if "critic" in d:
        out["critic"] = mirror_critic(d["critic"])
    return out


def _asym_batches(n_batches, n, dim, seed=0):
    """Random data with per-dim offsets and scales that differ between mirror partners."""
    g = torch.Generator().manual_seed(seed)
    loc = torch.linspace(-2.0, 3.0, dim)
    scale = torch.linspace(0.5, 2.5, dim)
    return [torch.randn(n, dim, generator=g) * scale + loc for _ in range(n_batches)]


def _wrapper_update(rms, x, mirror, symmetric):
    """Exactly what BaseIsaacLabWrapper._process_obs does to a scaler."""
    rms.update(torch.cat([x, mirror(x)], dim=0) if symmetric else x)


def _check_pair_relations(mean, var, perm, sign, tol=TOL):
    assert torch.allclose(mean, sign * mean[perm], atol=tol, rtol=0), (mean, sign * mean[perm])
    assert torch.allclose(var, var[perm], atol=tol, rtol=0), (var, var[perm])


def test_a_stage1_stats_satisfy_pair_relations():
    rms = RunningMeanStd(shape=(DIM_C,))
    for x in _asym_batches(25, 64, DIM_C):
        _wrapper_update(rms, x, mirror_critic, symmetric=True)
    _check_pair_relations(rms.mean, rms.var, C_PERM, C_SIGN)
    # a self-paired antisymmetric slot (s=-1, p(i)=i) must have exactly zero mean
    assert abs(float(rms.mean[2])) < TOL and abs(float(rms.mean[5])) < TOL


def test_b_normalise_then_mirror_equals_mirror_then_normalise():
    rms = RunningMeanStd(shape=(DIM_C,))
    for x in _asym_batches(10, 64, DIM_C, seed=1):
        _wrapper_update(rms, x, mirror_critic, symmetric=True)
    x = _asym_batches(1, 128, DIM_C, seed=2)[0]
    a = mirror_critic(rms.normalize(x))
    b = rms.normalize(mirror_critic(x))
    assert torch.allclose(a, b, atol=1e-5)


def test_c_flag_off_reproduces_old_stats_and_is_not_symmetric():
    batches = _asym_batches(10, 64, DIM_C, seed=3)
    old = RunningMeanStd(shape=(DIM_C,))
    off = RunningMeanStd(shape=(DIM_C,))
    for x in batches:
        old.update(x)                                     # pre-E15 behaviour
        _wrapper_update(off, x, mirror_critic, symmetric=False)
    assert torch.equal(old.mean, off.mean) and torch.equal(old.var, off.var)
    # ... and that old behaviour really violates the pair relations (so the test above is meaningful)
    assert float((old.mean - C_SIGN * old.mean[C_PERM]).abs().max()) > 0.1
    assert float((old.var - old.var[C_PERM]).abs().max()) > 0.1
    # mirrored-normalised vs normalised-mirrored samples are mis-scaled by O(sigma)
    x = batches[0]
    assert float((mirror_critic(old.normalize(x)) - old.normalize(mirror_critic(x))).abs().max()) > 0.3


def test_stage2_meanstd_is_symmetric_when_mirror_recorded():
    """DDPG.update records real + mirrored (normalised) obs: stage-2 mean/std keep the pair relations."""
    obs_space = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (DIM_A,)),
                             "critic": spaces.Box(-np.inf, np.inf, (DIM_C,))})
    act_space = spaces.Box(-1, 1, (ACT,))

    def run(use_symmetry):
        torch.manual_seed(0)
        model = ActorTwinCriticWithTargets(obs_space, act_space, actor_type="gaussian_multivariate")
        agent = DDPG(act_space, model, replay=_StubReplay(), exploration=_StubExploration(),
                     actor_updater=_StubUpdater(), critic_updater=_StubUpdater(),
                     symmetry_fn=(lambda o, a: (mirror_dict(o), a[:, ACT_PERM] * ACT_SIGN)) if use_symmetry else None)
        s1a = RunningMeanStd(shape=(DIM_A,))
        s1c = RunningMeanStd(shape=(DIM_C,))
        for xa, xc in zip(_asym_batches(20, 32, DIM_A, 5), _asym_batches(20, 32, DIM_C, 6)):
            for rms, x, m in ((s1a, xa, mirror_actor), (s1c, xc, mirror_critic)):
                _wrapper_update(rms, x, m, symmetric=True)      # stage 1 (flag on)
            obs = {"actor": s1a.normalize(xa), "critic": s1c.normalize(xc)}
            agent.step(obs, 0)
            agent.update(obs, torch.zeros(32), torch.zeros(32, dtype=torch.bool),
                         torch.zeros(32, dtype=torch.bool), steps=0)
            model.actor_observation_normalizer.update()
            model.critic_observation_normalizer.update()
        return model

    sym = run(True)
    for norm, perm, sign in ((sym.actor_observation_normalizer, A_PERM, A_SIGN),
                             (sym.critic_observation_normalizer, C_PERM, C_SIGN)):
        _check_pair_relations(norm._mean, norm._std ** 2, perm, sign, tol=2e-5)
    # control: without recording the mirror the stage-2 stats are NOT symmetric
    plain = run(False)
    assert float((plain.critic_observation_normalizer._mean
                  - C_SIGN * plain.critic_observation_normalizer._mean[C_PERM]).abs().max()) > 1e-3


def test_d_critic_1_and_2_normalisers_both_update():
    obs_space = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (DIM_A,)),
                             "critic": spaces.Box(-np.inf, np.inf, (DIM_C,))})
    act_space = spaces.Box(-1, 1, (ACT,))
    model = ActorTwinCriticWithTargets(obs_space, act_space, actor_type="gaussian_multivariate")
    n1 = model.critic_1.torso.observation_normalizer
    n2 = model.critic_2.torso.observation_normalizer
    assert n1 is n2 and n1 is model.critic_observation_normalizer       # one shared object
    assert model.target_critic_2.torso.observation_normalizer is not n1  # targets keep a Polyak copy

    for x in _asym_batches(5, 64, DIM_C, 7):
        model.critic_observation_normalizer.record(x)
    model.critic_observation_normalizer.update()
    for crit in (model.critic_1, model.critic_2):
        nm = crit.torso.observation_normalizer
        assert float(nm._std.sub(1).abs().max()) > 0.1 and float(nm._mean.abs().max()) > 0.1
    assert torch.equal(n1._mean, n2._mean) and torch.equal(n1._std, n2._std)

    # both critics and both targets track the stats through the Polyak sync
    for _ in range(2000):
        model.update_targets()
    for t in (model.target_critic_1, model.target_critic_2):
        assert torch.allclose(t.torso.observation_normalizer._mean, n1._mean, atol=1e-3)
        assert torch.allclose(t.torso.observation_normalizer._std, n1._std, atol=1e-3)

    # the two critics use the same input scaling (outputs differ only through their own weights)
    names = [k for k in model.state_dict() if k.endswith("observation_normalizer._mean")]
    assert "critic_2.torso.observation_normalizer._mean" in names


def test_d_legacy_checkpoint_does_not_clobber_critic_normaliser():
    """Old checkpoints hold an identity critic_2 normaliser; loading must not overwrite critic_1's."""
    obs_space = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (DIM_A,)),
                             "critic": spaces.Box(-np.inf, np.inf, (DIM_C,))})
    act_space = spaces.Box(-1, 1, (ACT,))
    src = ActorTwinCriticWithTargets(obs_space, act_space, actor_type="gaussian_multivariate")
    for x in _asym_batches(5, 64, DIM_C, 8):
        src.critic_observation_normalizer.record(x)
    src.critic_observation_normalizer.update()
    src.assign_targets()                                    # hard-sync the target copies
    sd = OrderedDict((k, v.clone()) for k, v in src.state_dict().items())
    for k in sd:                                            # emulate the legacy stale identity copies
        if k.startswith(("critic_2.torso.observation_normalizer", "target_critic_2.torso.observation_normalizer")):
            sd[k] = torch.zeros_like(sd[k]) if k.endswith("_mean") else torch.ones_like(sd[k])
    dst = ActorTwinCriticWithTargets(obs_space, act_space, actor_type="gaussian_multivariate")
    dst.load_state_dict(sd)
    ref = src.critic_1.torso.observation_normalizer
    for crit in (dst.critic_1, dst.critic_2, dst.target_critic_1, dst.target_critic_2):
        nm = crit.torso.observation_normalizer
        assert torch.allclose(nm._mean, ref._mean, atol=1e-3), crit
        assert torch.allclose(nm._std, ref._std, atol=1e-3), crit
    assert dst.critic_2.torso.observation_normalizer is dst.critic_observation_normalizer


class _StubReplay:
    def store(self, **kw):
        self.last = kw

    def ready(self, steps):
        return False


class _StubExploration:
    def __call__(self, observations, steps):
        return torch.zeros(observations["actor"].shape[0], ACT)

    def update(self, resets):
        pass


class _StubUpdater:
    def __call__(self, *a, **k):
        return {}
