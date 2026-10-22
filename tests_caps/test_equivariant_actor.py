"""Exactly mirror-equivariant actor (E16, train.equivariant_actor). CPU only, Isaac is NOT imported.

Uses the REAL V2 mirror tables: BipedEnv._build_mirror_transform / actor_mirror_tables are lifted out of
envs/isaaclab/biped_env.py with ast and run on a stand-in `self` (falls back to a transcription of the same
tables if that file is not reachable from here).

Run from torch_rl_algorithms/:  python -m pytest tests_caps/test_equivariant_actor.py -q
"""
import ast
import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from algorithms.utils import MeanStd, NoConfig, SquashedMultivariateNormalDiag
from models.equivariant import EquivariantActor, check_involution, find_actor_mirror_tables
from models.mlp import MLPActor
from models.networks import ActorTwinCriticWithTargets

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_biped_root():
    """BipedRobot/ (parent of torch_rl_algorithms): $BIPED_ROOT, else walk up from this file / the cwd."""
    cands = [os.environ.get("BIPED_ROOT"), os.path.dirname(os.path.dirname(HERE))]
    for start in (os.getcwd(), HERE):
        d = start
        for _ in range(4):
            cands.append(d)
            d = os.path.dirname(d)
    for c in cands:
        if c and os.path.exists(os.path.join(c, "envs", "isaaclab", "biped_env.py")):
            return c
    return os.path.dirname(os.path.dirname(HERE))


BIPED_ROOT = _find_biped_root()
BIPED_ENV = os.path.join(BIPED_ROOT, "envs", "isaaclab", "biped_env.py")
EXPORT_ONNX = os.path.join(BIPED_ROOT, "src", "isaaclab", "export_onnx.py")

PROPRIO, HIST, ACT = 50, 5, 12
OBS = PROPRIO * HIST
V2_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
V2_SIGNS = [1.0] * 12


def _real_tables():
    """(obs_perm, obs_sign, act_perm, act_sign) over the 250-dim actor obs, from the real BipedEnv code."""
    if os.path.exists(BIPED_ENV):
        src = open(BIPED_ENV, encoding="utf-8").read()
        names = {"_build_mirror_transform", "actor_mirror_tables"}
        fns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {f.name for f in fns} == names
        ns = {"torch": torch}
        exec(compile(ast.Module(fns, []), "biped_env_methods", "exec"), ns)

        class FakeEnv:
            device = torch.device("cpu")
            obs_proprio_dim = PROPRIO
            num_obs_policy = OBS
            cfg = SimpleNamespace(mirror_joint_perm=V2_PERM, mirror_joint_signs=V2_SIGNS, policy_has_privileged_info=False)
            _build_mirror_transform = ns["_build_mirror_transform"]
            actor_mirror_tables = ns["actor_mirror_tables"]

        env = FakeEnv()
        env._build_mirror_transform()
        return env.actor_mirror_tables()
    # transcription of BipedEnv._build_mirror_transform (V2 layout) + history tiling
    p, s = torch.arange(PROPRIO), torch.ones(PROPRIO)
    for i in (1, 3, 5, 7, 10, 11, 48, 49):
        s[i] = -1.0
    jp = torch.tensor(V2_PERM)
    for o in (12, 24, 36):
        p[o:o + 12] = o + jp
    perm = torch.cat([p + i * PROPRIO for i in range(HIST)])
    return perm, s.repeat(HIST), jp, torch.tensor(V2_SIGNS)


TABLES = _real_tables()


def _plain_actor(seed=0, obs_dim=OBS, act_dim=ACT, hidden=(64, 64), asym_norm=True):
    torch.manual_seed(seed)
    obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,))
    act_space = spaces.Box(-1, 1, (act_dim,))
    norm = MeanStd(shape=obs_space.shape)
    if asym_norm:   # deliberately NOT mirror-symmetric stage-2 stats: exactness must not depend on them
        with torch.no_grad():
            norm._mean.copy_(torch.randn(obs_dim))
            norm._std.copy_(torch.rand(obs_dim) + 0.5)
    return MLPActor(obs_space, act_space, list(hidden), norm, "gaussian_multivariate")


def _equiv_actor(seed=0):
    return EquivariantActor(_plain_actor(seed), *TABLES)


def _obs(n=128, seed=1, scale=2.0):
    return torch.randn(n, OBS, generator=torch.Generator().manual_seed(seed)) * scale


def _M_obs(x):
    return x[:, TABLES[0]] * TABLES[1]


def _M_act(a):
    return a[:, TABLES[2]] * TABLES[3]


def _squashed_log_prob(dist, u):
    """log p(tanh(u)) as SquashedMultivariateNormalDiag.rsample_with_log_prob computes it."""
    return (dist._distribution.log_prob(u) - torch.log(1 - torch.tanh(u) ** 2 + 1e-6)).sum(-1)


# ------------------------------------------------------------------------------------------------ tables
def test_real_tables_are_involutions_and_cover_history():
    perm, sign, ap, asg = TABLES
    assert perm.numel() == OBS and sign.numel() == OBS and ap.numel() == ACT
    check_involution(perm, sign, "obs")
    check_involution(ap, asg, "act")
    # per-frame structure: frame i only permutes within frame i; newest-frame phase clock is negated
    for i in range(HIST):
        assert (perm[i * PROPRIO:(i + 1) * PROPRIO] // PROPRIO == i).all()
    assert sign[48] == -1 and sign[49] == -1 and sign[1] == -1 and sign[12] == 1


def test_check_involution_rejects_bad_tables():
    with pytest.raises(ValueError, match="involution"):
        check_involution(torch.tensor([1, 2, 0]), torch.ones(3))
    with pytest.raises(ValueError, match="sign"):
        check_involution(torch.tensor([1, 0]), torch.tensor([1.0, -1.0]))
    with pytest.raises(ValueError, match="permutation"):
        check_involution(torch.tensor([0, 0]), torch.ones(2))


# --------------------------------------------------------------------------------------- (a) equivariance
def test_mean_action_is_exactly_equivariant():
    actor, s = _equiv_actor(), _obs()
    plain = actor.net
    with torch.no_grad():
        mu = actor(s).mean
        mu_of_mirror = actor(_M_obs(s)).mean
        gap_plain = (plain(_M_obs(s)).mean - _M_act(plain(s).mean)).abs().max()
    assert gap_plain > 1e-2, "test is vacuous: the plain actor must NOT be equivariant"
    assert (mu_of_mirror - _M_act(mu)).abs().max() < 1e-6
    assert (actor.get_action(_M_obs(s)) - _M_act(actor.get_action(s))).abs().max() < 1e-6


def test_scale_is_permutation_equivariant_and_positive():
    actor, s = _equiv_actor(), _obs()
    with torch.no_grad():
        loc, scale = actor.loc_scale(s)
        loc_m, scale_m = actor.loc_scale(_M_obs(s))
    assert (scale > 0).all()
    assert (scale_m - scale[:, TABLES[2]]).abs().max() < 1e-6
    assert (loc_m - _M_act(loc)).abs().max() < 1e-6


def test_log_prob_of_mirrored_action_given_mirrored_state_is_equal():
    actor, s = _equiv_actor(), _obs()
    with torch.no_grad():
        d, dm = actor(s), actor(_M_obs(s))
        u = d._distribution.sample()                       # pre-squash sample
        a = torch.tanh(u)
        lp = _squashed_log_prob(d, u)
        u_m = _M_act(u)                                    # M a = tanh(M u): tanh is odd and elementwise
        assert (torch.tanh(u_m) - _M_act(a)).abs().max() < 1e-6
        lp_m = _squashed_log_prob(dm, u_m)
        assert (lp - lp_m).abs().max() < 1e-4
        # same through the actions themselves (atanh is only float-accurate, hence the looser bound)
        u_m2 = torch.atanh(_M_act(a).clamp(-1 + 1e-6, 1 - 1e-6))
        sel = (u.abs() < 3).all(-1)
        assert (lp[sel] - _squashed_log_prob(dm, u_m2)[sel]).abs().max() < 1e-2


def test_rsample_and_log_prob_use_symmetrised_loc_scale():
    actor, s = _equiv_actor(), _obs(16)
    loc, scale = actor.loc_scale(s)
    d = actor(s)
    assert isinstance(d, SquashedMultivariateNormalDiag)
    assert torch.equal(d._distribution.loc, loc) and torch.equal(d._distribution.scale, scale)
    torch.manual_seed(0)
    a, lp = d.rsample_with_log_prob()
    assert a.shape == (16, ACT) and lp.shape == (16, ACT) and a.abs().max() <= 1
    a.sum().backward()
    assert all(p.grad is not None for p in actor.net.torso.net.parameters())


def test_equivariance_holds_for_any_inner_network_after_updates():
    actor, s = _equiv_actor(3), _obs(64, 5)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
    for _ in range(5):
        loss = actor(s).rsample().square().sum() + actor(s).mean.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        assert (actor(_M_obs(s)).mean - _M_act(actor(s).mean)).abs().max() < 1e-6


def test_passthrough_attributes_and_get_action_not_bypassed():
    actor = _equiv_actor()
    assert actor.torso is actor.net.torso and actor.action_size == ACT and actor.head_type == "gaussian_multivariate"
    s = _obs(8)
    with torch.no_grad():
        assert not torch.allclose(actor.get_action(s), actor.net.get_action(s))
    with pytest.raises(AttributeError):
        actor.does_not_exist


def test_ragged_inputs_work_on_extra_leading_dims():
    actor, s = _equiv_actor(), _obs(12)
    with torch.no_grad():
        flat = actor(s).mean
        seq = actor(s.reshape(3, 4, OBS)).mean
    assert torch.allclose(seq.reshape(12, ACT), flat, atol=1e-6)


# --------------------------------------------------------------------------------------- (b) flag off
def _model(equivariant, tables=None, seed=0):
    torch.manual_seed(seed)
    obs_space = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (OBS,)), "critic": spaces.Box(-np.inf, np.inf, (OBS + 10,))})
    act_space = spaces.Box(-1, 1, (ACT,))
    return ActorTwinCriticWithTargets(obs_space, act_space, actor_sizes=[32, 32], critic_sizes=[32, 32],
                                      actor_type="gaussian_multivariate", equivariant_actor=equivariant, mirror_tables=tables)


def test_flag_off_is_the_plain_actor():
    m = _model(False)
    assert type(m.actor) is MLPActor and type(m.target_actor) is MLPActor
    keys = m.state_dict().keys()
    assert not any("mirror" in k or k.startswith("actor.net.") for k in keys)
    assert "actor.torso.net.0.weight" in keys
    # same weights as the equivariant model's inner net -> same pre-squash distribution
    me = _model(True, TABLES)
    me.actor.net.load_state_dict(m.actor.state_dict())
    s = _obs(8)
    with torch.no_grad():
        d, de = m.actor(s)._distribution, me.actor.net(s)._distribution
    assert torch.equal(d.loc, de.loc) and torch.equal(d.scale, de.scale)


def test_equivariant_requires_gaussian_multivariate_head():
    obs_space, act_space = spaces.Box(-np.inf, np.inf, (OBS,)), spaces.Box(-1, 1, (ACT,))
    with pytest.raises(ValueError, match="gaussian_multivariate"):
        ActorTwinCriticWithTargets(obs_space, act_space, actor_type="gaussian", equivariant_actor=True, mirror_tables=TABLES)


def test_table_length_mismatch_is_rejected():
    bad = (TABLES[0][:-50], TABLES[1][:-50], TABLES[2], TABLES[3])
    with pytest.raises(ValueError, match="mirror tables"):
        _model(True, bad)


# ----------------------------------------------------------------------------- (c) checkpoints / buffers
def test_state_dict_has_buffers_and_round_trips(tmp_path):
    m = _model(True, TABLES)
    keys = m.state_dict().keys()
    for k in ("mirror_obs_perm", "mirror_obs_sign", "mirror_act_perm", "mirror_act_sign"):
        assert f"actor.{k}" in keys and f"target_actor.{k}" in keys
    assert "actor.net.torso.net.0.weight" in keys
    path = str(tmp_path / "ckpt.pt")
    torch.save(m.state_dict(), path)

    # env-less rebuild (sim2real / policy_gate): placeholder tables, filled from the checkpoint
    m2 = _model(True, None, seed=99)
    s = _obs(8)
    with pytest.raises(RuntimeError, match="placeholder"):
        m2.actor(s)
    m2.load_state_dict(torch.load(path))
    with torch.no_grad():
        assert torch.equal(m.actor(s).mean, m2.actor(s).mean)
        assert (m2.actor(_M_obs(s)).mean - _M_act(m2.actor(s).mean)).abs().max() < 1e-6
    for k in ("mirror_obs_perm", "mirror_obs_sign", "mirror_act_perm", "mirror_act_sign"):
        assert torch.equal(getattr(m.actor, k), getattr(m2.actor, k))
    assert m2.actor.mirror_obs_perm.dtype == torch.long and m2.actor.mirror_obs_sign.dtype == torch.float32


def test_old_plain_checkpoint_loads_into_equivariant_model_with_env_tables():
    plain = _model(False, seed=1)
    sd = plain.state_dict()
    m = _model(True, TABLES, seed=2)
    m.load_state_dict(sd)
    for (k, v) in plain.actor.state_dict().items():
        assert torch.equal(m.actor.net.state_dict()[k], v), k
    assert torch.equal(m.actor.mirror_obs_perm, TABLES[0])          # env tables kept
    s = _obs(8)
    with torch.no_grad():
        assert (m.actor(_M_obs(s)).mean - _M_act(m.actor(s).mean)).abs().max() < 1e-6


def test_old_plain_checkpoint_into_envless_equivariant_model_is_a_clear_error():
    sd = _model(False).state_dict()
    with pytest.raises(ValueError, match="no mirror tables"):
        _model(True, None).load_state_dict(sd)


def test_equivariant_checkpoint_into_plain_model_is_a_clear_error():
    sd = _model(True, TABLES).state_dict()
    with pytest.raises(ValueError, match="equivariant_actor"):
        _model(False).load_state_dict(sd)


def test_corrupt_tables_in_checkpoint_are_rejected():
    sd = _model(True, TABLES).state_dict()
    sd["actor.mirror_act_perm"] = torch.arange(ACT).roll(1)           # a cycle, not an involution
    with pytest.raises(RuntimeError, match="invalid"):
        _model(True, None).load_state_dict(sd)


def test_target_actor_is_also_symmetrised_and_soft_update_works():
    m = _model(True, TABLES)
    assert isinstance(m.target_actor, EquivariantActor)
    m.update_targets()
    s = _obs(8)
    with torch.no_grad():
        assert (m.target_actor(_M_obs(s)).mean - _M_act(m.target_actor(s).mean)).abs().max() < 1e-6


# ------------------------------------------------------------------------------- SAC updaters and model
def test_actor_updater_runs_with_wrapper_and_mirror_loss_is_zero():
    from algorithms.sac.sac import TwinCriticSoftDeterministicPolicyGradient
    from algorithms.utils import DecayingEntropyCoeff
    from test_caps import FakeModel, StepsAdam, OBS as TOBS
    from test_symmetry_loss import OBS_PERM, OBS_SIGNS, ACT_PERM, ACT_SIGNS, mirror_obs, mirror_action

    torch.manual_seed(0)
    model = FakeModel()
    model.actor = EquivariantActor(model.actor, OBS_PERM, OBS_SIGNS, ACT_PERM, ACT_SIGNS)
    upd = TwinCriticSoftDeterministicPolicyGradient(
        model=model, action_space=None, entropy_coeff=DecayingEntropyCoeff(),
        optimizer=lambda params: StepsAdam(params, lr=1e-2),
        mirror_fn=lambda o, a: (mirror_obs(o), mirror_action(a)),
        symmetry_loss_weight=2.0, caps_temporal_weight=1.0, caps_spatial_weight=1.0)
    obs = torch.randn(32, TOBS)
    before = [p.detach().clone() for p in model.actor.parameters()]
    info = upd(obs, next_observations=torch.randn(32, TOBS))
    assert info["symmetry_loss"] < 1e-10
    assert info["caps_spatial"] > 0 and torch.isfinite(info["loss"])
    assert any(not torch.equal(b, p) for b, p in zip(before, model.actor.parameters()))


def test_sac_model_builds_from_env_tables_and_from_checkpoint(tmp_path):
    from algorithms.sac.model import SAC

    class Cfg(NoConfig):
        def __init__(self, equivariant):
            self.train = SimpleNamespace(equivariant_actor=equivariant, symmetry_augmentation=False)
        def __bool__(self):
            return True

    class Env:
        observation_space = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (OBS,)),
                                         "critic": spaces.Box(-np.inf, np.inf, (OBS + 10,))})
        action_space = spaces.Box(-1, 1, (ACT,))
        device = torch.device("cpu")
        num_envs = 1

    class EnvWithTables(Env):
        def actor_mirror_tables(self):
            return TABLES

    class Wrapped(Env):                      # RL wrapper -> gym wrapper -> BipedEnv
        def __init__(self):
            gym_wrapper = SimpleNamespace(unwrapped=EnvWithTables())
            self.env = gym_wrapper

    assert find_actor_mirror_tables(Wrapped()) is not None and find_actor_mirror_tables(Env()) is None

    # flag off: plain actor, no tables needed
    assert type(SAC(Env(), config=Cfg(False)).model.actor) is MLPActor
    # flag on, no tables, no checkpoint: clear error
    with pytest.raises(ValueError, match="mirror tables"):
        SAC(Env(), config=Cfg(True))
    # flag on with tables through the wrapper chain
    m = SAC(Wrapped(), config=Cfg(True))
    assert isinstance(m.model.actor, EquivariantActor)
    path = str(tmp_path / "step_1.pt")
    torch.save(m.model.state_dict(), path)
    # env-less rebuild from the checkpoint (sim2real / policy_gate path)
    m2 = SAC(Env(), model_path=path, config=Cfg(True))
    s = _obs(8)
    with torch.no_grad():
        assert torch.equal(m.model.actor(s).mean, m2.model.actor(s).mean)
        a = m2.model.actor.get_action(s)
        assert (m2.model.actor.get_action(_M_obs(s)) - _M_act(a)).abs().max() < 1e-6


# --------------------------------------------------------------------------------------------- (d) ONNX
class _MeanAction(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, x):
        return self.actor.mean_action(x)


def _export(module, path, in_dim=OBS):
    torch.onnx.export(module, torch.zeros(1, in_dim), path, input_names=["obs"], output_names=["action"],
                      opset_version=17, do_constant_folding=True, dynamic_axes=None)


def test_onnx_of_wrapped_actor_matches_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    actor = _equiv_actor().eval()
    path = str(tmp_path / "wrapped.onnx")
    _export(_MeanAction(actor), path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        x = (rng.standard_normal((1, OBS)) * 3).astype(np.float32)
        with torch.no_grad():
            ref = actor.get_action(torch.from_numpy(x)).numpy()
        worst = max(worst, float(np.abs(ref - sess.run(["action"], {"obs": x})[0]).max()))
    assert worst < 1e-5
    # and the ONNX itself is equivariant
    x = (rng.standard_normal((1, OBS)) * 3).astype(np.float32)
    xm = (torch.from_numpy(x)[:, TABLES[0]] * TABLES[1]).numpy()
    a, am = sess.run(["action"], {"obs": x})[0], sess.run(["action"], {"obs": xm})[0]
    assert np.abs(am - (torch.from_numpy(a)[:, TABLES[2]] * TABLES[3]).numpy()).max() < 1e-5
    import onnx
    ops = {n.op_type for n in onnx.load(path).graph.node}
    assert "Gather" in ops


@pytest.mark.skipif(not os.path.exists(EXPORT_ONNX), reason="src/isaaclab/export_onnx.py not reachable from here")
def test_export_script_reconstruction_matches_wrapped_actor(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    spec = importlib.util.spec_from_file_location("_export_onnx_under_test", EXPORT_ONNX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    m = _model(True, TABLES, seed=4)
    # make the stage-2 normaliser non-trivial
    with torch.no_grad():
        m.actor.net.torso.observation_normalizer._mean.copy_(torch.randn(OBS))
        m.actor.net.torso.observation_normalizer._std.copy_(torch.rand(OBS) + 0.5)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    rebuilt, in_dim, act_dim, hidden = mod.build_actor_from_state_dict(sd)
    assert isinstance(rebuilt, mod.EquivariantDeterministicActor) and (in_dim, act_dim, hidden) == (OBS, ACT, [32, 32])
    s = _obs(64, 7, 3.0)
    with torch.no_grad():
        assert (rebuilt(s) - m.actor.get_action(s)).abs().max() < 1e-5
    path = str(tmp_path / "policy.onnx")
    _export(rebuilt, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst = max(float(np.abs(sess.run(["action"], {"obs": s[i:i + 1].numpy()})[0] - m.actor.get_action(s[i:i + 1]).detach().numpy()).max())
                for i in range(20))
    assert worst < 1e-5

    # a plain checkpoint still maps to the plain DeterministicActor (old path unchanged)
    plain_sd = _model(False, seed=5).state_dict()
    plain, *_ = mod.build_actor_from_state_dict({k: v.clone() for k, v in plain_sd.items()})
    assert isinstance(plain, mod.DeterministicActor)
