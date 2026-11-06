"""
GP hyperparameter fit (utils.fit_gp, run config key gp_fit) on the training data where the
"cold" fit fell into a flat optimum: the 2D L__p LogEI Matern-5/2 seed-0 run with fixed
noise 1e-5, BO iteration 54 (the 54 points the fit saw, tests/data/gp_fit_trap_L__p_seed0_iter54.csv).

At iteration 53 the fit found lengthscales (L, p) = (0.026, 0.46). At iteration 54 the cold
fit, which starts from gpytorch's initial values (0.693), ended at lengthscales (3e-8,
0.006) and outputscale 0.98: an iid-noise model whose log marginal likelihood (-76.1) is
about 49 nats below that of the iteration-53 lengthscales refitted on the same data
(-27.0). The run's surrogate was then flat for iterations 54-61 (SSIM against the 2D scan
0.01 instead of 0.8). A grid of the marginal likelihood showed one narrow peak near the
iteration-53 values on a large plateau at ls_L < 1e-3, so this is an optimizer trap, not a
preferred solution. gp_fit = "warm" refits from the previous iteration's hyperparameters
and keeps that fit only where it is better than the cold fit.

The cold-start test documents the trap (Linux, BoTorch 0.18.1 / gpytorch 1.15.2). Whether a
cold L-BFGS fit lands in it depends on the optimizer's path, so another BLAS (e.g. macOS
Accelerate) or library version may miss it; the test then skips instead of failing. The
warm-fit tests are the actual regression gate and do not depend on the cold fit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo import run_optimization
from harp_input import check_harp_installation
from utils import fit_gp, gp_log_marginal_likelihood, make_kernel, make_likelihood

HARP_WORKDIR = Path("~/HarP-main").expanduser().resolve()
DATA = Path(__file__).parent / "data" / "gp_fit_trap_L__p_seed0_iter54.csv"
SEED = 0

# Short end-to-end run: bounds and seeded initial point as in generate_2d_runs_for_fig1_fig2.py.
LZZR_BOUNDS = (1.5 / 100.0, 1.5)
PZR_BOUNDS = (300.0 / 100.0, 300.0)
INTENSITY_WCM2 = 2.5e14
TAU_FS = 22.0
INITIAL_RNG = np.random.default_rng(SEED)
INITIAL_PHYSICAL_VALUES = {
    "Lzzr": float(INITIAL_RNG.uniform(*LZZR_BOUNDS)),
    "pzr": float(INITIAL_RNG.uniform(*PZR_BOUNDS)),
    "intensity_Wcm2": INTENSITY_WCM2,
    "tau_fs": TAU_FS,
}

# Fitted hyperparameters of the same run at iteration 53 (model_iter_053.pt; a checkpoint
# holds the fit of its iteration and the training data including that iteration's new point).
ITER53 = {
    "lengthscale": torch.tensor([[0.02637676558694474, 0.46412435199142005]], dtype=torch.float64),
    "outputscale": torch.tensor(0.4024554559533836, dtype=torch.float64),
    "constant": torch.tensor(-0.5814998208699952, dtype=torch.float64),
}


def build_model():
    # As in bo.run_optimization for this run (identity objective transform, fixed noise 1e-5).
    data = np.loadtxt(DATA, delimiter=",", comments="#", skiprows=3)
    X = torch.tensor(data[:, :2], dtype=torch.float64)
    Y = torch.clamp(torch.tensor(data[:, 2:], dtype=torch.float64), min=1e-15)
    return SingleTaskGP(train_X=X, train_Y=Y, covar_module=make_kernel("matern52", 2),
                        likelihood=make_likelihood("fixed:1e-5"), outcome_transform=Standardize(m=1))


def lengthscales(model):
    return model.covar_module.base_kernel.lengthscale.detach().squeeze().tolist()


COLD_TRAP_NOT_REPRODUCED = ("cold fit did not reproduce the trap on this platform; harmless, "
                            "see test_warm_fit_reaches_the_iteration_53_optimum")


def in_flat_optimum(model):
    ls_L, ls_p = lengthscales(model)
    return ls_L < 1e-3 and ls_p < 0.05


def same_state(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_cold_fit_lands_in_the_flat_optimum() -> None:
    model = build_model()
    info = fit_gp(model, "cold")
    if not in_flat_optimum(model):
        pytest.skip(f"{COLD_TRAP_NOT_REPRODUCED} (lengthscales {lengthscales(model)})")
    assert info["mll_cold"] < -90.0, info  # about -76.1, plus the constant noise-prior term -17.6


def test_warm_fit_reaches_the_iteration_53_optimum() -> None:
    model = build_model()
    info = fit_gp(model, "warm", previous=ITER53)
    ls_L, ls_p = lengthscales(model)
    assert ls_L == pytest.approx(0.026, rel=0.05) and ls_p == pytest.approx(0.46, rel=0.05), (ls_L, ls_p)
    assert float(model.covar_module.outputscale.detach()) == pytest.approx(0.40, rel=0.05)
    # About -27.0, plus the constant noise-prior term -17.6 (the flat optimum: about -93.7).
    assert info["mll_warm"] > -50.0, info
    assert gp_log_marginal_likelihood(model) == info["mll_warm"]
    cold = build_model()
    fit_gp(cold, "cold")
    if in_flat_optimum(cold):  # where the cold fit is trapped, the warm fit must win
        assert info["start"] == "warm" and info["mll_warm"] > info["mll_cold"] + 45.0, info


def test_warm_fit_without_previous_is_the_cold_fit() -> None:
    cold, warm = build_model(), build_model()
    fit_gp(cold, "cold")
    info = fit_gp(warm, "warm", previous=None)
    assert info["start"] == "cold" and info["mll_warm"] is None
    assert same_state(cold, warm)


def test_warm_fit_is_deterministic() -> None:
    fits = []
    for _ in range(2):
        model = build_model()
        fits.append((fit_gp(model, "warm", previous=ITER53), model))
    assert fits[0][0] == fits[1][0]
    assert same_state(fits[0][1], fits[1][1])


def test_cold_fit_is_the_plain_fit() -> None:
    # gp_fit = "cold" must give exactly the fit of bo.py before gp_fit existed.
    plain = build_model()
    fit_gpytorch_mll(ExactMarginalLogLikelihood(plain.likelihood, plain))
    cold = build_model()
    fit_gp(cold, "cold")
    for (name, a), (_, b) in zip(plain.state_dict().items(), cold.state_dict().items()):
        assert torch.equal(a, b), name


def test_unknown_gp_fit_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_gp(build_model(), "lukewarm")


def short_2d_run(tmp_path, gp_fit):
    # A few real BO iterations on a tiny HarP grid, as in test_logei_2d_repro.py.
    config = {
        "dimension": 2,
        "seed": SEED,
        "n_initial": 1,
        "n_iterations": 6,
        "optimization_space": "L__p",
        "acquisition": "log_ei",
        "kernel": "matern52",
        "straddle_reference_mode": None,
        "straddle_reference_maximum_ce": None,
        "straddle_percentage": None,
        "straddle_beta": None,
        "Lzzr_bounds": LZZR_BOUNDS,
        "pzr_bounds": PZR_BOUNDS,
        "intensity_Wcm2_bounds": (INTENSITY_WCM2, INTENSITY_WCM2),
        "tau_fs_bounds": (TAU_FS, TAU_FS),
        "initial_physical_values": INITIAL_PHYSICAL_VALUES,
        # Deliberately tiny grid (production runs use nr=1024, nz=256) - keeps
        # each HarP call to a couple of seconds, same trick as test_harp_smoke.py.
        "nr": 64,
        "nz": 32,
        "lr_factor": 3.0,
        "lt_factor": 3.0,
        "dt_si": 1.2890625e-16,
        "temperature_K": 293.0,
        "wavelength_m": 0.81e-6,
        "zR_mm": 18.6,
        "harmonic_order": 23,
        "objective_scale": 1.0e-6,
        "harp_workdir": HARP_WORKDIR,
        "results_root": tmp_path,
        "gp_fit": gp_fit,
        "runner_file": Path(__file__).resolve(),
    }
    run_optimization(config)
    (run_folder,) = tmp_path.glob("bo_2d_*")
    return config, run_folder


def decisions(run_folder):
    # data.csv without the wall-clock columns and the archived-file paths (they contain the run folder)
    df = pd.read_csv(run_folder / "data.csv")
    return df.drop(columns=[c for c in df.columns if c.endswith("_s") or c == "diag_file"])


def test_short_2d_runs_with_warm_fit(tmp_path) -> None:
    # gp_fit = "warm" is recorded, every checkpoint says which fit was kept, the same run twice
    # gives identical data, and the data equal the cold run's as long as the warm fit never wins.
    try:
        check_harp_installation(HARP_WORKDIR)
    except FileNotFoundError:
        pytest.skip(f"No real HarP installation found at {HARP_WORKDIR}")
    config, warm_a = short_2d_run(tmp_path / "warm_a", "warm")
    _, warm_b = short_2d_run(tmp_path / "warm_b", "warm")
    _, cold = short_2d_run(tmp_path / "cold", "cold")
    assert "gp_fit = warm" in (warm_a / "run_config.txt").read_text().splitlines()
    assert "gp_fit = warm" in (warm_a / "summary.txt").read_text().splitlines()
    assert len(decisions(warm_a)) == 1 + config["n_iterations"]
    pd.testing.assert_frame_equal(decisions(warm_a), decisions(warm_b))
    kept = []
    for i, path in enumerate(sorted((warm_a / "models").glob("model_iter_*.pt"))):
        info = torch.load(path, map_location="cpu", weights_only=False)["gp_fit_info"]
        assert info["gp_fit"] == "warm" and np.isfinite(info["mll_cold"])
        assert (info["mll_warm"] is None) == (i == 0)  # no warm start in the first iteration
        kept.append(info["start"])
    if "warm" not in kept:
        pd.testing.assert_frame_equal(decisions(warm_a), decisions(cold))


def test_warm_fit_falls_back_when_the_cold_fit_fails(monkeypatch) -> None:
    # When the cold fit raises (fit_gpytorch_mll gives up, as can happen with Matern 1/2), gp_fit =
    # "warm" with a previous fit uses the warm fit instead, so the run continues; "cold" still raises.
    import utils

    real_fit = utils.fit_gpytorch_mll
    calls = []

    def fail_first(mll, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("simulated failure of the cold fit")
        return real_fit(mll, **kwargs)

    monkeypatch.setattr(utils, "fit_gpytorch_mll", fail_first)
    model = build_model()
    info = fit_gp(model, "warm", previous=ITER53)
    assert info["start"] == "warm (cold failed)" and "simulated failure" in info["cold_error"]
    assert info["mll_cold"] is None and info["mll_warm"] > -50.0, info
    ls_L, ls_p = lengthscales(model)
    assert ls_L == pytest.approx(0.026, rel=0.05) and ls_p == pytest.approx(0.46, rel=0.05), (ls_L, ls_p)

    calls.clear()
    with pytest.raises(RuntimeError):
        fit_gp(build_model(), "cold")
    calls.clear()
    with pytest.raises(RuntimeError):
        fit_gp(build_model(), "warm", previous=None)
