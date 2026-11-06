"""
Short end-to-end check of the 2D Straddle configuration of Figs. 4-5 in the log(p/L), log(pL) space.

The test runs generate_2d_runs_for_straddle_fig.py's configuration (same seed, bounds, initial point,
optimization_space="log_p_over_L__log_pL", acquisition="straddle") for 8 iterations with a
deliberately tiny HarP grid (nr=64, nz=32, as in test_harp_smoke.py), so it runs in about a minute.
This exercises the acquisition optimization under the nonlinear physical-box constraints, where
BoTorch may emit "Positive directional derivative for linesearch" OptimizationWarnings and then
continue. The test fails only if the run raises or writes the wrong number of rows; any warning
appears in the captured output.

Skipped (not failed) when no real HarP installation is available, as in test_harp_smoke.py.
"""

from pathlib import Path

import numpy as np
import pytest

from bo import run_optimization
from harp_input import check_harp_installation

HARP_WORKDIR = Path("~/HarP-main").expanduser().resolve()

# Same seed/bounds/space as generate_2d_runs_for_straddle_fig.py, so the
# initial physical point and acquisition setup match the real failing case.
SEED = 0
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


def test_straddle_log_p_over_L_log_pL_small_repro(tmp_path, recwarn) -> None:
    try:
        check_harp_installation(HARP_WORKDIR)
    except FileNotFoundError:
        pytest.skip(f"No real HarP installation found at {HARP_WORKDIR}")

    config = {
        "dimension": 2,
        "seed": SEED,
        "n_initial": 1,
        "n_iterations": 8,  # small: enough restarts to exercise optimize_acqf repeatedly
        "optimization_space": "log_p_over_L__log_pL",
        "acquisition": "straddle",
        "kernel": "matern52",
        "straddle_reference_mode": "fixed",
        "straddle_reference_maximum_ce": 3e-6,
        "straddle_percentage": 25,
        "straddle_beta": 1.96,
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
        "runner_file": Path(__file__).resolve(),
    }

    # Uses the `recwarn` fixture (not warnings.catch_warnings) specifically so
    # any triggered warning still reaches pytest's own end-of-session warnings
    # summary - a plain catch_warnings block would swallow it before pytest's
    # collector ever sees it, making it invisible in CI logs regardless of
    # -s/-v flags.
    run_optimization(config)

    linesearch_warnings = [
        w for w in recwarn.list
        if "directional derivative" in str(w.message).lower()
        or "linesearch" in str(w.message).lower()
    ]
    if linesearch_warnings:
        print(
            f"\n{len(linesearch_warnings)} linesearch/directional-derivative "
            "warning(s) triggered during this small repro run - see pytest's "
            "warnings summary below for BoTorch's own messages. Not treated "
            "as a hard failure here (see module docstring), just surfaced "
            "for a human to look at."
        )

    run_folders = list(tmp_path.glob("bo_2d_*"))
    assert len(run_folders) == 1, f"Expected exactly one run folder, found {run_folders}"
    data_csv = run_folders[0] / "data.csv"
    assert data_csv.exists()
    rows = data_csv.read_text().strip().splitlines()
    # header + 1 initial point + 8 BO iterations
    assert len(rows) == 1 + 1 + config["n_iterations"]
