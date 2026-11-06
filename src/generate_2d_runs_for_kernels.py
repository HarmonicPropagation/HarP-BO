"""
Reproduces the four 2D Bayesian-optimization runs that
`plot_kernels.py` compares against the direct grid scan
(`data/scan_results.csv`): one run per kernel in KERNELS, all using the
log_p_over_L__log_pL coordinate parameterization and the log
expected-improvement acquisition.

This mirrors run_bo_2d.py's physical setup (file 1 of 5 in the BO workflow)
but loops over every kernel instead of a single one, and fixes
OPTIMIZATION_SPACE = "log_p_over_L__log_pL" and ACQUISITION = "log_ei" so
the resulting run folders are named
`bo_2d_log_p_over_L__log_pL_log_ei_{kernel}`, exactly what plot_kernels.py
expects.

N_ITERATIONS defaults to 60 to match the paper's stated N_BO=60 (and what
plot_kernels.py's plot_iteration=60 expects), rather than the 101 used by
the dev-default run_bo_2d.py config. Each run takes roughly 60% as long as
`make run-2d`, so all four together take on the order of a few hours.
Use --kernels to run a subset and --skip-existing to resume after an
interruption without repeating already-completed runs.
"""

import argparse
from pathlib import Path

import numpy as np

from bo import run_optimization

KERNELS = [
    "rbf",
    "matern12",
    "matern32",
    "matern52",
]

# Seed for reproducibility
########################################################################################################
SEED = 0

# Path to the HarP installation on the computer
########################################################################################################
HARP_WORKDIR = Path("~/HarP-main").expanduser().resolve()

# BO parameters
########################################################################################################
N_INITIAL = 1  # Number of physical initial points
N_ITERATIONS = 60  # BO iterations (paper's N_BO=60)
OPTIMIZATION_SPACE = "log_p_over_L__log_pL"
ACQUISITION = "log_ei"

# HarP parameters
########################################################################################################
# Limits of BO in the physical variables used in the paper
LZZR_BOUNDS = (1.5 / 100.0, 1.5)  # Lzzr = L_gas / z_R
PZR_BOUNDS = (300.0 / 100.0, 300.0)  # pzr  = p_gas * z_R in mbar cm

# The 2D optimization keeps the laser intensity and duration fixed
INTENSITY_WCM2 = 2.5e14
TAU_FS = 22.0

# Generate one reproducible initial point directly in the physical Lzzr-pzr space.
# The same seed and the same physical bounds therefore produce exactly the same
# initial physical point for every kernel, matching run_bo_2d.py so the kernel
# runs here are directly comparable with one another.
INITIAL_RNG = np.random.default_rng(SEED)
INITIAL_LZZR = INITIAL_RNG.uniform(LZZR_BOUNDS[0], LZZR_BOUNDS[1])
INITIAL_PZR = INITIAL_RNG.uniform(PZR_BOUNDS[0], PZR_BOUNDS[1])
INITIAL_PHYSICAL_VALUES = {
    "Lzzr": float(INITIAL_LZZR),
    "pzr": float(INITIAL_PZR),
    "intensity_Wcm2": float(INTENSITY_WCM2),
    "tau_fs": float(TAU_FS),
}

# Numerical resolution.
# The temporal resolution is kept approximately fixed in SI units in 2D:
# Nt = ceil(LT_FACTOR * tau / DT_SI)
NR = 1024
NZ = 256
LR_FACTOR = 3.0  # box r-size * waist
LT_FACTOR = 3.0  # box time-size * duration
DT_SI = 1.2890625e-16

# Other physical parameters
TEMPERATURE_K = 293.0
WAVELENGTH_M = 0.81e-6
ZR_MM = 18.6
HARMONIC_ORDER = 23
OBJECTIVE_SCALE = 1.0e-6  # normalization for objective (CE)

# Path for saving outputs
########################################################################################################
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"


def build_config(kernel, n_initial, n_iterations):
    return {
        "dimension": 2,
        "seed": SEED,
        "n_initial": n_initial,
        "n_iterations": n_iterations,
        "optimization_space": OPTIMIZATION_SPACE,
        "acquisition": ACQUISITION,
        "kernel": kernel,
        "straddle_reference_mode": None,
        "straddle_reference_maximum_ce": None,
        "straddle_percentage": None,
        "straddle_beta": None,
        "Lzzr_bounds": LZZR_BOUNDS,
        "pzr_bounds": PZR_BOUNDS,
        "intensity_Wcm2_bounds": (INTENSITY_WCM2, INTENSITY_WCM2),
        "tau_fs_bounds": (TAU_FS, TAU_FS),
        "initial_physical_values": INITIAL_PHYSICAL_VALUES,
        "nr": NR,
        "nz": NZ,
        "lr_factor": LR_FACTOR,
        "lt_factor": LT_FACTOR,
        "dt_si": DT_SI,
        "temperature_K": TEMPERATURE_K,
        "wavelength_m": WAVELENGTH_M,
        "zR_mm": ZR_MM,
        "harmonic_order": HARMONIC_ORDER,
        "objective_scale": OBJECTIVE_SCALE,
        "harp_workdir": HARP_WORKDIR,
        "results_root": RESULTS_ROOT,
        "runner_file": Path(__file__).resolve(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=KERNELS,
        default=KERNELS,
        help="Subset of kernels to run (default: all four).",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=N_INITIAL,
        help="Number of physical initial points (default: matches run_bo_2d.py).",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=N_ITERATIONS,
        help="Number of BO iterations per run (default: matches run_bo_2d.py).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a kernel if its run folder already has a data.csv "
        "(lets a full sweep be resumed after an interruption).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Shared physical initial point (identical across all kernels below)")
    print("=" * 80)
    print(f"seed             = {SEED}")
    print(f"Lzzr             = {INITIAL_PHYSICAL_VALUES['Lzzr']:.16e}")
    print(f"pzr              = {INITIAL_PHYSICAL_VALUES['pzr']:.16e}")
    print(f"intensity_Wcm2   = {INITIAL_PHYSICAL_VALUES['intensity_Wcm2']:.16e}")
    print(f"tau_fs           = {INITIAL_PHYSICAL_VALUES['tau_fs']:.16e}")
    print("=" * 80)

    for i, kernel in enumerate(args.kernels, start=1):
        run_name = f"bo_2d_{OPTIMIZATION_SPACE}_log_ei_{kernel}"
        if args.skip_existing and (RESULTS_ROOT / run_name / "data.csv").exists():
            print(f"\n[{i}/{len(args.kernels)}] Skipping {run_name} (already exists)")
            continue
        print(f"\n[{i}/{len(args.kernels)}] Running {run_name}")
        config = build_config(kernel, args.n_initial, args.n_iterations)
        run_optimization(config)

    print("\nAll requested 2D log_ei kernel runs are ready for plot_kernels.py.")
