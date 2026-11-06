"""
Reproduces the three 2D Straddle Bayesian-optimization runs that
`plot_straddle.py` needs: one run per CE target threshold in
STRADDLE_PERCENTAGES (25%, 50%, 90% of a fixed 3e-6 reference CE, i.e.
targets 0.75e-6, 1.5e-6, 2.7e-6), all in the same optimization space and
kernel that plot_straddle.py hardcodes.

This mirrors run_bo_2d.py's physical setup (file 1 of 5 in the BO workflow)
but loops over the three Straddle percentage targets instead of a single one,
so the resulting run folders are named
`bo_2d_{space}_straddle{25,50,90}per_max3_matern52`, exactly what
plot_straddle.py expects.

N_ITERATIONS defaults to 60 to match the paper's stated N_BO=60 (and what
plot_straddle.py's plot_iteration=60 expects). Each run takes roughly 60% as
long as `make run-2d`, so all three together take on the order of two hours.
Use --percentages to run a subset and --skip-existing to resume after an
interruption without repeating already-completed runs.
"""

import argparse
from pathlib import Path

import numpy as np

from bo import run_optimization

# Must match plot_straddle.py's `optimization_space` and `kernel`.
OPTIMIZATION_SPACE = "log_p_over_L__log_pL"
KERNEL = "matern52"

# Must match plot_straddle.py's figure1_straddles/figure2_straddles and
# straddle_targets (25 -> 0.75e-6, 50 -> 1.5e-6, 90 -> 2.7e-6 of a 3e-6 reference).
STRADDLE_PERCENTAGES = [25, 50, 90]

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
ACQUISITION = "straddle"
STRADDLE_REFERENCE_MODE = "fixed"
STRADDLE_BETA = 1.96
STRADDLE_REFERENCE_MAXIMUM_CE = 3e-6

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
# initial physical point as run_bo_2d.py and generate_2d_runs_for_fig1_fig2.py.
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


def build_config(straddle_percentage, n_initial, n_iterations):
    return {
        "dimension": 2,
        "seed": SEED,
        "n_initial": n_initial,
        "n_iterations": n_iterations,
        "optimization_space": OPTIMIZATION_SPACE,
        "acquisition": ACQUISITION,
        "kernel": KERNEL,
        "straddle_reference_mode": STRADDLE_REFERENCE_MODE,
        "straddle_reference_maximum_ce": STRADDLE_REFERENCE_MAXIMUM_CE,
        "straddle_percentage": straddle_percentage,
        "straddle_beta": STRADDLE_BETA,
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
        "--percentages",
        nargs="+",
        type=int,
        choices=STRADDLE_PERCENTAGES,
        default=STRADDLE_PERCENTAGES,
        help="Subset of Straddle CE-target percentages to run (default: all three).",
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
        help="Number of BO iterations per run (default: paper's N_BO=60).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a percentage if its run folder already has a data.csv "
        "(lets a full sweep be resumed after an interruption).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Shared physical initial point (identical across all targets below)")
    print("=" * 80)
    print(f"seed             = {SEED}")
    print(f"Lzzr             = {INITIAL_PHYSICAL_VALUES['Lzzr']:.16e}")
    print(f"pzr              = {INITIAL_PHYSICAL_VALUES['pzr']:.16e}")
    print(f"intensity_Wcm2   = {INITIAL_PHYSICAL_VALUES['intensity_Wcm2']:.16e}")
    print(f"tau_fs           = {INITIAL_PHYSICAL_VALUES['tau_fs']:.16e}")
    print("=" * 80)

    for i, percentage in enumerate(args.percentages, start=1):
        run_name = f"bo_2d_{OPTIMIZATION_SPACE}_straddle{percentage:g}per_max3_{KERNEL}"
        if args.skip_existing and (RESULTS_ROOT / run_name / "data.csv").exists():
            print(f"\n[{i}/{len(args.percentages)}] Skipping {run_name} (already exists)")
            continue
        print(f"\n[{i}/{len(args.percentages)}] Running {run_name}")
        config = build_config(percentage, args.n_initial, args.n_iterations)
        run_optimization(config)

    print("\nAll requested 2D Straddle runs are ready for plot_straddle.py.")
