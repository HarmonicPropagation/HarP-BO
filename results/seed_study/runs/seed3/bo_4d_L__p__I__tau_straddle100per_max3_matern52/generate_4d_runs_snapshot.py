"""
Reproduces the 4 named 4D Bayesian-optimization runs that
`plot_4Dbo_convergence.py`'s RUNS_TO_PLOT compares (Cartesian
parameterization, physics mapping with raw CE, physics mapping with
inverse-square-root pulse duration and raw CE, physics mapping with
inverse-square-root pulse duration and sqrt(CE)), instead of hand-editing
run_bo_4d.py and rerunning per config.

acq_stopping_threshold is None (no early stopping) for all 4 runs, so each
runs the full budget of N_ITERATIONS=120: the paper compares the
parameterizations at equal iteration counts.

Each iteration is one real HarP simulation, on the order of 50-60s based on
past runs - i.e. up to a few hours per config, up to roughly half a day for
all three sequentially in the worst case. Use --configs to run a subset and
--skip-existing to resume after an interruption without repeating
already-completed runs.
"""

import argparse
from pathlib import Path
import numpy as np
from bo import run_optimization
from utils import make_run_name

# Seed for reproducibility
########################################################################################################
SEED                          = 0

# Path to the HarP installation on the computer
########################################################################################################
HARP_WORKDIR                  = Path("~/HarP-main").expanduser().resolve()

# BO parameters
########################################################################################################
N_INITIAL                     = 1
N_ITERATIONS                  = 120
KERNEL                        = "matern52"
ACQUISITION                   = "straddle"
STRADDLE_REFERENCE_MODE       = "fixed"
STRADDLE_BETA                 = 1.96
STRADDLE_PERCENTAGE           = 100
STRADDLE_REFERENCE_MAXIMUM_CE = 3e-6

# HarP parameters
########################################################################################################
LZZR_BOUNDS                   = (1.5 / 100.0, 1.5)
PZR_BOUNDS                    = (300.0 / 100.0, 300.0)
INTENSITY_WCM2_BOUNDS         = (0.01e14, 10e14)
TAU_0_FS                      = 22.0
TAU_FS_BOUNDS                 = (TAU_0_FS / 2.0, 4.0 * TAU_0_FS)

# Generate one reproducible initial point directly in the physical 4D space,
# identical across all 3 configs (same seed and bounds), matching the
# archived runs' shared initial_physical_values.
INITIAL_RNG = np.random.default_rng(SEED)
INITIAL_PHYSICAL_VALUES = {
    "Lzzr": float(INITIAL_RNG.uniform(LZZR_BOUNDS[0], LZZR_BOUNDS[1])),
    "pzr": float(INITIAL_RNG.uniform(PZR_BOUNDS[0], PZR_BOUNDS[1])),
    "intensity_Wcm2": float(INITIAL_RNG.uniform(INTENSITY_WCM2_BOUNDS[0], INTENSITY_WCM2_BOUNDS[1])),
    "tau_fs": float(INITIAL_RNG.uniform(TAU_FS_BOUNDS[0], TAU_FS_BOUNDS[1])),
}

# Numerical resolution
NR                            = 1024
NZ                            = 256
LR_FACTOR                     = 3.0
LT_FACTOR                     = 3.0
DT_SI                         = 1.2890625e-16  # chosen for power-of-two Nt at the reference taus - see utils.calculate_nt/points_per_optical_cycle

# Other physical parameters
TEMPERATURE_K                 = 293.0
WAVELENGTH_M                  = 0.81e-6
ZR_MM                         = 18.6
HARMONIC_ORDER                = 23
OBJECTIVE_SCALE               = 1.0e-6

# Path for saving outputs
########################################################################################################
PROJECT_ROOT                  = Path(__file__).resolve().parent.parent
RESULTS_ROOT                  = PROJECT_ROOT / "results"

# The 4 named configs, matching plot_4Dbo_convergence.py's RUNS_TO_PLOT
# exactly (folder name, label, and the settings that actually produced them).
########################################################################################################
CONFIGS = {
    "L__p__I__tau": {
        "label": "L, p, I, tau",
        "optimization_space": "L__p__I__tau",
        "acq_stopping_threshold": None,
        "objective_transform": None,
    },

    "log_p_over_L__log_pL__I__tau": {
        "label": "log(p/L), log(pL), I, tau",
        "optimization_space": "log_p_over_L__log_pL__I__tau",
        "acq_stopping_threshold": None,
        "objective_transform": None,
    },

    "log_p_over_L__log_pL__I__inv_sqrt_tau": {
        "label": "log(p/L), log(pL), I, 1/sqrt(tau)",
        "optimization_space": "log_p_over_L__log_pL__I__inv_sqrt_tau",
        "acq_stopping_threshold": None,
        "objective_transform": None,
    },

    "log_p_over_L__log_pL__I__inv_sqrt_tau__sqrtCE": {
        "label": "log(p/L), log(pL), I, 1/sqrt(tau), sqrt(CE)",
        "optimization_space": "log_p_over_L__log_pL__I__inv_sqrt_tau",
        "acq_stopping_threshold": None,
        "objective_transform": "sqrt",
    },
}

def build_config(name, n_initial, n_iterations):
    spec = CONFIGS[name]
    config = {
        "dimension": 4,
        "seed": SEED,
        "n_initial": n_initial,
        "n_iterations": n_iterations,
        "acq_stopping_threshold": spec["acq_stopping_threshold"],
        "optimization_space": spec["optimization_space"],
        "acquisition": ACQUISITION,
        "kernel": KERNEL,
        "straddle_reference_mode": STRADDLE_REFERENCE_MODE,
        "straddle_reference_maximum_ce": STRADDLE_REFERENCE_MAXIMUM_CE,
        "straddle_percentage": STRADDLE_PERCENTAGE,
        "straddle_beta": STRADDLE_BETA,
        "Lzzr_bounds": LZZR_BOUNDS,
        "pzr_bounds": PZR_BOUNDS,
        "intensity_Wcm2_bounds": INTENSITY_WCM2_BOUNDS,
        "tau_fs_bounds": TAU_FS_BOUNDS,
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
    if spec["objective_transform"] is not None:
        config["objective_transform"] = spec["objective_transform"]
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=list(CONFIGS),
        help="Subset of named configs to run (default: all three).",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=N_INITIAL,
        help="Number of physical initial points (default: matches the archived configs).",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=N_ITERATIONS,
        help="BO iteration budget per run (default: 256, matching the archived configs). "
        "mapping_raw has no early stop, so it always uses this full budget.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a config if its run folder already has a data.csv "
        "(lets a full sweep be resumed after an interruption).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Shared 4D physical initial point (identical across all 3 configs)")
    print("=" * 80)
    print(f"seed             = {SEED}")
    print(f"Lzzr             = {INITIAL_PHYSICAL_VALUES['Lzzr']:.16e}")
    print(f"pzr              = {INITIAL_PHYSICAL_VALUES['pzr']:.16e}")
    print(f"intensity_Wcm2   = {INITIAL_PHYSICAL_VALUES['intensity_Wcm2']:.16e}")
    print(f"tau_fs           = {INITIAL_PHYSICAL_VALUES['tau_fs']:.16e}")
    print("=" * 80)

    for i, name in enumerate(args.configs, start=1):
        config = build_config(name, args.n_initial, args.n_iterations)
        run_name = make_run_name(config)
        if args.skip_existing and (RESULTS_ROOT / run_name / "data.csv").exists():
            print(f"\n[{i}/{len(args.configs)}] Skipping {name} ({run_name}, already exists)")
            continue
        print(f"\n[{i}/{len(args.configs)}] Running {name} ({CONFIGS[name]['label']}) -> {run_name}")
        run_optimization(config)

    print("\nAll requested 4D runs are ready for plot_4Dbo_convergence.py.")
