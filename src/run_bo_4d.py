"""
This script defines the optimization settings and physical parameters for a
4D optimization (pressure, length, intensity, and pulse duration) of
conversion efficiency. It specifies the search space, Bayesian optimization
settings, numerical resolution, and HarP configuration before passing
everything to the main optimization routine.

This is file 2 of 5 in the BO workflow:
    1. run_bo_2d.py   - Defines the setup and starts the 2D Bayesian optimization.
    2. run_bo_4d.py   - Defines the setup and starts the 4D Bayesian optimization.
    3. harp_input.py  - Prepares, runs, and evaluates one HarP simulation.
    4. bo.py          - Bayesian optimization algorithm and acquisition strategy.
    5. utils.py       - Shared helper functions and numerical utilities.

This file is meant for one-off exploration - edit and rerun freely. That
includes a naive/raw-Cartesian-parameterization run (OPTIMIZATION_SPACE =
"L__p__I__tau", no coordinate mapping): there used to be a separate
run_bo_4d_naive.py for that specific variant, but it was just this file with
different parameter values, so it was retired - set the parameters below by
hand instead, same as for any other variant. To regenerate the exact 3 named
4D configs behind the paper's Fig. 4Dbo_convergence instead (without
hand-editing settings back and forth), see generate_4d_runs.py.
"""

from pathlib import Path
from bo import run_optimization
import numpy as np

# Seed for reproducibility
########################################################################################################
SEED                          = 0

# Path to the HarP installation on the computer
########################################################################################################
HARP_WORKDIR                  = Path("~/HarP-main").expanduser().resolve()

# BO parameters
########################################################################################################
N_INITIAL                     = 1                                       # BO random points
N_ITERATIONS                  = 256                                     # BO iterations
# OPTIMIZATION_SPACE            = "log_p_over_L__log_pL__I__inv_sqrt_tau" # "log_p_over_L__log_pL__I__tau", "log_p_over_L__log_pL__I__inv_sqrt_tau"
OPTIMIZATION_SPACE            = "log_p_over_L__log_pL__I__inv_sqrt_tau" # "log_p_over_L__log_pL__I__tau", "log_p_over_L__log_pL__I__inv_sqrt_tau"
ACQUISITION                   = "log_ei"                              # "log_ei", "straddle"
KERNEL                        = "matern52"                              # "matern12", "matern32", "matern52", "rbf"
ACQ_STOPPING_THRESHOLD        = None                                    # None = no early stopping (as for the paper figures); else a utility threshold (perhaps 1e-6 for LogEI, 1e-2 for Straddle)
# Straddle settings
# Ignored unless ACQUISITION = "straddle"
# STRADDLE_REFERENCE_MODE:
#       "fixed" - Uses a fixed reference CE specified below.
#       "gp"    - Uses the current GP-predicted maximum CE as the reference.
#       None    - Use when using different acq. function than "straddle"
# Note: In GP mode, the target contour moves every iteration because the estimated GP maximum may change.
# For "fixed", the reference maximum CE is given by the user in STRADDLE_REFERENCE_MAXIMUM_CE.
# For "gp", the reference maximum CE is estimated from the current GP model.
# The Straddle target is computed as STRADDLE_PERCENTAGE / 100 * reference CE.
# STRADDLE_BETA controls the exploration width around the target contour, e.g. 1.96 is 95% confidence interval

STRADDLE_REFERENCE_MODE       = "gp"           # None, "fixed", or "gp"
STRADDLE_BETA                 = 1.96           # for "fixed" and "gp"
STRADDLE_PERCENTAGE           = 75             # Target 75%
STRADDLE_REFERENCE_MAXIMUM_CE = 5e-6           # Target threshold (not used for gp mode)

# HarP parameters
########################################################################################################
# Limits of BO in the physical variables used in the paper
LZZR_BOUNDS                   = (1.5 / 100.0, 1.5)     # Lzzr = L_gas / z_R
PZR_BOUNDS                    = (300.0 / 100.0, 300.0) # pzr  = p_gas * z_R in mbar cm
INTENSITY_WCM2_BOUNDS         = (0.01e14, 10e14)       # laser intensity [W/cm^2], table is valid for 1e12-1e15 W/cm^2
TAU_0_FS                      = 22.0                   # fixed duration from 2D optimization as a reference
TAU_FS_BOUNDS                 = (TAU_0_FS / 2.0, 4.0 * TAU_0_FS)

# Generate one reproducible initial point directly in the physical 4D space.
INITIAL_RNG                   = np.random.default_rng(SEED)
INITIAL_PHYSICAL_VALUES       = {
    "Lzzr": float(INITIAL_RNG.uniform(LZZR_BOUNDS[0], LZZR_BOUNDS[1])),
    "pzr": float(INITIAL_RNG.uniform(PZR_BOUNDS[0], PZR_BOUNDS[1])),
    "intensity_Wcm2": float(INITIAL_RNG.uniform(INTENSITY_WCM2_BOUNDS[0], INTENSITY_WCM2_BOUNDS[1])),
    "tau_fs": float(INITIAL_RNG.uniform(TAU_FS_BOUNDS[0], TAU_FS_BOUNDS[1])),
}

# Numerical resolution.
# Nt = ceil(LT_FACTOR * tau / DT_SI), changes in 4D with time box size
NR                            = 1024
NZ                            = 256
LR_FACTOR                     = 3.0
LT_FACTOR                     = 3.0
DT_SI                         = 1.2890625e-16
# Other physical parameters
TEMPERATURE_K                 = 293.0
WAVELENGTH_M                  = 0.81e-6
ZR_MM                         = 18.6
HARMONIC_ORDER                = 23
OBJECTIVE_SCALE               = 1.0e-6 # normalization for objective (CE)

# Path for saving outputs
########################################################################################################
PROJECT_ROOT                  = Path(__file__).resolve().parent.parent
RESULTS_ROOT                  = PROJECT_ROOT / "results"
# RESULTS_ROOT = PROJECT_ROOT / "data" / "archived_runs"

# Main
########################################################################################################
# The configuration dictionary gathers all settings required for the optimization in 4D
# They are passed to the main BO routine, which executes the complete optimization loop
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("4D physical initial point")
    print("=" * 80)
    print(f"seed             = {SEED}")
    print(f"Lzzr             = {INITIAL_PHYSICAL_VALUES['Lzzr']:.16e}")
    print(f"pzr              = {INITIAL_PHYSICAL_VALUES['pzr']:.16e}")
    print(f"intensity_Wcm2   = {INITIAL_PHYSICAL_VALUES['intensity_Wcm2']:.16e}")
    print(f"tau_fs           = {INITIAL_PHYSICAL_VALUES['tau_fs']:.16e}")
    print("=" * 80)

    config = {
        "dimension": 4,
        "seed": SEED,
        "n_initial": N_INITIAL,
        "n_iterations": N_ITERATIONS,
        "acq_stopping_threshold": ACQ_STOPPING_THRESHOLD,
        "optimization_space": OPTIMIZATION_SPACE,
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
        "objective_transform": "sqrt",   # Was "log10" earlier
        "harp_workdir": HARP_WORKDIR,
        "results_root": RESULTS_ROOT,
        "runner_file": Path(__file__).resolve(),
    }

    run_optimization(config)
