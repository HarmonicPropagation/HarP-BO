"""
This script defines the optimization settings and physical parameters for a
2D optimization (pressure and length) of conversion efficiency.
It specifies the search space, Bayesian optimization settings,
numerical resolution, and HarP configuration before passing
everything to the main optimization routine.
This is file 1 of 5 in the BO workflow:
    1. run_bo_2d.py   - Defines the setup and starts the 2D Bayesian optimization.
    2. run_bo_4d.py   - Defines the setup and starts the 4D Bayesian optimization.
    3. harp_input.py  - Prepares, runs, and evaluates one HarP simulation.
    4. bo.py          - Bayesian optimization algorithm and acquisition strategy.
    5. utils.py       - Shared helper functions and numerical utilities.

This file is meant for one-off exploration - edit and rerun freely. To
regenerate the exact named 2D configs behind the paper's figures instead
(without hand-editing settings back and forth), see
generate_2d_runs_for_fig1_fig2.py and generate_2d_runs_for_straddle_fig.py.
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
N_INITIAL                     = 1                       # Number of physical initial points
N_ITERATIONS                  = 101                     # BO iterations
OPTIMIZATION_SPACE            = "log_p_over_L__log_pL"  # "L__p", "log_L__log_p", "p_over_L__pL", "log_p_over_L__log_pL", "sqrt_p_over_L__sqrt_pL"
ACQUISITION                   = "straddle"                # "log_ei", "straddle"
KERNEL                        = "matern52"              # "matern12", "matern32", "matern52", "rbf"
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

#STRADDLE_REFERENCE_MODE       = None           # None, "fixed", or "gp"
#STRADDLE_BETA                 = None          # for "fixed" and "gp", e.g. 1.96 for a central 95% interval assuming a Gaussian posterior
#STRADDLE_PERCENTAGE           = None            # for "fixed" and "gp", e.g. 40 for 40 %
#STRADDLE_REFERENCE_MAXIMUM_CE = None           # for "fixed" only, user given value e.g. 3.5e-6 as maximum CE, ignored for "gp"!

STRADDLE_REFERENCE_MODE       = "fixed"        # None, "fixed", or "gp"
STRADDLE_BETA                 = 1.96           # for "fixed" and "gp", e.g. 1.96 for a central 95% interval assuming a Gaussian posterior
STRADDLE_PERCENTAGE           = 100            # for "fixed" and "gp", e.g. 40 for 40 %
STRADDLE_REFERENCE_MAXIMUM_CE = 3e-6           # for "fixed" only, user given value e.g. 3.5e-6 as maximum CE, ignored for "gp"!

# HarP parameters
########################################################################################################
# Limits of BO in the physical variables used in the paper
LZZR_BOUNDS                   = (1.5 / 100.0, 1.5)     # Lzzr = L_gas / z_R
PZR_BOUNDS                    = (300.0 / 100.0, 300.0) # pzr  = p_gas * z_R in mbar cm

# The 2D optimization keeps the laser intensity and duration fixed
INTENSITY_WCM2                = 2.5e14
TAU_FS                        = 22.0

# Generate one reproducible initial point directly in the physical Lzzr-pzr space.
# The same seed and the same physical bounds therefore produce exactly the same
# initial physical point for every optimization parameterization.
INITIAL_RNG                   = np.random.default_rng(SEED)
INITIAL_LZZR                  = INITIAL_RNG.uniform(LZZR_BOUNDS[0],LZZR_BOUNDS[1],)
INITIAL_PZR                   = INITIAL_RNG.uniform(PZR_BOUNDS[0],PZR_BOUNDS[1],)
INITIAL_PHYSICAL_VALUES       = {"Lzzr": float(INITIAL_LZZR),"pzr": float(INITIAL_PZR),"intensity_Wcm2": float(INTENSITY_WCM2),"tau_fs": float(TAU_FS),}

# Numerical resolution.
# The temporal resolution is kept approximately fixed in SI units in 2D:
# Nt = ceil(LT_FACTOR * tau / DT_SI)
NR                            = 1024
NZ                            = 256
LR_FACTOR                     = 3.0 # box r-size * waist
LT_FACTOR                     = 3.0 # box time-size * duration
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
# The configuration dictionary gathers all settings required for the optimization in 2D.
# They are passed to the main BO routine, which executes the complete optimization loop.
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Shared physical initial point")
    print("=" * 80)
    print(f"seed             = {SEED}")
    print(f"Lzzr             = {INITIAL_PHYSICAL_VALUES['Lzzr']:.16e}")
    print(f"pzr              = {INITIAL_PHYSICAL_VALUES['pzr']:.16e}")
    print(
        f"intensity_Wcm2   = "
        f"{INITIAL_PHYSICAL_VALUES['intensity_Wcm2']:.16e}"
    )
    print(f"tau_fs           = {INITIAL_PHYSICAL_VALUES['tau_fs']:.16e}")
    print("=" * 80)
    config = {
        "dimension": 2,
        "seed": SEED,
        "n_initial": N_INITIAL,
        "n_iterations": N_ITERATIONS,
        "optimization_space": OPTIMIZATION_SPACE,
        "acquisition": ACQUISITION,
        "kernel": KERNEL,
        "straddle_reference_mode": STRADDLE_REFERENCE_MODE,
        "straddle_reference_maximum_ce": STRADDLE_REFERENCE_MAXIMUM_CE,
        "straddle_percentage": STRADDLE_PERCENTAGE,
        "straddle_beta": STRADDLE_BETA,
        "Lzzr_bounds": LZZR_BOUNDS,
        "pzr_bounds": PZR_BOUNDS,
        "intensity_Wcm2_bounds": (INTENSITY_WCM2, INTENSITY_WCM2,),
        "tau_fs_bounds": (TAU_FS,TAU_FS,),
        # One physical initial point shared by all 2D parameterizations.
        # bo.py will transform this same point into the selected BO coordinates.
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

    run_optimization(config)
