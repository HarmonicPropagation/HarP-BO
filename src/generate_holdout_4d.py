"""
Generate a static Sobol hold-out data set for the 4D optimization.

The output consists of 1024 Sobol points covering the normalized BO search
space together with their corresponding HarP evaluations.

The resulting CSV is later used for evaluating GP convergence (NRMSE and SSIM)
without requiring an exhaustive 4D grid scan.
"""

from pathlib import Path
import time

import pandas as pd
import torch
from scipy.stats import qmc

from harp_input import check_harp_installation, run_harp
from utils import x_to_physical

# Import the existing 4D configuration
from run_bo_4d import (
    SEED,
    HARP_WORKDIR,
    LZZR_BOUNDS,
    PZR_BOUNDS,
    INTENSITY_WCM2_BOUNDS,
    TAU_FS_BOUNDS,
    NR,
    NZ,
    LR_FACTOR,
    LT_FACTOR,
    DT_SI,
    TEMPERATURE_K,
    WAVELENGTH_M,
    ZR_MM,
    HARMONIC_ORDER,
    OBJECTIVE_SCALE,
    RESULTS_ROOT,
)

OPTIMIZATION_SPACE = "L__p__I__tau"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

################################################################################
# Configuration
################################################################################

M = 10
N_POINTS = 2**M

config = {
    "dimension": 4,
    "seed": SEED,
    "optimization_space": OPTIMIZATION_SPACE,
    "Lzzr_bounds": LZZR_BOUNDS,
    "pzr_bounds": PZR_BOUNDS,
    "intensity_Wcm2_bounds": INTENSITY_WCM2_BOUNDS,
    "tau_fs_bounds": TAU_FS_BOUNDS,
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
}


################################################################################

output_folder = PROJECT_ROOT / "data" / "holdout_4d"
diag_folder = output_folder / "diagnostics"

output_folder.mkdir(parents=True, exist_ok=True)
diag_folder.mkdir(exist_ok=True)

csv_file = output_folder / "holdout.csv"

paths = check_harp_installation(HARP_WORKDIR)

sampler = qmc.Sobol(
    d=4,
    scramble=True,
    seed=SEED,
)

X = sampler.random_base2(m=M)

rows = []

start_time = time.perf_counter()

for i, x in enumerate(X):

    x_tensor = torch.tensor(x, dtype=torch.double)

    values = x_to_physical(x_tensor, config)

    archive_path = diag_folder / f"DiagEnergy_{i:04d}.h5"

    result = run_harp(
        values=values,
        config=config,
        paths=paths,
        archive_path=archive_path,
    )

    elapsed = time.perf_counter() - start_time
    eta = elapsed / (i + 1) * (N_POINTS - i - 1)

    print(
        f"[{i+1:4d}/{N_POINTS}] "
        f"CE={result['conversion_efficiency']:.3e}   "
        f"elapsed={elapsed/3600:.2f} h   "
        f"ETA={eta/3600:.2f} h"
    )

    row = {
        "sobol_index": i,
        "x1_norm": float(x[0]),
        "x2_norm": float(x[1]),
        "x3_norm": float(x[2]),
        "x4_norm": float(x[3]),
        "Lzzr": values["Lzzr"],
        "pzr": values["pzr"],
        "pL": values["pzr"] * values["Lzzr"],
        "p_over_L": values["pzr"] / values["Lzzr"],
        "intensity_Wcm2": values["intensity_Wcm2"],
        "tau_fs": values["tau_fs"],
        "objective_scaled": result["objective"],
        "conversion_efficiency": result["conversion_efficiency"],
        "harmonic_energy_J": result["harmonic_energy_J"],
        "input_laser_energy_J": result["input_laser_energy_J"],
        "simulation_time_s": result["simulation_time_s"],
        "nt": result["nt"],
        "diag_file": archive_path.name,
    }

    rows.append(row)

    pd.DataFrame(rows).to_csv(csv_file, index=False)

print()
print(f"Finished {N_POINTS} HarP simulations.")
print(f"Results written to")
print(csv_file)
