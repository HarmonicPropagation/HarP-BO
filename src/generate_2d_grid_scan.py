"""
Generate the 100 x 100 direct grid scan used in 2D plots.
The HarP physics and numerical implementation are identical to the 2D BO.
Only the choice of evaluation points is different: here all 10,000 points
of the fixed L/z_R, p*z_R grid are evaluated, for comparison with the 
BO landscape.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from harp_input import check_harp_installation, run_harp


# Paths
########################################################################################################
HARP_WORKDIR = Path("~/HarP-main").expanduser().resolve()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV   = PROJECT_ROOT / "data" / "scan_results.csv"


# Physical parameters, identical to the 2D BO
########################################################################################################
INTENSITY_WCM2  = 2.5e14
TAU_FS          = 22.0
TEMPERATURE_K   = 293.0
WAVELENGTH_M    = 0.81e-6
ZR_MM           = 18.6
HARMONIC_ORDER  = 23
OBJECTIVE_SCALE = 1.0e-6

# Numerical parameters -- identical to the 2D BO
########################################################################################################
NR              = 1024
NZ              = 256
LR_FACTOR       = 3.0
LT_FACTOR       = 3.0
DT_SI           = 1.2890625e-16

# 100x100 grid 
########################################################################################################
N_L             = 100
N_P             = 100
L_MAX           = 1.5
P_MAX           = 300.0
L_AXIS          = np.arange(1, N_L+1) * (L_MAX/N_L)
P_AXIS          = np.arange(1, N_P+1) * (P_MAX/N_P)

# HarP config
########################################################################################################
CONFIG = {
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
}


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    paths = check_harp_installation(HARP_WORKDIR)
    # Resume an interrupted scan.
    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        completed = set(df["point_index"].astype(int))
        print(f"Resuming: {len(completed)} / {N_L * N_P} points already completed")
    else:
        completed = set()
    total_points = N_L*N_P
    for i_L, Lzzr in enumerate(L_AXIS):
        for i_p, pzr in enumerate(P_AXIS):
            point_index = i_L*N_P + i_p
            if point_index in completed:
                continue
            simulation_number = point_index + 1
            print(
                f"\n[{simulation_number}/{total_points}] "
                f"Lzzr={Lzzr:.6f}, pzr={pzr:.6f}"
            )
            values     = {
                "Lzzr": float(Lzzr),
                "pzr": float(pzr),
                "intensity_Wcm2": INTENSITY_WCM2,
                "tau_fs": TAU_FS,
            }
            start       = time.perf_counter()
            result      = run_harp(values=values,config=CONFIG,paths=paths)
            wall_time_s = time.perf_counter() - start
            row         = {
                "point_index": point_index,
                "simulation_number": simulation_number,
                "i_L": i_L,
                "i_p": i_p,
                "Lzzr": Lzzr,
                "pzr": pzr,
                "pL": pzr * Lzzr,
                "p_over_L": pzr / Lzzr,
                "cost": result["objective"],
                "ratio_order23_over_input_laser": result["conversion_efficiency"],
                "objective_scaled_ratio": result["objective"],
                "input_laser_energy": result["input_laser_energy_J"],
                "final_order_23": result["harmonic_energy_J"],
                "wall_time_s": wall_time_s,
                "diag_file": "DiagEnergy.h5",
            }
            pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", header=not OUTPUT_CSV.exists(), index=False)
    print("\nGrid scan complete.")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
