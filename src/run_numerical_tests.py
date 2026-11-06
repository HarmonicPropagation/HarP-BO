"""
Run the numerical-convergence study for the 32 Sobol points.
For each point:
  final : production numerical setup
  res   : higher numerical resolution at the same physical box size
  box   : doubled radial and temporal box at the production discretization
The same harp_input.run_harp() pathway as the BO calculations is used.
Results are saved after every simulation so an interrupted run can be resumed.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from harp_input import check_harp_installation, run_harp

# Physical and numerical settings
# =============================================================================

HARP_WORKDIR          = Path("~/HarP-main").expanduser().resolve()
INTENSITY_WCM2        = 2.5e14
TAU_FS                = 22.0
TEMPERATURE_K         = 293.0
WAVELENGTH_M          = 0.81e-6
ZR_MM                 = 18.6
HARMONIC_ORDER        = 23
OBJECTIVE_SCALE       = 1.0e-6
DT_FINAL_S            = 1.2890625e-16
CONVERGENCE_THRESHOLD = 5.0e-8

TESTS = {
    "final": {
        "nr": 1024,
        "nz": 256,
        "lr_factor": 3.0,
        "lt_factor": 3.0,
        "dt_si": DT_FINAL_S,
    },
    "res": {
        "nr": 2048,
        "nz": 2048,
        "lr_factor": 3.0,
        "lt_factor": 3.0,
        "dt_si": DT_FINAL_S / 4.0,
    },
    "box": {
        "nr": 2048,
        "nz": 256,
        "lr_factor": 6.0,
        "lt_factor": 6.0,
        "dt_si": DT_FINAL_S,
    },
}

TEST_ORDER      = ("final", "res", "box")
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
POINTS_FILE     = Path(__file__).resolve().parent / "sobol_32_points.txt"
RESULTS_DIR     = PROJECT_ROOT / "results" / "numerical_convergence"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
SIMULATIONS_CSV = RESULTS_DIR / "simulations.csv"
TABLE_CSV       = RESULTS_DIR / "numerical_convergence_table.csv"
SETTINGS_JSON   = RESULTS_DIR / "settings.json"

# Use None for all 32 points, or e.g. 3 for a short test.
MAX_POINTS      = None

# Small helper functions
# =============================================================================
def build_config(test_name):
    test = TESTS[test_name]
    return {
        "nr": test["nr"],
        "nz": test["nz"],
        "lr_factor": test["lr_factor"],
        "lt_factor": test["lt_factor"],
        "dt_si": test["dt_si"],
        "temperature_K": TEMPERATURE_K,
        "wavelength_m": WAVELENGTH_M,
        "zR_mm": ZR_MM,
        "harmonic_order": HARMONIC_ORDER,
        "objective_scale": OBJECTIVE_SCALE,
    }

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"

def save_simulations(df):
    df = df.sort_values(["point_index", "test"]).reset_index(drop=True)
    df.to_csv(SIMULATIONS_CSV, index=False)

def build_appendix_table(simulations):
    rows = []
    if simulations.empty:
        return pd.DataFrame()
    for point_index in sorted(simulations["point_index"].unique()):
        point_df = simulations[simulations["point_index"] == point_index]
        if not set(TEST_ORDER).issubset(set(point_df["test"])):
            continue

        ce_final = point_df.loc[
            point_df["test"] == "final", "conversion_efficiency"
        ].iloc[-1]
        ce_res = point_df.loc[
            point_df["test"] == "res", "conversion_efficiency"
        ].iloc[-1]
        ce_box = point_df.loc[
            point_df["test"] == "box", "conversion_efficiency"
        ].iloc[-1]

        first = point_df.iloc[0]
        delta_res = abs(ce_final - ce_res)
        delta_box = abs(ce_final - ce_box)

        rows.append(
            {
                "point_index": int(point_index),
                "Lzzr": float(first["Lzzr"]),
                "pzr": float(first["pzr"]),
                "CE_final": ce_final,
                "CE_res": ce_res,
                "delta_CE_res": delta_res,
                "relative_res_percent": delta_res / abs(ce_res) * 100.0,
                "CE_box": ce_box,
                "delta_CE_box": delta_box,
                "relative_box_percent": delta_box / abs(ce_final) * 100.0,
            }
        )

    return pd.DataFrame(rows)


def print_appendix_table(table):
    print("\n" + "=" * 112)
    print("APPENDIX TABLE — completed Sobol points")
    print("All CE columns below are in units of 1e-8.")
    print("=" * 112)
    header = (
        f"{'Point':>5s} "
        f"{'L [z_R]':>9s} "
        f"{'p z_R [mbar cm]':>16s} "
        f"{'CE_final':>10s} "
        f"{'CE_res':>10s} "
        f"{'Delta CE_res':>13s} "
        f"{'CE_box':>10s} "
        f"{'Delta CE_box':>13s}"
    )
    print(header)
    print("-" * len(header))
    for _, row in table.sort_values("point_index").iterrows():
        scale = 1.0e8
        print(
            f"{int(row['point_index']):5d} "
            f"{row['Lzzr']:9.2f} "
            f"{row['pzr']:16.2f} "
            f"{row['CE_final'] * scale:10.2f} "
            f"{row['CE_res'] * scale:10.2f} "
            f"{row['delta_CE_res'] * scale:13.2f} "
            f"{row['CE_box'] * scale:10.2f} "
            f"{row['delta_CE_box'] * scale:13.2f}"
        )
    print("=" * 112)


def print_summary(table, total_points):
    if table.empty:
        return
    worst_res = table.loc[table["delta_CE_res"].idxmax()]
    worst_box = table.loc[table["delta_CE_box"].idxmax()]
    print("\nCONVERGENCE SUMMARY")
    print(f"Completed points: {len(table)}/{total_points}")
    print(
        f"Max resolution difference: {worst_res['delta_CE_res']:.8e} "
        f"(point {int(worst_res['point_index'])}, "
        f"{worst_res['relative_res_percent']:.6f} %)"
    )
    print(
        f"Max box difference:        {worst_box['delta_CE_box']:.8e} "
        f"(point {int(worst_box['point_index'])}, "
        f"{worst_box['relative_box_percent']:.6f} %)"
    )
    print(f"Threshold:                 {CONVERGENCE_THRESHOLD:.8e}")


# Setup
# =============================================================================
if not POINTS_FILE.exists():
    raise FileNotFoundError(f"Sobol points file not found: {POINTS_FILE}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
paths      = check_harp_installation(HARP_WORKDIR)
points_raw = np.loadtxt(POINTS_FILE, comments="#", ndmin=2)
points     = pd.DataFrame(
    {
        "point_index": points_raw[:, 0].astype(int),
        "Lzzr": points_raw[:, 1].astype(float),
        "pzr": points_raw[:, 2].astype(float),
    }
)

if MAX_POINTS is not None:
    points = points.iloc[:MAX_POINTS].copy()

if SIMULATIONS_CSV.exists():
    simulations = pd.read_csv(SIMULATIONS_CSV)
    simulations["point_index"] = simulations["point_index"].astype(int)
    print(f"Resuming from {SIMULATIONS_CSV}")
else:
    simulations = pd.DataFrame()

settings = {
    "points_file": str(POINTS_FILE),
    "harp_workdir": str(HARP_WORKDIR),
    "intensity_Wcm2": INTENSITY_WCM2,
    "tau_fs": TAU_FS,
    "temperature_K": TEMPERATURE_K,
    "wavelength_m": WAVELENGTH_M,
    "zR_mm": ZR_MM,
    "harmonic_order": HARMONIC_ORDER,
    "objective_scale": OBJECTIVE_SCALE,
    "convergence_threshold": CONVERGENCE_THRESHOLD,
    "tests": TESTS,
}
SETTINGS_JSON.write_text(json.dumps(settings, indent=2))
print(f"\nRunning {len(points)} Sobol points")
print(f"Results: {RESULTS_DIR}")


# Run final, resolution, and box tests
# =============================================================================
for sequence_number, point in enumerate(points.itertuples(index=False), start=1):
    print("\n" + "#" * 92)
    print(
        f"SOBOL POINT {sequence_number}/{len(points)} | "
        f"L/z_R={point.Lzzr:.8f} | p z_R={point.pzr:.8f} mbar cm"
    )
    print("#" * 92)

    for test_name in TEST_ORDER:
        already_done = (
            not simulations.empty
            and (
                (simulations["point_index"] == int(point.point_index))
                & (simulations["test"] == test_name)
            ).any()
        )

        if already_done:
            saved = simulations[
                (simulations["point_index"] == int(point.point_index))
                & (simulations["test"] == test_name)
            ].iloc[-1]
            print(
                f"Skipping {test_name.upper()}: "
                f"CE={saved['conversion_efficiency']:.8e}"
            )
            continue

        config = build_config(test_name)
        values = {
            "Lzzr": float(point.Lzzr),
            "pzr": float(point.pzr),
            "intensity_Wcm2": INTENSITY_WCM2,
            "tau_fs": TAU_FS,
        }

        archive_path = DIAGNOSTICS_DIR / (
            f"DiagEnergy_point_{int(point.point_index):02d}_{test_name}.h5"
        )

        print(
            f"\n{test_name.upper()}: "
            f"Nr={config['nr']}, Nz={config['nz']}, "
            f"Lr={config['lr_factor']:g} w0, "
            f"Tsim={config['lt_factor']:g} tau, "
            f"target dt={config['dt_si'] * 1e18:.6f} as"
        )

        result = run_harp(
            values=values,
            config=config,
            paths=paths,
            archive_path=archive_path,
        )

        print(
            f"Finished {test_name.upper()}: "
            f"Nt={result['nt']}, "
            f"CE={result['conversion_efficiency']:.8e}, "
            f"wall time={format_time(result['simulation_time_s'])}"
        )

        row = {
            "point_index": int(point.point_index),
            "Lzzr": float(point.Lzzr),
            "pzr": float(point.pzr),
            "test": test_name,
            "nr": config["nr"],
            "nz": config["nz"],
            "nt": result["nt"],
            "lr_factor": config["lr_factor"],
            "lt_factor": config["lt_factor"],
            "dt_target_s": config["dt_si"],
            "conversion_efficiency": result["conversion_efficiency"],
            "harmonic_energy_J": result["harmonic_energy_J"],
            "input_laser_energy_J": result["input_laser_energy_J"],
            "objective": result["objective"],
            "simulation_time_s": result["simulation_time_s"],
            "diagnostic_file": str(archive_path),
        }

        simulations = pd.concat(
            [simulations, pd.DataFrame([row])],
            ignore_index=True,
        )

        # Save immediately after every HarP simulation.
        save_simulations(simulations)
        print(f"Saved: {SIMULATIONS_CSV}")

    # After all three tests for this point, update and print the appendix table.
    table = build_appendix_table(simulations)
    table.to_csv(TABLE_CSV, index=False)
    print_appendix_table(table)
    print_summary(table, len(points))


# Final report
# =============================================================================

table = build_appendix_table(simulations)
table.to_csv(TABLE_CSV, index=False)
print("\n" + "#" * 92)
print("NUMERICAL CONVERGENCE CAMPAIGN FINISHED")
print("#" * 92)
print(f"Raw simulations:     {SIMULATIONS_CSV}")
print(f"Appendix table:      {TABLE_CSV}")
print(f"Archived diagnostics:{DIAGNOSTICS_DIR}")
print_appendix_table(table)
print_summary(table, len(points))

