"""Numerical-convergence tests (Appendix A: final / res / box) at 4D conditions, i.e. with
per-point intensity and pulse duration. The tests "final", "res" and "box", the physical
constants and the table formulas are the same as in src/run_numerical_tests.py. "res_nz", "res_dt" and
"res_nr" apply one of the three changes of "res" at a time. The 32 2D Sobol points (ids s00..s31,
I 2.5e14 W/cm^2, tau 22 fs) were run with final and the single-change tests for comparison.

  conv4d.py worker QUEUE HARP_CLONE   pop "<point> <test>" jobs from QUEUE (flock) and run them in
                                      HARP_CLONE (a private HarP copy: BOinput.py / DiagEnergy.h5 are
                                      written there); each result is appended to simulations.csv
  conv4d.py table                     build numerical_convergence_table.csv from simulations.csv

Points (priority order):
  a  4D GP optimum of the paper (v1.0.0): L/z_R 0.0526, p z_R 300, I 3.71e14, tau 11 fs
  b  best sampled point of the seed-0 4D main run ((log p/L, log pL, I, 1/sqrt tau), Straddle;
     results/seed_study/runs/seed0, iteration 24, CE 6.4558577681044e-06)
  c  stress point: L/z_R 0.05, p z_R 300, I 1e15, tau 11 fs
  d  long pulse: L, p, I of point a, tau 88 fs
"""
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
from harp_input import check_harp_installation, run_harp  # noqa: E402

OUT = Path(__file__).resolve().parent
SIMULATIONS_CSV = OUT / "simulations.csv"
TABLE_CSV = OUT / "numerical_convergence_table.csv"
CONVERGENCE_THRESHOLD = 5.0e-8
TEST_ORDER = ("final", "res", "box")

# Same settings as src/run_numerical_tests.py (copied, since importing it would run it)
DT_FINAL_S = 1.2890625e-16
TESTS = {
    "final": {"nr": 1024, "nz": 256, "lr_factor": 3.0, "lt_factor": 3.0, "dt_si": DT_FINAL_S},
    "res": {"nr": 2048, "nz": 2048, "lr_factor": 3.0, "lt_factor": 3.0, "dt_si": DT_FINAL_S / 4.0},
    "box": {"nr": 2048, "nz": 256, "lr_factor": 6.0, "lt_factor": 6.0, "dt_si": DT_FINAL_S},
    # Single-change variants of "res": one of its three changes at a time
    "res_nz": {"nr": 1024, "nz": 2048, "lr_factor": 3.0, "lt_factor": 3.0, "dt_si": DT_FINAL_S},
    "res_dt": {"nr": 1024, "nz": 256, "lr_factor": 3.0, "lt_factor": 3.0, "dt_si": DT_FINAL_S / 4.0},
    "res_nr": {"nr": 2048, "nz": 256, "lr_factor": 3.0, "lt_factor": 3.0, "dt_si": DT_FINAL_S},
}
SINGLE = ("res_nz", "res_dt", "res_nr")
PHYSICS = {"temperature_K": 293.0, "wavelength_m": 0.81e-6, "zR_mm": 18.6, "harmonic_order": 23,
           "objective_scale": 1.0e-6}

POINTS = {
    "a": {"label": "4D GP optimum (paper)", "Lzzr": 0.0526, "pzr": 300.0, "intensity_Wcm2": 3.71e14,
          "tau_fs": 11.0},
    "b": {"label": "best sampled, seed-0 4D main run (iter 24)", "Lzzr": 0.08636716733525729,
          "pzr": 265.01391289296566, "intensity_Wcm2": 401290995251531.9, "tau_fs": 12.47920015790943},
    "c": {"label": "stress: short pulse, high intensity", "Lzzr": 0.05, "pzr": 300.0,
          "intensity_Wcm2": 1.0e15, "tau_fs": 11.0},
    "d": {"label": "long pulse at the optimum's L, p, I", "Lzzr": 0.0526, "pzr": 300.0,
          "intensity_Wcm2": 3.71e14, "tau_fs": 88.0},
}
# 2D comparison: the 32 Sobol points of Appendix A / run_numerical_tests.py, at the
# 2D laser settings (I 2.5e14 W/cm^2, tau 22 fs); point ids s00..s31
for _line in (ROOT / "src" / "sobol_32_points.txt").read_text().splitlines():
    if _line.strip() and not _line.startswith("#"):
        _k, _L, _p = _line.split()
        POINTS[f"s{int(_k):02d}"] = {"label": f"2D Sobol point {int(_k)}", "Lzzr": float(_L), "pzr": float(_p),
                                     "intensity_Wcm2": 2.5e14, "tau_fs": 22.0}


def pop_job(queue_file):
    with open(queue_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = [l for l in f.read().splitlines() if l.strip()]
        if not lines:
            return None
        f.seek(0)
        f.truncate()
        f.write("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""))
        return lines[0]


def append_row(row):
    with open(OUT / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pd.DataFrame([row]).to_csv(SIMULATIONS_CSV, mode="a", header=not SIMULATIONS_CSV.exists(), index=False)


def worker(queue_file, harp):
    paths = check_harp_installation(harp)
    diag_dir = OUT / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    while (job := pop_job(queue_file)) is not None:
        point_id, test = job.split()
        point = POINTS[point_id]
        config = {**TESTS[test], **PHYSICS}
        values = {k: point[k] for k in ("Lzzr", "pzr", "intensity_Wcm2", "tau_fs")}
        archive = diag_dir / f"DiagEnergy_point_{point_id}_{test}.h5"
        print(f"{time.strftime('%F %T')} start {point_id} {test} {values} in {harp}", flush=True)
        r = run_harp(values=values, config=config, paths=paths, archive_path=archive)
        append_row({"point_id": point_id, **values, "test": test, "nr": config["nr"], "nz": config["nz"],
                    "nt": r["nt"], "lr_factor": config["lr_factor"], "lt_factor": config["lt_factor"],
                    "dt_target_s": config["dt_si"], "conversion_efficiency": r["conversion_efficiency"],
                    "harmonic_energy_J": r["harmonic_energy_J"], "input_laser_energy_J": r["input_laser_energy_J"],
                    "objective": r["objective"], "simulation_time_s": r["simulation_time_s"],
                    "harp_workdir": str(paths["workdir"]), "diagnostic_file": archive.name})
        print(f"{time.strftime('%F %T')} done {point_id} {test}: nt {r['nt']}, CE {r['conversion_efficiency']:.8e}, "
              f"{r['simulation_time_s']:.0f} s", flush=True)


def table():
    sims = pd.read_csv(SIMULATIONS_CSV)
    rows = []
    for point_id, g in sims.groupby("point_id"):
        ce = {t: g.loc[g.test == t, "conversion_efficiency"].iloc[-1] for t in TEST_ORDER if (g.test == t).any()}
        p = POINTS[point_id]
        row = {"point_id": point_id, "label": p["label"], "Lzzr": p["Lzzr"], "pzr": p["pzr"],
               "intensity_Wcm2": p["intensity_Wcm2"], "tau_fs": p["tau_fs"],
               "CE_final": ce.get("final"), "CE_res": ce.get("res"), "CE_box": ce.get("box")}
        if "final" in ce and "res" in ce:  # relative_res = delta / CE_res, as run_numerical_tests.py
            row["delta_CE_res"] = abs(ce["final"] - ce["res"])
            row["relative_res_percent"] = row["delta_CE_res"] / abs(ce["res"]) * 100.0
            row["res_below_threshold"] = row["delta_CE_res"] < CONVERGENCE_THRESHOLD
        if "final" in ce and "box" in ce:  # relative_box = delta / CE_final, as run_numerical_tests.py
            row["delta_CE_box"] = abs(ce["final"] - ce["box"])
            row["relative_box_percent"] = row["delta_CE_box"] / abs(ce["final"]) * 100.0
            row["box_below_threshold"] = row["delta_CE_box"] < CONVERGENCE_THRESHOLD
        for test in SINGLE:  # relative to the variant's own CE, as relative_res
            ce_t = g.loc[g.test == test, "conversion_efficiency"]
            if "final" in ce and len(ce_t):
                row[f"CE_{test}"] = ce_t.iloc[-1]
                row[f"delta_CE_{test}"] = abs(ce["final"] - ce_t.iloc[-1])
                row[f"relative_{test}_percent"] = row[f"delta_CE_{test}"] / abs(ce_t.iloc[-1]) * 100.0
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(TABLE_CSV, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.drop(columns=["label"]).to_string(index=False, float_format="%.4g"))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    settings = {"script": "results/numerical_convergence_4d/conv4d.py (wrapper around harp_input.run_harp)",
                "tests_as_in": "src/run_numerical_tests.py", "tests": TESTS, "physics": PHYSICS,
                "points": POINTS, "convergence_threshold": CONVERGENCE_THRESHOLD,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset")}
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2))
    if sys.argv[1] == "worker":
        worker(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        table()
