"""2D comparison table: the 32 Sobol points of the 2D numerical convergence tests (I 2.5e14, tau 22 fs).
final / res_nz / res_dt / res_nr from simulations.csv here (ids s00..s31); res and box from
../numerical_convergence/simulations.csv (the 2D tests; their final values agree with ours to 4e-13).
Writes numerical_convergence_table_2d.csv and prints a summary."""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
s = pd.read_csv(OUT / "simulations.csv")
s2 = s[s.point_id.str.startswith("s")].copy()
s2["point_index"] = s2.point_id.str[1:].astype(int)
ce = s2.pivot_table(index="point_index", columns="test", values="conversion_efficiency")
ref = pd.read_csv(OUT.parent / "numerical_convergence" / "simulations.csv")
rce = ref.pivot_table(index="point_index", columns="test", values="conversion_efficiency")
pts = ref.groupby("point_index")[["Lzzr", "pzr"]].first()
t = pts.copy()
t["CE_final"] = ce["final"]
t["final_2d_tests_vs_here_rel"] = rce["final"] / ce["final"] - 1  # same HarP, two machines (2D tests on a Mac, these on Linux)
for test, col in (("res", rce["res"]), ("box", rce["box"]), ("res_nz", ce["res_nz"]), ("res_dt", ce["res_dt"]),
                  ("res_nr", ce["res_nr"])):
    t[f"CE_{test}"] = col
    t[f"delta_CE_{test}"] = (t.CE_final - col).abs()
    denom = t.CE_final if test == "box" else col  # as run_numerical_tests.py: res relative to CE_res, box to CE_final
    t[f"relative_{test}_percent"] = t[f"delta_CE_{test}"] / denom.abs() * 100
    t[f"signed_{test}_percent"] = (col - t.CE_final) / denom.abs() * 100
    t[f"{test}_below_5e-8"] = t[f"delta_CE_{test}"] < 5e-8
t.reset_index().to_csv(OUT / "numerical_convergence_table_2d.csv", index=False)

print("max |final of the 2D tests / final here - 1|:", t.final_2d_tests_vs_here_rel.abs().max())
for test in ("res", "res_nz", "res_dt", "res_nr", "box"):
    d, r = t[f"delta_CE_{test}"], t[f"signed_{test}_percent"]
    print(f"{test:7s} max |dCE| {d.max():.2e}  median {d.median():.2e}  pass {int(t[f'{test}_below_5e-8'].sum())}/32  "
          f"signed rel %: median {r.median():+.2f} min {r.min():+.2f} max {r.max():+.2f}")
hi = t.nlargest(5, "CE_final")
print(hi[["Lzzr", "pzr", "CE_final"] + [f"signed_{x}_percent" for x in ("res", "res_nz", "res_dt", "res_nr", "box")]]
      .to_string(float_format="%.3g"))
