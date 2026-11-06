"""
Seed study (v1.0.0 runs, seeds 0-4): best observed CE, NRMSE and SSIM per BO
iteration, per seed, and mean +- std over the seeds.

Reference data (same as the paper figures):
  2D: the direct grid scan data/scan_results.csv (NRMSE/SSIM of the GP posterior
      mean on the scan grid, as plot_fig1_fig2_paper.py / plot_straddle.py);
  4D: the 1024-point Sobol holdout data/holdout_4d/holdout.csv (as
      plot_4Dbo_convergence.py).
Runs are read from results/seed_study/runs/seed<k>/<run>/ (data.csv and models/; the models/
folders are in the raw-data archive of the release, see results/seed_study/README.md).

Writes next to this script:
  seed_study_curves.csv       every iteration, run and seed
  seed_study_per_seed.csv     the iterations the paper uses (2D: 60; 4D: 30, 60, 120)
  seed_study_summary.csv      mean and std (ddof = 1) over the seeds at those iterations
  seed_study_{2d,4d}_convergence.{pdf,png}   draft curves (not used in the paper)

Usage (from the repository root): PYTHONPATH=src python results/seed_study/seed_study_curves.py
"""
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize

from utils import Lp_to_Xnorm, get_objective_transform, global_ssim, make_kernel, normalized_rmse, physical_to_x

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.simplefilter("ignore")
dtype = torch.double
OUT = Path(__file__).resolve().parent
RUNS = OUT / "runs"
ROOT = OUT.parent.parent
SEEDS = range(5)
BOUNDS = {"Lzzr_bounds": (0.015, 1.5), "pzr_bounds": (3.0, 300.0),
          "intensity_Wcm2_bounds": (0.01e14, 10e14), "tau_fs_bounds": (11.0, 88.0)}

CONFIGS = {
    "2D LogEI": [("bo_2d_L__p_log_ei_matern52", "L, p"), ("bo_2d_log_L__log_p_log_ei_matern52", "log L, log p"),
                 ("bo_2d_p_over_L__pL_log_ei_matern52", "p/L, pL"),
                 ("bo_2d_sqrt_p_over_L__sqrt_pL_log_ei_matern52", "√(p/L), √(pL)"),
                 ("bo_2d_log_p_over_L__log_pL_log_ei_matern52", "log(p/L), log(pL)")],
    "2D Straddle": [(f"bo_2d_log_p_over_L__log_pL_straddle{p}per_max3_matern52", f"target {p}% of 3e-6")
                    for p in (25, 50, 90)],
    "4D": [("bo_4d_L__p__I__tau_straddle100per_max3_matern52", "naive (L, p, I, τ)"),
           ("bo_4d_log_p_over_L__log_pL__I__tau_straddle100per_max3_matern52", "mapping, τ"),
           ("bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle100per_max3_matern52", "mapping, 1/√τ, CE"),
           ("bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle100per_max3_matern52_sqrtCE",
            "mapping, 1/√τ, √CE")],
}
N_MAX = {"2D LogEI": 60, "2D Straddle": 60, "4D": 120}
N_PAPER = {"2D LogEI": (60,), "2D Straddle": (60,), "4D": (30, 60, 120)}

grid = pd.read_csv(ROOT / "data" / "scan_results.csv")
if "objective_scaled_ratio" not in grid.columns:
    grid["objective_scaled_ratio"] = grid["ratio_order23_over_input_laser"] / 1e-6
L_axis, p_axis = np.sort(grid["Lzzr"].unique()), np.sort(grid["pzr"].unique())
L_grid, p_grid = np.meshgrid(L_axis, p_axis, indexing="ij")
Z_direct = np.full(L_grid.shape, np.nan)
for _, r in grid.iterrows():
    Z_direct[np.where(np.isclose(L_axis, r["Lzzr"]))[0][0],
             np.where(np.isclose(p_axis, r["pzr"]))[0][0]] = r["objective_scaled_ratio"]
holdout = pd.read_csv(ROOT / "data" / "holdout_4d" / "holdout.csv")
G_ref = holdout["conversion_efficiency"].values / 1e-6
sobol_range = float(G_ref.max() - G_ref.min())


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {**BOUNDS, **ck.get("config", {})}
    dim = int(cfg.get("dimension", ck["X_normalized"].shape[1]))
    f_fwd, f_inv = get_objective_transform(cfg.get("objective_transform") or "id")
    X = ck["X_normalized"].to(dtype)
    Y = f_fwd(torch.clamp(ck["Y_objective"].to(dtype), min=1e-15))
    m = SingleTaskGP(X, Y, covar_module=make_kernel(cfg.get("kernel", "matern52"), dim),
                     outcome_transform=Standardize(m=1))
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, cfg, f_inv


def predict(m, X, f_inv):
    with torch.no_grad():
        return np.asarray(f_inv(m.posterior(torch.as_tensor(X, dtype=dtype).unsqueeze(-2)).mean.reshape(-1).numpy()))


def curves(rd, n_max, dim):
    d = pd.read_csv(rd / "data.csv")
    best = d.sort_values("iteration").set_index("iteration").conversion_efficiency.cummax()
    rows = []
    Xh = None
    for it in range(1, n_max + 1):
        path = rd / "models" / f"model_iter_{it:03d}.pt"
        if not path.exists():
            break
        m, cfg, f_inv = load_model(path)
        if dim == 2:
            Xg = Lp_to_Xnorm(L_grid, p_grid, cfg["optimization_space"]).reshape(-1, 2)
            valid = np.all((Xg >= 0) & (Xg <= 1), axis=1)
            Z = np.full(len(Xg), np.nan)
            Z[valid] = predict(m, Xg[valid], f_inv)
            Z = Z.reshape(L_grid.shape)
            nr, ss = normalized_rmse(Z_direct, Z), global_ssim(Z_direct, Z)
        else:
            if Xh is None:
                Xh = torch.stack([physical_to_x({k: r[k] for k in ("Lzzr", "pzr", "intensity_Wcm2", "tau_fs")}, cfg)
                                  for _, r in holdout.iterrows()]).numpy()
            Zh = predict(m, Xh, f_inv)
            nr, ss = normalized_rmse(G_ref, Zh, ref_range=sobol_range), global_ssim(G_ref, Zh)
        rows.append({"N": it, "best_ce": best.loc[:it].max(), "nrmse": nr, "ssim": ss})
    return pd.DataFrame(rows)


all_rows = []
for group, runs in CONFIGS.items():
    dim = 4 if group == "4D" else 2
    for run, lab in runs:
        for seed in SEEDS:
            c = curves(RUNS / f"seed{seed}" / run, N_MAX[group], dim)
            assert c.N.max() == N_MAX[group], (run, seed, c.N.max())
            c["group"], c["config"], c["run"], c["seed"] = group, lab, run, seed
            all_rows.append(c)
            print(f"{group:11s} {lab:22s} seed {seed}: {len(c)} checkpoints", flush=True)
df = pd.concat(all_rows)[["group", "config", "run", "seed", "N", "best_ce", "nrmse", "ssim"]]
df.to_csv(OUT / "seed_study_curves.csv", index=False)

per_seed = pd.concat([df[(df.group == g) & df.N.isin(ns)] for g, ns in N_PAPER.items()])
per_seed.to_csv(OUT / "seed_study_per_seed.csv", index=False)
summary = (per_seed.groupby(["group", "config", "run", "N"], sort=False)[["best_ce", "nrmse", "ssim"]]
           .agg(["mean", "std", "min", "max"]))
summary.columns = [f"{m}_{s}" for m, s in summary.columns]
summary.insert(0, "n_seeds", per_seed.groupby(["group", "config", "run", "N"], sort=False).size())
summary.reset_index().to_csv(OUT / "seed_study_summary.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(summary[[c for c in summary.columns if c.endswith(("mean", "std"))]].to_string(float_format="%.4g"))

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = [("best_ce", "best observed CE (×10⁻⁶)", 1e6), ("nrmse", "NRMSE", 1), ("ssim", "SSIM", 1)]


def panel(ax, group, metric, scale, show_legend):
    for i, (_, lab) in enumerate(CONFIGS[group]):
        g = df[(df.group == group) & (df.config == lab)].groupby("N")[metric]
        mu, sd = g.mean() * scale, g.std(ddof=1) * scale
        ax.fill_between(mu.index, mu - sd, mu + sd, color=SERIES[i], alpha=0.15, linewidth=0)
        ax.plot(mu.index, mu, color=SERIES[i], linestyle=STYLES[i], linewidth=2, label=lab,
                marker=MARKERS[i], markevery=max(1, N_MAX[group] // 6), markersize=5)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlim(1, N_MAX[group])
    if show_legend:
        ax.legend(fontsize=8.5, frameon=False, labelcolor=TEXT, loc="lower right" if metric != "nrmse" else "upper right")


def figure(groups, name, title):
    fig, axes = plt.subplots(len(METRICS), len(groups), figsize=(5.2 * len(groups), 9), squeeze=False, sharex="col")
    for j, group in enumerate(groups):
        axes[0, j].set_title(group, color=TEXT, fontsize=12)
        for i, (metric, ylab, scale) in enumerate(METRICS):
            panel(axes[i, j], group, metric, scale, show_legend=(i == 0))
            if j == 0:
                axes[i, j].set_ylabel(ylab, color=TEXT, fontsize=10)
        axes[-1, j].set_xlabel("BO iteration", color=TEXT, fontsize=10)
    fig.suptitle(title, color=TEXT, fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


figure(["2D LogEI", "2D Straddle"], "seed_study_2d_convergence",
       "v1.0.0, 2D: mean ± std over seeds 0-4 (draft, not a paper figure)")
figure(["4D"], "seed_study_4d_convergence",
       "v1.0.0, 4D, N = 120: mean ± std over seeds 0-4 (draft, not a paper figure)")
print("wrote", *sorted(p.name for p in OUT.iterdir() if p.suffix != ".py"))
