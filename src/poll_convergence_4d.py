import time
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
import matplotlib

# Import the shared metrics
from utils import normalized_rmse, global_ssim, get_objective_transform, make_kernel

matplotlib.use("TkAgg")  # Ensures interactive GUI windows work properly across platforms

# Settings & Paths
####################################################################################################################################
dtype = torch.double
dim = 4  # 4D BO dimensions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
results_root = PROJECT_ROOT / "results"
# RESULTS_ROOT = PROJECT_ROOT / "data" / "archived_runs"

# Adjust the run folder name to match the active 4D optimization run, or
# override with the RUN_DIR env var, e.g.:
#   RUN_DIR=results/bo_4d_..._straddle75per_gp_matern52_sqrtCE python src/poll_convergence_4d.py
_DEFAULT_RUN_DIR = results_root / "bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_log_ei_matern52_sqrtCE"
# _DEFAULT_RUN_DIR = results_root / "bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle75per_gp_matern52_sqrtCE"
# _DEFAULT_RUN_DIR = results_root / "bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle50per_gp_matern52_sqrtCE"
# _DEFAULT_RUN_DIR = results_root / "bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle100per_max3_matern52_sqrtCE"
# _DEFAULT_RUN_DIR = results_root / "bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle100per_max3_matern52_logCE"
# _DEFAULT_RUN_DIR = results_root / "bo_4d_L__p__I__tau_straddle100per_max3_matern52"
RUN_DIR = Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else _DEFAULT_RUN_DIR
DATA_CSV = RUN_DIR / "data.csv"
MODELS_DIR = RUN_DIR / "models"
HOLDOUT_CSV = PROJECT_ROOT / "data" / "holdout_4d" / "holdout.csv"

# Style configuration
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.5,
})

# Helper Functions
####################################################################################################################################
def evaluate_checkpoints(models_dir, X_test, G_ref, sobol_range, dtype, dim):
    """Iterates over all saved GP checkpoints and calculates convergence metrics."""
    metric_rows = []
    if not models_dir.exists() or X_test is None:
        return pd.DataFrame(metric_rows)

    model_paths = sorted(glob.glob(str(models_dir / "model_iter_*.pt")))
    for path in model_paths:
        iteration = int(os.path.basename(path).replace("model_iter_", "").replace(".pt", ""))

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            X_train = checkpoint["X_normalized"].to(dtype=dtype)
            Y_obj = checkpoint["Y_objective"].to(dtype=dtype)

            # Resolve the objective transform (with backwards compatibility for older checkpoints)
            if checkpoint.get("use_log_objective", False):
                transform_name = "log10"
            else:
                transform_name = checkpoint.get("config", {}).get("objective_transform", "id")
            f_fwd, f_inv = get_objective_transform(transform_name)

            Y_obj = f_fwd(torch.clamp(Y_obj, min=1e-15))

            # Reconstruct model dynamically
            kernel = make_kernel("matern52", dim)
            model = SingleTaskGP(
                train_X=X_train,
                train_Y=Y_obj,
                covar_module=kernel,
                outcome_transform=Standardize(m=1)
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            with torch.no_grad():
                posterior = model.posterior(X_test)
                Z_pred = posterior.mean.cpu().numpy().reshape(-1)

            # Invert the objective transform for physical scale comparison
            Z_pred = f_inv(Z_pred)

            metric_rows.append({
                "iteration": iteration,
                "nrmse": normalized_rmse(G_ref, Z_pred, ref_range=sobol_range),
                "ssim": global_ssim(G_ref, Z_pred),
            })
        except Exception as e:
            print(f"Error evaluating checkpoint {Path(path).name}: {e}")
            continue

    df = pd.DataFrame(metric_rows)
    if not df.empty:
        df = df.sort_values("iteration")
    return df


def update_dashboard(axes, fig, bo_df, metrics_df):
    """Clears and redraws the 3-panel matplotlib figure."""
    ax_bo, ax_nrmse, ax_ssim = axes
    ax_bo.clear()
    ax_nrmse.clear()
    ax_ssim.clear()

    # 1) Best-so-far CE
    y_best = bo_df["objective_scaled"].cummax().to_numpy()
    ax_bo.plot(bo_df["iteration"], y_best, "r-", linewidth=1.8, label="Best-so-far BO")
    ax_bo.scatter(bo_df["iteration"], bo_df["objective_scaled"], c="blue", s=12, alpha=0.6, label="Evaluated points")
    ax_bo.set_ylabel(r"Best BO CE / $10^{-6}$")
    ax_bo.set_title(f"4D BO Live Convergence (Latest Iteration: {bo_df['iteration'].max()})")
    ax_bo.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.8)
    ax_bo.set_ylim(bottom=0)

    # 2) NRMSE vs Sobol Holdout
    if not metrics_df.empty:
        ax_nrmse.plot(metrics_df["iteration"], metrics_df["nrmse"], "b-o", markersize=3, linewidth=1.5)
    ax_nrmse.axhline(0.0, color="0.7", linestyle="dotted", linewidth=1.2)
    ax_nrmse.set_ylabel("NRMSE (1024 Sobol)")
    ax_nrmse.set_ylim(-0.02, 0.4) # 0.25 is "lazy mean", 0 is "perfect match"

    # 3) Global SSIM vs Sobol Holdout
    if not metrics_df.empty:
        ax_ssim.plot(metrics_df["iteration"], metrics_df["ssim"], "g-o", markersize=3, linewidth=1.5)
    ax_ssim.axhline(1.0, color="0.7", linestyle="dotted", linewidth=1.2)
    ax_ssim.set_ylabel("SSIM (1024 Sobol)")
    ax_ssim.set_xlabel("Iteration")
    ax_ssim.set_ylim(-0.05, 1.05)

    # Formatting and Grids
    max_iter = max(bo_df["iteration"].max(), 10)
    for i, ax in enumerate(axes):
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(True, which="major", alpha=0.2, linewidth=0.6)
        ax.set_xlim(0, max_iter)
        ax.text(
            -0.08, 0.92, chr(97 + i) + ")",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=11, fontweight="bold",
            clip_on=False, zorder=100
        )

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.95, hspace=0.12)
    plt.draw()
    plt.pause(0.1)


# Main Polling Loop
####################################################################################################################################
def poll_and_plot():
    print(f"Starting 4D live convergence poller...")
    print(f"Watching data: {DATA_CSV}")
    print(f"Watching models: {MODELS_DIR}")

    if HOLDOUT_CSV.exists():
        holdout_df = pd.read_csv(HOLDOUT_CSV)
        G_ref = holdout_df["conversion_efficiency"].values / 1e-6
        X_test = torch.tensor(holdout_df[["x1_norm", "x2_norm", "x3_norm", "x4_norm"]].values, dtype=dtype)
        sobol_range = float(np.nanmax(G_ref) - np.nanmin(G_ref))
    else:
        print(f"Warning: Holdout CSV not found at {HOLDOUT_CSV}. NRMSE and SSIM panels will remain empty.")
        G_ref, X_test, sobol_range = None, None, 1.0

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.5), sharex=True)

    try:
        while True:
            if DATA_CSV.exists():
                try:
                    bo_df = pd.read_csv(DATA_CSV)
                    if not bo_df.empty and "conversion_efficiency" in bo_df.columns:
                        if "objective_scaled" not in bo_df.columns:
                            bo_df["objective_scaled"] = bo_df["conversion_efficiency"] / 1e-6

                        # 1. Extract data with absolute Sobol range scaling
                        metrics_df = evaluate_checkpoints(MODELS_DIR, X_test, G_ref, sobol_range, dtype, dim)

                        # 2. Render UI
                        update_dashboard(axes, fig, bo_df, metrics_df)

                except Exception as e:
                    print(f"Read/Plot warning (retrying shortly): {e}")

            time.sleep(10)

    except KeyboardInterrupt:
        print("\nPolling stopped by user.")
        plt.ioff()

if __name__ == "__main__":
    poll_and_plot()
