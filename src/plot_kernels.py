import os
import glob #file finder
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from matplotlib.ticker import FixedLocator, FixedFormatter, MultipleLocator
from matplotlib.legend_handler import HandlerTuple
from utils import make_kernel
from plotting_utils import (apply_paper_style, format_map_axis as paper_format_map_axis,
    add_physical_points as paper_add_physical_points, BO_CMAP as PAPER_BO_CMAP,
    INITIAL_MARKER as PAPER_INITIAL_MARKER, INITIAL_COLOR as PAPER_INITIAL_COLOR,
    POINT_EDGE_COLOR as PAPER_EDGE_COLOR, INITIAL_SIZE as PAPER_INITIAL_SIZE,
    BO_SIZE_MIN as PAPER_BO_SIZE_MIN, BO_SIZE_MAX as PAPER_BO_SIZE_MAX)

# Settings
####################################################################################################################################
dtype                    = torch.double
dim                      = 2 # BO dimensions
HARP_WORKDIR             = Path("~/HarP-main").expanduser().resolve()
PROJECT_ROOT             = Path(__file__).resolve().parent.parent
results_root             = PROJECT_ROOT / "results"
plots_root               = PROJECT_ROOT / "figures"
grid_scan_csv            = PROJECT_ROOT / "data" / "scan_results.csv" # where the whole grid scan is
plot_iteration           = 60  # Plot each kernel as far as its own data/checkpoints extend
L_min, L_max             = 1.5 / 100, 1.5   # Original dimensionless BO and grid limits
p_min, p_max             = 300 / 100, 300.0 # Original dimensionless BO and grid limits
optimization_space       = "log_p_over_L__log_pL"
figure1_kernels          = [ "matern12", "matern32", "rbf"]
figure2_kernels          = [ "matern12", "matern32", "matern52","rbf"]
kernel_titles            = {
    "matern12": r"Matérn $1/2$",
    "matern32": r"Matérn $3/2$",
    "matern52": r"Matérn $5/2$",
    "rbf": r"RBF",
}
acquisition        = "log_ei"
plots_root.mkdir(exist_ok=True)

# Consistent typography and plot styling as in the paper
####################################################################################################################################
apply_paper_style()
mpl.rcParams["font.family"] = ["serif"]
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
FONT_SIZE = 18
FIG1_FONT_SIZE = 18
FIG2_FONT_SIZE = 17
mpl.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
})
MAP_CMAP = "viridis"
BO_CMAP = PAPER_BO_CMAP
INITIAL_MARKER = PAPER_INITIAL_MARKER
INITIAL_COLOR = PAPER_INITIAL_COLOR
POINT_EDGE_COLOR = PAPER_EDGE_COLOR
INITIAL_SIZE = PAPER_INITIAL_SIZE
BO_SIZE_MIN = PAPER_BO_SIZE_MIN
BO_SIZE_MAX = PAPER_BO_SIZE_MAX
BEST_MARKER = "X"
BEST_BO_COLOR = "red"
BEST_SIZE = 60
FINAL_MARKER = "D"
FINAL_BO_COLOR = "black"
FINAL_SIZE = 40

def unify_figure_fonts(fig, font_size):
    from matplotlib.text import Text
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(font_size)
    fig.canvas.draw()

def pack_square_panels_left(fig, axes, left=0.105, gap=0.012):
    fig.canvas.draw()
    x = left
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([x, pos.y0, pos.width, pos.height], which="both")
        x += pos.width + gap
    fig.canvas.draw()

# Coordinate limits for different search spaces
####################################################################################################################################
log_L_min, log_L_max     = np.log10(L_min), np.log10(L_max)
log_p_min, log_p_max     = np.log10(p_min), np.log10(p_max)
pL_min                   = p_min * L_min
pL_max                   = p_max * L_max
p_over_L_min             = p_min / L_max
p_over_L_max             = p_max / L_min
log_pL_min, log_pL_max   = np.log10(pL_min), np.log10(pL_max)
log_p_over_L_min         = np.log10(p_over_L_min)
log_p_over_L_max         = np.log10(p_over_L_max)
sqrt_pL_min, sqrt_pL_max = np.sqrt(pL_min), np.sqrt(pL_max)
sqrt_p_over_L_min        = np.sqrt(p_over_L_min)
sqrt_p_over_L_max        = np.sqrt(p_over_L_max)

# Helpers
####################################################################################################################################
def L_to_plot(L):
    return np.asarray(L)
def p_to_plot(p):
    return np.asarray(p)
def Lp_to_plot_units(L, p):
    return L_to_plot(L), p_to_plot(p) # return L and p as numpy arrays

def clean_space_name(space):
    # Current optimization-space names used by run_bo_2d.py.
    return space

def normalize_kernel_name(kernel):
    # Normalize possible kernel spellings used in data.csv or folder names.
    name = str(kernel).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "rbf": "rbf",
        "se": "rbf",
        "squaredexponential": "rbf",
        "matern12": "matern12",
        "matern0.5": "matern12",
        "matern1/2": "matern12",
        "matern32": "matern32",
        "matern1.5": "matern32",
        "matern3/2": "matern32",
        "matern52": "matern52",
        "matern2.5": "matern52",
        "matern5/2": "matern52",
    }
    return aliases.get(name, name)

def make_model(X, Y_objective, kernel_name): # creates a Gaussian Process model from the training data
    kernel_name = normalize_kernel_name(kernel_name)
    kernel = make_kernel(kernel_name, dim)  # same kernels as the BO loop (utils.make_kernel)
    # Standardize: shifts the objective values to mean 0 and standard deviation 1 for GP to learn the model
    # m=1 only one value to predict (CE)
    model  = SingleTaskGP(train_X=X,train_Y=Y_objective,covar_module=kernel,outcome_transform=Standardize(m=1))
    return model

def best_so_far(df): # cummulative maximum for the best so far point
    return df["objective_scaled_ratio"].cummax().to_numpy()

def format_map_axis(ax, show_ylabel=True):
    paper_format_map_axis(ax, show_ylabel=show_ylabel)
    ax.set_xlim(L_min, L_max)
    ax.set_ylim(p_min, p_max)

def add_physical_points(ax, df):
    # Use the same point colors/sizes as the paper's 2D and Straddle maps.
    paper_add_physical_points(ax, df, show_best=True)

def Lp_to_Xnorm(L, p, optimization_space):
    # Take L,p, transform them into the chosen optimization parameterization
    # Normalize them to the BO coordinate system
    # Return them in the shape expected by the GP
    base_space = clean_space_name(optimization_space) # just gets rid of the filename suffix
    if base_space == "L__p":
        x0 = (L - L_min) / (L_max - L_min)
        x1 = (p - p_min) / (p_max - p_min)
    elif base_space == "log_L__log_p":
        x0 = (np.log10(L) - log_L_min) / (log_L_max - log_L_min)
        x1 = (np.log10(p) - log_p_min) / (log_p_max - log_p_min)
    elif base_space == "p_over_L__pL":
        pL = p * L
        p_over_L = p / L
        x0 = (p_over_L - p_over_L_min) / (p_over_L_max - p_over_L_min)
        x1 = (pL - pL_min) / (pL_max - pL_min)
    elif base_space == "log_p_over_L__log_pL":
        pL = p * L
        p_over_L = p / L
        x0 = (np.log10(p_over_L) - log_p_over_L_min) / (log_p_over_L_max - log_p_over_L_min)
        x1 = (np.log10(pL) - log_pL_min) / (log_pL_max - log_pL_min)
    elif base_space == "sqrt_p_over_L__sqrt_pL":
        pL = p * L
        p_over_L = p / L
        x0 = (np.sqrt(p_over_L) - sqrt_p_over_L_min) / (sqrt_p_over_L_max - sqrt_p_over_L_min)
        x1 = (np.sqrt(pL) - sqrt_pL_min) / (sqrt_pL_max - sqrt_pL_min)
    else:
        raise ValueError(f"Unknown optimization space: {optimization_space}")
    return np.stack([x0.reshape(-1), x1.reshape(-1)], axis=1) #reshape to the expected shape

def predict_mean(kernel_name, X_grid, shape, iteration, extrapolate=False):
    """Restore the GP fitted before evaluation `iteration` and predict on a grid.
    Checkpoint i stores X/Y through i, but the GP state was fitted through i-1.
    Dropping the last stored observation is essential for matching GP state.
    """
    model_path = run_folders[kernel_name] / "models" / f"model_iter_{iteration:03d}.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    X_saved = checkpoint["X_normalized"].to(dtype=dtype)
    Y_saved = checkpoint["Y_objective"].to(dtype=dtype)
    if len(X_saved) != iteration + 1 or len(Y_saved) != len(X_saved):
        raise ValueError(f"Unexpected stored point count in {model_path}: "
                         f"X={len(X_saved)}, Y={len(Y_saved)}, iteration={iteration}")
    transform = (checkpoint.get("config") or {}).get("objective_transform", "id")
    if transform not in (None, "id"):
        raise ValueError(f"Unexpected objective transform {transform!r} in {model_path}")
    model = make_model(X_saved[:-1], Y_saved[:-1], kernel_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    valid = np.ones(len(X_grid), dtype=bool) if extrapolate else (
        (X_grid[:, 0] >= 0.0) & (X_grid[:, 0] <= 1.0) &
        (X_grid[:, 1] >= 0.0) & (X_grid[:, 1] <= 1.0)
    )
    mean = np.full(len(X_grid), np.nan)
    if np.any(valid):
        valid_ids = np.flatnonzero(valid)
        with torch.no_grad():
            for indices in np.array_split(valid_ids, max(1, int(np.ceil(len(valid_ids) / 4096)))):
                X_valid = torch.as_tensor(X_grid[indices], dtype=dtype)
                mean[indices] = model.posterior(X_valid).mean.detach().cpu().numpy().reshape(-1)
    return mean.reshape(shape)

def normalized_rmse(Z_ref, Z_pred):
    # normalized root mean squared error (NRMSE) between the direct grid Z_ref and GP prediction Z_pred
    valid = np.isfinite(Z_ref) & np.isfinite(Z_pred)
    if np.sum(valid) < 2:
        return np.nan
    rmse  = np.sqrt(np.mean((Z_pred[valid] - Z_ref[valid]) ** 2))
    scale = np.nanmax(Z_ref[valid]) - np.nanmin(Z_ref[valid])
    if scale == 0:
        return np.nan
    return rmse / scale

def global_ssim(Z_ref, Z_pred):
    # single global SSIM-like similarity score between the direct grid Z_ref and GP prediction Z_pred
    valid = np.isfinite(Z_ref) & np.isfinite(Z_pred)
    if np.sum(valid) < 2:
        return np.nan
    x           = Z_ref[valid].reshape(-1)
    y           = Z_pred[valid].reshape(-1)
    data_range  = np.nanmax(x) - np.nanmin(x)
    if data_range == 0:
        return np.nan
    C1          = (0.01 * data_range) ** 2
    C2          = (0.03 * data_range) ** 2
    mu_x        = np.mean(x)
    mu_y        = np.mean(y)
    var_x       = np.var(x)
    var_y       = np.var(y)
    cov_xy      = np.mean((x - mu_x) * (y - mu_y))
    numerator   = (2.0 * mu_x * mu_y + C1) * (2.0 * cov_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
    if denominator == 0:
        return np.nan
    return numerator / denominator

def get_bo_iteration_range(all_data, plot_iteration):
    # what is the largest number of actual BO points in any of my runs
    bo_counts = []
    for df in all_data.values():
        d    = df[df["iteration"] <= plot_iteration].copy()
        d_bo = d[d["phase"] == "bo"]
        if len(d_bo) > 0:
            bo_counts.append(len(d_bo))
    if len(bo_counts) == 0:
        return 0
    return int(max(bo_counts))

# Load grid scan
####################################################################################################################################
# Loads the direct grid scan, makes sure the objective column exists
# Reconstructs the 2D L,p grid, finds the best grid point
if not grid_scan_csv.exists():
    raise FileNotFoundError(f"Missing grid scan CSV: {grid_scan_csv}")
grid_df = pd.read_csv(grid_scan_csv)
if "objective_scaled_ratio" not in grid_df.columns:
    if "ratio_order23_over_input_laser" in grid_df.columns:
        grid_df["objective_scaled_ratio"] = grid_df["ratio_order23_over_input_laser"] / 1e-6
    elif "ratio_order23_over_order1" in grid_df.columns:
        grid_df["objective_scaled_ratio"] = grid_df["ratio_order23_over_order1"] / 1e-6
    elif "cost" in grid_df.columns:
        grid_df["objective_scaled_ratio"] = grid_df["cost"]
    else:
        raise ValueError("Grid CSV needs objective_scaled_ratio, CE ratio, or cost.")
L_axis                   = np.sort(grid_df["Lzzr"].unique())
p_axis                   = np.sort(grid_df["pzr"].unique())
L_grid, p_grid           = np.meshgrid(L_axis, p_axis, indexing="ij")
L_grid_plot, p_grid_plot = Lp_to_plot_units(L_grid, p_grid)
Z_direct                 = np.full((len(L_axis), len(p_axis)), np.nan)
for _, row in grid_df.iterrows():
    iL               = np.where(np.isclose(L_axis, row["Lzzr"]))[0][0]
    ip               = np.where(np.isclose(p_axis, row["pzr"]))[0][0]
    Z_direct[iL, ip] = row["objective_scaled_ratio"]
best_direct          = grid_df.loc[grid_df["objective_scaled_ratio"].idxmax()]

# Load BO files
####################################################################################################################################
# Only these exact runs are allowed:
# results/bo_2d_<optimization_space>_<acquisition>_<kernel>/
# This kernel-comparison script does not search across optimization spaces
# or acquisition functions. Only the kernel name changes.

all_data         = {}
run_folders      = {}
required_kernels = list(figure2_kernels)

for kernel_name in required_kernels:

    run_name = (
        f"bo_2d_{optimization_space}_{acquisition}_{kernel_name}"
    )

    run_folder = results_root / run_name
    data_path  = run_folder / "data.csv"
    models_dir = run_folder / "models"

    if not run_folder.exists():
        raise FileNotFoundError(
            f"Missing required kernel-comparison run folder:\n"
            f"    {run_folder}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing data.csv for kernel '{kernel_name}':\n"
            f"    {data_path}"
        )

    if not models_dir.exists():
        raise FileNotFoundError(
            f"Missing models folder for kernel '{kernel_name}':\n"
            f"    {models_dir}"
        )

    df = (
        pd.read_csv(data_path)
        .sort_values("iteration", kind="mergesort")
        .copy()
    )

    if len(df) == 0:
        raise ValueError(
            f"Empty data.csv for kernel '{kernel_name}':\n"
            f"    {data_path}"
        )

    # Verify the stored optimization space when the column is available.
    if "optimization_space" in df.columns:
        stored_spaces = (
            df["optimization_space"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(stored_spaces) > 0 and np.any(
            stored_spaces != optimization_space
        ):
            raise ValueError(
                f"Optimization-space mismatch in:\n"
                f"    {data_path}\n"
                f"Expected: {optimization_space}\n"
                f"Found:    {stored_spaces.tolist()}"
            )

    # Verify the stored acquisition when the column is available.
    if "acquisition" in df.columns:
        stored_acquisitions = (
            df["acquisition"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )

        if len(stored_acquisitions) > 0 and np.any(
            stored_acquisitions != acquisition.lower()
        ):
            raise ValueError(
                f"Acquisition mismatch in:\n"
                f"    {data_path}\n"
                f"Expected: {acquisition}\n"
                f"Found:    {stored_acquisitions.tolist()}"
            )

    # Objective used for plotting.
    if "objective_scaled" in df.columns:
        df["objective_scaled_ratio"] = df["objective_scaled"]

    elif "conversion_efficiency" in df.columns:
        df["objective_scaled_ratio"] = (
            df["conversion_efficiency"] / 1e-6
        )

    elif "objective_scaled_ratio" not in df.columns:
        raise ValueError(
            f"Cannot find an objective column in:\n"
            f"    {data_path}"
        )

    # Wall time used in the printed summary.
    if "cumulative_wall_time_s" not in df.columns:
        if "iteration_wall_time_s" in df.columns:
            df["cumulative_wall_time_s"] = (
                df["iteration_wall_time_s"].cumsum()
            )
        else:
            df["cumulative_wall_time_s"] = np.nan

    # Verify the kernel using the newest checkpoint.
    checkpoint_paths = sorted(
        models_dir.glob("model_iter_*.pt")
    )

    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(
            f"No model checkpoints found for kernel '{kernel_name}':\n"
            f"    {models_dir}"
        )

    newest_checkpoint = torch.load(
        checkpoint_paths[-1],
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_config = newest_checkpoint.get("config", {})
    checkpoint_kernel = checkpoint_config.get("kernel")

    if checkpoint_kernel is not None:
        checkpoint_kernel = normalize_kernel_name(
            checkpoint_kernel
        )

        if checkpoint_kernel != kernel_name:
            raise ValueError(
                f"Kernel mismatch in:\n"
                f"    {checkpoint_paths[-1]}\n"
                f"Expected: {kernel_name}\n"
                f"Found:    {checkpoint_kernel}"
            )

    all_data[kernel_name]    = df
    run_folders[kernel_name] = run_folder

# Independent checkpoint ranges: do NOT truncate all kernels to the shortest run.
####################################################################################################
checkpoint_iterations = {}
print(f"\nKernel comparison: best CE through evaluation {plot_iteration}; GP models at most checkpoint {plot_iteration} (trained through {plot_iteration - 1}).")
for kernel_name in figure2_kernels:
    folder = run_folders[kernel_name] / "models"
    available = set()
    for path in folder.glob("model_iter_*.pt"):
        try:
            available.add(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            continue
    checkpoint_iterations[kernel_name] = sorted(i for i in available if 1 <= i <= plot_iteration)
    if not checkpoint_iterations[kernel_name]:
        print(f"  {kernel_name:10s}: no GP checkpoints within requested range")
    else:
        print(f"  {kernel_name:10s}: GP checkpoints through {checkpoint_iterations[kernel_name][-1]}, "
              f"accuracy through observation {checkpoint_iterations[kernel_name][-1] - 1}")
    evaluations = all_data[kernel_name].loc[
        lambda df: (df["iteration"] >= 1) & (df["iteration"] <= plot_iteration), "iteration"
    ]
    print(f"{'':14s} CE observations through {int(evaluations.max()) if not evaluations.empty else 'none'}")

# The metrics evaluate each model at the grid scan, exactly as in the previous
# figures. Checkpoint i stores the GP trained through observation i-1.
####################################################################################################
X_grid_metrics = Lp_to_Xnorm(L_grid, p_grid, optimization_space)
metric_rows = []
for kernel_name in figure2_kernels:
    iterations = checkpoint_iterations[kernel_name]
    print(f"Computing {kernel_titles[kernel_name]} GP accuracy ({len(iterations)} checkpoints)...")
    for ix, checkpoint_iteration in enumerate(iterations, 1):
        Z_pred = predict_mean(kernel_name, X_grid_metrics, Z_direct.shape, checkpoint_iteration)
        metric_rows.append({
            "kernel": kernel_name,
            "iteration": checkpoint_iteration - 1,
            "nrmse": normalized_rmse(Z_direct, Z_pred),
            "ssim": global_ssim(Z_direct, Z_pred),
        })
        if ix % 10 == 0 or ix == len(iterations):
            print(f"  {ix}/{len(iterations)} checkpoints processed")
metrics_df = pd.DataFrame(metric_rows, columns=["kernel", "iteration", "nrmse", "ssim"])
if metrics_df.empty:
    raise RuntimeError("No saved GP models to compute accuracy for any kernel.")

# Three-panel paper-style comparison with right-side kernel legend
fig, axes = plt.subplots(3, 1, figsize=(11.69, 5.9671 * 0.95), sharex=True)
ax_bo_conv, ax_nrmse, ax_ssim = axes
curve_colors = ["#482475", "#0C9F8D", "#C98A00", "#C34675"]
line_styles = ["-", (0, (6.5, 2, 1.5, 2)), (0, (5.0, 2.2)), (0, (8, 3))]
line_styles = [
    "-",                          # Matérn 1/2
    (0, (9.0, 3.0, 1.3, 3.0)),    # Matérn 3/2
    (0, (5.0, 2.2)),              # Matérn 5/2
    (0, (1.0, 2.8)),              # RBF
]
for i, kernel_name in enumerate(figure2_kernels):
    color, style = curve_colors[i], line_styles[i]
    df = all_data[kernel_name].loc[lambda d: d["iteration"] <= plot_iteration].sort_values("iteration").copy()
    if df.empty:
        continue
    y_best = best_so_far(df)
    visible = df["iteration"].to_numpy() >= 1
    ax_bo_conv.plot(df.loc[visible, "iteration"], y_best[visible],
                    color=color, linestyle=style, linewidth=3.1,
                    alpha=0.98, zorder=4)
    m = metrics_df.loc[
        (metrics_df["kernel"] == kernel_name) &
        (metrics_df["iteration"] >= 1) &
        (metrics_df["iteration"] <= plot_iteration)
    ].sort_values("iteration")
    ax_nrmse.plot(m["iteration"], m["nrmse"], color=color, linestyle=style,
                  linewidth=3.1, alpha=0.98, zorder=4)
    ax_ssim.plot(m["iteration"], m["ssim"], color=color, linestyle=style,
                 linewidth=3.1, alpha=0.98, zorder=4)

GRID_COLOR = "0.72"
ax_nrmse.axhline(0, color=GRID_COLOR, linestyle="dotted", linewidth=1.3, zorder=2)
ax_ssim.axhline(1, color=GRID_COLOR, linestyle="dotted", linewidth=1.3, zorder=2)
ax_nrmse.annotate("Grid", xy=(0.05, -0.01), xycoords=("axes fraction", "data"),
                  xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                  color="0.48", zorder=10)
ax_ssim.annotate("Grid", xy=(0.05, 1.0), xycoords=("axes fraction", "data"),
                 xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                 color="0.48", zorder=10)
ax_bo_conv.set_ylim(-0.1, 4.0)
ax_nrmse.set_ylim(-0.05, 0.3)
ax_ssim.set_ylim(-0.1, 1.1)
ax_bo_conv.set_ylabel(r"$\mathrm{CE}_{\mathrm{best}}\,/\,10^{-6}$")
ax_nrmse.set_ylabel(r"$\mathrm{NRMSE}_{\mathrm{grid}}$")
ax_ssim.set_ylabel(r"$\mathrm{SSIM}_{\mathrm{grid}}$")
ax_ssim.set_xlabel("Iteration")
for i, ax in enumerate(axes):
    ax.set_xlim(1, plot_iteration)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(direction="in", top=True, right=True)
    ax.tick_params(which="minor", direction="in", top=True, right=True,
                   length=2.8, width=0.7)
    ax.tick_params(axis="x", which="major", length=5, width=0.9)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax.grid(False, which="minor")
    ax.yaxis.set_label_coords(-0.070, 0.5)
    ax.text(
     0.99, 0.13, chr(97 + i),
     transform=ax.transAxes,
     ha="right",
     va="top",
     fontweight="bold",
     clip_on=True,
     zorder=100,
    )
ax_bo_conv.tick_params(axis="x", which="both", labelbottom=False)
ax_nrmse.tick_params(axis="x", which="both", labelbottom=False)

# Four kernel entries in a blue-gray inset, matching the Straddle line figure.
plot_right = 0.773
legend_left = 0.797
legend_width = 0.183
fig.subplots_adjust(left=0.120, right=plot_right, bottom=0.115,
                    top=0.955, hspace=0.055)
fig.canvas.draw()
plot_top = max(ax.get_position().y1 for ax in axes)
plot_bottom = min(ax.get_position().y0 for ax in axes)
legend_height = 0.57 * (plot_top - plot_bottom)
legend_bottom = (plot_bottom + plot_top - legend_height) / 2
legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.set_facecolor("#b9c7d9")
legend_ax.patch.set_alpha(0.95)
legend_ax.set_xticks([])
legend_ax.set_yticks([])
for spine in legend_ax.spines.values():
    spine.set_visible(False)
# Top-to-bottom: Matern 1/2, Matern 3/2, Matern 5/2, RBF.
legend_rows = [0.83, 0.61, 0.39, 0.17]
for y, kernel_name, color, style in zip(legend_rows, figure2_kernels, curve_colors, line_styles):
    legend_ax.plot([0.085, 0.36], [y, y], color=color,
                   linestyle=style, linewidth=3.1, solid_capstyle="butt",
                   dash_capstyle="butt", clip_on=False)
    legend_ax.text(0.44, y, kernel_titles[kernel_name], ha="left", va="center",
                   fontsize=FIG2_FONT_SIZE)

fig_path_2 = plots_root / "KERNELfigure2.pdf"
unify_figure_fonts(fig, FIG2_FONT_SIZE)
fig.savefig(fig_path_2, dpi=300, bbox_inches="tight", pad_inches=0.025)
fig.savefig(fig_path_2.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.025)
plt.close(fig)
print(f"Saved: {fig_path_2} and .png")

####################################################################################################################################
# Final print
####################################################################################################################################
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

# Grid-search wall time
grid_total_evals = len(grid_df)
if "wall_time_s" in grid_df.columns:
    grid_total_wall_time_s = grid_df["wall_time_s"].sum()
else:
    grid_total_wall_time_s = np.nan
print("\nWall times")
print("-" * 72)
if np.isfinite(grid_total_wall_time_s):
    print(
        f"{'Direct grid scan':35s} "
        f"{grid_total_evals:5d} evals   "
        f"{grid_total_wall_time_s / 60.0:9.2f} min   "
        f"({grid_total_wall_time_s / 3600.0:7.2f} h)"
    )
else:
    print(
        f"{'Direct grid scan':35s} "
        f"{grid_total_evals:5d} evals   "
        f"wall time unavailable"
    )

for kernel_name in figure2_kernels:
    if kernel_name not in all_data:
        continue
    df = all_data[kernel_name]
    if "cumulative_wall_time_s" in df.columns:
        finite_times = df["cumulative_wall_time_s"].to_numpy(dtype=float)
        finite_times = finite_times[np.isfinite(finite_times)]
        if len(finite_times) > 0:
            total_s = finite_times[-1]
        else:
            total_s = np.nan
    else:
        total_s = np.nan
    if np.isfinite(total_s):
        print(
            f"{kernel_titles[kernel_name]:35s} "
            f"{len(df):5d} evals   "
            f"{total_s / 60.0:9.2f} min   "
            f"({total_s / 3600.0:7.2f} h)"
        )
    else:
        print(
            f"{kernel_titles[kernel_name]:35s} "
            f"{len(df):5d} evals   "
            f"wall time unavailable"
        )

print("\nBest direct grid-scan value")
print("-" * 72)
print(f"CE / 1e-6 = {best_direct['objective_scaled_ratio']:.6f}")
print(f"L/z_R     = {best_direct['Lzzr']:.6f}")
print(f"p z_R     = {best_direct['pzr']:.6f}")
print("\nBest actually evaluated BO value in each run")
print("-" * 72)

for kernel_name in figure2_kernels:
    if kernel_name not in all_data:
        continue
    df = all_data[kernel_name]
    best = df.loc[df["objective_scaled_ratio"].idxmax()]
    print(
        f"{kernel_titles[kernel_name]:35s} "
        f"CE/1e-6 = {best['objective_scaled_ratio']:.6f}   "
        f"iteration = {int(best['iteration'])}"
    )

print("\nKERNELfigure2 was generated (no colormap figure).")
print("Best-CE curves include available evaluations through 60; GP metrics end at each kernel's latest selected checkpoint minus one.")
print("GP metrics are indexed by the last observation used for training.")
print("Done.")

# Final displayed value of each of the three curves. GP metrics can stop
# one iteration earlier than the best-CE curve because checkpoint i stores
# the GP fitted through observation i-1. Each kernel is handled separately.
print("\nFINAL PLOTTED VALUES BY KERNEL")
print("-" * 101)
print(f"{'Kernel':18s} {'Best CE/1e-6':>15s} {'CE iter':>8s} "
      f"{'NRMSE_grid':>14s} {'GP iter':>8s} {'SSIM_grid':>14s} {'GP iter':>8s}")
print("-" * 101)
for kernel_name in figure2_kernels:
    df_end = (all_data[kernel_name]
              .loc[lambda d: d["iteration"] <= plot_iteration]
              .sort_values("iteration", kind="mergesort"))
    metric_end = (metrics_df
                  .loc[lambda d: (d["kernel"] == kernel_name) &
                                  (d["iteration"] >= 1) &
                                  (d["iteration"] <= plot_iteration)]
                  .sort_values("iteration", kind="mergesort"))

    if df_end.empty:
        ce_str, ce_iter = "N/A", "--"
    else:
        # Final value of the convergence curve = cumulative best CE,
        # not necessarily the CE of the last evaluated point.
        ce_str = f"{df_end['objective_scaled_ratio'].cummax().iloc[-1]:.6f}"
        ce_iter = str(int(df_end["iteration"].iloc[-1]))

    if metric_end.empty:
        nrmse_str = ssim_str = "N/A"
        nrmse_iter = ssim_iter = "--"
    else:
        last_nrmse = metric_end.loc[metric_end["nrmse"].notna()]
        last_ssim = metric_end.loc[metric_end["ssim"].notna()]
        nrmse_str = f"{last_nrmse['nrmse'].iloc[-1]:.6f}" if not last_nrmse.empty else "N/A"
        ssim_str = f"{last_ssim['ssim'].iloc[-1]:.6f}" if not last_ssim.empty else "N/A"
        nrmse_iter = str(int(last_nrmse["iteration"].iloc[-1])) if not last_nrmse.empty else "--"
        ssim_iter = str(int(last_ssim["iteration"].iloc[-1])) if not last_ssim.empty else "--"

    label = kernel_titles[kernel_name].replace("$", "")
    print(f"{label:18s} {ce_str:>15s} {ce_iter:>8s} "
          f"{nrmse_str:>14s} {nrmse_iter:>8s} {ssim_str:>14s} {ssim_iter:>8s}")
print("-" * 101)

print("\n KERNELfigure2 was generated (no colormap figure).")
print("Best-CE curves include available evaluations through 60; GP metrics end at each kernel's latest selected checkpoint minus one.")
print("GP metrics are indexed by the last observation used for training.")
print("Done.")
