import os
import glob #file finder
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from matplotlib.ticker import FixedLocator, FixedFormatter, MultipleLocator
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patheffects as pe
from utils import (
    make_kernel,
    normalized_rmse,
    global_ssim,
    clean_space_name,
    L_to_plot,
    p_to_plot,
    Lp_to_plot_units,
    best_so_far,
    get_bo_iteration_range,
    Lp_to_Xnorm,
)
from plotting_utils import (
    apply_paper_style,
    format_map_axis,
    add_physical_points,
    BO_CMAP,
    INITIAL_MARKER,
    INITIAL_COLOR,
    POINT_EDGE_COLOR,
    INITIAL_SIZE,
    BO_SIZE_MIN,
    BO_SIZE_MAX,
    make_initial_marker,
)
# Settings
####################################################################################################################################
dtype                    = torch.double
dim                      = 2 # BO dimensions
HARP_WORKDIR             = Path("~/HarP-main").expanduser().resolve()
PROJECT_ROOT             = Path(__file__).resolve().parent.parent
results_root             = PROJECT_ROOT / "results"
plots_root               = PROJECT_ROOT / "figures"
grid_scan_csv            = PROJECT_ROOT / "data" / "scan_results.csv" # where the whole grid scan is
plot_iteration           = 60 #None             # how many iterations, if none put the last iteration that exists for every optimization
L_min, L_max             = 1.5 / 100, 1.5   # Original dimensionless BO and grid limits
p_min, p_max             = 300 / 100, 300.0 # Original dimensionless BO and grid limits
optimization_space       = "log_p_over_L__log_pL"
figure1_straddles        = [25, 50, 90]
figure2_straddles        = [25, 50, 90]
straddle_titles = {
    25:  r"$\mathrm{CE}_{\mathrm{target}} = 0.75 \times 10^{-6}$",
    50:  r"$\mathrm{CE}_{\mathrm{target}} = 1.5 \times 10^{-6}$",
    90: r"$\mathrm{CE}_{\mathrm{target}} = 2.7 \times 10^{-6}$",
}
# CE target values in the plotted units (CE / 10^{-6})
straddle_targets = {
    25: 0.75,
    50: 1.5,
    90: 2.7,
}
kernel                   = "matern52"
acquisition_base         = "straddle"
plots_root.mkdir(exist_ok=True)

# Style of the plots
####################################################################################################################################
apply_paper_style()
mpl.rcParams["font.family"] = ["serif"]
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
FONT_SIZE      = 18
FIG1_FONT_SIZE = 18
FIG2_FONT_SIZE = 17  # Uniform Figure 2 typography, including its side legend
mpl.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
})
def unify_figure_fonts(fig, font_size):
    """Apply size to all text including helper-created axis and colorbar labels."""
    from matplotlib.text import Text
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(font_size)
    fig.canvas.draw()
def pack_square_panels_left(fig, axes, left=0.105, gap=0.012):
    """Pack fixed-aspect panels without changing their height or data aspect."""
    fig.canvas.draw()
    x = left
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([x, pos.y0, pos.width, pos.height], which="both")
        x += pos.width + gap
    fig.canvas.draw()
MAP_CMAP                 = "viridis"     # colormap for grid
FINAL_MARKER             = "D"           # marker for the final evaluated BO point
FINAL_BO_COLOR           = "black" #"#D62728"   # red: final evaluated BO point
FINAL_SIZE               = 40
# Helpers
####################################################################################################################################
def make_model(X, Y_objective): # creates a Gaussian Process model from the training data
    # All compared Straddle runs use the same Matérn 5/2 kernel.
    kernel_module = make_kernel("matern52", dim)  # same kernel as the BO loop (bo.py)
    # Standardize: shifts the objective values to mean 0 and standard deviation 1 for GP to learn the model
    # m=1 only one value to predict (CE)
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y_objective,
        covar_module=kernel_module,
        outcome_transform=Standardize(m=1),
    )
    return model

def predict_mean(straddle_value, X_grid, shape, iteration, extrapolate=False):
    # Takes one saved GP model at one iteration
    # Evaluates its predicted mean at all points in grid
    # returns the predictions reshaped back into the original 2D grid.
    # straddle_value: target code used in the result-folder name (50, 100, or 300)
    # X_grid: all grid points transformed and normalized for this GP, shape (N, 2)
    # shape: original 2D grid shape
    # iteration: which saved GP model to load
    model_path  = run_folders[straddle_value] / "models" / f"model_iter_{iteration:03d}.pt" # the model filename
    checkpoint  = torch.load(model_path, map_location="cpu", weights_only=False)
    X_saved = checkpoint["X_normalized"].to(dtype=dtype)
    Y_saved = checkpoint["Y_objective"].to(dtype=dtype)
    # Checkpoint i was written after evaluation i, but the GP was fitted
    # before that evaluation, using observations 0,...,i-1.
    if len(X_saved) != iteration + 1 or len(Y_saved) != len(X_saved):
        raise ValueError(f"Unexpected training-data size in {model_path}: "
                         f"X={len(X_saved)}, Y={len(Y_saved)}, iteration={iteration}")
    X = X_saved[:-1]
    Y_objective = Y_saved[:-1]
    if len(X) < 1:
        raise ValueError(f"No observations in GP reconstruction: {model_path}")
    # Straddle runs use the identity objective transform; check for safety.
    transform = checkpoint.get("config", {}).get("objective_transform", "id")
    if transform not in (None, "id"):
        raise ValueError(f"Non-identity objective transform {transform!r} in {model_path}; "
                         "the plotting routine expects CE-scaled units")
    model = make_model(X, Y_objective)
    model.load_state_dict(checkpoint["model_state_dict"]) # restores paramaters that GP learned, like lenght scales
    model.eval() # model is evaluated

    # The rectangular physical grid may extend beyond the normalized BO domain.
    # Allow GP extrapolation for visualization, but restrict quantitative
    # comparisons to points inside the optimization domain.
    finite = np.all(np.isfinite(X_grid), axis=1)
    inside = np.all((X_grid >= 0.0) & (X_grid <= 1.0), axis=1)
    valid = finite if extrapolate else (finite & inside)
    mean        = np.full(len(X_grid), np.nan) # create empty array
    if np.any(valid): # predict the model if valid points exist
        X_valid = torch.tensor(X_grid[valid], dtype=dtype)
        with torch.no_grad():
            mean[valid] = model.posterior(X_valid).mean.cpu().numpy().reshape(-1) # GP posterior for every grid point
    return mean.reshape(shape)

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
# results/bo_2d_<optimization_space>_straddle<target_code>_matern52/
# The optimization space and kernel stay fixed. Only the Straddle target changes.
all_data          = {}
run_folders       = {}
required_straddles = list(dict.fromkeys(figure1_straddles + figure2_straddles))
for straddle_value in required_straddles:
    run_name = (
        f"bo_2d_{optimization_space}_"
        f"{acquisition_base}{straddle_value}per_max3_{kernel}"
    )
    # Reads results/ directly, like plot_fig1_fig2_paper.py and the 4D scripts, so
    # the figure always shows the runs just made (not an older archived copy).
    run_folder = results_root / run_name
    data_path  = run_folder / "data.csv"
    models_dir = run_folder / "models"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing data.csv for STRADDLE target {straddle_value}:\n"
            f"    {data_path}"
        )
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Missing models folder for STRADDLE target {straddle_value}:\n"
            f"    {models_dir}"
        )
    df = (
        pd.read_csv(data_path)
        .sort_values("iteration", kind="mergesort")
        .copy()
    )
    if len(df) == 0:
        raise ValueError(
            f"Empty data.csv for STRADDLE target {straddle_value}:\n"
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
    # Verify that this is a Straddle run when the acquisition column is available.
    # The stored value may be either "straddle" or include the numeric target suffix.
    if "acquisition" in df.columns:
        stored_acquisitions = (
            df["acquisition"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )
        valid_acquisitions = {
            acquisition_base,
            f"{acquisition_base}{straddle_value}",
        }
        invalid = [
            value for value in stored_acquisitions
            if value not in valid_acquisitions
        ]
        if len(invalid) > 0:
            raise ValueError(
                f"Acquisition mismatch in:\n"
                f"    {data_path}\n"
                f"Expected one of: {sorted(valid_acquisitions)}\n"
                f"Found:           {stored_acquisitions.tolist()}"
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
    # Verify the fixed Matérn 5/2 kernel using the newest checkpoint when possible.
    checkpoint_paths = sorted(
        models_dir.glob("model_iter_*.pt")
    )
    if len(checkpoint_paths) == 0:
        raise FileNotFoundError(
            f"No model checkpoints found for STRADDLE target {straddle_value}:\n"
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
        normalized_checkpoint_kernel = (
            str(checkpoint_kernel)
            .strip()
            .lower()
            .replace("_", "")
            .replace("-", "")
        )
        if normalized_checkpoint_kernel not in {
            "matern52", "matern2.5", "matern5/2"
        }:
            raise ValueError(
                f"Kernel mismatch in:\n"
                f"    {checkpoint_paths[-1]}\n"
                f"Expected: matern52\n"
                f"Found:    {checkpoint_kernel}"
            )
    all_data[straddle_value]    = df
    run_folders[straddle_value] = run_folder

# Choose iteration
####################################################################################################################################
# Find which iterations exist for all kernels used in Figure 1 and Figure 2
iteration_sets = []
# first collect available iterations separately for every kernel
for straddle_value in required_straddles:
    folder = run_folders[straddle_value] / "models"
    paths  = sorted(glob.glob(str(folder / "model_iter_*.pt")))
    if len(paths) == 0:
        print("Missing model checkpoints:", folder)
        continue
    its = []
    for path in paths:
        name = os.path.basename(path)
        its.append(int(name.replace("model_iter_", "").replace(".pt", "")))
    iteration_sets.append(set(its))
if len(iteration_sets) == 0:
    raise RuntimeError("No model checkpoints found.")
# Intersection keeps only iterations that are present for every required STRADDLE target
common_iterations = sorted(set.intersection(*iteration_sets))
if len(common_iterations) == 0:
    raise RuntimeError("No common model checkpoint exists for all required STRADDLE targets.")
if plot_iteration is None:
    plot_iteration = common_iterations[-1]
elif plot_iteration not in common_iterations:
    raise ValueError(
        f"Requested plot_iteration={plot_iteration} is not available for every required STRADDLE target. "
        f"Available common iterations are {common_iterations[0]} to {common_iterations[-1]}."
    )
print("Plotting iteration:", plot_iteration)
print("Figure 1: showing GP extrapolation outside the normalized BO domain "
      "to fill the physical (L,p) plotting rectangle; interpret this region cautiously.")

# GP maps in original L/z_R, p z_R domain
####################################################################################################################################
# The reference CSV does not necessarily sample the entire displayed physical
# rectangle: in particular its largest L/z_R can be smaller than the plotting
# limit (1.5). Removing NaNs alone cannot fill that un-sampled strip. Build a
# dedicated Figure 1 mesh reaching the specified physical limits and evaluate
# the saved GP there. Figure 2 still uses the original scan grid unchanged.
L_axis_map = np.asarray(L_axis, dtype=float)
p_axis_map = np.asarray(p_axis, dtype=float)
if L_axis_map[-1] < L_max - 1e-10:
    step_L = float(np.median(np.diff(L_axis_map)))
    n_extra = max(2, int(np.ceil((L_max - L_axis_map[-1]) / step_L)))
    L_axis_map = np.r_[L_axis_map, np.linspace(L_axis_map[-1], L_max, n_extra + 1)[1:]]
if p_axis_map[-1] < p_max - 1e-10:
    step_p = float(np.median(np.diff(p_axis_map)))
    n_extra = max(2, int(np.ceil((p_max - p_axis_map[-1]) / step_p)))
    p_axis_map = np.r_[p_axis_map, np.linspace(p_axis_map[-1], p_max, n_extra + 1)[1:]]
L_grid_map, p_grid_map = np.meshgrid(L_axis_map, p_axis_map, indexing="ij")
L_grid_plot_map, p_grid_plot_map = Lp_to_plot_units(L_grid_map, p_grid_map)
X_grid_map = Lp_to_Xnorm(L_grid_map, p_grid_map, optimization_space)
plot_data = {}
for straddle_value in figure1_straddles:
    plot_data[straddle_value] = predict_mean(
        straddle_value,
        X_grid_map,
        L_grid_map.shape,
        plot_iteration,
        extrapolate=True,
    )
# Keep the reference-domain scaling: do not allow extrapolated values
# to change the established colormap limits.
inside_bo = np.all((X_grid_map >= 0.0) & (X_grid_map <= 1.0), axis=1).reshape(L_grid_map.shape)
print(f"Figure 1 map extent: L={L_axis_map[0]:.4g}..{L_axis_map[-1]:.4g}, "
      f"p={p_axis_map[0]:.4g}..{p_axis_map[-1]:.4g}")
all_maps = [Z_direct[np.isfinite(Z_direct)]]  # preserve direct-scan scale
for straddle_value in figure1_straddles:
    map_values = plot_data[straddle_value]
    all_maps.append(map_values[inside_bo & np.isfinite(map_values)])
all_maps = np.concatenate(all_maps) # make 1d array of maps from grid scan and BO
vmin     = np.nanmin(all_maps)
vmax     = np.nanmax(all_maps)

# Legend handles for Figure 1
####################################################################################################################################
n_bo_for_legend = get_bo_iteration_range(all_data, plot_iteration - 1) #largest number of BO evaluations up to the selected iteration
# How the Initial BO point marker will look in the legend
initial_handle = make_initial_marker()
# How BO point markers will look like in the legend
bo_legend_points = []
for k in range(5):
    frac = k / 4
    marker_area = BO_SIZE_MIN + frac * (BO_SIZE_MAX - BO_SIZE_MIN)
    bo_legend_points.append(
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=plt.get_cmap(BO_CMAP)(frac),
            markeredgecolor=POINT_EDGE_COLOR,
            markeredgewidth=0.55,
            markersize=np.sqrt(marker_area),
        )
    )
bo_size_handle = tuple(bo_legend_points)

# Final BO point visualization
final_bo_handle = plt.Line2D(
    [0], [0],
    marker=FINAL_MARKER,
    linestyle="",
    markerfacecolor=FINAL_BO_COLOR,
    markeredgecolor=POINT_EDGE_COLOR,
    markeredgewidth=0.8,
    markersize=np.sqrt(FINAL_SIZE),
)
# Red dashed CE-target contour used in every Figure 1 panel.
target_contour_handle = plt.Line2D(
    [0], [0],
    color="red",
    linestyle="--",
    linewidth=1.8,
)
legend_handles = [
    initial_handle,
    bo_size_handle,
    target_contour_handle,
]
legend_labels = [
    r"Initial BO point $i=0$",
    rf"BO points ($i=1$ $\rightarrow$ $i={n_bo_for_legend}$)",
    r"$\mathrm{CE}_{\mathrm{target}}$ contour",
]
# Draw all items in the tuple (legend_handles) together inside one legend entry
legend_handler_map = {tuple: HandlerTuple(ndivide=None, pad=0.08)}

####################################################################################################################################
####################################################################################################################################
# Figure 1: three Straddle-target maps for one optimization space, 1 x 3 panels
####################################################################################################################################
####################################################################################################################################
fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.8), sharex=True, sharey=True)
axes = np.asarray(axes).reshape(-1)
# Panels a-c: BO maps for the three Straddle CE targets
####################################################################################################################################
im = None
for i, straddle_value in enumerate(figure1_straddles):
    ax   = axes[i]
    mean = plot_data[straddle_value]
    im = ax.pcolormesh(
        L_grid_plot_map,
        p_grid_plot_map,
        mean,
        shading    = "auto",
        cmap       = MAP_CMAP,
        vmin       = vmin,
        vmax       = vmax,
        zorder     = 1,
        rasterized = True
    )
    # Red dashed level-set contour at this run's CE target.
    target_level = straddle_targets[straddle_value]
    finite_mean = mean[np.isfinite(mean)]
    if (
        len(finite_mean) > 0
        and np.nanmin(finite_mean) <= target_level <= np.nanmax(finite_mean)
    ):
        ax.contour(
            L_grid_plot_map,
            p_grid_plot_map,
            mean,
            levels=[target_level],
            colors="red",
            linewidths=1.8,
            linestyles="--",
            zorder=250000,
        )
    else:
        print(
            f"Warning: target contour CE/1e-6={target_level:g} "
            f"is outside the predicted range for STRADDLE {straddle_value} "
            f"at iteration {plot_iteration}."
        )
    # Plot the evaluated BO points. No best-point star is used.
    df = all_data[straddle_value]
    df = df[df["iteration"] < plot_iteration].copy()
    add_physical_points(ax, df, show_best=False)
    ax.set_title(straddle_titles[straddle_value])
    ax.text(0.98, 0.97, chr(97 + i), transform=ax.transAxes,
            ha="right", va="top", fontsize=FIG1_FONT_SIZE, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.2),
            clip_on=False, zorder=100)
    format_map_axis(ax, show_ylabel=(i == 0))

# Axis labels only where needed
####################################################################################################################################
for j, ax in enumerate(axes):
    if j == 0:
        ax.set_ylabel(r"$p z_R$ [mbar cm]")
        ax.tick_params(labelleft=True)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_xlabel(r"$L/z_R$")
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.245, top=0.78,
                    wspace=0.12)
for ax in axes:
    ax.set_box_aspect(1)
pack_square_panels_left(fig, axes, left=0.105, gap=0.012)
fig.canvas.draw()
panel_left = min(ax.get_position().x0 for ax in axes)
panel_right = max(ax.get_position().x1 for ax in axes)
panel_center = (panel_left + panel_right) / 2

# Colorbar
####################################################################################################################################
# Colorbar position = [left, bottom, width, height].
# Lowered bottom from 0.064 to 0.038 (figure-relative coordinates).
cax  = fig.add_axes([panel_center - 0.28, 0.02, 0.56, 0.028])
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
cbar.set_label(r"CE/$10^{-6}$", labelpad=2)
# Keep ordinary colorbar ticks and add all three target values.
default_ticks  = np.asarray(cbar.get_ticks(), dtype=float)
target_ticks   = np.asarray([straddle_targets[value] for value in figure1_straddles], dtype=float,)
combined_ticks = np.unique(
    np.concatenate([
        default_ticks[(default_ticks >= vmin) & (default_ticks <= vmax)],
        target_ticks[(target_ticks >= vmin) & (target_ticks <= vmax)],
    ])
)
# Keep numeric ticks
cbar.set_ticks(np.arange(0, 3.6, 0.5))
# Draw the red target lines
for label, target_level in zip(["a)", "b)", "c)"], target_ticks):
    if vmin <= target_level <= vmax:
        cbar.ax.axvline(
            target_level,
            color="red",
            linewidth=1.8,
            linestyle="-",
            zorder=20,
        )
        # Put the label above the colorbar
        cbar.ax.text(
            target_level,
            1.18,                    # above the bar
            label,
            color="red",
            fontsize=FONT_SIZE,
            #fontweight="bold",
            ha="center",
            va="bottom",
            transform=cbar.ax.get_xaxis_transform(),
            clip_on=False,
        )
# Legend
####################################################################################################################################
leg = fig.legend(
    legend_handles,
    legend_labels,
    handler_map    = legend_handler_map,
    loc            = "upper center",
    ncol           = 3,
    frameon        = True,
    bbox_to_anchor = (panel_center, 0.955),
    columnspacing  = 0.95,
    handlelength   = 1.6,
    handletextpad  = 0.35,
    borderaxespad  = 0.0,
)
frame = leg.get_frame()
frame.set_facecolor("#b9c7d9")
frame.set_alpha(0.95)
frame.set_edgecolor("none")

# Save Figure 1
####################################################################################################################################
fig_path_1 = plots_root / "STRADDLEfigure1.pdf"
unify_figure_fonts(fig, FIG1_FONT_SIZE)
fig.savefig(
    fig_path_1,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.025,
)
fig.savefig(
    fig_path_1.with_suffix(".png"),
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.025,
)
plt.close(fig)
print(f"Saved: {fig_path_1} and .png")



####################################################################################################################################
####################################################################################################################################
# Figure 2: three full-width convergence/prediction-quality panels
####################################################################################################################################
####################################################################################################################################
# Metrics versus direct grid scan
####################################################################################################################################
# A checkpoint k represents the GP fitted through iteration k-1.
# Use at most checkpoint 60: its GP was fitted through BO iteration 59.
metric_iterations = [
    iteration
    for iteration in common_iterations
    if 1 <= iteration <= plot_iteration
]
if len(metric_iterations) == 0:
    raise RuntimeError(
        f"No common model checkpoints were found up to iteration {plot_iteration}."
    )
print(
    f"Calculating Figure 2 GP metrics through displayed iteration {plot_iteration - 1} "
    f"({len(metric_iterations)} common checkpoints)..."
)
# Transform the physical grid into the selected optimization space only once.
X_grid_metrics = Lp_to_Xnorm(L_grid, p_grid, optimization_space)
metric_rows = []
for iteration_number, iteration in enumerate(metric_iterations, start=1):
    for straddle_value in figure2_straddles:
        model_path = (
            run_folders[straddle_value]
            / "models"
            / f"model_iter_{iteration:03d}.pt"
        )
        if not model_path.exists():
            continue
        Z_pred = predict_mean(
            straddle_value=straddle_value,
            X_grid=X_grid_metrics,
            shape=Z_direct.shape,
            iteration=iteration,
        )
        metric_rows.append({
            "straddle": straddle_value,
            "iteration": iteration - 1,  # completed BO iterations used in fitted GP
            "nrmse": normalized_rmse(Z_direct, Z_pred),
            "ssim": global_ssim(Z_direct, Z_pred),
        })
    # Print progress after every five completed checkpoint iterations
    # and once after the final selected checkpoint.
    if iteration_number % 5 == 0 or iteration_number == len(metric_iterations):
        print(
            f"Figure 2 metrics: completed {iteration_number}/"
            f"{len(metric_iterations)} checkpoint iterations "
            f"(latest model iteration: {iteration})"
        )
metrics_df = pd.DataFrame(metric_rows)
if len(metrics_df) == 0:
    raise RuntimeError(
        "No Figure 2 prediction-quality metrics could be calculated."
    )

####################################################################################################
####################################################################################################
# Figure 2: convergence and GP predictive accuracy, matching the 2D paper style
####################################################################################################
# At displayed iteration i, best CE includes simulation i, and GP metrics
# use checkpoint i+1 (whose GP was fitted with observations through i).
# Match the canvas height and width of the logEI 2D Figure 2.
fig, axes = plt.subplots(3, 1, figsize=(11.69, 5.9671 * 0.95), sharex=True)
ax_bo_conv, ax_nrmse, ax_ssim = axes

# Distinct, matching 2D figure palette, with three Straddle thresholds.
line_styles  = ["-", (0, (6.5, 2, 1.5, 2)), (0, (2, 2.5))]
curve_colors = ["#482475", "#0C9F8D", "#D48A18"]
curve_colors = ["#C34675", "#0C9F8D", "#D48A18"]

for k, straddle_value in enumerate(figure2_straddles):
    color = curve_colors[k]
    style = line_styles[k]
    df = (all_data[straddle_value]
          .loc[lambda d: d["iteration"] <= plot_iteration]
          .sort_values("iteration", kind="mergesort").copy())
    if df.empty:
        continue
    # Initial point contributes to best-so-far but is not displayed (starts at 1).
    y_best = best_so_far(df)
    visible = df["iteration"].to_numpy() >= 1
    ax_bo_conv.plot(df.loc[visible, "iteration"], y_best[visible],
                    color=color, linestyle=style, linewidth=3.1,
                    alpha=0.98, marker=None, zorder=4)
    d = metrics_df.loc[(metrics_df["straddle"] == straddle_value) &
                       (metrics_df["iteration"] >= 1) &
                       (metrics_df["iteration"] <= plot_iteration)].sort_values("iteration")
    ax_nrmse.plot(d["iteration"], d["nrmse"], color=color, linestyle=style,
                  linewidth=3.1, alpha=0.98, zorder=4)
    ax_ssim.plot(d["iteration"], d["ssim"], color=color, linestyle=style,
                 linewidth=3.1, alpha=0.98, zorder=4)

# Direct-grid reference lines, identical convention to the main 2D figure.
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
ssim_values = metrics_df["ssim"].to_numpy(dtype=float)
ssim_values = ssim_values[np.isfinite(ssim_values)]
ssim_lower = min(-0.1, float(ssim_values.min()) - 0.05) if len(ssim_values) else -0.1
ssim_upper = max(1.1, float(ssim_values.max()) + 0.05) if len(ssim_values) else 1.1
ax_ssim.set_ylim(ssim_lower, ssim_upper)
ax_ssim.set_ylim(ssim_lower, ssim_upper)

ax_bo_conv.set_ylabel(r"$\mathrm{CE}_{\mathrm{best}}\,/\,10^{-6}$")
ax_ssim.set_ylim(ssim_lower, ssim_upper)
ax_nrmse.set_ylabel(r"$\mathrm{NRMSE}_{\mathrm{grid}}$")
ax_ssim.set_ylabel(r"$\mathrm{SSIM}_{\mathrm{grid}}$")
ax_ssim.set_xlabel("Iteration")

for i, ax in enumerate(axes):
    ax.set_xlim(1, plot_iteration)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(direction="in", top=True, right=True)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.8, width=0.7)
    ax.tick_params(axis="x", which="major", length=5.0, width=0.9)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax.grid(False, which="minor")
    ax.yaxis.set_label_coords(-0.070, 0.5)
    ax.text(0.985, 0.14, chr(97 + i), transform=ax.transAxes,
            ha="right", va="top", fontweight="bold", clip_on=True,
            zorder=100, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

ax_bo_conv.tick_params(axis="x", which="both", labelbottom=False)
ax_nrmse.tick_params(axis="x", which="both", labelbottom=False)

# Slightly widen the centered side legend to keep its complete title on one line.
# Give the line swatches room for at least two visible dashes while preserving
# the same plot height, data, typography and GP checkpoint indexing.
legend_width_previous = 0.158 * 0.95
legend_width = legend_width_previous * 0.95  # another 5% narrower than CURRENT version
legend_left = 0.823 + (0.158 - legend_width)  # preserve right edge
plot_right = legend_left - 0.020  # give reclaimed width to the plots
fig.subplots_adjust(left=0.120, right=plot_right, bottom=0.115,
                    top=0.955, hspace=0.055)
fig.canvas.draw()
plot_top = max(ax.get_position().y1 for ax in axes)
plot_bottom = min(ax.get_position().y0 for ax in axes)
plot_height = plot_top - plot_bottom
legend_height = 0.50 * plot_height
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

# Title centered at 0.50. The full row (swatch + value) is centered
# after rendering, with a SMALL swatch-to-label gap and comfortable margins.
legend_ax.text(0.50, 0.85, r"$\mathrm{CE}_{\mathrm{target}}\,/\,10^{-6}$",
               fontsize=FIG2_FONT_SIZE, ha="center", va="center")
legend_values = [r"$0.75$", r"$1.5$", r"$2.7$"]
legend_rows = [0.62, 0.40, 0.18]
for k, (y, value, style, color) in enumerate(
    zip(legend_rows, legend_values, line_styles, curve_colors)
):
    # The last (amber) swatch gets a shorter dash cycle than the plotted curve,
    # making multiple dashes visible even in this compact legend.
    legend_style = (0, (5.0, 2.2)) if k == 2 else style
    # Compact gap between swatch and number; center their complete visible
    # width rather than imposing large empty space between them.
    swatch, = legend_ax.plot([0.11, 0.41], [y, y], linestyle=legend_style,
                             color=color, linewidth=3.1, solid_capstyle="butt",
                             dash_capstyle="butt", clip_on=False)
    value_text = legend_ax.text(0.63, y, value, ha="center", va="center",
                                fontsize=FIG2_FONT_SIZE)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    line_bbox = swatch.get_window_extent(renderer)
    text_bbox = value_text.get_window_extent(renderer)
    combined_mid_px = (min(line_bbox.x0, text_bbox.x0) +
                       max(line_bbox.x1, text_bbox.x1)) / 2
    box_bbox = legend_ax.get_window_extent(renderer)
    shift_axes = (box_bbox.x0 + box_bbox.width / 2 - combined_mid_px) / box_bbox.width
    swatch.set_xdata(np.asarray(swatch.get_xdata()) + shift_axes)
    value_text.set_x(value_text.get_position()[0] + shift_axes)

fig_path_2 = plots_root / "STRADDLEfigure2.pdf"
unify_figure_fonts(fig, FIG2_FONT_SIZE)
fig.savefig(fig_path_2, dpi=300, bbox_inches="tight", pad_inches=0.025)
fig.savefig(fig_path_2.with_suffix(".png"), dpi=300,
            bbox_inches="tight", pad_inches=0.025)
plt.close(fig)
print(f"Saved: {fig_path_2} and .png")
####################################################################################################
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
for straddle_value in figure2_straddles:
    if straddle_value not in all_data:
        continue
    df = all_data[straddle_value]
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
            f"{straddle_titles[straddle_value]:35s} "
            f"{len(df):5d} evals   "
            f"{total_s / 60.0:9.2f} min   "
            f"({total_s / 3600.0:7.2f} h)"
        )
    else:
        print(
            f"{straddle_titles[straddle_value]:35s} "
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
for straddle_value in figure2_straddles:
    if straddle_value not in all_data:
        continue
    df = all_data[straddle_value]
    best = df.loc[df["objective_scaled_ratio"].idxmax()]
    print(
        f"{straddle_titles[straddle_value]:35s} "
        f"CE/1e-6 = {best['objective_scaled_ratio']:.6f}   "
        f"iteration = {int(best['iteration'])}"
    )
print(f"\nFigure 1: GP from checkpoint {plot_iteration} (fitted through iteration {plot_iteration-1}).")
print("Figure 2: best CE through BO iteration 60 (when present); GP metrics at checkpoint minus one, at most 59.")
metric_last = int(metrics_df["iteration"].max())
if metric_last < plot_iteration:
    print(f"NOTE: GP metrics end at iteration {metric_last}; a checkpoint for "
          f"iteration {plot_iteration + 1} would be required to include BO iteration {plot_iteration} in the GP. "
          f"The best-CE curve still includes simulation {plot_iteration}.")
print("Done.")
