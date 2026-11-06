import os
import glob # file finder
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
import matplotlib
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
    add_panel_label,
    format_map_axis,
    evaluation_sizes,
    add_physical_points,
    make_initial_marker,
    BO_CMAP,
    INITIAL_MARKER,
    INITIAL_COLOR,
    BEST_MARKER,
    BEST_BO_COLOR,
    POINT_EDGE_COLOR,
    INITIAL_SIZE,
    BEST_SIZE,
    BO_SIZE_MIN,
    BO_SIZE_MAX,
)
# Settings
####################################################################################################################################
dtype                    = torch.double
dim                      = 2 # BO dimensions
HARP_WORKDIR             = Path("~/HarP-main").expanduser().resolve()
PROJECT_ROOT             = Path(__file__).resolve().parent.parent
results_root             = PROJECT_ROOT / "results"
plots_root               = PROJECT_ROOT / "figures" # paper figures saved as figure1.pdf,png and figure2.pdf,png
grid_scan_csv            = PROJECT_ROOT / "data" / "scan_results.csv" # where the whole grid scan is
plot_iteration           = 60                  # how many iterations, if none put the last iteration that exists for every optimization
L_min, L_max             = 1.5 / 100, 1.5      # Original dimensionless BO and grid limits
p_min, p_max             = 300 / 100, 300.0   # Original dimensionless BO and grid limits
optimization_spaces      = ["L__p", "log_L__log_p", "p_over_L__pL", "sqrt_p_over_L__sqrt_pL", "log_p_over_L__log_pL"]
case_titles              = [r"$p,L$", r"$\log p,\log L$", r"$pL,p/L$", r"$\sqrt{pL},\sqrt{p/L}$", r"$\log(pL),\log(p/L)$"]
plots_root.mkdir(exist_ok=True)
# Style of the plots
####################################################################################################################################
apply_paper_style()
mpl.rcParams["font.family"] = ["serif"]
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
# Unified font size for both figures
FONT_SIZE      = 18  
FIG1_FONT_SIZE = 18
FIG2_FONT_SIZE = 18
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
    """Set all text in a completed figure to the requested font size.
    Includes axis tick labels, colorbar labels/ticks, panel markers,
    legend text, annotations, and text created by plotting_utils.
    """
    from matplotlib.text import Text
    # Render first to instantiate tick labels (also those on colorbars).
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(font_size)
    fig.canvas.draw()

def pack_square_panels_left(fig, axes, left=0.095, gap=0.012):
    """Left-align square panels and minimize horizontal inter-panel whitespace.
    Uses the axes' rendered dimensions after set_box_aspect(1), so it does
    not distort the square plot areas or change their y positions.
    """
    fig.canvas.draw()
    axes = list(axes)
    # Group by the rendered row position, with numerical tolerance.
    rows = []
    for ax in axes:
        pos = ax.get_position()
        for row in rows:
            if abs(row[0].get_position().y0 - pos.y0) < 0.025:
                row.append(ax)
                break
        else:
            rows.append([ax])
    for row in rows:
        row.sort(key=lambda ax: ax.get_position().x0)
        x = left
        for ax in row:
            pos = ax.get_position()
            ax.set_position([x, pos.y0, pos.width, pos.height], which="both")
            x += pos.width + gap
    fig.canvas.draw()

MAP_CMAP                 = "viridis"     # colormap for grid
FINAL_MARKER             = "D"           # marker for the final evaluated BO point
FINAL_BO_COLOR           = "black"       #  final evaluated BO point
FINAL_SIZE               = 40

# Helpers
####################################################################################################################################
def make_model(X, Y_objective): # creates a Gaussian Process model from the training data
    kernel = make_kernel("matern52", dim)  # same kernel as the BO loop (bo.py)
    # Standardize: shifts the objective values to mean 0 and standard deviation 1 for GP to learn the model
    # m=1 only one value to predict (CE)
    model  = SingleTaskGP(train_X=X,train_Y=Y_objective,covar_module=kernel,outcome_transform=Standardize(m=1))
    return model

def predict_mean(space, X_grid, shape, iteration):
    # Takes one saved GP model at one iteration
    # Evaluates its predicted mean at all points in grid
    # returns the predictions reshaped back into the original 2D grid.
    # space: which optimization parameterization space
    # X_grid: all grid points transformed and normalized for this GP, shape (N, 2)
    # shape: original 2D grid shape
    # iteration: which saved GP model to load
    model_path  = run_folders[space] / "models" / f"model_iter_{iteration:03d}.pt" # the model filename
    checkpoint  = torch.load(model_path, map_location="cpu", weights_only=False)
    # bo.py saves the checkpoint after appending the newly evaluated BO point,
    # but its GP was fitted before that evaluation. Reconstruct the exact
    # fitted GP by excluding the final, not-yet-used observation.
    X_saved     = checkpoint["X_normalized"].to(dtype=dtype)
    Y_saved     = checkpoint["Y_objective"].to(dtype=dtype)
    if iteration < 1 or len(X_saved) != iteration + 1 or len(Y_saved) != len(X_saved):
        raise ValueError(
            f"Unexpected checkpoint data length at iteration {iteration}: "
            f"X={len(X_saved)}, Y={len(Y_saved)}; expected {iteration + 1}."
        )
    X           = X_saved[:-1]
    Y_objective = Y_saved[:-1]
    model       = make_model(X, Y_objective) # GP fitted to observations available before BO point i
    model.load_state_dict(checkpoint["model_state_dict"]) # restores paramaters that GP learned, like lenght scales
    model.eval() # model is evaluated
    valid       = (
        (X_grid[:, 0] >= 0.0) & (X_grid[:, 0] <= 1.0) &
        (X_grid[:, 1] >= 0.0) & (X_grid[:, 1] <= 1.0)
    ) # Checks whether both normalized coordinates lie between 0 and 1.
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

# Load BO files, Matern 5/2 only
####################################################################################################################################
all_data    = {}
run_folders = {}
for space in optimization_spaces:
    # Reads results/ directly, so the figure always shows the runs just made.
    run_folder = results_root / f"bo_2d_{space}_log_ei_matern52"
    data_path  = run_folder / "data.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing data file for {space}:\n"
            f"{data_path}"
        )
    df = pd.read_csv(data_path).sort_values("iteration").copy()
    if "objective_scaled" in df.columns:
        df["objective_scaled_ratio"] = df["objective_scaled"]
    elif "conversion_efficiency" in df.columns:
        df["objective_scaled_ratio"] = (df["conversion_efficiency"] / 1e-6)
    else:
        raise ValueError(f"Cannot find objective column in {data_path}")
    if "cumulative_wall_time_s" not in df.columns:
        if "iteration_wall_time_s" in df.columns:
            df["cumulative_wall_time_s"] = (df["iteration_wall_time_s"].cumsum())
        else:
            df["cumulative_wall_time_s"] = np.nan
    all_data[space]    = df
    run_folders[space] = run_folder
    print(f"Loaded {space}: {run_folder}")
# Choose iteration
####################################################################################################################################
# Find which iterations exist for all optimization spaces
iteration_sets = []
# first collect available iterations separately for every optimization space
for space in optimization_spaces:
    folder = run_folders[space] / "models"
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
# Intersection keeps only iterations that are in every optimization space
common_iterations = sorted(set.intersection(*iteration_sets))
if plot_iteration is None:
    plot_iteration = common_iterations[-1]
print(f"Selected checkpoint: {plot_iteration}; GP and map markers through BO iteration {plot_iteration - 1}; best CE through BO iteration {plot_iteration}.")

# GP maps in original L/z_R, p z_R domain
####################################################################################################################################
plot_data = {}
for space in optimization_spaces:
    X_grid           = Lp_to_Xnorm(L_grid, p_grid, space)
    plot_data[space] = predict_mean(space, X_grid, Z_direct.shape, plot_iteration)
all_maps = [Z_direct[np.isfinite(Z_direct)]] # all values from the grid scan
for space in optimization_spaces: # all values from different optimization spaces
    all_maps.append(plot_data[space][np.isfinite(plot_data[space])])
all_maps = np.concatenate(all_maps) # make 1d array of maps from grid scan and BO
vmin     = np.nanmin(all_maps)
vmax     = np.nanmax(all_maps)

# Legend handles for Figure 1
####################################################################################################################################
n_bo_for_legend = get_bo_iteration_range(all_data, plot_iteration - 1) #largest number of BO evaluations up to the selected iteratin
# How the initial BO point marker will look in the legend
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

# Best point found by BO visualization
best_bo_handle = plt.Line2D(
    [0], [0],
    marker=BEST_MARKER,
    linestyle="",
    markerfacecolor=BEST_BO_COLOR,
    markeredgecolor='white',# POINT_EDGE_COLOR,
    markeredgewidth=0.7,
    markersize=11,
)
legend_handles = [initial_handle, bo_size_handle, best_bo_handle]
legend_labels = [
    r"Initial BO point $i=0$",
    rf"BO points ($i=1$ $\rightarrow$ $i={n_bo_for_legend}$)",
    r"Best BO point",
]

# Draw all items in the tuple (legend_handles) together inside one legend entry
legend_handler_map = {tuple: HandlerTuple(ndivide=None, pad=0.08)}

####################################################################################################################################
####################################################################################################################################
# Figure 1: grid scan + five optimization maps, 2x3 panels
####################################################################################################################################
####################################################################################################################################

fig, axes = plt.subplots(2, 3, figsize=(13.4, 10.7), sharex=True, sharey=True)
# Reshape axes to 1D array
# axes[0]: top left, axes[1]: top middle, axes[2]: top right
# axes[3]: bottom left, axes[4]: bottom middle, axes[5]: bottom right
axes = axes.reshape(-1)

# Panel a: direct grid scan
####################################################################################################################################
im = axes[0].pcolormesh(
    L_grid_plot,
    p_grid_plot,
    Z_direct,
    shading    = "auto",
    cmap       = MAP_CMAP,
    vmin       = vmin,
    vmax       = vmax,
    zorder     = 1,
    rasterized = True
)
axes[0].set_title("Grid scan")
add_panel_label(axes[0], "a")
format_map_axis(axes[0], show_ylabel=True)

# Panels b-f: BO maps
####################################################################################################################################
for i, space in enumerate(optimization_spaces):
    ax   = axes[i+1]
    mean = plot_data[space]
    ax.pcolormesh(
        L_grid_plot,
        p_grid_plot,
        mean,
        shading    = "auto",
        cmap       = MAP_CMAP,
        vmin       = vmin,
        vmax       = vmax,
        zorder     = 1,
        rasterized = True
    )
    # Plot the BO points and the best point on the grid
    df = all_data[space]
    df = df[df["iteration"] < plot_iteration].copy()
    add_physical_points(ax, df)
    ax.set_title(case_titles[i])
    add_panel_label(ax, chr(98 + i))
    show_ylabel = (i == 2)
    format_map_axis(ax, show_ylabel=show_ylabel)

# Axis labels only where needed
for j, ax in enumerate(axes):
    if j in [0, 3]:
        ax.set_ylabel(r"$p z_R$ [mbar cm]")
        ax.tick_params(labelleft=True)
    else: # we won't show y axis label and tick labels for b c e f
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    if j < 3: # for top row, don't show x axis labels and tick labels
        ax.set_xlabel("")
    else:
        ax.set_xlabel(r"$L/z_R$")
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.185, top=0.885,
                    wspace=0.16, hspace=0.18)
for ax in axes:
    ax.set_box_aspect(1)
pack_square_panels_left(fig, axes, left=0.095, gap=0.012)
# Center supplementary elements relative to the panel region.
fig.canvas.draw()
panel_left = min(ax.get_position().x0 for ax in axes)
panel_right = max(ax.get_position().x1 for ax in axes)
panel_center = (panel_left + panel_right) / 2

# Colorbar
####################################################################################################################################
cax  = fig.add_axes([panel_center - 0.28, 0.075, 0.56, 0.023])
cbar = fig.colorbar(im, cax = cax, orientation = "horizontal")
cbar.set_label(r"CE/$10^{-6}$", labelpad=3)
cbar.ax.tick_params(direction="in", labelsize=FONT_SIZE)

# Legend
####################################################################################################################################
leg = fig.legend(
    legend_handles,
    legend_labels,
    handler_map    = legend_handler_map,
    loc            = "upper center",
    ncol           = 4,
    frameon        = True,
    bbox_to_anchor = (panel_center, 0.98),
    columnspacing  = 0.95, #0.85
    handlelength   = 1.6,
    handletextpad  = 0.35,
    borderaxespad  = 0.0,
)
frame = leg.get_frame()
frame.set_facecolor("#b9c7d9")
frame.set_alpha(0.95)
frame.set_edgecolor("none")

# Save figure 1
####################################################################################################################################
fig_path_1 = plots_root / "figure1.pdf"
unify_figure_fonts(fig, FIG1_FONT_SIZE)
fig.savefig(fig_path_1, dpi=600, bbox_inches="tight", pad_inches=0.025)
fig.savefig(fig_path_1.with_suffix(".png"), dpi=600,
            bbox_inches="tight", pad_inches=0.025)
plt.close(fig)
print(f"Saved: {fig_path_1} and .png")
####################################################################################################################################
####################################################################################################################################



# Figure 2: convergence and GP prediction accuracy
####################################################################################################################################
# Calculate NRMSE and SSIM against the direct grid scan.
metric_iterations = [i for i in common_iterations if 1 <= i <= plot_iteration]

if not metric_iterations:
    raise RuntimeError(
        f"No common GP checkpoints up to iteration {plot_iteration}."
    )

print(
    f"Calculating Figure 2 metrics for "
    f"{len(metric_iterations)} checkpoints..."
)

X_grids = {
    space: Lp_to_Xnorm(L_grid, p_grid, space)
    for space in optimization_spaces
}

metric_rows = []

for k, iteration in enumerate(metric_iterations, start=1):
    for space in optimization_spaces:
        Z_pred = predict_mean(
            space=space,
            X_grid=X_grids[space],
            shape=Z_direct.shape,
            iteration=iteration,
        )

        metric_rows.append({
            "space": space,
            "iteration": iteration - 1,  # checkpoint i: GP fitted through BO iteration i-1
            "nrmse": normalized_rmse(Z_direct, Z_pred),
            "ssim": global_ssim(Z_direct, Z_pred),
        })

    if k % 5 == 0 or k == len(metric_iterations):
        print(
            f"Figure 2 metrics: "
            f"{k}/{len(metric_iterations)} checkpoints completed"
        )

metrics_df = pd.DataFrame(metric_rows)

# Three vertically stacked panels with one shared x-axis
####################################################################################################################################
fig, axes = plt.subplots(3, 1, figsize=(11.69, 5.9671*0.95), sharex=True)
ax_bo_conv, ax_nrmse, ax_ssim = axes

# Line styles and colors.
line_styles = [
    "-",                  # purple: solid
    (0, (5, 2)),          # blue: dashed
    (0, (6.5, 2, 1.5, 2)),  # teal: clearly dash-dot
    (0, (9, 3.5)),        # orange: long dashed
    (0, (2, 2)),          # pink: short dashed
]
curve_colors = ["#482475", "#2786C0", "#0C9F8D", "#E18324", "#BD4999"]
curve_colors = [
    "#482475",  # purple
    "#2786C0",  # blue
    "#0C9F8D",  # teal
    "#D48A18",  # amber
    "#C34675",  # raspberry
]
# line_styles/curve_colors (and legend_coordinates below) are in legend order, which lists
# log(pL), log(p/L) before sqrt(pL), sqrt(p/L); optimization_spaces has these two the other way round.
legend_order = ["L__p", "log_L__log_p", "p_over_L__pL", "log_p_over_L__log_pL", "sqrt_p_over_L__sqrt_pL"]
for space, label in zip(optimization_spaces, case_titles):
    linestyle = line_styles[legend_order.index(space)]
    color = curve_colors[legend_order.index(space)]
    df = (
        all_data[space]
        .loc[all_data[space]["iteration"] <= plot_iteration]
        .sort_values("iteration", kind="mergesort")
        .copy()
    )
    if df.empty:
        continue

    # Compute best-so-far INCLUDING the initial point (iteration 0).
    # Only its DISPLAY starts at iteration 1.
    y_best = np.asarray(best_so_far(df))
    visible = (df["iteration"] >= 1).to_numpy()
    ax_bo_conv.plot(
        df.loc[visible, "iteration"], y_best[visible],
        linestyle=linestyle, color=color, linewidth=3.1,
        alpha=0.98, marker=None, label=label, zorder=4,
    )

    d = (
        metrics_df.loc[metrics_df["space"] == space]
        .sort_values("iteration", kind="mergesort")
    )
    ax_nrmse.plot(
        d["iteration"], d["nrmse"], linestyle=linestyle,
        color=color, linewidth=3.1, alpha=0.98, zorder=4,
    )
    ax_ssim.plot(
        d["iteration"], d["ssim"], linestyle=linestyle,
        color=color, linewidth=3.1, alpha=0.98, zorder=4,
    )

# Perfect agreement with the direct grid scan: NRMSE = 0, SSIM = 1
####################################################################################################################################
GRID_SCAN_COLOR = "0.72"
ax_nrmse.axhline(0.0, color=GRID_SCAN_COLOR, linestyle="dotted",
                 linewidth=1.3, zorder=2)
ax_ssim.axhline(1.0, color=GRID_SCAN_COLOR, linestyle="dotted",
                linewidth=1.3, zorder=2)
reference_bbox = dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1)

ax_nrmse.annotate(
    "Grid", xy=(0.06, -0.0135), xycoords=("axes fraction", "data"),
    xytext=(0, 4), textcoords="offset points",
    ha="center", va="bottom",
    color="0.48", zorder=10,
)

ax_ssim.annotate(
    "Grid", xy=(0.06, 0.997), xycoords=("axes fraction", "data"),
    xytext=(0, -4), textcoords="offset points",
    ha="center", va="top",
    color="0.48", zorder=10,
)

# Limits, tick marks and scientific labels.
####################################################################################################################################
ax_bo_conv.set_ylim(-0.5, 4)
ax_nrmse.set_ylim(-0.1, 0.5)
ax_ssim.set_ylim(-0.1, 1.1)
ax_nrmse.yaxis.set_major_locator(mpl.ticker.FixedLocator([0.0, 0.2, 0.4]))
ax_nrmse.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
ax_ssim.yaxis.set_major_locator(MultipleLocator(0.2))
ax_ssim.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

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
    ax.tick_params(axis="x", which="major", length=5.0, width=0.9)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax.grid(False, which="minor")
    ax.yaxis.set_label_coords(-0.058, 0.5)
    # Lower-right panel letters, inside each axes.
    ax.text(
      0.99, 0.002, chr(97 + i),
      transform=ax.transAxes,
      ha="right",
      va="bottom",
      fontweight="bold",
      clip_on=True,
      zorder=100,
      path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

ax_bo_conv.tick_params(axis="x", which="both", labelbottom=False)
ax_nrmse.tick_params(axis="x", which="both", labelbottom=False)

# Adjust panel and legend widths while maintaining consistent spacing
####################################################################################################################################
old_legend_width = 0.16205
legend_width = old_legend_width * 0.90
plot_right = 0.739 + (old_legend_width - legend_width)
legend_left = plot_right + 0.022
fig.subplots_adjust(
    left=0.119, right=plot_right,
    bottom=0.110, top=0.945, hspace=0.03,
)

# Legend panel shares exact top and bottom with the three plotting axes.
# Five equally spaced entries with equal external (top/bottom) padding.
####################################################################################################################################
fig.canvas.draw()
plot_top = max(ax.get_position().y1 for ax in axes)
plot_bottom = min(ax.get_position().y0 for ax in axes)
legend_ax = fig.add_axes([
    legend_left, plot_bottom, legend_width, plot_top - plot_bottom
])
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)
legend_ax.set_facecolor("#b9c7d9")
legend_ax.patch.set_alpha(0.95)
legend_ax.set_xticks([])
legend_ax.set_yticks([])
for spine in legend_ax.spines.values():
    spine.set_visible(False)

legend_coordinates = [
    (r"$p$,", r"$L$"),
    (r"$\log p$,", r"$\log L$"),
    (r"$pL$,", r"$p/L$"),
    (r"$\log(pL)$,", r"$\log(p/L)$"),
    (r"$\sqrt{pL}$,", r"$\sqrt{p/L}$"),
]

entry_height = 0.12
entry_gap = (1.0 - len(legend_coordinates) * entry_height) / (len(legend_coordinates) + 1)
for i, ((first, second), style, color) in enumerate(
    zip(legend_coordinates, line_styles, curve_colors)
):
    block_top = 1.0 - entry_gap - i * (entry_height + entry_gap)
    center_y = block_top - 0.5 * entry_height
    upper_y = center_y + 0.040
    lower_y = center_y - 0.040
    # Balanced left/right margins around the legend sample + two-line text.
    legend_ax.plot(
    [0.06, 0.36], [upper_y, upper_y],
    linestyle=style,
    color=color,
    linewidth=3.1,
    solid_capstyle="butt",
    dash_capstyle="butt",
    clip_on=False,
    )


    legend_ax.text(
      0.42, upper_y, first,
      ha="left", va="center",
      fontsize=FIG2_FONT_SIZE,
    )

    legend_ax.text(
      0.42, lower_y, second,
      ha="left", va="center",
      fontsize=FIG2_FONT_SIZE,
    )


# Save Figure 2
####################################################################################################################################
print(f"Figure 2: best-so-far CE through {plot_iteration}; NRMSE and SSIM through {int(metrics_df['iteration'].max())}.")
fig_path_2 = plots_root / "figure2.pdf"
unify_figure_fonts(fig, FIG2_FONT_SIZE)
fig.savefig(fig_path_2, dpi=300, bbox_inches="tight", pad_inches=0.025)
fig.savefig(fig_path_2.with_suffix(".png"), dpi=300,
            bbox_inches="tight", pad_inches=0.025)
plt.close(fig)
print(f"Saved: {fig_path_2} and .png")

