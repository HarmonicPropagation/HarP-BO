"""4D BO paper plot: best observed CE and held-out Sobol NRMSE.
Run from the repository root with the virtual environment active
(python src/plot_4Dbo_convergence.py). No HarP simulations are run; the script
reads existing BO checkpoints and data/holdout_4d/holdout.csv.
Uses the same visual style as the 2D convergence figure.
"""

import glob
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from plotting_utils import apply_paper_style
from utils import (
    get_objective_transform,
    make_kernel,
    normalized_rmse,
    physical_to_x,
)
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_ROOT   = PROJECT_ROOT / "results"
HOLDOUT_CSV    = PROJECT_ROOT / "data" / "holdout_4d" / "holdout.csv"
OUTPUT_PLOT    = PROJECT_ROOT / "figures" / "figure_4Dbo_convergence.pdf"
MAX_ITERATIONS = 120
DIM            = 4
DTYPE          = torch.double
# Match the original 4D runs and the 2D convergence-plot line styles.
RUNS_TO_PLOT   = [
    (
        "bo_4d_L__p__I__tau_straddle100per_max3_matern52",
        r"$p,L,I,\tau$",
        "L__p__I__tau",
    ),
    (
        "bo_4d_log_p_over_L__log_pL__I__tau_straddle100per_max3_matern52",
        r"$\log(p/L),\log(pL),I,\tau$",
        "log_p_over_L__log_pL__I__tau",
    ),
    (
        "bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle100per_max3_matern52",
        r"$\log(p/L),\log(pL),I,1/\sqrt{\tau}$",
        "log_p_over_L__log_pL__I__inv_sqrt_tau",
    ),
    (
        "bo_4d_log_p_over_L__log_pL__I__inv_sqrt_tau_straddle100per_max3_matern52_sqrtCE",
        r"$\log(p/L),\log(pL),I,1/\sqrt{\tau},\sqrt{\mathrm{CE}}$",
        "log_p_over_L__log_pL__I__inv_sqrt_tau",
    ),
]
# Match the 2D convergence figure: first four colors and line styles.
LINE_STYLES = ["-", (0, (6.5, 2, 1.5, 2)), (0, (9, 3.5)), (0, (2, 2))]
CURVE_COLORS = ["#482475", "#D48A18", "#0C9F8D", "#C34675"]
apply_paper_style()
mpl.rcParams["font.family"] = ["serif"]
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
# A slightly smaller native canvas and larger type improve readability
# when the PDF is scaled to the manuscript text width in Overleaf.
FONT_SIZE        = 16
LEGEND_FONT_SIZE = 16
mpl.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
})

def unify_figure_fonts(fig, font_size):
    from matplotlib.text import Text
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(font_size)
    fig.canvas.draw()

def evaluate_run(folder, optimization_space, holdout_df, reference, reference_range, base_config):
    """Evaluate the BO-trained GP against independently simulated HarP values."""
    run_dir = RESULTS_ROOT / folder
    data_csv = run_dir / "data.csv"
    if not data_csv.exists():
        raise FileNotFoundError(f"Missing BO data: {data_csv}")
    bo_df = (pd.read_csv(data_csv)
             .query("iteration <= @MAX_ITERATIONS")
             .sort_values("iteration", kind="mergesort")
             .copy())
    if bo_df.empty:
        raise ValueError(f"No BO evaluations in {data_csv}")
    bo_df["best_ce"] = bo_df["conversion_efficiency"].cummax() / 1e-6
    model_paths = sorted(
        (Path(p) for p in glob.glob(str(run_dir / "models" / "model_iter_*.pt"))
         if 1 <= int(Path(p).stem.replace("model_iter_", "")) <= MAX_ITERATIONS),
        key=lambda p: int(p.stem.replace("model_iter_", "")),
    )
    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints in {run_dir / 'models'}")
    # The last stored model in checkpoint 120 is fitted through iteration 119.
    last_checkpoint = int(model_paths[-1].stem.replace("model_iter_", ""))
    print(f"  {folder}: last GP checkpoint {last_checkpoint}, "
          f"trained through BO iteration {last_checkpoint - 1}")
    first_checkpoint = torch.load(model_paths[0], map_location="cpu", weights_only=False)
    config = dict(first_checkpoint.get("config") or {})
    for key, value in base_config.items():
        config.setdefault(key, value)
    config.setdefault("optimization_space", optimization_space)
    x_test = torch.stack([
        physical_to_x({
            "Lzzr": row.Lzzr,
            "pzr": row.pzr,
            "intensity_Wcm2": row.intensity_Wcm2,
            "tau_fs": row.tau_fs,
        }, config)
        for row in holdout_df.itertuples(index=False)
    ]).to(dtype=DTYPE)
    rows = []
    for k, path in enumerate(model_paths, start=1):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Checkpoint i is written AFTER simulation i, but the stored GP was
        # fitted BEFORE that simulation. Drop the final, newly appended result.
        checkpoint_iteration = int(path.stem.replace("model_iter_", ""))
        x_saved = checkpoint["X_normalized"].to(dtype=DTYPE)
        y_saved = checkpoint["Y_objective"].to(dtype=DTYPE)
        if len(x_saved) != checkpoint_iteration + 1 or len(y_saved) != len(x_saved):
            raise ValueError(
                f"Unexpected checkpoint size at iteration {checkpoint_iteration}: "
                f"X={len(x_saved)}, Y={len(y_saved)}"
            )
        x_train = x_saved[:-1]
        y_obj = y_saved[:-1]
        if checkpoint.get("use_log_objective", False):
            transform_name = "log10"
        else:
            transform_name = checkpoint.get("config", {}).get("objective_transform") or "id"
        forward, inverse = get_objective_transform(transform_name)
        y_obj = forward(torch.clamp(y_obj, min=1e-15))
        model = SingleTaskGP(
            train_X=x_train,
            train_Y=y_obj,
            covar_module=make_kernel(config.get("kernel", "matern52"), DIM),
            outcome_transform=Standardize(m=1),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        # Chunk posterior evaluations to avoid unnecessarily large test tensors.
        predictions = []
        with torch.no_grad():
            for x_chunk in x_test.split(256):
                predictions.append(model.posterior(x_chunk).mean.cpu().numpy().reshape(-1))
        pred_ce = np.asarray(inverse(np.concatenate(predictions)), dtype=float)
        if not np.all(np.isfinite(pred_ce)):
            raise ValueError(f"Non-finite GP predictions in {path}")
        rows.append({
            # The GP is fitted through BO iteration i-1, not iteration i.
            "iteration": checkpoint_iteration - 1,
            "nrmse": normalized_rmse(reference, pred_ce, ref_range=reference_range),
        })
        if k % 20 == 0 or k == len(model_paths):
            print(f"  {folder}: evaluated {k}/{len(model_paths)} checkpoints")
    return bo_df, pd.DataFrame(rows).sort_values("iteration", kind="mergesort")
def main():
    base_config = {
        "Lzzr_bounds": (1.5 / 100.0, 1.5),
        "pzr_bounds": (300.0 / 100.0, 300.0),
        "intensity_Wcm2_bounds": (0.01e14, 10e14),
        "tau_fs_bounds": (22.0 / 2.0, 4.0 * 22.0),
    }
    if not HOLDOUT_CSV.is_file():
        raise FileNotFoundError(f"Missing Sobol validation data: {HOLDOUT_CSV}")
    holdout_df = pd.read_csv(HOLDOUT_CSV)
    reference = holdout_df["conversion_efficiency"].to_numpy(dtype=float) / 1e-6
    reference_range = float(np.max(reference) - np.min(reference))
    if reference_range <= 0 or not np.all(np.isfinite(reference)):
        raise ValueError("Invalid Sobol reference CE values")
    print(f"Sobol reference: {len(reference)} independent HarP values")

    # Match Figure 2 (2D): two vertically stacked panels and a right legend.
    # CE includes completed evaluations through 120; NRMSE uses the
    # last fitted GP, trained through 119 when checkpoint 120 is the latest.
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 4.85), sharex=True)
    ax_ce, ax_nrmse = axes
    summaries = []
    for i, (folder, label, space) in enumerate(RUNS_TO_PLOT):
        print(f"Loading: {folder}")
        bo_df, metric_df = evaluate_run(
            folder, space, holdout_df, reference, reference_range, base_config
        )
        # Original BO best-so-far: after each completed simulation.
        visible = bo_df["iteration"] >= 1
        ax_ce.plot(
            bo_df.loc[visible, "iteration"], bo_df.loc[visible, "best_ce"],
            color=CURVE_COLORS[i], linestyle=LINE_STYLES[i],
            linewidth=3.1, alpha=0.98, marker=None, label=label, zorder=4,
        )
        # GP checkpoint i was fitted through iteration i-1.
        ax_nrmse.plot(
            metric_df.loc[metric_df["iteration"] >= 1, "iteration"],
            metric_df.loc[metric_df["iteration"] >= 1, "nrmse"],
            color=CURVE_COLORS[i], linestyle=LINE_STYLES[i],
            linewidth=3.1, alpha=0.98, marker=None, zorder=4,
        )
        # Each run may have a different final checkpoint. Report the
        # actual last iteration included in its saved fitted GP.
        valid_metrics = metric_df.loc[metric_df["iteration"] >= 1]
        if valid_metrics.empty:
            final_nrmse = float("nan")
            final_metric_iteration = None
        else:
            last_metric = valid_metrics.iloc[-1]
            final_nrmse = float(last_metric["nrmse"])
            final_metric_iteration = int(last_metric["iteration"])
        final_ce_iteration = int(bo_df["iteration"].iloc[-1])
        summaries.append((label, bo_df["best_ce"].iloc[-1],
                          final_ce_iteration, final_nrmse, final_metric_iteration))

    ax_nrmse.axhline(0.0, color="0.72", linestyle="dotted", linewidth=1.3, zorder=2)
    ax_nrmse.annotate(
        "Sobol", xy=(0.085, 0.00), xycoords=("axes fraction", "data"),
        xytext=(0, 1), textcoords="offset points", ha="center", va="bottom",
        color="0.48", zorder=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1),
    )
    ax_ce.set_ylim(0, 8)
    ax_nrmse.set_ylim(-0.02, 0.40)
    ax_ce.set_ylabel(r"$\mathrm{CE}_{\mathrm{best}}\,/\,10^{-6}$")
    ax_nrmse.set_ylabel(r"$\mathrm{NRMSE}_{\mathrm{Sobol}}$")
    ax_nrmse.set_xlabel("Iteration")

    for i, ax in enumerate(axes):
        ax.set_xlim(1, MAX_ITERATIONS)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.tick_params(direction="in", top=True, right=True)
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       length=2.8, width=0.7)
        ax.tick_params(axis="x", which="major", length=5.0, width=0.9)
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.grid(False, which="minor")
        ax.yaxis.set_label_coords(-0.067, 0.5)
        ax.text(
            0.99, 0.965, chr(97 + i), transform=ax.transAxes,
            ha="right", va="top", fontweight="bold", clip_on=True,
            zorder=100, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )
    ax_ce.tick_params(axis="x", which="both", labelbottom=False)

    # Narrow side legend: approximately one fifth of the complete figure width.
    # The longer labels are broken into three/four lines instead of forcing
    # a wide panel. Keep the gap between plots and legend modest.
    legend_width = 0.145
    plot_right = 0.785
    legend_left = plot_right + 0.018
    # More room between panels tto prevent overlaping.
    fig.subplots_adjust(left=0.115, right=plot_right,
                        bottom=0.155, top=0.955, hspace=0.16)
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

    # Compact, deliberately wrapped legend labels. Four entries spread evenly
    # across the panel height; each line sample starts the first text line.
    # Each parameter on its own line, with proper math rendering.
    # Explicit group centers and enough line spacing for 12-point math labels.
    # A line step of 0.075 gives about 21 points of baseline separation.
    # Keep the four-line last group inside the legend without overlapping.
    legend_coordinates = [
    [r"$p$, $L,$", r"$I$, $\tau$"],
    [r"$\log(p/L)$,", r"$\log(pL)$,", r"$I$, $\tau$"],
    [r"$\log(p/L)$,", r"$\log(pL)$,", r"$I$, $1/\sqrt{\tau}$"],
    [r"$\log(p/L)$,", r"$\log(pL)$,", r"$I$, $1/\sqrt{\tau}$", r"$\sqrt{\mathrm{CE}}$"],
    ]

    step = 0.075
    group_gap = 0.085

    total_height = sum((len(lines) - 1) * step for lines in legend_coordinates)
    total_height += (len(legend_coordinates) - 1) * group_gap

    top_y = (1.0 + total_height) / 2

    for i, (lines, style, color) in enumerate(
      zip(legend_coordinates, LINE_STYLES, CURVE_COLORS)
    ):
      legend_ax.plot(
        [0.045, 0.29], [top_y, top_y],
        linestyle=style,
        color=color,
        linewidth=2.5,
        solid_capstyle="butt",
        dash_capstyle="butt",
        clip_on=False,
      )

      for j, label in enumerate(lines):
        legend_ax.text(
            0.34, top_y - j * step, label,
            ha="left",
            va="center",
            fontsize=LEGEND_FONT_SIZE,
        )

      top_y -= (len(lines) - 1) * step + group_gap



    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    # Do not override the intentionally smaller side-legend typography.
    unify_figure_fonts(fig, FONT_SIZE)
    for artist in legend_ax.texts:
        artist.set_fontsize(LEGEND_FONT_SIZE)
    fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUTPUT_PLOT.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_PLOT} and .png")
    print(f"{'Optimization space':66s} {'Best CE / 1e-6':>17s} {'CE iter':>8s} "
          f"{'NRMSE_Sobol':>14s} {'GP iter':>8s}")
    for label, best, ce_iteration, nrmse, metric_iteration in summaries:
        print(f"{label:66s} {best:17.6f} {ce_iteration:8d} "
              f"{nrmse:14.6f} {str(metric_iteration):>8s}")

if __name__ == "__main__":
    main()


