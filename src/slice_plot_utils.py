"""Publication-style 4D GP slices"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator, MultipleLocator
from plotting_utils import apply_paper_style, format_map_axis
from utils import physical_to_x

def get_extended_cmap(v_ref=3.5, v_max=6.0):
    """Extended-viridis construction (CE in units of 1e-6)."""
    viridis = plt.get_cmap("viridis")
    n_bins = 256
    fraction_base = v_ref / v_max
    n_base_bins = max(2, int(n_bins * fraction_base))
    n_over_bins = max(2, n_bins - n_base_bins)
    base_colors = viridis(np.linspace(0, 1, n_base_bins))
    highlight_color = np.array([1.0, 0.4, 0.3, 1.0])  # coral endpoint
    over_colors = np.linspace(base_colors[-1], highlight_color, n_over_bins)
    combined_colors = np.vstack([base_colors, over_colors])
    return mcolors.LinearSegmentedColormap.from_list("extended_viridis", combined_colors)


RESOLUTION       = 100
# Typography / paper style
FONT_SIZE        = 20
LEGEND_FONT_SIZE = FONT_SIZE  # Every text artist uses 20 pt

# Canvas size for this 2-panel figure
FIG_WIDTH        = 13.4  # Full-width paper figure; aligned with earlier colormaps
FIG_HEIGHT       = 5.0   # Wide manuscript figure; panels preserve 3.55-inch square size

TITLE_PAD        = 11
NEARBY_SIZE      = 30
SLICE_OPT_COLOR  = "white"
REFERENCE_LABEL  = "Intersection of slices (a) and (b)"


# Figure layout parameters
PANEL_SIZE_IN     = 3.30   # square panel dimensions
PANEL_GAP_IN      = 0.85   # spacing between panels
PANEL_LEFT        = 0.078  # left margin (figure coordinates)
PANEL_BOTTOM      = 0.180  # bottom margin (figure coordinates)
COLORBAR_LEFT_IN  = 9.06   # colorbar horizontal position
COLORBAR_WIDTH_IN = 0.17
LEGEND_LEFT_IN    = 9.54   # legend left boundary
LEGEND_RIGHT_IN   = 12.97  # legend right boundary


# Legend / colorbar distances from panels
TOP_CLEARANCE    = 0.070
BOTTOM_CLEARANCE = 0.095
COLORBAR_HEIGHT  = 0.022

apply_paper_style()
mpl.rcParams["font.family"]      = ["serif"]
mpl.rcParams["font.serif"]       = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["pdf.fonttype"]     = 42
mpl.rcParams["ps.fonttype"]      = 42
mpl.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
})


def _fmt_ce(value_abs, objective_scale):
    """Format one CE value in scientific units used throughout the paper."""
    return rf"$\mathrm{{CE}}={value_abs / objective_scale:.2f}\times10^{{-6}}$"


def _fmt_ce_asym(mean_abs, lower_abs, upper_abs, objective_scale):
    """Format CE with asymmetric GP uncertainty in the same units."""
    mean = mean_abs / objective_scale
    lower = (mean_abs - lower_abs) / objective_scale
    upper = (upper_abs - mean_abs) / objective_scale
    return (
        rf"$\mathrm{{CE}}=\left({mean:.2f}^{{+{upper:.2f}}}_{{-{lower:.2f}}}\right)"
        rf"\times10^{{-6}}$"
    )


def _fmt_pzr_title(value):
    """Make p z_R title visually consistent."""
    return f"{value:.0f}"


def unify_figure_fonts(fig, font_size):
    """Apply one consistent font size."""
    fig.canvas.draw()
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(font_size)
    fig.canvas.draw()


def physical_grids_to_normalized_tensor(L, p, I, tau, config, dtype=torch.float64):
    """Map physical grid points using the same transform as the BO runner."""
    L, p, I, tau = np.broadcast_arrays(L, p, I, tau)
    points = (
        physical_to_x(
            {
                "Lzzr": float(l),
                "pzr": float(pr),
                "intensity_Wcm2": float(intensity),
                "tau_fs": float(duration),
            },
            config,
        )
        for l, pr, intensity, duration in zip(
            L.ravel(), p.ravel(), I.ravel(), tau.ravel()
        )
    )
    return torch.stack(list(points)).to(dtype=dtype)


def _predict(model, f_inv, X, shape):
    with torch.no_grad():
        pred = f_inv(model.posterior(X).mean)
    return np.asarray(pred.detach().cpu().numpy()).reshape(shape)


def _nearby_points(ax, df, distance, x, y, epsilon, norm_ce, objective_scale, cmap):
    """Preserve CE-colored samples with alpha fading by distance to the plane."""
    if df is None or df.empty or epsilon <= 0:
        return False
    mask = distance < epsilon
    if not mask.any():
        return False

    selected = df.loc[mask]
    alpha = np.clip(1.0 - distance.loc[mask].to_numpy() / epsilon, 0, 1)
    colors = cmap(norm_ce(selected["conversion_efficiency"].to_numpy() / objective_scale))
    colors[:, 3] = alpha
    edges = np.ones_like(colors)
    edges[:, 3] = 0.8 * alpha

    ax.scatter(
        x(selected),
        y(selected),
        c=colors,
        edgecolors=edges,
        linewidth=0.7,
        s=NEARBY_SIZE,
        zorder=4,
        clip_on=False,
    )
    return True


def _marker(color, marker, size, edge="white", edgewidth=0.7):
    return plt.Line2D(
        [],
        [],
        marker=marker,
        linestyle="",
        markerfacecolor=color,
        markeredgecolor=edge,
        markeredgewidth=edgewidth,
        markersize=size,
    )


def _add_panel_label(ax, label, fontsize=FONT_SIZE):
    ax.text(
        0.975,          # slightly more to the right
        0.965,
        f"{label}",     # no parenthesis
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.20),
        clip_on=False,
        zorder=100,
    )


def _place_shared_legend(fig, axes, entries):
    """Vertically centered, blue-grey legend to the right of the square panels."""
    handles, labels = zip(*entries)
    leg = fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(LEGEND_LEFT_IN / FIG_WIDTH, PANEL_BOTTOM + PANEL_SIZE_IN / (2 * FIG_HEIGHT)),
        bbox_transform=fig.transFigure,
        ncol=1,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.15,
        handletextpad=0.50,
        labelspacing=1.20,
        borderpad=0.80,
        borderaxespad=0.0,
    )
    leg.get_frame().set_facecolor("#b9c7d9")
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_edgecolor("none")
    return leg



def plot_gp_slices(
    model,
    f_inv,
    best_row,
    config,
    out_path_base,
    reference_i=2.5e14,
    reference_tau=22.0,
    reference_l=None,
    reference_p=None,
    dtype=torch.float64,
    df_valid=None,
    objective_scale=1e-6,
    epsilon=0.15,
    confirmed_ce=None,
    ce_band=None,
    reference_confirmed_ce=None,
):
    """Plot 4D GP slices with a layout visually matched to the 2D paper figures.

    Panel (a): (L, p) slice at the reference I and tau.
    Panel (b): (I, tau) slice at the GP-optimal L and p.
    """

    best_L = float(best_row["Lzzr"])
    best_p = float(best_row["pzr"])
    best_I = float(best_row.get("intensity_Wcm2", best_row.get("I", reference_i)))
    best_tau = float(best_row.get("tau_fs", best_row.get("tau", reference_tau)))

    # By default b is cut at the saved GP optimum spatial coordinates.
    reference_l = best_L if reference_l is None else float(reference_l)
    reference_p = best_p if reference_p is None else float(reference_p)

    best_ce = float(best_row["conversion_efficiency"])
    headline_ce = best_ce if confirmed_ce is None else float(confirmed_ce)

    L_axis = np.linspace(*config["Lzzr_bounds"], RESOLUTION)
    p_axis = np.linspace(*config["pzr_bounds"], RESOLUTION)
    I_axis = np.linspace(*config["intensity_Wcm2_bounds"], RESOLUTION)
    tau_axis = np.linspace(*config["tau_fs_bounds"], RESOLUTION)

    # Left slice: (L, p) at the 2D reference intensity and pulse duration.
    L_grid, p_grid = np.meshgrid(L_axis, p_axis, indexing="ij")
    X1 = physical_grids_to_normalized_tensor(
        L_grid,
        p_grid,
        np.full_like(L_grid, reference_i),
        np.full_like(L_grid, reference_tau),
        config,
        dtype=dtype,
    )
    Z1 = _predict(model, f_inv, X1, L_grid.shape)
    i1 = np.unravel_index(np.nanargmax(Z1), Z1.shape)
    slice_L, slice_p, slice_ce1 = L_axis[i1[0]], p_axis[i1[1]], Z1[i1]

    # Right slice: (I, tau) at the GP-optimal spatial coordinates.
    I_grid, tau_grid = np.meshgrid(I_axis, tau_axis, indexing="ij")
    X2 = physical_grids_to_normalized_tensor(
        np.full_like(I_grid, reference_l),
        np.full_like(I_grid, reference_p),
        I_grid,
        tau_grid,
        config,
        dtype=dtype,
    )
    Z2 = _predict(model, f_inv, X2, I_grid.shape)
    i2 = np.unravel_index(np.nanargmax(Z2), Z2.shape)
    slice_I, slice_tau, slice_ce2 = I_axis[i2[0]], tau_axis[i2[1]], Z2[i2]

    # Shared 4D reference point: the unique intersection of the two slices.
    X_ref = physical_grids_to_normalized_tensor(
        np.array([reference_l]),
        np.array([reference_p]),
        np.array([reference_i]),
        np.array([reference_tau]),
        config,
        dtype=dtype,
    )
    with torch.no_grad():
        posterior_ref = model.posterior(X_ref)
        ref_mean_transformed = float(posterior_ref.mean.reshape(-1)[0].item())
        ref_std_transformed = float(
            posterior_ref.variance.clamp_min(0).sqrt().reshape(-1)[0].item()
        )

    ref_ce = float(f_inv(ref_mean_transformed))
    ref_ce_lower = float(f_inv(max(0.0, ref_mean_transformed - ref_std_transformed)))
    ref_ce_upper = float(f_inv(ref_mean_transformed + ref_std_transformed))

    # Shared color scaling across both panels; allow confirmed optimum to extend range.
    vmin = min(float(np.nanmin(Z1)), float(np.nanmin(Z2)))
    vmax = max(
        float(np.nanmax(Z1)),
        float(np.nanmax(Z2)),
        headline_ce / objective_scale,
        4.5,
    )
    if vmax <= vmin:
        vmax = vmin + 1e-9

    color_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    map_cmap = get_extended_cmap(v_ref=3.5, v_max=vmax)

    fig, (ax_lp, ax_it) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), sharey=False)

    im = ax_lp.pcolormesh(
        L_grid,
        p_grid,
        Z1,
        shading="auto",
        cmap=map_cmap,
        norm=color_norm,
        rasterized=True,
        linewidth=0.0,
        edgecolors="none",
        antialiased=False,
    )
    ax_it.pcolormesh(
        I_grid / 1e14,
        tau_grid,
        Z2,
        shading="auto",
        cmap=map_cmap,
        norm=color_norm,
        rasterized=True,
        linewidth=0.0,
        edgecolors="none",
        antialiased=False,
    )

    has_near_1 = has_near_2 = False
    if df_valid is not None and not df_valid.empty:
        I_range = np.diff(config["intensity_Wcm2_bounds"])[0]
        tau_range = np.diff(config["tau_fs_bounds"])[0]
        L_range = np.diff(config["Lzzr_bounds"])[0]
        p_range = np.diff(config["pzr_bounds"])[0]

        d1 = np.sqrt(
            ((df_valid["intensity_Wcm2"] - reference_i) / I_range) ** 2
            + ((df_valid["tau_fs"] - reference_tau) / tau_range) ** 2
        )
        d2 = np.sqrt(
            ((df_valid["Lzzr"] - reference_l) / L_range) ** 2
            + ((df_valid["pzr"] - reference_p) / p_range) ** 2
        )

        has_near_1 = _nearby_points(
            ax_lp,
            df_valid,
            d1,
            lambda d: d["Lzzr"],
            lambda d: d["pzr"],
            epsilon,
            color_norm,
            objective_scale,
            map_cmap,
        )
        has_near_2 = _nearby_points(
            ax_it,
            df_valid,
            d2,
            lambda d: d["intensity_Wcm2"] / 1e14,
            lambda d: d["tau_fs"],
            epsilon,
            color_norm,
            objective_scale,
            map_cmap,
        )

    # Markers
    ax_lp.plot(
        slice_L,
        slice_p,
        marker="D",
        color=SLICE_OPT_COLOR,
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=0.9,
        linestyle="",
        zorder=90,
        clip_on=False,
    )
    ax_it.plot(
        slice_I / 1e14,
        slice_tau,
        marker="D",
        color=SLICE_OPT_COLOR,
        markersize=8,
        markeredgecolor="black",
        markeredgewidth=0.9,
        linestyle="",
        zorder=90,
        clip_on=False,
    )
    ax_lp.plot(
        best_L,
        best_p,
        marker="x",
        color="red",
        markersize=10,
        markeredgewidth=1.8,
        linestyle="",
        zorder=100,
        clip_on=False,
    )
    ax_it.plot(
        best_I / 1e14,
        best_tau,
        marker="x",
        color="red",
        markersize=10,
        markeredgewidth=1.8,
        linestyle="",
        zorder=100,
        clip_on=False,
    )

    # Axis formatting
    format_map_axis(ax_lp, show_ylabel=True)
    ax_lp.xaxis.labelpad = 1

    ax_it.set_xlim(np.array(config["intensity_Wcm2_bounds"]) / 1e14)
    ax_it.set_ylim(config["tau_fs_bounds"])
    ax_it.set_xlabel(r"$I$ [$10^{14}$ W cm${^{-2}}$]", labelpad=1)
    ax_it.set_ylabel(r"$\tau$ [fs]", labelpad=2)
    ax_it.xaxis.set_major_locator(FixedLocator([0, 2.5, 5, 7.5, 10]))
    ax_it.xaxis.set_major_formatter(FixedFormatter(["0", "2.5", "5", "7.5", "10"]))
    ax_it.xaxis.set_minor_locator(MultipleLocator(1.25))
    ax_it.yaxis.set_major_locator(FixedLocator([20, 40, 60, 80]))
    ax_it.yaxis.set_minor_locator(MultipleLocator(10))
    # The larger panel gap keeps the label clear of panel (a).
    # Place tau near the middle of the gap, with comfortable separation from both squares.
    ax_it.yaxis.set_label_coords(-0.09, 0.5)  # close to panel b; clear of panel a

    for ax in (ax_lp, ax_it):
        ax.tick_params(direction="in", top=True, right=True)
        ax.tick_params(which="minor", direction="in", top=True, right=True)
        ax.grid(False, which="major")
        ax.grid(False, which="minor")

    # Manual panel placement
    panel_w = PANEL_SIZE_IN / FIG_WIDTH
    panel_h = PANEL_SIZE_IN / FIG_HEIGHT
    gap = PANEL_GAP_IN / FIG_WIDTH
    left0 = PANEL_LEFT
    bottom = PANEL_BOTTOM

    ax_lp.set_position([left0, bottom, panel_w, panel_h])
    ax_it.set_position([left0 + panel_w + gap, bottom, panel_w, panel_h])

    for ax in (ax_lp, ax_it):
        ax.set_box_aspect(1)

    _add_panel_label(ax_lp, "a", fontsize=FONT_SIZE)
    _add_panel_label(ax_it, "b", fontsize=FONT_SIZE)

    ax_lp.set_title(
     rf"$I={reference_i / 1e14:g}\times10^{{14}}\,\mathrm{{W\,cm^{{-2}}}},\ "
     rf"\tau={reference_tau:g}\,\mathrm{{fs}}$",
     pad=TITLE_PAD,
    )


    ax_it.set_title(
        rf"$L/z_R={reference_l:.4f}$, "
        rf"$p z_R={_fmt_pzr_title(reference_p)}$ mbar cm",
        pad=TITLE_PAD,
    )

    # Legend text
    if ce_band is not None:
        gp_opt_text = _fmt_ce_asym(best_ce, ce_band[0], ce_band[1], objective_scale)
    else:
        gp_opt_text = _fmt_ce(best_ce, objective_scale)

    # Stack longer labels into short lines so the side legend stays compact.
    if confirmed_ce is not None:
        global_text = (
            "4D optimum projection:"
            + "\nHarP " + _fmt_ce(confirmed_ce, objective_scale)
            + "\nGP " + gp_opt_text
        )
    else:
        global_text = "4D optimum projection:\nGP " + gp_opt_text

    ref_gp_text = _fmt_ce_asym(
        ref_ce * objective_scale,
        ref_ce_lower * objective_scale,
        ref_ce_upper * objective_scale,
        objective_scale,
    )
    if reference_confirmed_ce is not None:
        reference_text = (
            f"{REFERENCE_LABEL}: HarP "
            + _fmt_ce(reference_confirmed_ce, objective_scale)
            + ", GP "
            + ref_gp_text
        )
    else:
        reference_text = f"{REFERENCE_LABEL}: GP " + ref_gp_text

    slice_text = (
        "Slice optima:"
        + "\n" + rf"(a) {_fmt_ce(slice_ce1 * objective_scale, objective_scale)}"
        + "\n" + rf"(b) {_fmt_ce(slice_ce2 * objective_scale, objective_scale)}"
    )

    sample_h = _marker("none", "o", 7, "white", edgewidth=0.7)
    cross_h = plt.Line2D(
        [], [], marker="x", linestyle="", color="red", markersize=10, markeredgewidth=1.8
    )
    diamond_h = _marker(SLICE_OPT_COLOR, "D", 8, "black", edgewidth=0.9)

    shared_entries = []
    if has_near_1 or has_near_2:
        shared_entries.append((sample_h, rf"Nearby BO points ($\epsilon < {epsilon:g}$)"))
    shared_entries.extend([
        (cross_h, global_text),
        (diamond_h, slice_text),
        # (_marker("white", "s", 7, "black", edgewidth=1.0), reference_text),
    ])

    # Right-hand legend, aligned to the height of the square panels.
    # The full asymmetric superscript/subscript GP uncertainty is kept on ONE
    # mathematical line.  Its preceding heading occupies a separate line.
    legend_left = LEGEND_LEFT_IN / FIG_WIDTH
    legend_width = (LEGEND_RIGHT_IN - LEGEND_LEFT_IN) / FIG_WIDTH
    legend_ax = fig.add_axes([legend_left, bottom, legend_width, panel_h])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_facecolor("#b9c7d9")
    legend_ax.patch.set_alpha(0.95)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    # Legend layout: consistent spacing within entries and between groups,
    # with balanced vertical margins.
    marker_x     = 0.065       # horizontal marker position
    label_x      = 0.125        # horizontal text position
    row_step     = 0.10        # spacing between lines within an entry
    group_gap    = 0.16       # spacing between legend groups
    nearby_y     = 0.86        # vertical position of the first group
    projection_y = nearby_y - row_step - group_gap
    slice_y     = projection_y - row_step - group_gap

    if has_near_1 or has_near_2:
        legend_ax.plot(marker_x, nearby_y, marker="o", markersize=7,
                       markerfacecolor="none", markeredgecolor="white",
                       linestyle="", markeredgewidth=0.9)
        legend_ax.text(label_x, nearby_y, "Nearby BO points",
                       ha="left", va="center", fontsize=FONT_SIZE)
        legend_ax.text(label_x, nearby_y - row_step,
                       rf"($\epsilon < {epsilon:g}$)",
                       ha="left", va="center", fontsize=FONT_SIZE)

    legend_ax.plot(marker_x, projection_y, marker="x", color="red",
                   markersize=9, markeredgewidth=1.7, linestyle="")
    legend_ax.text(label_x, projection_y, "4D optimum projection:",
                   ha="left", va="center", fontsize=FONT_SIZE)
    # Show only the GP prediction at the red-cross location.
    # The confirmatory HarP result is still computed and printed by the runner.
    legend_ax.text(label_x, projection_y - row_step, gp_opt_text,
                   fontsize=FONT_SIZE, ha="left", va="center")

    legend_ax.plot(marker_x, slice_y, marker="D", color="white",
                   markeredgecolor="black", markeredgewidth=0.9,
                   markersize=7, linestyle="")
    legend_ax.text(label_x, slice_y, "Slice optima:",
                   fontsize=FONT_SIZE, ha="left", va="center")
    legend_ax.text(label_x, slice_y - row_step,
                   rf"(a) {_fmt_ce(slice_ce1 * objective_scale, objective_scale)}",
                   fontsize=FONT_SIZE, ha="left", va="center")
    legend_ax.text(label_x, slice_y - 2 * row_step,
                   rf"(b) {_fmt_ce(slice_ce2 * objective_scale, objective_scale)}",
                   fontsize=FONT_SIZE, ha="left", va="center")

    # Vertical CE scale aligned EXACTLY with the height of the square panels.
    # Colorbar has its own narrow column between the two squares and legend.
    cax = fig.add_axes([COLORBAR_LEFT_IN / FIG_WIDTH, bottom,
                        COLORBAR_WIDTH_IN / FIG_WIDTH, panel_h])
    bar = fig.colorbar(im, cax=cax, orientation="vertical")
    bar.ax.yaxis.set_ticks_position("right")
    bar.ax.tick_params(direction="in", labelsize=FONT_SIZE, pad=3)
    bar.set_label(r"Predicted CE / $10^{-6}$", fontsize=FONT_SIZE,
                  rotation=90, labelpad=5)
    bar.ax.yaxis.set_label_position("left")

    unify_figure_fonts(fig, FONT_SIZE)


    base = Path(out_path_base)
    base.parent.mkdir(parents=True, exist_ok=True)

    # Export the exact canvas geometry; cropping would alter paper proportions.
    fig.savefig(
      base.with_suffix(".pdf"),
      dpi=600,
      bbox_inches=None,
    )

    fig.savefig(
      base.with_suffix(".png"),
      dpi=600,
      bbox_inches=None,
   )
    plt.close(fig)
    print(f"Saved: {base.with_suffix('.pdf')} and {base.with_suffix('.png')}")

