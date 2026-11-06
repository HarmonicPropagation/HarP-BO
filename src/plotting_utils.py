"""
Shared plotting helpers for the 2D BO comparison figures.
This file contains only plotting-specific functionality that is reused across the figure scripts.
BO mathematics, coordinate transforms, metrics, and other functionalities remain
in utils.py or in the individual plotting scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator

# Common physical plotting domain used by the 2D figures
L_MIN            = 1.5 / 100
L_MAX            = 1.5
P_MIN            = 300 / 100
P_MAX            = 300.0

# Common BO-point styling
BO_CMAP          = "gist_heat_r"
INITIAL_MARKER   = "P"
INITIAL_COLOR    = "#D900E8"
BEST_MARKER      = "X"
BEST_BO_COLOR    = "red"
POINT_EDGE_COLOR = "white"
INITIAL_SIZE     = 6.5
INITIAL_LINEWIDTH = 0.45
INITIAL_OUTLINE_WIDTH = 3.5
BEST_SIZE        = 60
BO_SIZE_MIN      = 10
BO_SIZE_MAX      = 30

__all__          = ["apply_paper_style", "add_panel_label", "format_map_axis", "evaluation_sizes", "add_physical_points", "make_initial_marker"]

def apply_paper_style():
    # Apply the Matplotlib style shared by the paper figures.
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

def add_panel_label(ax, label):
    # Add an a), b), ... style panel label to an axes.
    ax.text(
        0.96,
        0.97,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        zorder=100,
    )


def format_map_axis(
    ax,
    show_ylabel=True,
    *,
    L_min=L_MIN,
    L_max=L_MAX,
    p_min=P_MIN,
    p_max=P_MAX,
):
    # Format a physical L/z_R versus p z_R map axis.
    ax.set_xlim(L_min, L_max)
    ax.set_ylim(p_min, p_max)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1.0, 1.5]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "1.5"]))
    ax.yaxis.set_major_locator(FixedLocator([100, 200, 300]))
    ax.yaxis.set_major_formatter(FixedFormatter(["100", "200", "300"]))
    ax.xaxis.set_minor_locator(FixedLocator([0.25, 0.75, 1.25]))
    ax.yaxis.set_minor_locator(FixedLocator([50, 150, 250]))
    ax.set_xlabel(r"$L/z_R$")
    if show_ylabel:
        ax.set_ylabel(r"$p z_R$ [mbar cm]")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.tick_params(direction="in", top=True, right=True)
    ax.tick_params(which="minor", direction="in", top=True, right=True)

def evaluation_sizes(
    n_points,
    size_min=BO_SIZE_MIN,
    size_max=BO_SIZE_MAX,
):
    # Return marker sizes that increase linearly with BO evaluation order.
    if n_points == 0:
        return np.array([], dtype=float)
    if n_points == 1:
        return np.array([size_min], dtype=float)
    return np.linspace(size_min, size_max, n_points)


def make_initial_marker(x=(), y=(), markersize=INITIAL_SIZE):
    """Create the same outlined magenta initial-point marker for plots and legends."""
    marker = Line2D(
        x, y,
        marker=INITIAL_MARKER,
        linestyle="None",
        markerfacecolor=INITIAL_COLOR,
        markeredgecolor="white",
        markersize=markersize,
        markeredgewidth=INITIAL_LINEWIDTH,
    )
    return marker


def add_physical_points(
    ax,
    df,
    *,
    show_best=True,
    bo_cmap=BO_CMAP,
    initial_marker=INITIAL_MARKER,
    initial_color=INITIAL_COLOR,
    best_marker=BEST_MARKER,
    best_color=BEST_BO_COLOR,
    point_edge_color=POINT_EDGE_COLOR,
    initial_size=INITIAL_SIZE,
    best_size=BEST_SIZE,
    bo_size_min=BO_SIZE_MIN,
    bo_size_max=BO_SIZE_MAX,
):
    """
    Plot evaluated BO points on a physical L, p prediction map.
    Initial points are shown as outlined magenta plus markers. BO evaluations are shown as circles
    whose size and color progress with evaluation order.
    Set show_best=False for figures, such as the Straddle comparison,
    where the best-point marker is intentionally omitted.
    """
    df_plot    = df.copy()
    if len(df_plot) == 0:
        return

    df_initial = df_plot[df_plot["phase"].isin(["initial", "shared_initial"])]
    df_bo      = (df_plot[df_plot["phase"] == "bo"].sort_values("iteration", kind="mergesort").copy())

    if len(df_initial) > 0:
        initial_artist = make_initial_marker(
            np.asarray(df_initial["Lzzr"]),
            np.asarray(df_initial["pzr"]),
            markersize=initial_size,
        )
        initial_artist.set_zorder(35)
        initial_artist.set_clip_on(False)
        ax.add_line(initial_artist)

    if len(df_bo) > 0:
        sizes = evaluation_sizes(
            len(df_bo),
            size_min=bo_size_min,
            size_max=bo_size_max,
        )
        evaluation_scale = np.linspace(0.0, 1.0, len(df_bo))

        ax.scatter(
            np.asarray(df_bo["Lzzr"]),
            np.asarray(df_bo["pzr"]),
            s=sizes,
            marker="o",
            c=evaluation_scale,
            cmap=bo_cmap,
            vmin=0.0,
            vmax=1.0,
            edgecolor=point_edge_color,
            linewidth=0.55,
            alpha=0.90,
            zorder=36,
            clip_on=False,
        )

    if show_best:
        best_bo = df_plot.loc[df_plot["objective_scaled_ratio"].idxmax()]
        ax.scatter(
            best_bo["Lzzr"],
            best_bo["pzr"],
            s=best_size,
            marker=best_marker,
            facecolor=best_color,
            edgecolor="white",
            linewidth=0.4,
            zorder=47,
            clip_on=False,
        )

