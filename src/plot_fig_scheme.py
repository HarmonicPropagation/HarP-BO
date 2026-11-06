from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Journal-style font settings
#################################################################
#plt.rcParams.update({
    #"text.usetex": False,
   # "font.family": "STIXGeneral",
   # "mathtext.fontset": "stix",
   # "font.size": 10,
   # "axes.labelsize": 10,
   # "xtick.labelsize": 9,
   # "ytick.labelsize": 9,
   # "legend.fontsize": 9,
  #  "pdf.fonttype": 42,
 #   "ps.fonttype": 42,
#})

# Initial settings
#################################################################
steps = [
    "Initial\nCE evaluation",
    "Update\nGP",
    "Optimise\nacquisition",
    "Run HHG\nsimulation",
    "Extract\nCE",
    "Update\nGP",
]

# Initialization colors
init_light = "#FFF3CD"
init_dark  = "#8A6500"

# Repeated BO-loop colors
loop_light = "#E8F5E9"
loop_dark  = "#1B5E20"

# Arrow colors
forward_arrow = "#4A4A4A"
loop_arrow    = "#2E7D32"

# Color assigned to each box
facecolors = [
    init_light,
    init_light,
    loop_light,
    loop_light,
    loop_light,
    loop_light,
]

edgecolors = [
    init_dark,
    init_dark,
    loop_dark,
    loop_dark,
    loop_dark,
    loop_dark,
]

textcolors = edgecolors

# Plotting
#################################################################
fig, ax = plt.subplots(figsize=(12.5, 2.8))

ax.set_xlim(0, 12.5)
ax.set_ylim(0.65, 2.5)
ax.axis("off")

bw, bh = 1.55, 0.78
yc = 1.25
xs = [0.35, 2.35, 4.35, 6.35, 8.35, 10.35]
fs = 13

# Boxes
for x, label, facecolor, edgecolor, textcolor in zip(
    xs,
    steps,
    facecolors,
    edgecolors,
    textcolors,
):
    ax.add_patch(
        FancyBboxPatch(
            (x, yc - bh / 2),
            bw,
            bh,
            boxstyle="round,pad=0.05,rounding_size=0.22",
            linewidth=2,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
    )

    ax.text(
        x + bw / 2,
        yc,
        label,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="semibold",
        color=textcolor,
    )

# Forward arrows
for i in range(len(xs) - 1):
    ax.add_patch(
        FancyArrowPatch(
            (xs[i] + bw + 0.08, yc),
            (xs[i + 1] - 0.08, yc),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=2,
            color=forward_arrow,
        )
    )

# Repetition arrow
ax.add_patch(
    FancyArrowPatch(
        (xs[-1] + bw / 2, yc + bh / 2 + 0.03),
        (xs[2] + bw / 2, yc + bh / 2 + 0.03),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.1,
        color=loop_arrow,
        connectionstyle="arc3,rad=0.33",
    )
)

ax.text(
    8.15,
    2.02,
    r"Repeat until $N_{\mathrm{BO}}$ simulations",
    ha="center",
    va="center",
    fontsize=fs,
    fontweight="semibold",
    color=loop_dark,
)

# Saving the figure
#################################################################
PROJECT_ROOT = Path(__file__).resolve().parent.parent
plots_root = PROJECT_ROOT / "figures"
plots_root.mkdir(parents=True, exist_ok=True)

pdf = plots_root / "setup.pdf"
png = plots_root / "setup.png"

plt.tight_layout(pad=0.4)
plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
plt.close(fig)