"""Plot utilities: set the shared style and save the figures."""

import os
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Times New Roman first. The rest are metric-compatible clones for hosts without it.
FONT_SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"]

# Subdirectory of a log or evaluation directory that holds the drawn figures.
FIGURE_DIR = "figures"

# Curve color shared by every metric, and the highlight color of the best point.
LINE_COLOR, BEST_COLOR = "#1f77b4", "#2ca02c"

# One combined figure per x axis. The epoch figure takes only end-of-epoch records.
X_AXES = (
    {"key": "epoch", "name": "epoch", "label": "training epochs"},
    {"key": "total_step", "name": "step", "label": "training steps"},
)


def set_plot_style() -> None:
    """Use a Times New Roman serif style with readable font sizes."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_SERIF,
        "mathtext.fontset": "stix",
        "grid.alpha": 0.35,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })


def grid_figure(n: int, width: float, height: float, max_cols: int = 3):
    """Make a figure of n sub-figures, max_cols per row. Return it and the n axes."""
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    fig = Figure(figsize=(width * cols, height * rows))
    axes = fig.subplots(rows, cols, squeeze=False).flatten()
    for ax in axes[n:]:  # hide the unused cells of the last row
        ax.set_axis_off()
    return fig, axes[:n]


def save_figure(fig, out_dir: str, file_name: str) -> str:
    """Save one figure and close it."""
    fig.tight_layout()
    path = os.path.join(out_dir, file_name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
