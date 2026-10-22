"""Plot utilities: the shared look and the saving of every drawn figure."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Times New Roman first; the rest are metric-compatible clones for hosts without it.
FONT_SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"]

# Sub-directory of a log dir holding the drawn figures.
FIGURE_DIR = "figures"

# Curve color shared by every metric, and the highlight color of the best point.
LINE_COLOR, BEST_COLOR = "#1f77b4", "#2ca02c"

# One combined figure per x axis; the epoch one only takes the end-of-epoch records.
X_AXES = (
    {"key": "total_step", "name": "step", "label": "training steps"},
    {"key": "epoch", "name": "epoch", "label": "training epochs"},
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


def save_figure(fig, out_dir: str, file_name: str) -> str:
    """Save one figure and close it."""
    fig.tight_layout()
    path = os.path.join(out_dir, file_name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
