"""Evaluation summary: aggregate per-checkpoint metrics into a JSON file and curves."""

import json
import math
import os

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from nexus_align.utils.plot import (
    FIGURE_DIR, LINE_COLOR, BEST_COLOR, X_AXES, set_plot_style, save_figure
)


# One figure per metric: which result keys to draw and how to name and label them.
METRIC_PLOTS = (
    {"key": "fid", "std_key": None, "name": "fid", "label": "FID ↓", "lower_is_better": True},
    {"key": "is_mean", "std_key": "is_std", "name": "is", "label": "Inception Score ↑",
     "lower_is_better": False},
)

MAX_COLS = 3  # sub-figures per row of a combined figure


def fid_is_result(metrics: dict) -> dict | None:
    """Map the FID/IS metrics of one checkpoint to result keys; None without FID."""
    if metrics.get("frechet_inception_distance") is None:
        return None
    return {
        "fid": metrics["frechet_inception_distance"],
        "is_mean": metrics.get("inception_score_mean"),
        "is_std": metrics.get("inception_score_std"),
    }


def collect_results(
    ckpt_metrics: list[tuple[dict, str, str]], parse=fid_is_result
) -> list[dict]:
    """Read the metrics of already-evaluated checkpoints, sorted by total training step."""
    results = []
    for train_state, ckpt, metrics_file in ckpt_metrics:
        if not os.path.exists(metrics_file):
            continue
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"⚠️ WARNING: could not parse <{metrics_file}>: {e}")
            continue
        result = parse(metrics)
        if result is None:
            continue
        results.append({**train_state, "ckpt": os.path.relpath(ckpt), **result})
    results.sort(key=lambda r: r["total_step"])
    return results


def save_summary(results: list[dict], out_dir: str) -> str:
    """Save all collected results to a single JSON file."""
    path = os.path.join(out_dir, "eval_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


def draw_metric(ax, results: list[dict], spec: dict, axis: dict) -> None:
    """Draw one metric against one x axis on one sub-figure."""
    points = [r for r in results if r.get(spec["key"]) is not None]
    x_values = [r[axis["key"]] for r in points]
    values = [r[spec["key"]] for r in points]

    ax.plot(x_values, values, marker="o", markersize=6, linewidth=1.6, color=LINE_COLOR)

    stds = [r.get(spec["std_key"]) for r in points] if spec["std_key"] else []
    if stds and all(s is not None for s in stds):
        ax.fill_between(x_values, [v - s for v, s in zip(values, stds)],
                        [v + s for v, s in zip(values, stds)],
                        color=LINE_COLOR, alpha=0.15, linewidth=0, label="± std")

    # Label a thinned subset when the points get dense, keeping the best and the last one.
    pick = min if spec["lower_is_better"] else max
    best = pick(range(len(values)), key=lambda i: values[i])
    stride = max(1, round(len(values) / 10))
    marked = {best, len(values) - 1}
    labeled = marked | {i for i in range(0, len(values), stride)
                        if all(abs(i - j) >= stride for j in marked)}
    for i in sorted(labeled):
        ax.annotate(f"{values[i]:.2f}", (x_values[i], values[i]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=13, color=LINE_COLOR)

    ax.scatter([x_values[best]], [values[best]], s=180, facecolors="none", edgecolors=BEST_COLOR,
               linewidths=1.8, zorder=5,
               label=f"best = {values[best]:.2f} @ {x_values[best]} {axis['name']}s")

    ax.set_xlabel(axis["label"])
    ax.set_ylabel(spec["label"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # steps and epochs are integers
    ax.margins(x=0.06, y=0.14)
    ax.legend(loc="best", framealpha=0.9)


def plot_metric(results: list[dict], out_dir: str, spec: dict, axis: dict) -> str:
    """Draw one metric against one x axis on its own figure."""
    fig = Figure(figsize=(8, 5))
    draw_metric(fig.subplots(), results, spec, axis)
    return save_figure(fig, out_dir, f"{spec['name']}_{axis['name']}.png")


def axis_results(results: list[dict], axis: dict) -> list[dict]:
    """Keep the results carrying this x value; epochs only take end-of-epoch checkpoints."""
    return [r for r in results if r.get(axis["key"]) is not None
            and (axis["key"] != "epoch" or r.get("step") == 0)]


def plot_axis(
    results: list[dict], out_dir: str, axis: dict, metric_plots=METRIC_PLOTS
) -> list[str]:
    """Draw every metric against one x axis: one combined figure and one per metric."""
    points = axis_results(results, axis)
    specs = [s for s in metric_plots if any(r.get(s["key"]) is not None for r in points)]
    if not specs:
        return []

    cols = min(len(specs), MAX_COLS)
    rows = math.ceil(len(specs) / cols)
    fig = Figure(figsize=(8 * cols, 5 * rows))
    axes = fig.subplots(rows, cols, squeeze=False).flatten()
    for ax, spec in zip(axes, specs):
        draw_metric(ax, points, spec, axis)
    for ax in axes[len(specs):]:  # empty cells of the last row
        ax.set_axis_off()

    paths = [save_figure(fig, out_dir, f"{axis['name']}.png")]
    return paths + [plot_metric(points, out_dir, spec, axis) for spec in specs]


def plot_results(
    results: list[dict], out_dir: str, metric_plots=METRIC_PLOTS
) -> list[str]:
    """Draw the combined figure and the per-metric figures of every x axis."""
    if not results:
        return []
    set_plot_style()
    fig_dir = os.path.join(out_dir, FIGURE_DIR)
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for axis in X_AXES:
        paths += plot_axis(results, fig_dir, axis, metric_plots)
    return paths


def refresh_summary(
    ckpt_metrics: list[tuple[dict, str, str]],
    out_dir: str,
    parse=fid_is_result,
    metric_plots=METRIC_PLOTS,
) -> None:
    """Refresh the summary JSON file and the evaluation curves."""
    results = collect_results(ckpt_metrics, parse)
    summary_path = save_summary(results, out_dir)
    plot_paths = plot_results(results, out_dir, metric_plots)
    print(f"💾 Saved {len(results)} results to <{summary_path}>")
    for path in plot_paths:
        print(f"💾 Saved figure to <{path}>")
