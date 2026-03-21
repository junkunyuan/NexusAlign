"""Draw command: plot metrics from results.jsonl to results.png."""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Exclude these from being metrics (they are index/step fields)
INDEX_KEYS = {"epoch", "step", "total_step"}

# Default nice blue: RGB (59, 130, 246) -> normalized
DEFAULT_COLOR = (59 / 255, 130 / 255, 246 / 255)


def _load_results(path: Path) -> list[dict]:
    """Load JSONL results file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _get_metrics(records: list[dict]) -> list[str]:
    """Get all metric keys (exclude epoch, step, total_step)."""
    if not records:
        return []
    keys = set(records[0].keys())
    return sorted(k for k in keys if k not in INDEX_KEYS)


def _parse_color(color_str: str) -> tuple[float, float, float]:
    """Parse 'r,g,b' (0-255) to normalized (0-1) tuple."""
    parts = [int(x.strip()) for x in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("❌ Color must be three comma-separated integers (0-255)")
    return tuple(max(0, min(255, int(x))) / 255 for x in parts)


def _deduplicate_by_x(records: list[dict], x_key: str, y_keys: list[str]) -> tuple[list, dict[str, list]]:
    """
    When x_key values are not unique, keep last occurrence per unique x.
    Returns (x_values, {metric: y_values}).
    """
    x_to_idx: dict = {}
    x_list = []
    y_lists = {k: [] for k in y_keys}

    for rec in records:
        x_val = rec.get(x_key)
        if x_val is None:
            continue
        if x_val in x_to_idx:
            idx = x_to_idx[x_val]
            for k in y_keys:
                y_lists[k][idx] = rec.get(k)
        else:
            x_to_idx[x_val] = len(x_list)
            x_list.append(x_val)
            for k in y_keys:
                y_lists[k].append(rec.get(k))

    return x_list, y_lists


def _compute_ema(values: list[float], alpha: float) -> list[float]:
    """
    Compute Exponential Moving Average.
    EMA_t = alpha * EMA_{t-1} + (1 - alpha) * value_t
    alpha: smoothing factor, 0 < alpha < 1. Higher alpha = more smoothing.
    """
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * result[-1] + (1 - alpha) * v)
    return result


def draw(
    result_path: Path,
    x_data: str = "total_step",
    y_data: list[str] | None = None,
    color: tuple[float, float, float] = DEFAULT_COLOR,
    fontname: str = "Times New Roman",
    ema_alpha: float | None = None,
    dpi: int = 150,
    fontsize: int = 15,
) -> Path:
    """
    Plot metrics from results.jsonl and save to results.png in the same directory.

    Args:
        result_path: Path to results.jsonl.
        x_data: Key for x-axis (default: total_step).
        y_data: List of metric keys for y-axis (default: all except epoch/step/total_step).
        color: RGB tuple (0-1) for line color.
        fontname: Font family for plot (default: Times New Roman).
        ema_alpha: If set (0 < alpha < 1), plot EMA overlay; raw data becomes semi-transparent.
        dpi: Output image DPI (default: 150).
        fontsize: Base font size (default: 15).
    Returns:
        Path to saved results.png
    """
    records = _load_results(result_path)
    if not records:
        raise ValueError(f"❌ No valid records in <{result_path}>")

    all_metrics = _get_metrics(records)
    if not all_metrics:
        raise ValueError(f"❌ No metrics found in <{result_path}>")

    y_keys = y_data if y_data is not None else all_metrics
    # Validate y_keys
    for k in y_keys:
        if k not in all_metrics:
            raise ValueError(f"❌ Unknown metric '{k}'. Available: {all_metrics}")

    if x_data not in records[0]:
        raise ValueError(
            f"❌ X-axis key '{x_data}' not found. Available: {list(records[0].keys())}"
        )

    x_list, y_lists = _deduplicate_by_x(records, x_data, y_keys)

    n_plots = len(y_keys)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    plt.rcParams["font.family"] = fontname
    plt.rcParams["font.size"] = fontsize

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, metric in enumerate(y_keys):
        ax = axes_flat[idx]
        y_vals = y_lists[metric]
        # Filter out None
        valid_pairs = [(x, y) for x, y in zip(x_list, y_vals) if y is not None]
        if not valid_pairs:
            ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center")
            ax.set_title(metric, fontname=fontname)
            continue

        xs, ys = zip(*valid_pairs)
        ys_list = list(ys)
        if ema_alpha is not None:
            ax.plot(
                xs, ys_list,
                color=color, linewidth=1.5, alpha=0.35, label=f"raw"
            )
            ema_vals = _compute_ema(ys_list, ema_alpha)
            ax.plot(
                xs, ema_vals,
                color=color, linewidth=2.5, alpha=1.0, label=f"EMA"
            )
        else:
            ax.plot(xs, ys_list, color=color, linewidth=2, label=metric)
        ax.set_xlabel(x_data, fontname=fontname, fontsize=fontsize)
        ax.set_ylabel(metric, fontname=fontname, fontsize=fontsize)
        # ax.set_title(metric, fontname=fontname, fontsize=fontsize)
        # ax.legend(loc="best", prop={"family": fontname})
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fontsize)

    # Hide unused subplots
    for idx in range(len(y_keys), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=2.0)
    out_path = result_path.parent / "results.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def run_draw(args: argparse.Namespace) -> int:
    """Run draw with parsed args."""
    result_path = Path(args.result).resolve()
    if not result_path.exists():
        print(f"❌ Error: File not found: <{result_path}>")
        return 1

    y_data = None
    if args.y_data:
        y_data = [k.strip() for k in args.y_data.split(",") if k.strip()]

    color = DEFAULT_COLOR
    if args.color:
        try:
            color = _parse_color(args.color)
        except ValueError as e:
            print(f"❌ Error: {e}")
            return 1

    ema_alpha = getattr(args, "ema", None)
    if ema_alpha is not None and not (0 < ema_alpha < 1):
        print("❌ Error: --ema must be between 0 and 1 (exclusive)")
        return 1

    try:
        out_path = draw(
            result_path=result_path,
            x_data=args.x_data,
            y_data=y_data,
            color=color,
            ema_alpha=ema_alpha,
            dpi=args.dpi,
            fontsize=args.fontsize,
        )
        print(f"💾 Saved: <{out_path}>")
        return 0
    except (ValueError, KeyError) as e:
        print(f"❌ Error: {e}")
        return 1


def main() -> int:
    """CLI entry for draw command (standalone)."""
    parser = argparse.ArgumentParser(description="Plot metrics")
    parser.add_argument(
        "--result",
        "-r",
        type=str,
        required=True,
        help="Path to results.jsonl file",
    )
    parser.add_argument(
        "--x_data",
        "-x",
        type=str,
        default="total_step",
        help="Key for x-axis (default: total_step). Options: epoch, step, total_step, etc.",
    )
    parser.add_argument(
        "--y_data",
        "-y",
        type=str,
        default=None,
        help="Comma-separated metric keys for y-axis (default: all metrics), e.g., loss,lr",
    )
    parser.add_argument(
        "--color",
        "-c",
        type=str,
        default=None,
        help="Line color as RGB 0-255, comma-separated (default: blue), e.g., 59,130,246",
    )
    parser.add_argument(
        "--ema",
        "-e",
        type=float,
        default=0.8,
        metavar="ALPHA",
        help="EMA smoothing factor (0 < alpha < 1). Raw data is semi-transparent and EMA is overlaid (default: 0.8).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI (default: 150).",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=15,
        help="Base font size (default: 15).",
    )

    args = parser.parse_args()
    return run_draw(args)
