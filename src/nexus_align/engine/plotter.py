"""Training curves: metric figures drawn in the background, off the training process."""

import os
import math
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from nexus_align.utils.plot import (
    FIGURE_DIR, LINE_COLOR, X_AXES, set_plot_style, save_figure
)


# Record keys that are x axes or redundant duplicates, never drawn as a curve.
AXIS_KEYS = ("epoch", "step", "total_step")
SKIP_SUFFIXES = ("_mean", "_peak", "_total")


def metric_keys(records: list[dict]) -> list[str]:
    """List the metrics to draw, in the order the records introduce them."""
    keys = []
    for record in records:
        for key, value in record.items():
            if key in AXIS_KEYS or key.endswith(SKIP_SUFFIXES) or key in keys:
                continue
            if isinstance(value, (int, float)):
                keys.append(key)
    return keys


def axis_points(records: list[dict], axis: dict) -> list[dict]:
    """Sort the records by this x value, one record per x; epochs take the end-of-epoch ones."""
    points = {}
    for record in records:
        if axis["key"] == "epoch" and record.get("step") != 0:
            continue
        x = record.get(axis["key"])
        if x is not None:
            points[x] = record  # a repeated x keeps the latest record
    return [points[x] for x in sorted(points)]


def draw_metric(ax, points: list[dict], key: str, axis: dict) -> None:
    """Draw one metric against one x axis on one sub-figure."""
    values = [(p[axis["key"]], p[key]) for p in points
              if isinstance(p.get(key), (int, float)) and not math.isnan(p[key])]
    ax.plot([x for x, _ in values], [y for _, y in values], linewidth=1.2, color=LINE_COLOR)
    ax.set_xlabel(axis["label"])
    ax.set_ylabel(key)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # steps and epochs are integers
    ax.margins(x=0.03, y=0.12)


def plot_axis(records: list[dict], out_dir: str, axis: dict) -> list[str]:
    """Draw every metric against one x axis: one combined figure and one per metric."""
    points = axis_points(records, axis)
    keys = metric_keys(points)
    if len(points) < 2 or not keys:
        return []

    ncols = min(3, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig = Figure(figsize=(7 * ncols, 4.5 * nrows))
    axes = [ax for row in fig.subplots(nrows, ncols, squeeze=False) for ax in row]
    for ax, key in zip(axes, keys):
        draw_metric(ax, points, key, axis)
    for ax in axes[len(keys):]:  # hide the unused cells of the last row
        ax.axis("off")
    paths = [save_figure(fig, out_dir, f"{axis['name']}.png")]

    for key in keys:
        fig = Figure(figsize=(8, 5))
        draw_metric(fig.subplots(), points, key, axis)
        paths.append(save_figure(fig, out_dir, f"{key}_{axis['name']}.png"))
    return paths


def draw_figures(records: list[dict], fig_dir: str) -> list[str]:
    """Draw every figure from the records; runs in the drawing process."""
    set_plot_style()
    paths = []
    for axis in X_AXES:
        paths += plot_axis(records, fig_dir, axis)
    return paths


class MetricPlotter:
    """
    Background plotter of the training metric curves (rank 0 only).

    - update() only appends the latest record, so the training loop never blocks.
    - A daemon thread redraws every figure at most once every interval seconds, in a
      child process, as drawing in the training process holds its GIL for seconds and
      stalls the kernel launches of training.
    - close() draws the final figures and stops the thread.
    """

    def __init__(self, log_dir: str, interval: float = 300.0) -> None:
        self.fig_dir = os.path.join(log_dir, FIGURE_DIR)
        self.interval = interval
        self.records = []
        self.drawn = 0
        self.lock = threading.Lock()  # guards the records, never held while drawing
        self.draw_lock = threading.Lock()  # serializes the drawing itself
        self._stop = threading.Event()
        self._thread = None
        self._pool = None

    def start(self) -> None:
        """Prepare the figure dir and start the background drawing thread."""
        if self._thread is not None:
            return
        os.makedirs(self.fig_dir, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        self._pool = ProcessPoolExecutor(1, mp_context=context)
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"📌 Metric figures refreshed every {self.interval:.0f}s in <{self.fig_dir}>")

    def update(self, record: dict) -> None:
        """Append one log record; the drawing itself happens in the background."""
        with self.lock:
            self.records.append(dict(record))

    def _worker(self) -> None:
        """Redraw on every wake-up until close() stops the thread."""
        while not self._stop.wait(self.interval):
            self.draw()

    def draw(self) -> list[str]:
        """Redraw every figure from the records collected so far."""
        with self.draw_lock:
            with self.lock:
                records = list(self.records)
            if len(records) == self.drawn:
                return []
            if self._pool is None:
                paths = draw_figures(records, self.fig_dir)
            else:
                paths = self._pool.submit(draw_figures, records, self.fig_dir).result()
            self.drawn = len(records)  # only after success, so a failed draw is retried
            return paths

    def close(self) -> None:
        """Stop the drawing thread, then draw the final figures."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        paths = self.draw()
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if paths:
            print(f"💾 Saved {len(paths)} metric figures to <{self.fig_dir}>")
