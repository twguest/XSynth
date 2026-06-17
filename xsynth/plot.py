from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _as_2d_scan_points(scan_output) -> np.ndarray:
    scan_points = getattr(scan_output, "scan_points_executed", None)
    if scan_points is None:
        scan_points = getattr(scan_output, "scan_points", None)

    scan_points = np.asarray(scan_points, dtype=float)
    if scan_points.ndim == 1:
        scan_points = scan_points.reshape(-1, 1)

    return scan_points


def _signal_matrix(scan_output, generator_index: int) -> np.ndarray:
    signals = getattr(scan_output, "signals", None)
    if not signals:
        raise ValueError("scan_output.signals is empty. Execute the scan before plotting.")

    traces = [np.asarray(point_signals[generator_index], dtype=float).reshape(-1) for point_signals in signals]
    lengths = {len(trace) for trace in traces}
    if len(lengths) != 1:
        raise ValueError("Cannot summarize signals with different lengths.")

    return np.vstack(traces)


def _time_axis(scan_output, generator_index: int, signal_length: int) -> np.ndarray:
    generators = getattr(scan_output, "generators", [])
    if generator_index < len(generators):
        t = np.asarray(getattr(generators[generator_index], "t", []), dtype=float).reshape(-1)
        if len(t) == signal_length:
            return t

    return np.arange(signal_length, dtype=float)


def _default_label(scan_output, generator_index: int) -> str:
    generators = getattr(scan_output, "generators", [])
    if generator_index < len(generators):
        generator = generators[generator_index]
        server = getattr(generator, "server", getattr(generator, "kicker", None))
        label = getattr(server, "__name__", None)
        if label:
            return str(label)

    return f"Signal {generator_index + 1}"


def _scan_label(scan_output, index: int) -> str:
    scan_variables = getattr(scan_output, "scan_variables", [])
    if index < len(scan_variables):
        return str(scan_variables[index])
    return f"scan variable {index + 1}"


def _plot_signal_family(ax, time, signal_matrix, color_values, color_label, title, cmap):
    norm = Normalize(vmin=float(np.min(color_values)), vmax=float(np.max(color_values)))
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)

    for trace, color_value in zip(signal_matrix, color_values):
        ax.plot(time, trace, color=scalar_mappable.to_rgba(color_value), alpha=0.75, linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("time / pulseId")
    ax.set_ylabel("signal")
    ax.grid(True, alpha=0.25)

    colorbar = ax.figure.colorbar(scalar_mappable, ax=ax)
    colorbar.set_label(color_label)


def plot_scan_summary(
    scan_output,
    generator_labels: Sequence[str] | None = None,
    max_phase_traces: int = 30,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
):
    """Plot a summary of a completed Scan/MeshScan output.

    For a one-variable scan, this plots signal traces over time and colors each
    trace by scan-point index.

    For a two-variable scan, such as an IBFB X/Y scan, this creates three
    panels: signal family for generator 1, signal family for generator 2, and a
    signal-1-vs-signal-2 scatter panel. All traces/points are colored by
    scan-point index.

    Parameters
    ----------
    scan_output:
        A completed ``MeshScan``/``MiniScan`` output with ``signals`` and
        ``scan_points_executed`` populated.
    generator_labels:
        Optional labels for generator/signal panels.
    max_phase_traces:
        Maximum number of scan-point clouds to overlay for two-dimensional
        signal-vs-signal summaries.
    cmap:
        Matplotlib colormap name.
    figsize:
        Optional figure size.

    Returns
    -------
    tuple
        ``(fig, axes)`` from Matplotlib.
    """

    scan_points = _as_2d_scan_points(scan_output)
    if scan_points.size == 0:
        raise ValueError("scan_output does not contain scan points.")

    n_scan_dims = scan_points.shape[1]
    scan_index = np.arange(len(scan_points), dtype=float)
    n_generators = len(getattr(scan_output, "generators", []))
    if n_generators == 0:
        n_generators = len(scan_output.signals[0])

    labels = list(generator_labels or [])
    while len(labels) < n_generators:
        labels.append(_default_label(scan_output, len(labels)))

    if n_scan_dims == 1 or n_generators == 1:
        signal_matrix = _signal_matrix(scan_output, 0)
        time = _time_axis(scan_output, 0, signal_matrix.shape[1])

        fig, ax = plt.subplots(1, 1, figsize=figsize or (8, 5))
        _plot_signal_family(
            ax,
            time,
            signal_matrix,
            scan_index,
            "scan point",
            labels[0],
            cmap,
        )
        fig.tight_layout()
        return fig, ax

    if n_scan_dims < 2 or n_generators < 2:
        raise ValueError("Two-dimensional scan summaries require at least two scan variables and two generators.")

    signal_a = _signal_matrix(scan_output, 0)
    signal_b = _signal_matrix(scan_output, 1)
    time_a = _time_axis(scan_output, 0, signal_a.shape[1])
    time_b = _time_axis(scan_output, 1, signal_b.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=figsize or (16, 4.8))

    _plot_signal_family(
        axes[0],
        time_a,
        signal_a,
        scan_index,
        "scan point",
        labels[0],
        cmap,
    )
    _plot_signal_family(
        axes[1],
        time_b,
        signal_b,
        scan_index,
        "scan point",
        labels[1],
        cmap,
    )

    shared_length = min(signal_a.shape[1], signal_b.shape[1], len(time_a), len(time_b))
    norm = Normalize(vmin=float(np.min(scan_index)), vmax=float(np.max(scan_index)))
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    trace_indices = np.linspace(
        0,
        len(scan_points) - 1,
        min(max_phase_traces, len(scan_points)),
        dtype=int,
    )

    for index in np.unique(trace_indices):
        color = scalar_mappable.to_rgba(float(index))
        axes[2].scatter(
            signal_a[index, :shared_length],
            signal_b[index, :shared_length],
            color=color,
            alpha=0.45,
            s=12,
            linewidths=0,
        )

    axes[2].set_title(f"{labels[0]} vs {labels[1]}")
    axes[2].set_xlabel(labels[0])
    axes[2].set_ylabel(labels[1])
    axes[2].grid(True, alpha=0.25)
    axes[2].autoscale()

    colorbar = fig.colorbar(scalar_mappable, ax=axes[2])
    colorbar.set_label("scan point")

    fig.tight_layout()
    return fig, axes


scan_summary = plot_scan_summary
